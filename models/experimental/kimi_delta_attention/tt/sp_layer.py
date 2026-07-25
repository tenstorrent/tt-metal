# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Two-span, four-way tensor-parallel KDA orchestration for LoudBox.

This module intentionally uses two physical TP=4 submeshes rather than
pretending an eight-device mesh is TP=8.  LoudBox is exposed as a 1x8 mesh, so
the groups are the first and second contiguous sets of four chips.  This is a
logical SP=2 x TP=4 topology.  The only data crossing the sequence boundary is
KDA's causal cache, transferred through fabric sockets directly from the first
TP group to the matching rank of the second group.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

import ttnn
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.tt_transformers.tt.ccl import TT_CCL

_SP_SIZE = 2
_TP_SIZE = 4


def _socket_config(mesh_shape: ttnn.MeshShape) -> ttnn.SocketConfig:
    """Create one rank-aligned fabric lane per LoudBox device.

    This is the established 1x8-to-two-1x4 socket shape used by the CCL
    send/receive test.  A single worker-core pair avoids the runtime warning
    and fabric-resource ambiguity associated with multiple sender cores on a
    device.  Striping the carry over the second physical link is deliberately
    deferred until the single-lane functional path is measured.
    """
    sender_cores = [ttnn.CoreCoord(0, 0)]
    receiver_cores = [ttnn.CoreCoord(0, 1)]
    connections = [
        ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, sender), ttnn.MeshCoreCoord(coord, receiver))
        for coord in ttnn.MeshCoordinateRange(mesh_shape)
        for sender, receiver in zip(sender_cores, receiver_cores, strict=True)
    ]
    return ttnn.SocketConfig(
        connections,
        ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, 10 * 1024),
    )


class SP2TP4KimiDeltaAttention:
    """Execute KDA over two ordered sequence spans on a 2x4 mesh.

    ``forward`` takes one already-device-resident activation tensor per span;
    both use a 1x4 submesh.  It returns the two TP=4 output shards in sequence
    order.  Callers concatenate only for host-side validation or later model
    integration—the KDA cache handoff itself never materializes on the host.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: KDAConfig,
        state_dict: Mapping[str, torch.Tensor],
        tensor_cache_path: Path | None = None,
    ) -> None:
        if tuple(mesh_device.shape) != (1, _SP_SIZE * _TP_SIZE):
            raise ValueError(f"SP2TP4 requires a 1x8 LoudBox mesh, got {tuple(mesh_device.shape)}")
        self.mesh_device = mesh_device
        self.span_devices = tuple(
            mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(0, span * _TP_SIZE))
            for span in range(_SP_SIZE)
        )
        self.layers = tuple(
            KimiDeltaAttention(
                span_device,
                config,
                state_dict,
                tensor_cache_path=tensor_cache_path,
                tt_ccl=TT_CCL(span_device),
            )
            for span_device in self.span_devices
        )
        socket_config = _socket_config(self.span_devices[0].shape)
        # A socket endpoint is a single ordered fabric stream.  Reuse it for
        # the recurrent cache and then the convolution history rather than
        # allocating competing socket pairs on the same worker core.
        self._send_socket, self._recv_socket = ttnn.create_socket_pair(
            self.span_devices[0], self.span_devices[1], socket_config
        )

    @property
    def first_layer(self) -> KimiDeltaAttention:
        return self.layers[0]

    @property
    def second_layer(self) -> KimiDeltaAttention:
        return self.layers[1]

    def reset_state(self, batch_size: int) -> None:
        for layer in self.layers:
            layer.reset_state(batch_size)

    def enable_trace_stable_state(self) -> None:
        """Keep both span caches at fixed addresses for captured execution.

        The sequence handoff overwrites the second span's cache before its
        recurrence consumes it, so fixed destination buffers are compatible
        with the causal ordering as well as required for trace replay.
        """
        for layer in self.layers:
            assert layer.recurrent_state is not None
            assert layer.convolution_state is not None
            layer.set_external_state(layer.recurrent_state, layer.convolution_state)

    def _handoff_causal_state(self) -> None:
        """Copy the completed first-span cache into the second-span cache."""
        source, destination = self.first_layer, self.second_layer
        assert source.recurrent_state is not None
        assert source.convolution_state is not None
        assert destination.recurrent_state is not None
        assert destination.convolution_state is not None
        ttnn.experimental.send_async(source.recurrent_state, self._send_socket)
        ttnn.experimental.recv_async(destination.recurrent_state, self._recv_socket)
        ttnn.experimental.send_async(source.convolution_state, self._send_socket)
        ttnn.experimental.recv_async(destination.convolution_state, self._recv_socket)

    def forward(
        self,
        first_span: ttnn.Tensor,
        second_span: ttnn.Tensor,
        *,
        mode: str = "chunk",
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run span zero, transfer its causal cache, then run span one."""
        first_output = self.first_layer.forward(first_span, mode=mode)
        self._handoff_causal_state()
        second_output = self.second_layer.forward(second_span, mode=mode)
        return first_output, second_output
