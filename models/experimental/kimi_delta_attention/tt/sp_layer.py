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

import os
from collections.abc import Mapping
from pathlib import Path

import torch

import ttnn
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.tt_transformers.tt.ccl import TT_CCL

_SP_SIZE = 2
_TP_SIZE = 4
_SP8_SIZE = 8
_TP1_SIZE = 1


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
    # A full per-rank FP32 recurrent state is 512 KiB.  Matching the socket
    # FIFO to that payload eliminates the multi-round-trip backpressure of
    # the old 10 KiB default; callers can still sweep it through the env var.
    fifo_size = int(os.getenv("KDA_SP_SOCKET_FIFO_BYTES", str(512 * 1024)))
    if fifo_size <= 0 or fifo_size % 1024:
        raise ValueError(f"KDA_SP_SOCKET_FIFO_BYTES must be a positive KiB multiple, got {fifo_size}")
    return ttnn.SocketConfig(connections, ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, fifo_size))


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

    def _forward_affine_sp2(
        self,
        first_span: ttnn.Tensor,
        second_span: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Use the first span's affine summary to release the second scan early.

        The short convolution remains an ordered three-sample dependency, but
        it is available as soon as first-span preparation completes.  The
        large recurrent boundary is instead derived from `(A, B)` before the
        first span's token-output scan, allowing the two final scans to run on
        their separate TP=4 submeshes concurrently.  This is the SP=2 base
        case of the Galaxy log-depth affine prefix; larger SP will prefix the
        same per-span transforms instead of relaying a materialized state.
        """
        first = self.first_layer
        second = self.second_layer
        assert first.recurrent_state is not None
        assert first.convolution_state is not None
        assert second.recurrent_state is not None
        assert second.convolution_state is not None

        first_prepared = first.prepare_chunk(first_span)
        # Queue the small convolution handoff immediately. The second prepare
        # consumes its destination buffer, so its projection/convolution work
        # can overlap the first span's affine-summary construction.
        ttnn.experimental.send_async(first_prepared.new_convolution_state, self._send_socket)
        ttnn.experimental.recv_async(second.convolution_state, self._recv_socket)
        second_prepared = second.prepare_chunk(second_span)

        span_end_state = first.affine_span_end_state(first_prepared)
        ttnn.experimental.send_async(span_end_state, self._send_socket)
        ttnn.experimental.recv_async(second.recurrent_state, self._recv_socket)

        first_output = first.forward_prepared(first_prepared)
        second_output = second.forward_prepared(second_prepared)
        return first_output, second_output

    def _forward_pipelined_sp2(
        self,
        first_span: ttnn.Tensor,
        second_span: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Overlap span one's recurrence with span two's input preparation.

        Unlike the affine-summary prototype, this reuses the first span's
        ordinary recurrence result for the state handoff, so it introduces no
        second scan of that span.
        """
        first, second = self.first_layer, self.second_layer
        first_prepared = first.prepare_chunk(first_span)
        ttnn.experimental.send_async(first_prepared.new_convolution_state, self._send_socket)
        ttnn.experimental.recv_async(second.convolution_state, self._recv_socket)
        second_prepared = second.prepare_chunk(second_span)

        first_output = first.forward_prepared(first_prepared)
        assert first.recurrent_state is not None
        ttnn.experimental.send_async(first.recurrent_state, self._send_socket)
        ttnn.experimental.recv_async(second.recurrent_state, self._recv_socket)
        second_output = second.forward_prepared(second_prepared)
        return first_output, second_output

    def _forward_split_affine_sp2(
        self,
        first_span: ttnn.Tensor,
        second_span: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the SP affine prefix without repeating either span's KDA prep.

        Each span first builds the state-independent eight-chunk preparation
        tensors.  The first prefix produces the boundary state while both
        spans' final output scans remain unscheduled; once the state is on
        fabric, the two seeded scans can run independently.
        """
        first, second = self.first_layer, self.second_layer
        groups = first_span.shape[1] // 256
        first_prepared = first.prepare_chunk(first_span)
        ttnn.experimental.send_async(first_prepared.new_convolution_state, self._send_socket)
        ttnn.experimental.recv_async(second.convolution_state, self._recv_socket)
        second_prepared = second.prepare_chunk(second_span)

        first_grouped = first.group_prepare(first_prepared)
        second_grouped = second.group_prepare(second_prepared)
        first_a, first_b = first.group_summary(first_grouped)
        second_a, second_b = second.group_summary(second_grouped)
        first_entries = first.group_entry_states(first_a, first_b, groups)
        first_end_state = first.group_end_state(first_a, first_b, first_entries, groups)
        ttnn.experimental.send_async(first_end_state, self._send_socket)
        ttnn.experimental.recv_async(second.recurrent_state, self._recv_socket)
        second_entries = second.group_entry_states(second_a, second_b, groups)

        first_raw_output, first_state = first.group_scan(first_prepared, first_grouped, first_entries)
        second_raw_output, second_state = second.group_scan(second_prepared, second_grouped, second_entries)
        first_output = first._finish_prepared(first_prepared, first_raw_output, first_state, fuse_scan_rms=False)
        second_output = second._finish_prepared(second_prepared, second_raw_output, second_state, fuse_scan_rms=False)
        return first_output, second_output

    def forward(
        self,
        first_span: ttnn.Tensor,
        second_span: ttnn.Tensor,
        *,
        mode: str = "chunk",
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run span zero, transfer its causal cache, then run span one."""
        if (
            os.getenv("KDA_SP_AFFINE", "0") == "1"
            and mode == "chunk"
            and first_span.shape[1] == second_span.shape[1]
            and first_span.shape[1] % 256 == 0
        ):
            return self._forward_affine_sp2(first_span, second_span)
        if (
            os.getenv("KDA_SP_SPLIT_AFFINE", "0") == "1"
            and mode == "chunk"
            and first_span.shape[1] == second_span.shape[1]
            and first_span.shape[1] % 256 == 0
        ):
            return self._forward_split_affine_sp2(first_span, second_span)
        if os.getenv("KDA_SP_PIPELINED", "0") == "1" and mode == "chunk":
            return self._forward_pipelined_sp2(first_span, second_span)
        first_output = self.first_layer.forward(first_span, mode=mode)
        self._handoff_causal_state()
        second_output = self.second_layer.forward(second_span, mode=mode)
        return first_output, second_output


class SP8TP1KimiDeltaAttention:
    """Eight-rank sequence-parallel KDA protocol probe for LoudBox.

    Each chip owns all heads for one contiguous sequence span. This is not a
    production-performance topology--a Galaxy rank will instead own one
    quarter of the heads--but it exercises all seven ordered causal handoffs
    with the real KDA layer and no host materialization of either cache.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: KDAConfig,
        state_dict: Mapping[str, torch.Tensor],
        tensor_cache_path: Path | None = None,
    ) -> None:
        if tuple(mesh_device.shape) != (1, _SP8_SIZE):
            raise ValueError(f"SP8TP1 requires a 1x8 LoudBox mesh, got {tuple(mesh_device.shape)}")
        self.mesh_device = mesh_device
        self.span_devices = tuple(
            mesh_device.create_submesh(ttnn.MeshShape(1, _TP1_SIZE), ttnn.MeshCoordinate(0, span))
            for span in range(_SP8_SIZE)
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
        self._sockets = tuple(
            ttnn.create_socket_pair(
                self.span_devices[span], self.span_devices[span + 1], _socket_config(self.span_devices[span].shape)
            )
            for span in range(_SP8_SIZE - 1)
        )

    def reset_state(self, batch_size: int) -> None:
        for layer in self.layers:
            layer.reset_state(batch_size)

    def _handoff_causal_state(self, source_index: int) -> None:
        source = self.layers[source_index]
        destination = self.layers[source_index + 1]
        assert source.recurrent_state is not None
        assert source.convolution_state is not None
        assert destination.recurrent_state is not None
        assert destination.convolution_state is not None
        send_socket, recv_socket = self._sockets[source_index]
        ttnn.experimental.send_async(source.recurrent_state, send_socket)
        ttnn.experimental.recv_async(destination.recurrent_state, recv_socket)
        ttnn.experimental.send_async(source.convolution_state, send_socket)
        ttnn.experimental.recv_async(destination.convolution_state, recv_socket)

    def forward(self, *spans: ttnn.Tensor, mode: str = "chunk") -> tuple[ttnn.Tensor, ...]:
        if len(spans) != _SP8_SIZE:
            raise ValueError(f"SP8TP1 expects {_SP8_SIZE} spans, got {len(spans)}")
        outputs = []
        for span_index, (layer, span) in enumerate(zip(self.layers, spans, strict=True)):
            outputs.append(layer.forward(span, mode=mode))
            if span_index + 1 < _SP8_SIZE:
                self._handoff_causal_state(span_index)
        return tuple(outputs)
