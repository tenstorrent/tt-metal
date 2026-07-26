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
from models.experimental.kimi_delta_attention.tt.sp_affine_prefix import SP8AffinePrefixProbe, _prefix_socket_config
from models.tt_transformers.tt.ccl import TT_CCL

_SP_SIZE = 2
_TP_SIZE = 4
_SP8_SIZE = 8
_TP1_SIZE = 1


def _socket_config(mesh_shape: ttnn.MeshShape) -> ttnn.SocketConfig:
    """Create one rank-aligned fabric lane per LoudBox device.

    This is the established 1x8-to-two-1x4 socket shape used by the CCL
    send/receive test.  One worker-core pair is the conservative default;
    a two-lane opt-in is available to evaluate fabric striping despite the
    runtime's experimental multi-sender warning.
    """
    # send_async/recv_async stripe tensor pages over the socket's worker pairs,
    # and map each pair onto a distinct fabric link.  Keep one lane as the
    # conservative default; the two-lane variant is the relevant LoudBox
    # boundary-transport experiment for a 512 KiB recurrent shard.
    lanes = int(os.getenv("KDA_SP_SOCKET_LANES", "1"))
    if lanes not in (1, 2):
        raise ValueError(f"KDA_SP_SOCKET_LANES must be 1 or 2, got {lanes}")
    sender_cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)][:lanes]
    receiver_cores = [ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1)][:lanes]
    connections = [
        ttnn.SocketConnection(ttnn.MeshCoreCoord(coord, sender), ttnn.MeshCoreCoord(coord, receiver))
        for coord in ttnn.MeshCoordinateRange(mesh_shape)
        for sender, receiver in zip(sender_cores, receiver_cores, strict=True)
    ]
    # A full per-rank FP32 recurrent state is 512 KiB.  Match the aggregate
    # FIFO to that payload so each striped lane can make one uninterrupted
    # pass, without multiplying socket memory by the number of lanes.
    fifo_size = int(os.getenv("KDA_SP_SOCKET_FIFO_BYTES", str((512 * 1024) // lanes)))
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

        Each span first builds state-independent four- or eight-chunk preparation
        tensors.  The first prefix produces the boundary state while both
        spans' final output scans remain unscheduled; once the state is on
        fabric, the two seeded scans can run independently.
        """
        first, second = self.first_layer, self.second_layer
        group_tokens = 256 if first_span.shape[1] % 256 == 0 else 128
        groups = first_span.shape[1] // group_tokens
        first_convolution = first.prepare_chunk_convolution(first_span)
        ttnn.experimental.send_async(first_convolution.new_convolution_state, self._send_socket)
        ttnn.experimental.recv_async(second.convolution_state, self._recv_socket)
        second_convolution = second.prepare_chunk_convolution(second_span)
        first_prepared = first.complete_chunk_preparation(first_convolution)
        second_prepared = second.complete_chunk_preparation(second_convolution)

        first_grouped = first.group_prepare(first_prepared)
        second_grouped = second.group_prepare(second_prepared)
        first_a, first_b = first.group_summary(first_grouped)
        second_a, second_b = second.group_summary(second_grouped)
        first_entries, first_end_state = first.group_entry_states_and_end(first_a, first_b, groups)
        ttnn.experimental.send_async(first_end_state, self._send_socket)
        ttnn.experimental.recv_async(second.recurrent_state, self._recv_socket)
        second_entries = second.group_entry_states(second_a, second_b, groups)

        first_raw_output, _ = first.group_scan(first_prepared, first_grouped, first_entries, output_final_state=False)
        second_raw_output, second_state = second.group_scan(second_prepared, second_grouped, second_entries)
        assert second_state is not None
        first_output = first._finish_prepared(first_prepared, first_raw_output, first_end_state, fuse_scan_rms=False)
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
            and first_span.shape[1] % 128 == 0
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


class SP8AffineTP1KimiDeltaAttention:
    """Correctness-first SP=8 affine scheduler on a 1x8 LoudBox.

    This path is deliberately a TP=1 protocol proof, not a performance model
    for Galaxy's SP=8 x TP=4 topology.  It keeps short-convolution history in
    causal order, derives each span's terminal affine map on device, prefixes
    those maps through the real fabric, and installs the resulting recurrent
    entry states without a host tensor handoff.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        config: KDAConfig,
        state_dict: Mapping[str, torch.Tensor],
        tensor_cache_path: Path | None = None,
    ) -> None:
        if tuple(mesh_device.shape) != (1, _SP8_SIZE):
            raise ValueError(f"SP8 affine TP1 requires a 1x8 LoudBox mesh, got {tuple(mesh_device.shape)}")
        self.mesh_device = mesh_device
        self.prefix = SP8AffinePrefixProbe(mesh_device)
        self.span_devices = self.prefix.span_devices
        self.layers = tuple(
            KimiDeltaAttention(span_device, config, state_dict, tensor_cache_path=tensor_cache_path)
            for span_device in self.span_devices
        )
        # Keep the two ordered causal handoffs on cores disjoint from all three
        # affine-prefix distances. The convolution transfer is tiny; the entry
        # state carries one FP32 state per TP1 rank.
        self._convolution_sockets = tuple(
            ttnn.create_socket_pair(self.span_devices[span], self.span_devices[span + 1], _prefix_socket_config(3))
            for span in range(_SP8_SIZE - 1)
        )
        self._entry_sockets = tuple(
            ttnn.create_socket_pair(self.span_devices[span], self.span_devices[span + 1], _prefix_socket_config(4))
            for span in range(_SP8_SIZE - 1)
        )

    def reset_state(self, batch_size: int) -> None:
        for layer in self.layers:
            layer.reset_state(batch_size)

    def _synchronize(self) -> None:
        for device in self.span_devices:
            ttnn.synchronize_device(device)

    def _prepare_convolutions(self, spans: tuple[ttnn.Tensor, ...]) -> tuple:
        """Prepare the short-convolution boundary in causal order on device."""
        convolutions = []
        for span, (layer, hidden) in enumerate(zip(self.layers, spans, strict=True)):
            convolution = layer.prepare_chunk_convolution(hidden)
            convolutions.append(convolution)
            if span + 1 < _SP8_SIZE:
                destination = self.layers[span + 1]
                assert destination.convolution_state is not None
                send_socket, recv_socket = self._convolution_sockets[span]
                ttnn.experimental.send_async(convolution.new_convolution_state, send_socket)
                ttnn.experimental.recv_async(destination.convolution_state, recv_socket)
                self._synchronize()
        return tuple(convolutions)

    def _broadcast_initial_recurrent_state(self) -> None:
        """Install rank zero's carry on every rank without a host copy.

        An inclusive affine prefix maps the same incoming state through each
        rank's terminal transform.  After a previous invocation only rank zero
        necessarily owns that global carry, so propagate it over the ordered
        chain before deriving the per-rank endpoint states.
        """
        for span in range(_SP8_SIZE - 1):
            source = self.layers[span]
            destination = self.layers[span + 1]
            assert source.recurrent_state is not None
            assert destination.recurrent_state is not None
            send_socket, recv_socket = self._entry_sockets[span]
            ttnn.experimental.send_async(source.recurrent_state, send_socket)
            ttnn.experimental.recv_async(destination.recurrent_state, recv_socket)
            self._synchronize()

    def forward(self, *spans: ttnn.Tensor, mode: str = "chunk") -> tuple[ttnn.Tensor, ...]:
        """Run eight equal chunk spans with a device-resident affine prefix."""
        if mode != "chunk":
            raise ValueError("SP8 affine TP1 currently supports chunk mode only")
        if len(spans) != _SP8_SIZE:
            raise ValueError(f"SP8 affine TP1 expects {_SP8_SIZE} spans, got {len(spans)}")
        span_length = spans[0].shape[1]
        if any(span.shape[1] != span_length for span in spans):
            raise ValueError("SP8 affine TP1 requires equal sequence spans")
        if span_length % 128:
            raise ValueError(f"SP8 affine TP1 requires 128-token-aligned spans, got T={span_length}")
        groups = span_length // (256 if span_length % 256 == 0 else 128)
        self._broadcast_initial_recurrent_state()
        convolutions = self._prepare_convolutions(spans)
        prepared = tuple(
            layer.complete_chunk_preparation(convolution) for layer, convolution in zip(self.layers, convolutions)
        )
        grouped = tuple(layer.group_prepare(rank_prepared) for layer, rank_prepared in zip(self.layers, prepared))
        grouped_transforms = tuple(
            layer.group_summary(rank_grouped) for layer, rank_grouped in zip(self.layers, grouped)
        )
        span_transforms = tuple(
            layer.group_terminal_affine_transform(transform_a, transform_b, groups, batch_size=spans[0].shape[0])
            for layer, (transform_a, transform_b) in zip(self.layers, grouped_transforms, strict=True)
        )
        span_a, span_b = zip(*span_transforms, strict=True)
        prefix_a, prefix_b = self.prefix.run(span_a, span_b, synchronize_stages=True)

        # Each inclusive prefix maps the common initial state to the state at
        # the end of its rank. Send that completed state one hop in parallel so
        # every following rank owns the exact exclusive scan entry.
        span_end_states = tuple(
            ttnn.reshape(
                ttnn.add(
                    ttnn.matmul(
                        transform_a,
                        ttnn.reshape(
                            layer.recurrent_state,
                            (
                                layer.recurrent_state.shape[0] * layer.recurrent_state.shape[1],
                                layer.recurrent_state.shape[2],
                                layer.recurrent_state.shape[3],
                            ),
                        ),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    transform_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                layer.recurrent_state.shape,
            )
            for layer, transform_a, transform_b in zip(self.layers, prefix_a, prefix_b, strict=True)
        )
        for span, span_end_state in enumerate(span_end_states[:-1]):
            destination = self.layers[span + 1]
            assert destination.recurrent_state is not None
            send_socket, recv_socket = self._entry_sockets[span]
            ttnn.experimental.send_async(span_end_state, send_socket)
            ttnn.experimental.recv_async(destination.recurrent_state, recv_socket)
        self._synchronize()

        entries = tuple(
            layer.group_entry_states(transform_a, transform_b, groups)
            for layer, (transform_a, transform_b) in zip(self.layers, grouped_transforms, strict=True)
        )
        outputs = []
        for layer, rank_prepared, rank_grouped, rank_entries in zip(
            self.layers, prepared, grouped, entries, strict=True
        ):
            raw_output, final_state = layer.group_scan(rank_prepared, rank_grouped, rank_entries)
            assert final_state is not None
            outputs.append(layer._finish_prepared(rank_prepared, raw_output, final_state, fuse_scan_rms=False))
        return tuple(outputs)
