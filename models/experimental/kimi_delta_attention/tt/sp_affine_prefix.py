# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Eight-rank device-resident affine prefix probe for KDA sequence parallelism.

This deliberately models the *per-TP4-rank* Galaxy payload on an eight-chip
LoudBox: one FP32 ``A[K,K]`` and one FP32 ``B[K,V]`` transform for each of
eight local heads.  It is not an SP8xTP4 implementation--LoudBox has only
eight devices--but it exercises the real three-stage fabric schedule and the
same 1 MiB/rank transform payload a Galaxy prefix must carry.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

import ttnn

_SP_SIZE = 8
_PREFIX_STAGES = (1, 2, 4)


def _prefix_socket_config(stage_index: int) -> ttnn.SocketConfig:
    """Give each prefix stage disjoint endpoint cores and a one-tensor FIFO."""
    lanes = int(os.getenv("KDA_SP_PREFIX_LANES", "1"))
    if lanes not in (1, 2):
        raise ValueError(f"KDA_SP_PREFIX_LANES must be 1 or 2, got {lanes}")
    fifo_bytes = int(os.getenv("KDA_SP_PREFIX_FIFO_BYTES", str((512 * 1024) // lanes)))
    if fifo_bytes <= 0 or fifo_bytes % 1024:
        raise ValueError(f"KDA_SP_PREFIX_FIFO_BYTES must be a positive KiB multiple, got {fifo_bytes}")
    mesh_coord = ttnn.MeshCoordinate(0, 0)
    # An intermediate rank sends and receives in each stage.  Distinct cores
    # avoid sharing one fabric endpoint between those two directions.  Stages
    # are also assigned distinct rows so their persistent socket objects do
    # not alias endpoint resources under trace capture.
    sender_cores = [ttnn.CoreCoord(0, stage_index * 2), ttnn.CoreCoord(2, stage_index * 2)][:lanes]
    receiver_cores = [ttnn.CoreCoord(1, stage_index * 2), ttnn.CoreCoord(3, stage_index * 2)][:lanes]
    return ttnn.SocketConfig(
        [
            ttnn.SocketConnection(ttnn.MeshCoreCoord(mesh_coord, sender), ttnn.MeshCoreCoord(mesh_coord, receiver))
            for sender, receiver in zip(sender_cores, receiver_cores, strict=True)
        ],
        ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, fifo_bytes),
    )


def _barrier_socket_config(stage_index: int) -> ttnn.SocketConfig:
    """Use the reverse prefix endpoints for a tiny tree-barrier token."""
    lanes = int(os.getenv("KDA_SP_PREFIX_LANES", "1"))
    sender_cores = [ttnn.CoreCoord(0, stage_index * 2), ttnn.CoreCoord(2, stage_index * 2)][:lanes]
    receiver_cores = [ttnn.CoreCoord(1, stage_index * 2), ttnn.CoreCoord(3, stage_index * 2)][:lanes]
    fifo_bytes = int(os.getenv("KDA_SP_BARRIER_FIFO_BYTES", "4096"))
    if fifo_bytes <= 0 or fifo_bytes % 1024:
        raise ValueError(f"KDA_SP_BARRIER_FIFO_BYTES must be a positive KiB multiple, got {fifo_bytes}")
    mesh_coord = ttnn.MeshCoordinate(0, 0)
    return ttnn.SocketConfig(
        [
            ttnn.SocketConnection(ttnn.MeshCoreCoord(mesh_coord, sender), ttnn.MeshCoreCoord(mesh_coord, receiver))
            for sender, receiver in zip(sender_cores, receiver_cores, strict=True)
        ],
        ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, fifo_bytes),
    )


class SP8AffinePrefixProbe:
    """Compose eight KDA span transforms with a three-stage inclusive prefix.

    Inputs and outputs are tuples ordered by sequence span.  ``transform_a[i]``
    and ``transform_b[i]`` describe ``T_i(S) = A_i @ S + B_i``.  The result at
    rank ``i`` is the inclusive composition ``T_i o ... o T_0``.  The default
    eager path fences each stage on the host; the opt-in device-barrier path
    instead queues a fabric gather/release boundary between prefix distances.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice) -> None:
        if tuple(mesh_device.shape) != (1, _SP_SIZE):
            raise ValueError(f"SP8 affine prefix requires a 1x8 LoudBox mesh, got {tuple(mesh_device.shape)}")
        self.mesh_device = mesh_device
        self.span_devices = tuple(
            mesh_device.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, span)) for span in range(_SP_SIZE)
        )
        self._stage_sockets = tuple(
            tuple(
                ttnn.create_socket_pair(
                    self.span_devices[source], self.span_devices[source + distance], _prefix_socket_config(stage)
                )
                for source in range(_SP_SIZE - distance)
            )
            for stage, distance in enumerate(_PREFIX_STAGES)
        )
        self._barrier_gather_sockets = None
        self._barrier_release_sockets = None
        self._barrier_tokens = None

    def _ensure_device_barrier(self) -> None:
        """Allocate the opt-in tree barrier without perturbing the baseline."""
        if self._barrier_gather_sockets is not None:
            return
        # A stage must finish on every rank before any rank starts the next
        # distance.  Host fences satisfy that requirement but preclude a
        # monolithic trace.  These reverse trees gather a completion token at
        # rank zero; the existing forward prefix trees then release that token
        # back to all ranks.  Every send/receive is queued after the local
        # stage matmuls, so command-queue order turns the token into a global
        # device-side stage boundary.
        self._barrier_gather_sockets = tuple(
            tuple(
                ttnn.create_socket_pair(
                    self.span_devices[source + distance],
                    self.span_devices[source],
                    _barrier_socket_config(stage),
                )
                for source in range(0, _SP_SIZE, 2 * distance)
            )
            for stage, distance in enumerate(_PREFIX_STAGES)
        )
        self._barrier_release_sockets = tuple(
            tuple(self._stage_sockets[stage][source] for source in range(0, _SP_SIZE, 2 * distance))
            for stage, distance in enumerate(_PREFIX_STAGES)
        )
        # The value is irrelevant: this tiled page is a command-queue and
        # fabric dependency token.  Keep one fixed-address token per rank so
        # repeated eager calls and future trace capture use the same buffers.
        self._barrier_tokens = tuple(
            ttnn.from_torch(
                torch.zeros((1, 1, 32, 32), dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            for device in self.span_devices
        )

    def _synchronize(self) -> None:
        for device in self.span_devices:
            ttnn.synchronize_device(device)

    def queue_device_barrier(self) -> None:
        """Queue a tree gather/release barrier on the eight device queues.

        The gather is ordered from distance one to four.  An intermediate
        rank therefore cannot notify its parent until its child has arrived.
        The release runs in the reverse order, so a child cannot release its
        descendants before it has itself been released by rank zero.
        """
        self._ensure_device_barrier()
        assert self._barrier_gather_sockets is not None
        assert self._barrier_release_sockets is not None
        assert self._barrier_tokens is not None
        for stage, stage_sockets in enumerate(self._barrier_gather_sockets):
            distance = _PREFIX_STAGES[stage]
            for pair_index, (send_socket, recv_socket) in enumerate(stage_sockets):
                source = (2 * distance) * pair_index
                destination = source + distance
                ttnn.experimental.send_async(self._barrier_tokens[destination], send_socket)
                ttnn.experimental.recv_async(self._barrier_tokens[source], recv_socket)
        for stage, stage_sockets in reversed(tuple(enumerate(self._barrier_release_sockets))):
            distance = _PREFIX_STAGES[stage]
            for pair_index, (send_socket, recv_socket) in enumerate(stage_sockets):
                source = (2 * distance) * pair_index
                destination = source + distance
                ttnn.experimental.send_async(self._barrier_tokens[source], send_socket)
                ttnn.experimental.recv_async(self._barrier_tokens[destination], recv_socket)

    def run_stage(
        self,
        prefix_a: tuple[ttnn.Tensor, ...],
        prefix_b: tuple[ttnn.Tensor, ...],
        stage: int,
        *,
        retain_buffers: bool = False,
        synchronize_transfer: bool = False,
        after_enqueued: Callable[[int], None] | None = None,
    ) -> tuple[tuple[ttnn.Tensor, ...], tuple[ttnn.Tensor, ...]]:
        """Enqueue one prefix distance.

        Callers that invoke more than one stage must establish a rank-wide
        completion boundary before the following stage.  ``synchronize_transfer``
        is the eager implementation of that boundary; stage-sliced trace
        callers synchronize after replaying this stage on every rank.
        """
        if stage < 0 or stage >= len(_PREFIX_STAGES):
            raise ValueError(f"expected prefix stage in [0, {len(_PREFIX_STAGES)}), got {stage}")
        if len(prefix_a) != _SP_SIZE or len(prefix_b) != _SP_SIZE:
            raise ValueError(f"expected {_SP_SIZE} transforms, got A={len(prefix_a)}, B={len(prefix_b)}")
        distance = _PREFIX_STAGES[stage]
        received_a: dict[int, ttnn.Tensor] = {}
        received_b: dict[int, ttnn.Tensor] = {}
        # Snapshot the preceding stage at every source before any rank is
        # overwritten. A socket is an ordered stream, so A then B are sent and
        # received in the same order on every edge.
        for source, (send_socket, recv_socket) in enumerate(self._stage_sockets[stage]):
            destination = source + distance
            received_a[destination] = ttnn.allocate_tensor_on_device(
                prefix_a[source].spec, self.span_devices[destination]
            )
            received_b[destination] = ttnn.allocate_tensor_on_device(
                prefix_b[source].spec, self.span_devices[destination]
            )
            ttnn.experimental.send_async(prefix_a[source], send_socket)
            ttnn.experimental.recv_async(received_a[destination], recv_socket)
            ttnn.experimental.send_async(prefix_b[source], send_socket)
            ttnn.experimental.recv_async(received_b[destination], recv_socket)
        if after_enqueued is not None:
            after_enqueued(stage)
        if synchronize_transfer:
            self._synchronize()

        next_a = list(prefix_a)
        next_b = list(prefix_b)
        for destination in range(distance, _SP_SIZE):
            # T_destination o T_prefix: A_d @ A_p, A_d @ B_p + B_d.
            next_a[destination] = ttnn.matmul(
                prefix_a[destination], received_a[destination], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            carried_b = ttnn.matmul(
                prefix_a[destination], received_b[destination], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            next_b[destination] = ttnn.add(carried_b, prefix_b[destination], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if not retain_buffers:
            for destination in range(distance, _SP_SIZE):
                ttnn.deallocate(prefix_a[destination])
                ttnn.deallocate(prefix_b[destination])
                ttnn.deallocate(received_a[destination])
                ttnn.deallocate(received_b[destination])
        return tuple(next_a), tuple(next_b)

    def run(
        self,
        transform_a: tuple[ttnn.Tensor, ...],
        transform_b: tuple[ttnn.Tensor, ...],
        *,
        retain_buffers: bool = False,
        synchronize_stages: bool = True,
        device_barrier: bool = False,
        after_stage_enqueued: Callable[[int], None] | None = None,
    ) -> tuple[tuple[ttnn.Tensor, ...], tuple[ttnn.Tensor, ...]]:
        """Run all three Hillis--Steele stages and return inclusive transforms.

        ``after_stage_enqueued`` can queue independent local work once a
        stage's socket operations are in flight.  It deliberately requires
        the eager global stage barriers: the callback is not yet sequenced
        against the device-barrier token protocol.
        """
        if len(transform_a) != _SP_SIZE or len(transform_b) != _SP_SIZE:
            raise ValueError(f"expected {_SP_SIZE} transforms, got A={len(transform_a)}, B={len(transform_b)}")
        if device_barrier and synchronize_stages:
            raise ValueError("device_barrier requires synchronize_stages=False")
        if after_stage_enqueued is not None and not synchronize_stages:
            raise ValueError("after_stage_enqueued requires synchronize_stages=True")
        prefix_a, prefix_b = transform_a, transform_b
        for stage in range(len(_PREFIX_STAGES)):
            prefix_a, prefix_b = self.run_stage(
                prefix_a,
                prefix_b,
                stage,
                retain_buffers=retain_buffers,
                synchronize_transfer=synchronize_stages,
                after_enqueued=after_stage_enqueued,
            )
            if synchronize_stages:
                self._synchronize()
            elif device_barrier and stage + 1 < len(_PREFIX_STAGES):
                self.queue_device_barrier()
        return prefix_a, prefix_b
