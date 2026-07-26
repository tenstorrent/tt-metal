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


class SP8AffinePrefixProbe:
    """Compose eight KDA span transforms with a three-stage inclusive prefix.

    Inputs and outputs are tuples ordered by sequence span.  ``transform_a[i]``
    and ``transform_b[i]`` describe ``T_i(S) = A_i @ S + B_i``.  The result at
    rank ``i`` is the inclusive composition ``T_i o ... o T_0``.  The host
    queues all independent sends/receives within a stage before synchronizing,
    then performs the stage's local compositions in parallel across devices.
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

    def _synchronize(self) -> None:
        for device in self.span_devices:
            ttnn.synchronize_device(device)

    def run(
        self,
        transform_a: tuple[ttnn.Tensor, ...],
        transform_b: tuple[ttnn.Tensor, ...],
        *,
        retain_buffers: bool = False,
        synchronize_stages: bool = True,
        after_stage_enqueued: Callable[[int], None] | None = None,
    ) -> tuple[tuple[ttnn.Tensor, ...], tuple[ttnn.Tensor, ...]]:
        """Run all three Hillis--Steele stages and return inclusive transforms.

        ``after_stage_enqueued`` can queue independent local work once a
        stage's socket operations are in flight.  It deliberately requires
        the global stage barriers: a callback with unsynchronized stages would
        let device traces advance the fabric protocol out of order.
        """
        if len(transform_a) != _SP_SIZE or len(transform_b) != _SP_SIZE:
            raise ValueError(f"expected {_SP_SIZE} transforms, got A={len(transform_a)}, B={len(transform_b)}")
        if after_stage_enqueued is not None and not synchronize_stages:
            raise ValueError("after_stage_enqueued requires synchronize_stages=True")
        prefix_a = list(transform_a)
        prefix_b = list(transform_b)

        for stage, distance in enumerate(_PREFIX_STAGES):
            received_a: dict[int, ttnn.Tensor] = {}
            received_b: dict[int, ttnn.Tensor] = {}
            # Snapshot the preceding stage at every source before any rank is
            # overwritten.  A socket is an ordered stream, so A then B are
            # sent/received in the same order on every edge.
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
            if after_stage_enqueued is not None:
                after_stage_enqueued(stage)
            if synchronize_stages:
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
            if synchronize_stages:
                self._synchronize()

            if not retain_buffers:
                for destination in range(distance, _SP_SIZE):
                    ttnn.deallocate(prefix_a[destination])
                    ttnn.deallocate(prefix_b[destination])
                    ttnn.deallocate(received_a[destination])
                    ttnn.deallocate(received_b[destination])
            prefix_a, prefix_b = next_a, next_b

        return tuple(prefix_a), tuple(prefix_b)
