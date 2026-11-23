# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Pipeline-stage boundaries carried over the host.

Two stages with no fabric link between them, such as stages on galaxies that are not cabled to each
other, hand the activation over through their hosts: the sending stage writes it to a
D2HStreamService, its rank sends the page to the next rank over MPI, and that rank writes it to an
H2DStreamService the receiving stage drains. HostHopSender and HostHopReceiver stand in for the D2D
services of a fabric boundary in the request loop.
"""

import math
import os

import torch

import ttnn

HOST_HOP_RANKS_ENV = "PREFILL_HOST_HOP_RANKS"
# MPI tag of the boundary out of rank r is HOST_HOP_TAG_BASE + r, under the 32767 every MPI guarantees.
HOST_HOP_TAG_BASE = 0x4800
D2H_FIFO_SIZE_BYTES = 1 << 20
# One page is one MPI message, whose byte count is an int.
MAX_PAGE_BYTES = 2**31 - 1


def host_hop_sender_ranks(num_ranks: int) -> frozenset:
    """Ranks whose outbound boundary crosses the host, from PREFILL_HOST_HOP_RANKS (comma-separated)."""
    raw = os.environ.get(HOST_HOP_RANKS_ENV, "").strip()
    try:
        ranks = frozenset(int(r) for r in raw.split(",") if r.strip())
    except ValueError:
        raise ValueError(f"{HOST_HOP_RANKS_ENV}={raw!r} is not a comma-separated list of ranks") from None
    bad = sorted(r for r in ranks if not 0 <= r < num_ranks - 1)
    if bad:
        raise ValueError(f"{HOST_HOP_RANKS_ENV}={raw!r}: ranks {bad} have no outbound boundary among {num_ranks} ranks")
    return ranks


class _HostHopEnd:
    """What both ends share with a D2D service. A host boundary holds no fabric links, so the request
    loop's lease calls do nothing."""

    def __init__(self, mesh_device, service, peer_rank: int, sender_rank: int, metadata_size_bytes: int):
        self._service = service
        self._peer_rank = peer_rank
        self._tag = HOST_HOP_TAG_BASE + sender_rank
        self._metadata_size_bytes = metadata_size_bytes
        self._mesh_shape = mesh_device.shape
        shard_spec = service.get_per_shard_spec()
        if shard_spec.dtype != ttnn.bfloat16:
            raise ValueError(f"a host hop carries bfloat16 activations, not {shard_spec.dtype}")
        self._shard_spec = shard_spec
        self._shard_shape = [int(d) for d in shard_spec.shape]
        self._shard_bytes = 2 * math.prod(self._shard_shape)
        self._page_bytes = self._shard_bytes * mesh_device.get_num_devices() + metadata_size_bytes
        if self._page_bytes > MAX_PAGE_BYTES:
            raise ValueError(f"a host hop page of {self._page_bytes} B does not fit one MPI message")

    def get_backing_tensor(self):
        return self._service.get_backing_tensor()

    def get_backing_tensors(self) -> list:
        return [self._service.get_backing_tensor()]

    def release_fabric_links(self) -> None:
        pass

    def wait_for_fabric_links(self) -> None:
        pass


class HostHopSender(_HostHopEnd):
    """The sending end: sync() queues the device copy into the D2H service, forward() moves that page
    to the next rank."""

    def __init__(
        self,
        mesh_device,
        *,
        global_spec,
        mapper_config,
        worker_cores,
        metadata_size_bytes,
        peer_rank: int,
        sender_rank: int,
    ):
        service = ttnn.D2HStreamService(
            mesh_device,
            global_spec=global_spec,
            fifo_size_bytes=D2H_FIFO_SIZE_BYTES,
            mapper=ttnn.create_mesh_mapper(mesh_device, mapper_config),
            worker_cores=worker_cores,
            metadata_size_bytes=metadata_size_bytes,
        )
        super().__init__(mesh_device, service, peer_rank, sender_rank, metadata_size_bytes)
        self._page = ttnn.from_torch(
            torch.zeros([int(d) for d in global_spec.shape], dtype=torch.bfloat16),
            spec=global_spec,
            mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mapper_config),
        )

    def sync(self, payload, metadata):
        return ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(
            self._service, input=payload, metadata=metadata
        )

    def forward(self) -> None:
        metadata = self._service.read_from_tensor(self._page)
        shards = [ttnn.to_torch(s).view(torch.int16).numpy().tobytes() for s in ttnn.get_device_tensors(self._page)]
        ttnn.distributed_context_send_bytes(b"".join([*shards, metadata]), self._peer_rank, self._tag)


class HostHopReceiver(_HostHopEnd):
    """The receiving end: receive() takes the next page from the previous rank and writes it to the
    H2D service, sync() queues the device drain of it."""

    def __init__(
        self,
        mesh_device,
        *,
        global_spec,
        mapper_config,
        worker_cores,
        metadata_size_bytes,
        peer_rank: int,
        sender_rank: int,
    ):
        service = ttnn.H2DStreamService(
            mesh_device=mesh_device,
            global_spec=global_spec,
            mapper=ttnn.create_mesh_mapper(mesh_device, mapper_config),
            worker_cores=worker_cores,
            metadata_size_bytes=metadata_size_bytes,
            parallel_host_push=True,
        )
        super().__init__(mesh_device, service, peer_rank, sender_rank, metadata_size_bytes)

    def receive(self) -> None:
        # torch.frombuffer wants a writable buffer.
        page = bytearray(ttnn.distributed_context_recv_bytes(self._page_bytes, self._peer_rank, self._tag))
        payload_bytes = self._page_bytes - self._metadata_size_bytes
        shards = [
            ttnn.from_torch(
                torch.frombuffer(page, dtype=torch.int16, count=self._shard_bytes // 2, offset=offset)
                .view(torch.bfloat16)
                .reshape(self._shard_shape),
                spec=self._shard_spec,
            )
            for offset in range(0, payload_bytes, self._shard_bytes)
        ]
        self._service.forward_to_tensor(ttnn.from_host_shards(shards, self._mesh_shape), bytes(page[payload_bytes:]))

    def sync(self, *, metadata_out, tokens_out):
        return ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
            self._service,
            metadata_size_bytes=self._metadata_size_bytes,
            metadata_out=metadata_out,
            tokens_out=tokens_out,
        )
