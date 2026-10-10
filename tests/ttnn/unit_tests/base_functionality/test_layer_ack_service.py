# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""LayerAckService on a galaxy: real D2H ack records through the device-side ack op, labelled in
ACK space and pushed into a layer-completion ring this test consumes."""

import os
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_slow_dispatch, skip_with_llk_assert
from ttnn._experimental.layer_completion import LayerCompletionQueueV2, LayerCompletionRouter

pytestmark = [
    skip_for_slow_dispatch(),
    skip_with_llk_assert("D2HStreamService fails to pin host DMA buffers. Issue: #47909"),
    pytest.mark.skipif(
        ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
        reason="D2HStreamService runs on Blackhole Galaxy clusters only (#47750)",
    ),
]

RECORD_BYTES = 12  # {slot_id, pos_start, pos_end}
NUM_ACK_LAYERS = 6  # two ranks of three KV-writing layers each (a hybrid stack)
ACK_FIRST_IDX = 3
ACK_LOCAL_COUNT = 3
LAYER_IDS = [15, 19, 23]
SOURCE_RANK = 1


def _unlink(shm_name: str) -> None:
    try:
        os.remove(f"/dev/shm/{shm_name.lstrip('/')}")
    except FileNotFoundError:
        pass


def _d2h_service(mesh_device):
    return ttnn.D2HStreamService(
        mesh_device,
        global_spec=None,
        fifo_size_bytes=4 * 64,
        worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        metadata_size_bytes=RECORD_BYTES,
    )


def _record(mesh_device, slot: int, start: int, end: int):
    return ttnn.from_torch(
        torch.tensor([slot, start, end], dtype=torch.int64).reshape(1, 1, 1, 3),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _fire(d2h, record, times: int) -> None:
    for _ in range(times):
        ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h, metadata=record)


def _pop(ring, n: int, timeout_s: float = 60.0) -> list:
    out = []
    deadline = time.monotonic() + timeout_s
    while len(out) < n and time.monotonic() < deadline:
        m = ring.try_pop()
        if m is None:
            time.sleep(0.005)
            continue
        out.append(m)
    assert len(out) == n, f"got {len(out)} of {n} messages"
    return out


def test_v2_messages_carry_the_record_identity_and_the_mapped_layer(mesh_device):
    ring_name = f"/tt_lcq_ack_v2_{os.getpid()}"
    _unlink(ring_name)
    ring = LayerCompletionQueueV2.create(ring_name)  # this test is the consumer; the ring must exist before start()
    d2h = _d2h_service(mesh_device)
    service = ttnn.LayerAckService(
        d2h,
        ring_name,
        source_rank=SOURCE_RANK,
        num_ack_layers=NUM_ACK_LAYERS,
        ack_first_idx=ACK_FIRST_IDX,
        ack_local_count=ACK_LOCAL_COUNT,
        ack_layer_ids=LAYER_IDS,
        protocol=2,
    )
    service.start()
    try:
        chunks = [(0, 0, 128), (1, 128, 256), (0, 256, 384), (2, 0, 512)]
        records = [_record(mesh_device, *c) for c in chunks]
        ttnn.synchronize_device(mesh_device)
        for rec in records:
            _fire(d2h, rec, ACK_LOCAL_COUNT)
        msgs = _pop(ring, len(chunks) * ACK_LOCAL_COUNT)
        for i, chunk in enumerate(chunks):
            for j in range(ACK_LOCAL_COUNT):
                seq, source_rank, request_id, slot_id, pos_start, pos_end, layer_start, layer_end = msgs[
                    i * ACK_LOCAL_COUNT + j
                ]
                assert (source_rank, request_id) == (SOURCE_RANK, i)
                assert (slot_id, pos_start, pos_end) == chunk
                assert (layer_start, layer_end) == (LAYER_IDS[j], LAYER_IDS[j] + 1)
                assert seq == i * NUM_ACK_LAYERS + ACK_FIRST_IDX + j
        service.check()
    finally:
        service.stop()
        ring.shutdown()


def test_v1_records_count_through_the_router(mesh_device):
    ring_name = f"/tt_lcq_ack_v1_{os.getpid()}"
    sched_name = f"/tt_lcq_ack_v1_sched_{os.getpid()}"
    _unlink(ring_name)
    _unlink(sched_name)
    router = LayerCompletionRouter(
        rank=0, world_size=1, master_rank=0, ring_shm_name=ring_name, scheduler_shm_name=sched_name, protocol=1
    )
    consumer = ttnn.InterProcessCounterChannel.connect(sched_name, connect_timeout_ms=10_000)
    d2h = _d2h_service(mesh_device)
    service = ttnn.LayerAckService(
        d2h, ring_name, source_rank=0, num_ack_layers=ACK_LOCAL_COUNT, ack_first_idx=0, ack_local_count=ACK_LOCAL_COUNT
    )
    service.start()
    try:
        records = [_record(mesh_device, 0, 0, 128), _record(mesh_device, 0, 128, 256)]
        ttnn.synchronize_device(mesh_device)
        for rec in records:
            _fire(d2h, rec, ACK_LOCAL_COUNT)
        got = 0
        deadline = time.monotonic() + 60
        while got < 2 * ACK_LOCAL_COUNT and time.monotonic() < deadline:
            got += consumer.try_consume_all()
            time.sleep(0.005)
        assert got == 2 * ACK_LOCAL_COUNT
        service.check()
    finally:
        service.stop()
        consumer.shutdown()
        router.stop()


def test_a_lost_record_is_reported_and_the_fifo_keeps_draining(mesh_device):
    """Chunk A sends two of its three records (one 'lost'); chunk B's identity then appears at slice 2,
    which the service reports through check() and stop() while still draining the D2H FIFO."""
    ring_name = f"/tt_lcq_ack_desync_{os.getpid()}"
    _unlink(ring_name)
    ring = LayerCompletionQueueV2.create(ring_name)
    d2h = _d2h_service(mesh_device)
    service = ttnn.LayerAckService(
        d2h,
        ring_name,
        source_rank=SOURCE_RANK,
        num_ack_layers=NUM_ACK_LAYERS,
        ack_first_idx=ACK_FIRST_IDX,
        ack_local_count=ACK_LOCAL_COUNT,
        ack_layer_ids=LAYER_IDS,
        protocol=2,
    )
    service.start()
    try:
        chunk_a = _record(mesh_device, 0, 0, 128)
        chunk_b = _record(mesh_device, 1, 128, 256)
        ttnn.synchronize_device(mesh_device)
        _fire(d2h, chunk_a, ACK_LOCAL_COUNT - 1)
        _fire(d2h, chunk_b, ACK_LOCAL_COUNT)
        _fire(d2h, chunk_b, ACK_LOCAL_COUNT)  # more records after the desync: must be drained, never emitted
        ttnn.synchronize_device(mesh_device)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and any(s.has_data() for s in d2h.get_sockets()):
            time.sleep(0.01)
        assert not any(s.has_data() for s in d2h.get_sockets()), "the service stopped draining the D2H FIFO"
        with pytest.raises(RuntimeError, match="saw a new chunk"):  # allow-pytest.raises: host-side error
            service.check()
        assert len(_pop(ring, ACK_LOCAL_COUNT - 1)) == ACK_LOCAL_COUNT - 1  # chunk A's records went out
        assert ring.try_pop() is None  # nothing after the desync
    finally:
        service.stop()  # the error was already taken by check(), so this does not raise
        ring.shutdown()
