# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""LayerAckService on a galaxy: real D2H ack records through the device-side ack op, labelled with
their global layer and pushed into a layer-completion ring this test consumes."""

import os
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_slow_dispatch, skip_with_llk_assert
from ttnn._experimental.layer_completion import LayerCompletionQueue

pytestmark = [
    skip_for_slow_dispatch(),
    skip_with_llk_assert("D2HStreamService fails to pin host DMA buffers. Issue: #47909"),
    pytest.mark.skipif(
        ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
        reason="D2HStreamService runs on Blackhole Galaxy clusters only (#47750)",
    ),
]

RECORD_BYTES = 12  # {slot_id, pos_start, pos_end}
FIRST_RECORD = 3  # second rank of a hybrid stack with three KV-writing layers per rank
RECORD_COUNT = 3
LAYER_IDS = [15, 19, 23]


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
        metadata_ring=True,
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


def test_messages_carry_the_record_identity_and_the_mapped_layer(mesh_device):
    ring_name = f"/tt_lcq_ack_{os.getpid()}"
    _unlink(ring_name)
    ring = LayerCompletionQueue.create(
        ring_name, RECORD_COUNT
    )  # this test is the consumer; the ring must exist before start()
    d2h = _d2h_service(mesh_device)
    service = ttnn.LayerAckService(
        d2h, ring_name, first_record=FIRST_RECORD, record_count=RECORD_COUNT, layer_ids=LAYER_IDS
    )
    service.start()
    try:
        chunks = [(0, 0, 128), (1, 128, 256), (0, 256, 384), (2, 0, 512)]
        records = [_record(mesh_device, *c) for c in chunks]
        ttnn.synchronize_device(mesh_device)
        for rec in records:
            _fire(d2h, rec, RECORD_COUNT)
        msgs = _pop(ring, len(chunks) * RECORD_COUNT)
        for i, chunk in enumerate(chunks):
            for j in range(RECORD_COUNT):
                slot_id, pos_start, pos_end, layer_start, layer_end, host_timestamp_ns = msgs[i * RECORD_COUNT + j]
                assert (slot_id, pos_start, pos_end) == chunk
                assert (layer_start, layer_end) == (LAYER_IDS[j], LAYER_IDS[j] + 1)
                assert host_timestamp_ns > 0
        service.check()
    finally:
        service.stop()
        ring.shutdown()


def test_a_lost_record_is_reported_and_the_fifo_keeps_draining(mesh_device):
    """Chunk A sends two of its three records (one 'lost'); chunk B's identity then appears at slice 2,
    which the service reports through check() and stop() while still draining the D2H FIFO."""
    ring_name = f"/tt_lcq_ack_desync_{os.getpid()}"
    _unlink(ring_name)
    ring = LayerCompletionQueue.create(ring_name, RECORD_COUNT)
    d2h = _d2h_service(mesh_device)
    service = ttnn.LayerAckService(
        d2h, ring_name, first_record=FIRST_RECORD, record_count=RECORD_COUNT, layer_ids=LAYER_IDS
    )
    service.start()
    try:
        chunk_a = _record(mesh_device, 0, 0, 128)
        chunk_b = _record(mesh_device, 1, 128, 256)
        ttnn.synchronize_device(mesh_device)
        _fire(d2h, chunk_a, RECORD_COUNT - 1)
        _fire(d2h, chunk_b, RECORD_COUNT)
        _fire(d2h, chunk_b, RECORD_COUNT)  # more records after the desync: must be drained, never emitted
        ttnn.synchronize_device(mesh_device)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and any(s.has_data() for s in d2h.get_sockets()):
            time.sleep(0.01)
        assert not any(s.has_data() for s in d2h.get_sockets()), "the service stopped draining the D2H FIFO"
        with pytest.raises(RuntimeError, match="saw a new chunk"):  # allow-pytest.raises: host-side error
            service.check()
        assert len(_pop(ring, RECORD_COUNT - 1)) == RECORD_COUNT - 1  # chunk A's records went out
        assert ring.try_pop() is None  # nothing after the desync
    finally:
        service.stop()  # the error was already taken by check(), so this does not raise
        ring.shutdown()
