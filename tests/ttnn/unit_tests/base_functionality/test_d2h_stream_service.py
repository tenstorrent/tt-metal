# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Python correctness sweeps for ttnn.D2HStreamService."""

import pytest
import torch

import ttnn
from models.common.utility_functions import skip_for_slow_dispatch, skip_with_llk_assert

# Only run on Blackhole Galaxy; smaller Blackhole boards (P150/P300) fail to pin host pages for
# the DMA buffer. See #47750.
_D2H_SUPPORTED_CLUSTER_TYPES = frozenset({ttnn.cluster.ClusterType.BLACKHOLE_GALAXY})
pytestmark = [
    skip_for_slow_dispatch(),
    skip_with_llk_assert("D2HStreamService fails to pin host DMA buffers. Issue: #47909"),
    pytest.mark.skipif(
        ttnn.cluster.get_cluster_type() not in _D2H_SUPPORTED_CLUSTER_TYPES,
        reason="D2HStreamService sweeps are only run on Blackhole Galaxy clusters (#47750)",
    ),
]

_DTYPE_TORCH = torch.int32
_DTYPE_TTNN = ttnn.uint32
_DTYPE_SIZE = 4
_IO_LOOPS = 10
_RANDINT_HIGH = 2**31


def _make_global_spec(shape: ttnn.Shape) -> ttnn.TensorSpec:
    return ttnn.TensorSpec(
        shape=shape,
        dtype=_DTYPE_TTNN,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _run_io_loop(
    service: ttnn.D2HStreamService,
    iter_mapper: ttnn.CppTensorToMesh,
    global_spec: ttnn.TensorSpec,
    shape_list: list,
    input_path: str,
    mesh_device,
    num_iters: int = _IO_LOOPS,
) -> None:
    if input_path == "tensor":
        drain_host = ttnn.from_torch(
            torch.zeros(shape_list, dtype=_DTYPE_TORCH), spec=global_spec, mesh_mapper=iter_mapper
        )
        service.read_from_tensor(drain_host)
    else:
        service.read_from_tensor_bytes()
    service.barrier()

    for i in range(num_iters):
        gen = torch.Generator()
        gen.manual_seed(i)
        src_torch = torch.randint(low=0, high=_RANDINT_HIGH, size=shape_list, dtype=_DTYPE_TORCH, generator=gen)
        expected_host = ttnn.from_torch(src_torch, spec=global_spec, mesh_mapper=iter_mapper)
        ttnn.copy_host_to_device_tensor(expected_host, service.get_backing_tensor())
        ttnn.synchronize_device(mesh_device)

        if input_path == "tensor":
            read_host = ttnn.from_torch(
                torch.zeros(shape_list, dtype=_DTYPE_TORCH),
                spec=global_spec,
                mesh_mapper=iter_mapper,
            )
            service.read_from_tensor(read_host)
            service.barrier()
            _verify_readback(service, expected_host, read_host)
        else:
            raw = service.read_from_tensor_bytes()
            service.barrier()
            read_vec = torch.frombuffer(raw, dtype=torch.int32).clone()
            expected_vec = src_torch.view(-1)
            assert torch.equal(read_vec.to(torch.int64), expected_vec.to(torch.int64)), f"iter {i}: bytes mismatch"


def _verify_readback(service, expected_host, actual_host) -> None:
    expected_subs = ttnn.get_device_tensors(expected_host)
    actual_subs = ttnn.get_device_tensors(actual_host)
    assert len(actual_subs) == len(expected_subs)
    for i, (actual, expected) in enumerate(zip(actual_subs, expected_subs)):
        a_t = ttnn.to_torch(actual).view(-1).to(torch.int64)
        e_t = ttnn.to_torch(expected).view(-1).to(torch.int64)
        assert torch.equal(a_t, e_t), f"device {i}: contents mismatch"


@pytest.mark.parametrize(
    "shape_list, scratch_cb_pages, fifo_pages",
    [
        ([1, 1, 1, 640], 1, 1),
        ([1, 1, 16, 640], 4, 16),
        ([1, 1, 7, 640], 4, 8),
    ],
)
@pytest.mark.parametrize("input_path", ["tensor", "bytes"])
def test_d2h_stream_service_replicated_sweep(
    mesh_device,
    shape_list,
    scratch_cb_pages,
    fifo_pages,
    input_path,
):
    shape = ttnn.Shape(shape_list)
    per_row_bytes = shape_list[-1] * _DTYPE_SIZE
    global_spec = _make_global_spec(shape)

    placements = [ttnn.PlacementReplicate() for _ in range(mesh_device.shape.dims())]
    iter_mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(placements=placements))

    service = ttnn.D2HStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=fifo_pages * per_row_bytes,
        max_socket_page_size_bytes=scratch_cb_pages * per_row_bytes,
    )

    _run_io_loop(service, iter_mapper, global_spec, shape_list, input_path, mesh_device)


def _sharded_sweep_patterns(mesh_shape, N, per_row):
    """Return (label, placements, shape_list) for each sharded placement pattern supported by mesh_shape."""
    num_rows, num_cols = mesh_shape[0], mesh_shape[1]
    patterns = []
    if num_rows >= 2:
        patterns.append(
            (
                "shard_rows_replicate_cols",
                [ttnn.PlacementShard(3), ttnn.PlacementReplicate()],
                [1, 1, N, num_rows * per_row],
            )
        )
    if num_cols >= 2:
        patterns.append(
            (
                "replicate_rows_shard_cols",
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)],
                [1, 1, N, num_cols * per_row],
            )
        )
    if num_rows >= 2 and num_cols >= 2:
        patterns.append(
            (
                "full_shard_2d",
                [ttnn.PlacementShard(2), ttnn.PlacementShard(3)],
                [1, 1, num_rows * N, num_cols * per_row],
            )
        )
    return patterns


@pytest.mark.parametrize(
    "pattern",
    ["shard_rows_replicate_cols", "replicate_rows_shard_cols", "full_shard_2d"],
)
@pytest.mark.parametrize(
    "N, scratch_cb_pages, fifo_pages",
    [
        (1, 1, 1),
        (16, 4, 16),
        (7, 4, 8),
    ],
)
def test_d2h_stream_service_sharded_sweep(mesh_device, pattern, N, scratch_cb_pages, fifo_pages):
    """Mesh-sharded tensors over the D2H Tensor read path (mirrors the C++ Sharded_Sweep).

    Requires a 2D mesh. Each device's shard reads back independently. The bytes path
    is skipped here: under sharded placements it goes through the auto-derived
    composer, which has known limitations; the Tensor path is composer-free per device.
    """
    mesh_shape = mesh_device.shape
    if mesh_shape.dims() != 2:
        pytest.skip(f"sharded sweep requires a 2D mesh; got {mesh_shape}")
    if mesh_shape[0] < 2 and mesh_shape[1] < 2:
        pytest.skip(f"no shardable mesh axis on {mesh_shape}")

    per_row = 640
    per_row_bytes = per_row * _DTYPE_SIZE

    patterns = _sharded_sweep_patterns(mesh_shape, N, per_row)
    pattern_map = {label: (placements, shape_list) for label, placements, shape_list in patterns}
    if pattern not in pattern_map:
        pytest.skip(f"pattern {pattern!r} not supported on mesh shape {mesh_shape}")

    placements, shape_list = pattern_map[pattern]
    global_spec = _make_global_spec(ttnn.Shape(shape_list))
    mapper_config = ttnn.MeshMapperConfig(placements=placements)
    iter_mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)
    service_mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)

    service = ttnn.D2HStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=fifo_pages * per_row_bytes,
        max_socket_page_size_bytes=scratch_cb_pages * per_row_bytes,
        mapper=service_mapper,
    )

    _run_io_loop(service, iter_mapper, global_spec, shape_list, "tensor", mesh_device)


# ---- metadata-only acks against a host that is not reading -------------------------------------
#
# The layer-ack path: a device op forwards one record per transfer into the service core and
# bumps its ack counter without waiting. The service core keeps a ring of records and the ack
# op refuses to overwrite one the service has not sent yet, so a host that stops reading sees
# every record, in order, and the service core never wedges on a counter it can no longer match.

_ACK_RECORD_BYTES = 12  # {slot_id, pos_start, pos_end}


def _metadata_only_service(mesh_device, fifo_pages: int) -> ttnn.D2HStreamService:
    # The socket page is the PCIe-aligned record (<= 64 B); the ring is sized off the real page count.
    return ttnn.D2HStreamService(
        mesh_device,
        global_spec=None,
        fifo_size_bytes=fifo_pages * 64,
        worker_cores=ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
        metadata_size_bytes=_ACK_RECORD_BYTES,
    )


def _ack_record(mesh_device, i: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.tensor([i, 1000 + i, 2000 + i], dtype=torch.int64).reshape(1, 1, 1, 3),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _read_records(service: ttnn.D2HStreamService, n: int, timeout_s: float) -> list:
    import struct
    import threading

    out = []

    def run():
        for _ in range(n):
            out.append(struct.unpack("<3I", service.read_metadata()))

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_s)
    assert not t.is_alive(), f"host read stalled after {len(out)} of {n} records: the service core stopped acking"
    return out


def _fire_acks(service, mesh_device, n: int) -> list:
    records = [_ack_record(mesh_device, i) for i in range(n)]
    ttnn.synchronize_device(mesh_device)
    for rec in records:
        ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(service, metadata=rec)
    return records


def test_metadata_acks_queue_on_the_service_core_while_the_host_is_not_reading(mesh_device):
    service = _metadata_only_service(mesh_device, fifo_pages=4)
    ring = service.get_metadata_ring_slots()
    fifo_pages = ring - service.get_slot_count()
    # More acks than the socket FIFO and the staging CB can hold, but still within the ring, so no
    # ack op has to wait. Before the ring this wedged the service core on its exact-match ack check
    # and sent later records under earlier acks.
    n = ring + fifo_pages - 2
    keep = _fire_acks(service, mesh_device, n)
    ttnn.synchronize_device(mesh_device)
    records = _read_records(service, n, timeout_s=60)
    assert records == [(i, 1000 + i, 2000 + i) for i in range(n)]
    del keep


def test_metadata_ack_waits_for_the_service_once_the_ring_is_full(mesh_device):
    service = _metadata_only_service(mesh_device, fifo_pages=4)
    ring = service.get_metadata_ring_slots()
    fifo_pages = ring - service.get_slot_count()
    # The last few ack ops find every slot holding an unsent record and must wait for the host's
    # reads to free them; an overwrite would show up as a wrong or missing record below.
    n = ring + fifo_pages + 3
    keep = _fire_acks(service, mesh_device, n)
    records = _read_records(service, n, timeout_s=60)
    ttnn.synchronize_device(mesh_device)
    assert records == [(i, 1000 + i, 2000 + i) for i in range(n)]
    del keep
