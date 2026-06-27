# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Python correctness sweeps for ttnn.H2DStreamService."""

import pytest
import torch

import ttnn

from models.common.utility_functions import skip_for_slow_dispatch

# H2DStreamService claims FD dispatch-column service cores and DMA-pins host memory; it is only
# supported on Blackhole Galaxy and requires fast dispatch. Skip the whole module on any other
# configuration so unsupported runners skip cleanly instead of hitting the claim TT_FATAL.
pytestmark = [
    skip_for_slow_dispatch(),
    pytest.mark.skipif(
        ttnn.cluster.get_cluster_type() != ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
        reason="H2DStreamService is only supported on Blackhole Galaxy",
    ),
]


# int32 source viewed as UINT32; compared as int64 (lossless for 0..INT32_MAX).
_DTYPE_TORCH = torch.int32
_DTYPE_TTNN = ttnn.uint32
_DTYPE_SIZE = 4

_IO_LOOPS = 20

# Exclusive upper bound for torch.randint; negatives break the int64 comparison.
_RANDINT_HIGH = 2**31


def _run_io_loop(
    service: ttnn.H2DStreamService,
    iter_mapper: ttnn.CppTensorToMesh,
    global_spec: ttnn.TensorSpec,
    shape_list: list,
    input_path: str,
    num_iters: int = _IO_LOOPS,
) -> None:
    """Drive `num_iters` write+barrier+verify cycles on a persistent service."""
    for i in range(num_iters):
        gen = torch.Generator()
        gen.manual_seed(i)
        src_torch = torch.randint(low=0, high=_RANDINT_HIGH, size=shape_list, dtype=_DTYPE_TORCH, generator=gen)
        expected_host = ttnn.from_torch(src_torch, spec=global_spec, mesh_mapper=iter_mapper)
        _push_source(service, expected_host, src_torch, input_path)
        service.barrier()
        try:
            _verify_readback(service, expected_host)
        except AssertionError as e:
            raise AssertionError(f"iter {i}: {e}") from e


def _make_global_spec(shape: ttnn.Shape) -> ttnn.TensorSpec:
    return ttnn.TensorSpec(
        shape=shape,
        dtype=_DTYPE_TTNN,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _push_source(
    service: ttnn.H2DStreamService,
    host_tensor: ttnn.Tensor,
    src_torch: torch.Tensor,
    input_path: str,
) -> None:
    """Push the source through the service via the requested input path."""
    if input_path == "tensor":
        service.forward_to_tensor(host_tensor)
    elif input_path == "bytes":
        service.forward_to_tensor_bytes(src_torch.contiguous().numpy())
    else:
        raise ValueError(f"unknown input_path: {input_path!r}")


def _verify_readback(service: ttnn.H2DStreamService, expected_host: ttnn.Tensor) -> None:
    """Compare service's per-device readback against `expected_host`'s shards."""
    expected_subs = ttnn.get_device_tensors(expected_host)
    actual_subs = ttnn.get_device_tensors(service.get_backing_tensor())
    assert len(actual_subs) == len(
        expected_subs
    ), f"got {len(actual_subs)} actual vs {len(expected_subs)} expected sub-tensors"
    # Compare as int64: torch.equal refuses to promote between Int and UInt.
    for i, (actual, expected) in enumerate(zip(actual_subs, expected_subs)):
        a_t = ttnn.to_torch(actual).view(-1).to(torch.int64)
        e_t = ttnn.to_torch(expected).view(-1).to(torch.int64)
        assert a_t.shape == e_t.shape, f"device {i}: shape mismatch {a_t.shape} vs {e_t.shape}"
        assert torch.equal(a_t, e_t), f"device {i}: contents mismatch"


# Replicated sweep: mapper=None exercises the C++ replicate-on-every-mesh-dim default.
@pytest.mark.parametrize(
    "shape_list, scratch_cb_pages, fifo_pages",
    [
        # Single chunk, big innermost dim.
        ([1, 1, 1, 65536], 1, 1),
        # Single chunk, small innermost dim.
        ([1, 1, 1, 640], 1, 1),
        # Multi-chunk even split: 16 pages, CB for 4 -> 4 chunks of 4.
        ([1, 1, 16, 640], 4, 16),
        # Divisor fallback: 7 (prime) pages -> pages_per_chunk == 1.
        ([1, 1, 7, 640], 4, 8),
        # Larger innermost dim with a smaller CB budget.
        ([1, 1, 4, 4096], 2, 4),
    ],
)
@pytest.mark.parametrize("input_path", ["tensor", "bytes"])
def test_h2d_stream_service_replicated_sweep(
    mesh_device,
    shape_list,
    scratch_cb_pages,
    fifo_pages,
    input_path,
):
    shape = ttnn.Shape(shape_list)
    per_row_bytes = shape_list[-1] * _DTYPE_SIZE
    global_spec = _make_global_spec(shape)

    # `iter_mapper` matches the service's internal replicate-on-all default.
    placements = [ttnn.PlacementReplicate() for _ in range(mesh_device.shape.dims())]
    iter_mapper = ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(placements=placements))

    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=fifo_pages * per_row_bytes,
        scratch_cb_size_bytes=scratch_cb_pages * per_row_bytes,
        # mapper omitted -> C++ replicate-on-every-mesh-dim default.
    )

    _run_io_loop(service, iter_mapper, global_spec, shape_list, input_path)


# Compute the global tensor shape from a per-device shape + placement vector.
def _global_shape_for_placements(per_device_shape, placements, mesh_device):
    global_shape = list(per_device_shape)
    mesh_shape = mesh_device.shape
    for mesh_dim_idx, placement in enumerate(placements):
        if isinstance(placement, ttnn.PlacementShard):
            global_shape[placement.dim] *= mesh_shape[mesh_dim_idx]
    return global_shape


# Sharded sweep across placement patterns × per-device shapes × chunking. Requires an 8x4 mesh.
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("input_path", ["tensor", "bytes"])
@pytest.mark.parametrize(
    "placements, per_device_shape, cb_pages, fifo_pages",
    [
        # Shard innermost (dim 3) across long mesh edge (8), replicate short (4).
        pytest.param(
            [ttnn.PlacementShard(3), ttnn.PlacementReplicate()],
            [1, 1, 1, 640],
            1,
            1,
            id="shard_innermost_long_replicate_short",
        ),
        # Flipped: shard innermost across the short mesh edge (4), replicate the long edge.
        pytest.param(
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)],
            [1, 1, 1, 640],
            1,
            1,
            id="replicate_long_shard_innermost_short",
        ),
        # Shard on a non-innermost tensor dim; each device sees 2 tensor pages.
        pytest.param(
            [ttnn.PlacementShard(2), ttnn.PlacementReplicate()],
            [1, 1, 2, 640],
            1,
            1,
            id="shard_dim2_long_replicate_short",
        ),
        # Full 2D shard: distinct slice at every coord along two tensor dims.
        # Global = [1, 1, 64, 2560]; per-device = [1, 1, 8, 640]. No replication.
        pytest.param(
            [ttnn.PlacementShard(2), ttnn.PlacementShard(3)],
            [1, 1, 8, 640],
            1,
            1,
            id="full_2d_shard",
        ),
        # Multi-chunk sharded transfer: per-device 16 pages, CB for 4 -> 4 chunks of 4.
        pytest.param(
            [ttnn.PlacementShard(3), ttnn.PlacementReplicate()],
            [1, 1, 16, 640],
            4,
            16,
            id="shard_innermost_long_multi_chunk",
        ),
    ],
)
def test_h2d_stream_service_sharded_sweep(mesh_device, placements, per_device_shape, cb_pages, fifo_pages, input_path):
    shape_list = _global_shape_for_placements(per_device_shape, placements, mesh_device)
    shape = ttnn.Shape(shape_list)
    global_spec = _make_global_spec(shape)
    per_row_bytes = per_device_shape[-1] * _DTYPE_SIZE

    # `service_mapper` is consumed by the ctor; `iter_mapper` stays alive for the loop.
    mapper_config = ttnn.MeshMapperConfig(placements=placements)
    iter_mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)
    service_mapper = ttnn.create_mesh_mapper(mesh_device, mapper_config)

    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=fifo_pages * per_row_bytes,
        scratch_cb_size_bytes=cb_pages * per_row_bytes,
        mapper=service_mapper,  # ownership transferred; handle now invalidated.
    )

    _run_io_loop(service, iter_mapper, global_spec, shape_list, input_path)


# The tests below validate the worker_cores + metadata binding surface only (no worker-sync data
# flow, which needs a C++ worker kernel); they never run a successful forward when worker_cores is set.


def _make_minimal_service(mesh_device, **kwargs):
    """Smallest viable service for API-surface tests (a [1,1,1,640] uint32 tensor)."""
    shape = ttnn.Shape([1, 1, 1, 640])
    global_spec = _make_global_spec(shape)
    per_row_bytes = 640 * _DTYPE_SIZE
    return ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=per_row_bytes,
        scratch_cb_size_bytes=per_row_bytes,
        **kwargs,
    )


def _mesh_coords(mesh_device):
    """Enumerate every MeshCoordinate in the mesh shape. Supports 1D and 2D."""
    shape = mesh_device.shape
    n_dims = shape.dims()
    if n_dims == 1:
        return [ttnn.MeshCoordinate(i) for i in range(shape[0])]
    if n_dims == 2:
        return [ttnn.MeshCoordinate(row, col) for row in range(shape[0]) for col in range(shape[1])]
    raise NotImplementedError(f"unsupported mesh dim count: {n_dims}")


def _dummy_source_bytes(shape_list):
    """Zero-filled uint32 numpy source for negative tests; never expected to land on device."""
    src = torch.zeros(shape_list, dtype=_DTYPE_TORCH)
    return src.contiguous().numpy()


def test_worker_sync_getters_addresses_are_nonzero(mesh_device):
    """worker_cores set: the worker-sync getters return live L1 addresses for every coord."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores)

    assert service.get_data_ready_sem_addr() != 0
    for coord in _mesh_coords(mesh_device):
        assert service.get_consumed_counter_addr(coord) != 0
        # `get_service_core` returns a CoreCoord; smoke-check x/y access.
        sc = service.get_service_core(coord)
        _ = (sc.x, sc.y)


def test_metadata_getter_address_is_nonzero(mesh_device):
    """worker_cores + metadata_size_bytes set: get_metadata_addr returns a live L1 address."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores, metadata_size_bytes=16)
    assert service.get_metadata_addr() != 0


def test_worker_sync_getters_raise_when_disabled(mesh_device):
    """worker_cores unset -> the three worker-sync getters all raise."""
    service = _make_minimal_service(mesh_device)
    with pytest.raises(RuntimeError):
        service.get_data_ready_sem_addr()
    # Per-coord getters need a coord; pick coord 0.
    coord = _mesh_coords(mesh_device)[0]
    with pytest.raises(RuntimeError):
        service.get_consumed_counter_addr(coord)


def test_metadata_getter_raises_when_disabled(mesh_device):
    """worker_cores set but metadata_size_bytes=0 -> get_metadata_addr raises."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores)
    with pytest.raises(RuntimeError):
        service.get_metadata_addr()


def test_ctor_rejects_metadata_without_worker_cores(mesh_device):
    """metadata_size_bytes > 0 requires worker_cores; ctor rejects otherwise."""
    with pytest.raises(RuntimeError):
        _make_minimal_service(mesh_device, metadata_size_bytes=16)


def test_ctor_rejects_metadata_larger_than_socket_page(mesh_device):
    """metadata_size_bytes larger than the socket page size is rejected at construction."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    with pytest.raises(RuntimeError):
        _make_minimal_service(mesh_device, worker_cores=worker_cores, metadata_size_bytes=1 << 20)


def test_forward_bytes_rejects_wrong_metadata_size(mesh_device):
    """Service expects 16 B of metadata, caller passes 8 B."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores, metadata_size_bytes=16)
    data = _dummy_source_bytes([1, 1, 1, 640])
    with pytest.raises(RuntimeError):
        service.forward_to_tensor_bytes(data, metadata=b"x" * 8)


def test_forward_bytes_rejects_metadata_when_disabled(mesh_device):
    """metadata_size_bytes=0 at construction, caller still passes bytes."""
    service = _make_minimal_service(mesh_device)
    data = _dummy_source_bytes([1, 1, 1, 640])
    with pytest.raises(RuntimeError):
        service.forward_to_tensor_bytes(data, metadata=b"x" * 16)


def test_forward_bytes_rejects_missing_metadata_when_required(mesh_device):
    """Service expects 16 B of metadata, caller passes none (default b'')."""
    worker_cores = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
    service = _make_minimal_service(mesh_device, worker_cores=worker_cores, metadata_size_bytes=16)
    data = _dummy_source_bytes([1, 1, 1, 640])
    with pytest.raises(RuntimeError):
        service.forward_to_tensor_bytes(data)
