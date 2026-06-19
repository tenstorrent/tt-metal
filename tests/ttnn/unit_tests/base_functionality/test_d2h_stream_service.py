# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Python correctness sweeps for ttnn.D2HStreamService."""

import pytest
import torch

import ttnn

from models.common.utility_functions import skip_for_slow_dispatch

# D2HStreamService claims FD dispatch-column service cores, which the runtime only permits when
# BOTH conditions hold (see tt_metal/impl/internal/service/service_core_manager.cpp):
#   1. Fast Dispatch is active (get_claimable_cores / claim TT_FATAL otherwise), and
#   2. the cluster is a Blackhole board or a UBB Galaxy (claim TT_FATAL otherwise).
#
# This is an allowlist of real clusters that satisfy condition 2. Simulator cluster types
# (SIMULATOR_*) are deliberately excluded: the simulator has no Fast Dispatch, so it can never
# satisfy condition 1 and would hit the "requires Fast Dispatch" TT_FATAL even on Blackhole arch.
# Slow Dispatch on real hardware is handled separately by skip_for_slow_dispatch() below.
# Anything not listed here (simulators, non-Galaxy Wormhole, future cluster types) skips cleanly.
_D2H_SUPPORTED_CLUSTER_TYPES = frozenset(
    {
        # Blackhole single/multi-card
        ttnn.cluster.ClusterType.P100,
        ttnn.cluster.ClusterType.P150,
        ttnn.cluster.ClusterType.P150_X2,
        ttnn.cluster.ClusterType.P150_X4,
        ttnn.cluster.ClusterType.P150_X8,
        ttnn.cluster.ClusterType.P300,
        ttnn.cluster.ClusterType.P300_X2,
        # UBB Galaxy (Wormhole and Blackhole)
        ttnn.cluster.ClusterType.GALAXY,
        ttnn.cluster.ClusterType.BLACKHOLE_GALAXY,
    }
)
pytestmark = [
    skip_for_slow_dispatch(),
    pytest.mark.skipif(
        ttnn.cluster.get_cluster_type() not in _D2H_SUPPORTED_CLUSTER_TYPES,
        reason="D2HStreamService is only supported on Blackhole and UBB Galaxy clusters",
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
        scratch_cb_size_bytes=scratch_cb_pages * per_row_bytes,
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
        scratch_cb_size_bytes=scratch_cb_pages * per_row_bytes,
        mapper=service_mapper,
    )

    _run_io_loop(service, iter_mapper, global_spec, shape_list, "tensor", mesh_device)
