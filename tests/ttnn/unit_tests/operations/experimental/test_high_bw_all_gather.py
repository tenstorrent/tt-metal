# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Production-shaped correctness and bandwidth coverage for high_bw_all_gather."""

import csv
import gc
import math
import os
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


def _fabric_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = 14 * 1024
    return config


def _device_params(fabric_config):
    return {
        "fabric_config": fabric_config,
        "fabric_router_config": _fabric_router_config(),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        "l1_small_size": 2048,
    }


# A line has boundary ranks and fewer usable paths than the bidirectional ring
# schedule, so its qualification floor is half the ring floor.
_LINE_PERF_DEVICE_PARAMS = [
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_1D),
        45.0,
        45.0,
        id="fabric_1d_line",
    ),
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_1D_RING),
        90.0,
        90.0,
        id="fabric_1d_ring",
    ),
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_2D),
        45.0,
        45.0,
        id="fabric_2d",
    ),
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_X),
        45.0,
        90.0,
        id="fabric_2d_torus_x",
    ),
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_Y),
        90.0,
        45.0,
        id="fabric_2d_torus_y",
    ),
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY),
        90.0,
        90.0,
        id="fabric_2d_torus_xy",
    ),
]

_FABRIC_2D_LINE_DEVICE_PARAMS = pytest.param(
    _device_params(ttnn.FabricConfig.FABRIC_2D),
    id="fabric_2d_line",
)

_FABRIC_2D_TORUS_XY_DEVICE_PARAMS = pytest.param(
    _device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY),
    id="fabric_2d_torus_xy",
)

_TEST_CASES = [
    ("bf16_1152b_rows", ttnn.bfloat16, 576, ttnn.ROW_MAJOR_LAYOUT, 1152),
    ("scaled_fp8_704b_rows", ttnn.fp8_e4m3, 656, ttnn.ROW_MAJOR_LAYOUT, 704),
    ("bf16_tiles", ttnn.bfloat16, 576, ttnn.TILE_LAYOUT, 2048),
    ("bfloat8_b_tiles", ttnn.bfloat8_b, 576, ttnn.TILE_LAYOUT, 1088),
]

_PERF_ROWS_PER_DEVICE = 65536
_ACCURACY_ROWS_PER_DEVICE = 1024
_MEDIUM_BF16_ROWS_PER_DEVICE = 1200  # 5.5296 MB/link on the 8-device, 2-link qualification mesh.
_NUM_LINKS = 2

# CI performance targets are global sequence lengths. Both divide exactly by
# the four-rank QuietBox ring and eight-rank Blackhole Galaxy line.
_CI_PERF_GLOBAL_ROWS = (55_000, 512 * 1024)
_CI_PERF_TEST_CASES = [
    ("bf16_row_major", ttnn.bfloat16, 576, ttnn.ROW_MAJOR_LAYOUT, 1152),
    ("fp8_row_major", ttnn.fp8_e4m3, 656, ttnn.ROW_MAJOR_LAYOUT, 704),
    ("bf16_tiled", ttnn.bfloat16, 576, ttnn.TILE_LAYOUT, 2048),
    # fp8_e4m3 tile tensors are unsupported. bfloat8_b is the supported FP8
    # tile representation and has a 64-byte tile exponent header.
    ("bfloat8_tiled", ttnn.bfloat8_b, 576, ttnn.TILE_LAYOUT, 1088),
]

# These floors are specific to a physical four-chip QuietBox.  The 55K and
# 512K gates measured 85--90 GB/s and 119--121 GB/s respectively across every
# supported payload on the local QuietBox. CI hardware measures about 103--104
# GB/s at 512K, so retain enough headroom while still catching a lost
# wraparound path (which falls to about 64--67 GB/s).
_QUIETBOX_CI_MIN_BANDWIDTH_GBPS = {
    55_000: 70.0,
    512 * 1024: 100.0,
}


def _is_physical_quietbox(mesh_device):
    """Whether ``mesh_device`` is the complete four-chip QuietBox topology.

    A four-rank submesh may be opened on a larger system, but its bandwidth is
    not representative of a QuietBox.  Require both the physical system and
    the opened mesh to contain exactly four devices before applying the
    QuietBox performance gate.
    """
    return tuple(mesh_device.shape) == (4, 1) and mesh_device.get_num_devices() == ttnn.get_num_devices() == 4


def _rank_line_mesh(mesh_device):
    """Return a four-rank Blackhole line, preferring an eight-rank line when available.

    QuietBox exposes its four Blackhole devices as a 4x1 mesh. LoudBox exposes
    eight devices as 2x4, so the corresponding physical rank line is a 1x4
    submesh. Larger systems retain an 8-rank line where one is available.
    """
    mesh_shape = tuple(mesh_device.shape)
    if mesh_shape == (4, 1):
        return mesh_device, 0
    if mesh_shape == (1, 4):
        return mesh_device, 1
    if mesh_shape == (2, 4):
        return mesh_device.create_submesh(ttnn.MeshShape(1, 4)), 1
    if mesh_shape[0] >= 8:
        return mesh_device.create_submesh(ttnn.MeshShape(8, 1)), 0
    if mesh_shape[1] >= 8:
        return mesh_device.create_submesh(ttnn.MeshShape(1, 8)), 1
    if mesh_shape[0] >= 4:
        return mesh_device.create_submesh(ttnn.MeshShape(4, 1)), 0
    if mesh_shape[1] >= 4:
        return mesh_device.create_submesh(ttnn.MeshShape(1, 4)), 1
    pytest.skip(
        "high_bw_all_gather requires a 4-rank QuietBox/LoudBox line or an 8-rank logical line; "
        f"got physical mesh {mesh_shape}"
    )


def _make_tensor(mesh_device, host_tensor, dtype, layout, mesh_mapper):
    tensor = ttnn.from_torch(
        host_tensor,
        dtype=ttnn.bfloat16 if dtype == ttnn.fp8_e4m3 else dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
        device=mesh_device,
    )
    return ttnn.typecast(tensor, dtype) if dtype == ttnn.fp8_e4m3 else tensor


def _fp8_payloads(tensor, mesh_device):
    assert tensor.dtype == ttnn.fp8_e4m3
    device_tensors = ttnn.get_device_tensors(tensor)
    local_shape = tuple(device_tensors[0].shape)
    host_bytes = (
        ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).contiguous().view(torch.uint8)
    )
    return host_bytes.reshape(len(device_tensors), *local_shape)


def _profile_high_bw_all_gather(mesh_device, run):
    _, records = profile_realtime_program(mesh_device, run, collect_all=True, record_timeout_seconds=5.0)
    programs = {}
    for record in records:
        sources = [source.replace("\\", "/") for source in record["kernel_sources"]]
        if not any("/experimental/high_bw_all_gather/" in source for source in sources):
            continue
        assert any(source.endswith("/unicast_writer.cpp") for source in sources)
        runtime_id = record["runtime_id"]
        programs[runtime_id] = max(programs.get(runtime_id, 0.0), record["duration_ns"])
    assert programs, "realtime profiler returned no high_bw_all_gather program"
    return sum(programs.values())


def _to_torch_with_device_padding(tensor):
    host_tensor = ttnn.to_torch(tensor)
    padded_shape = tuple(tensor.padded_shape)
    padding = []
    for logical_extent, padded_extent in reversed(list(zip(host_tensor.shape, padded_shape))):
        padding.extend((0, padded_extent - logical_extent))
    return torch.nn.functional.pad(host_tensor, padding)


def _assert_exact_all_gather(device_input, persistent_output, mesh_device, dtype):
    """The collective performs no arithmetic, so compare the gathered device values exactly."""
    if dtype == ttnn.fp8_e4m3:
        expected = torch.cat(list(_fp8_payloads(device_input, mesh_device)), dim=2)
        actual_outputs = _fp8_payloads(persistent_output, mesh_device)
    else:
        # Compare the device representation so quantized types and padding inserted
        # independently into each local tile shard are both represented exactly.
        expected = torch.cat(
            [_to_torch_with_device_padding(tensor) for tensor in ttnn.get_device_tensors(device_input)],
            dim=2,
        )
        actual_outputs = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(persistent_output)]
    for actual in actual_outputs:
        assert torch.equal(actual, expected)


def _assert_exact_replicated_output(host_input, persistent_output, mesh_device, dtype, layout):
    """Compare rank-local outputs with an independently quantized replicated host reference."""
    expected_output = _make_tensor(
        mesh_device,
        host_input,
        dtype,
        layout,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    if dtype == ttnn.fp8_e4m3:
        expected_outputs = _fp8_payloads(expected_output, mesh_device)
        actual_outputs = _fp8_payloads(persistent_output, mesh_device)
    else:
        expected_outputs = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(expected_output)]
        actual_outputs = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(persistent_output)]
    assert len(actual_outputs) == len(expected_outputs)
    for actual, expected in zip(actual_outputs, expected_outputs):
        assert torch.equal(actual, expected)


def _run_high_bw_all_gather_accuracy(
    mesh_device, dtype, width, layout, expected_page_size, cluster_axis, rows_per_device
):
    global_shape = (1, 1, rows_per_device * mesh_device.shape[cluster_axis], width)
    torch.manual_seed(0)
    host_input = torch.rand(global_shape, dtype=torch.bfloat16)
    device_input = _make_tensor(
        mesh_device,
        host_input,
        dtype,
        layout,
        ttnn.ShardTensor2dMesh(
            mesh_device, dims=(2, None) if cluster_axis == 0 else (None, 2), mesh_shape=tuple(mesh_device.shape)
        ),
    )
    local_padded_shape = ttnn.get_device_tensors(device_input)[0].padded_shape
    output_shape = list(local_padded_shape)
    output_shape[2] *= mesh_device.shape[cluster_axis]
    persistent_output = _make_tensor(
        mesh_device,
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype,
        layout,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    assert ttnn.get_device_tensors(device_input)[0].buffer_aligned_page_size() == expected_page_size

    ttnn.experimental.high_bw_all_gather(
        device_input,
        dim=2,
        output_tensor=persistent_output,
        cluster_axis=cluster_axis,
        num_links=_NUM_LINKS,
    )
    ttnn.synchronize_device(mesh_device)

    _assert_exact_all_gather(device_input, persistent_output, mesh_device, dtype)


@pytest.mark.parametrize("device_params", [_FABRIC_2D_TORUS_XY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_high_bw_all_gather_preserves_same_dim_sharding_on_other_axis(mesh_device, expect_error):
    """Gathering TP must preserve SP when both mesh axes shard the same tensor dimension."""
    global_shape = (1, 4, 256, 32)
    torch.manual_seed(0)
    host_input = torch.rand(global_shape, dtype=torch.bfloat16)
    source_placements = [ttnn.PlacementShard(2), ttnn.PlacementShard(1)]
    gathered_placements = [ttnn.PlacementShard(2), ttnn.PlacementReplicate()]
    sp_gathered_placements = [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)]
    duplicate_placements = [ttnn.PlacementShard(2), ttnn.PlacementShard(2)]

    source = _make_tensor(
        mesh_device,
        host_input,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(source_placements, mesh_device.shape)),
    )
    inverse_output = _make_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(gathered_placements, mesh_device.shape)),
    )
    flattened_source = _make_tensor(
        mesh_device,
        host_input,
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    with expect_error(RuntimeError, "tensor distribution axes to match the device mesh axes"):
        ttnn.experimental.high_bw_all_gather(
            flattened_source,
            dim=2,
            output_tensor=inverse_output,
            cluster_axis=1,
        )

    inverse_output = ttnn.experimental.high_bw_all_gather(
        source,
        dim=1,
        output_tensor=inverse_output,
        cluster_axis=1,
    )
    duplicate_sharded_input = ttnn.mesh_partition(
        inverse_output,
        dim=2,
        cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    duplicate_topology = duplicate_sharded_input.tensor_topology()
    duplicate_sharded_input.update_tensor_topology(
        ttnn.TensorTopology(
            duplicate_topology.distribution_shape(),
            duplicate_placements,
            duplicate_topology.mesh_coords(),
        )
    )
    assert [
        placement.dim if isinstance(placement, ttnn.PlacementShard) else None
        for placement in duplicate_sharded_input.tensor_topology().placements()
    ] == [2, 2]

    persistent_output = _make_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(gathered_placements, mesh_device.shape)),
    )
    gathered = ttnn.experimental.high_bw_all_gather(
        duplicate_sharded_input,
        dim=2,
        output_tensor=persistent_output,
        cluster_axis=1,
    )
    ttnn.synchronize_device(mesh_device)

    assert [
        placement.dim if isinstance(placement, ttnn.PlacementShard) else None
        for placement in gathered.tensor_topology().placements()
    ] == [2, None]
    local_outputs = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(gathered)]
    rows_per_sp_shard = global_shape[2] // mesh_device.shape[0]
    for device_idx, actual in enumerate(local_outputs):
        sp_rank = device_idx // mesh_device.shape[1]
        expected = host_input[
            :,
            :,
            sp_rank * rows_per_sp_shard : (sp_rank + 1) * rows_per_sp_shard,
            :,
        ]
        assert torch.equal(actual, expected)

    # A size-two Torus dimension is a line, not a ring. Exercise that schedule
    # with one link while preserving the independent TP shard.
    sp_output = _make_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=torch.bfloat16),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.create_mesh_mapper(mesh_device, ttnn.MeshMapperConfig(sp_gathered_placements, mesh_device.shape)),
    )
    # Queue a burst without host synchronization to cover persistent semaphore
    # reuse across adjacent invocations. The first call is a cache miss and the
    # remainder are cache hits.
    for _ in range(16):
        sp_gathered = ttnn.experimental.high_bw_all_gather(
            source,
            dim=2,
            output_tensor=sp_output,
            cluster_axis=0,
            num_links=1,
        )
    ttnn.synchronize_device(mesh_device)
    assert [
        placement.dim if isinstance(placement, ttnn.PlacementShard) else None
        for placement in sp_gathered.tensor_topology().placements()
    ] == [None, 1]
    columns = mesh_device.shape[1]
    for device_idx, actual in enumerate(ttnn.get_device_tensors(sp_gathered)):
        tp_rank = device_idx % columns
        expected = host_input[:, tp_rank : tp_rank + 1, :, :]
        assert torch.equal(ttnn.to_torch(actual), expected)

    # Gathering the innermost dimension in row-major layout exercises the
    # concat path, where several input chunks share one output page.
    concat_host = torch.rand((1, 1, 32, 128), dtype=torch.bfloat16)
    concat_input = _make_tensor(
        mesh_device,
        concat_host,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.create_mesh_mapper(
            mesh_device,
            ttnn.MeshMapperConfig(
                [ttnn.PlacementReplicate(), ttnn.PlacementShard(3)],
                mesh_device.shape,
            ),
        ),
    )
    concat_output = _make_tensor(
        mesh_device,
        torch.zeros_like(concat_host),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    concat_output = ttnn.experimental.high_bw_all_gather(
        concat_input,
        dim=3,
        output_tensor=concat_output,
        cluster_axis=1,
        num_links=1,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_exact_replicated_output(
        concat_host,
        concat_output,
        mesh_device,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    )


def _run_high_bw_all_gather_perf(
    mesh_device,
    dtype,
    width,
    layout,
    expected_page_size,
    min_bandwidth_gbps,
    cluster_axis,
    rows_per_device=_PERF_ROWS_PER_DEVICE,
    profile_samples=7,
):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("high_bw_all_gather bandwidth test requires the realtime device profiler")

    axis_size = mesh_device.shape[cluster_axis]
    global_shape = (1, 1, rows_per_device * axis_size, width)
    torch.manual_seed(0)
    host_input = torch.rand(global_shape, dtype=torch.bfloat16)
    device_input = _make_tensor(
        mesh_device,
        host_input,
        dtype,
        layout,
        ttnn.ShardTensor2dMesh(
            mesh_device, dims=(2, None) if cluster_axis == 0 else (None, 2), mesh_shape=tuple(mesh_device.shape)
        ),
    )
    # The op gathers each device tensor's padded shape. For TILE layout, a logical shard whose height is not a
    # multiple of 32 therefore produces a correspondingly padded gathered output (for example, 625 rows/device
    # produces 640 * ring_size output rows).
    local_padded_shape = ttnn.get_device_tensors(device_input)[0].padded_shape
    output_shape = list(local_padded_shape)
    output_shape[2] *= axis_size
    persistent_output = _make_tensor(
        mesh_device,
        torch.zeros(output_shape, dtype=torch.bfloat16),
        dtype,
        layout,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    page_size = ttnn.get_device_tensors(device_input)[0].buffer_aligned_page_size()
    assert page_size == expected_page_size
    assert ttnn.get_tt_fabric_max_payload_size_bytes() == 14 * 1024

    def run():
        return ttnn.experimental.high_bw_all_gather(
            device_input,
            dim=2,
            output_tensor=persistent_output,
            cluster_axis=cluster_axis,
            num_links=_NUM_LINKS,
        )

    run()
    ttnn.synchronize_device(mesh_device)
    run()
    ttnn.synchronize_device(mesh_device)
    _profile_high_bw_all_gather(mesh_device, run)
    durations_ns = [_profile_high_bw_all_gather(mesh_device, run) for _ in range(profile_samples)]

    median_ns = statistics.median(durations_ns)
    if layout == ttnn.TILE_LAYOUT:
        pages_per_device = math.prod(local_padded_shape) // (32 * 32)
    else:
        pages_per_device = rows_per_device
    bandwidth_gbps = pages_per_device * page_size * (axis_size - 1) / median_ns
    assert bandwidth_gbps >= min_bandwidth_gbps
    print(
        f"HIGH_BW_ALL_GATHER fabric={ttnn.get_fabric_config()} dtype={dtype} "
        f"layout={layout} num_links={_NUM_LINKS} rows_per_device={rows_per_device} page_size={page_size}B "
        f"median={median_ns / 1e6:.3f}ms effective_receive_bw={bandwidth_gbps:.3f}GB/s "
        f"samples_ms={[round(duration / 1e6, 3) for duration in durations_ns]}"
    )
    return bandwidth_gbps, median_ns


def _run_high_bw_all_gather(mesh_device, dtype, width, layout, expected_page_size, min_bandwidth_gbps, cluster_axis):
    if ttnn.device.IsProgramRealtimeProfilerActive():
        _run_high_bw_all_gather_perf(
            mesh_device,
            dtype,
            width,
            layout,
            expected_page_size,
            min_bandwidth_gbps,
            cluster_axis,
        )
        _run_high_bw_all_gather_accuracy(
            mesh_device,
            dtype,
            width,
            layout,
            expected_page_size,
            cluster_axis,
            rows_per_device=_ACCURACY_ROWS_PER_DEVICE,
        )
    else:
        _run_high_bw_all_gather_accuracy(
            mesh_device,
            dtype,
            width,
            layout,
            expected_page_size,
            cluster_axis,
            rows_per_device=int(
                os.getenv("TT_METAL_HIGH_BW_ALL_GATHER_ACCURACY_ROWS_PER_DEVICE", _PERF_ROWS_PER_DEVICE)
            ),
        )


def _run_high_bw_all_gather_test_cases(mesh_device, min_bandwidth_gbps, cluster_axis):
    selected_formats = set(
        os.getenv(
            "TT_METAL_HIGH_BW_ALL_GATHER_TEST_FORMATS",
            ",".join(case_name for case_name, *_ in _TEST_CASES),
        ).split(",")
    )
    for case_name, dtype, width, layout, expected_page_size in _TEST_CASES:
        if case_name not in selected_formats:
            continue
        print(f"HIGH_BW_ALL_GATHER_CASE {case_name}")
        _run_high_bw_all_gather(
            mesh_device,
            dtype,
            width,
            layout,
            expected_page_size,
            min_bandwidth_gbps,
            cluster_axis,
        )


def _run_high_bw_all_gather_ci_perf(mesh_device, cluster_axis, min_bandwidth_gbps):
    """Run the compact CI matrix and pair every bandwidth measurement with correctness."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("high_bw_all_gather CI performance coverage requires the realtime device profiler")

    axis_size = mesh_device.shape[cluster_axis]
    for global_rows in _CI_PERF_GLOBAL_ROWS:
        assert global_rows % axis_size == 0
        rows_per_device = global_rows // axis_size
        required_bandwidth_gbps = (
            min_bandwidth_gbps[global_rows] if isinstance(min_bandwidth_gbps, dict) else min_bandwidth_gbps
        )
        for case_name, dtype, width, layout, expected_page_size in _CI_PERF_TEST_CASES:
            print(f"HIGH_BW_ALL_GATHER_CI_PERF global_rows={global_rows} case={case_name}")
            _run_high_bw_all_gather_perf(
                mesh_device,
                dtype,
                width,
                layout,
                expected_page_size,
                required_bandwidth_gbps,
                cluster_axis,
                rows_per_device=rows_per_device,
            )
            # Use a compact reference run after the full-size measurement. This
            # validates the same dtype/layout/route without doubling CI memory
            # pressure for the 512K target.
            _run_high_bw_all_gather_accuracy(
                mesh_device,
                dtype,
                width,
                layout,
                expected_page_size,
                cluster_axis,
                rows_per_device=min(rows_per_device, _ACCURACY_ROWS_PER_DEVICE),
            )


@run_for_blackhole("Blackhole Galaxy perf gate requires Blackhole")
@pytest.mark.skipif(os.getenv("MESH_DEVICE") != "TG", reason="Blackhole Galaxy perf gate requires MESH_DEVICE=TG")
@pytest.mark.parametrize("device_params", [_FABRIC_2D_TORUS_XY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
def test_high_bw_all_gather_galaxy_ci_perf(mesh_device):
    """Blackhole Galaxy perf gate: 8x1 Fabric2D torus at 55K and 512K targets."""
    _run_high_bw_all_gather_ci_perf(
        mesh_device,
        cluster_axis=0,
        min_bandwidth_gbps={55_000: 65.0, 512 * 1024: 90.0},
    )


@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                **_device_params(ttnn.FabricConfig.FABRIC_1D_RING),
                # A 4-device logical ring cannot safely be opened as a subset
                # of an 8-device LoudBox. This is intentionally a QuietBox-only
                # gate and must use its physical 4-device ring.
                "require_exact_physical_num_devices": True,
            },
            id="fabric_1d_ring",
        ),
        pytest.param(
            {
                **_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY),
                # As above, exercise the actual four-device QuietBox topology
                # rather than an unsupported subset of a LoudBox.
                "require_exact_physical_num_devices": True,
            },
            id="fabric_2d_torus_xy",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 1)], indirect=True)
def test_high_bw_all_gather_quietbox_ci_perf(mesh_device):
    """QuietBox perf gate: 4x1 ring and 2D torus at 55K and 512K targets."""
    if not _is_physical_quietbox(mesh_device):
        pytest.skip(
            "QuietBox perf gate requires the complete physical 4x1, four-device mesh; "
            f"got mesh={tuple(mesh_device.shape)}, physical_devices={ttnn.get_num_devices()}"
        )

    _run_high_bw_all_gather_ci_perf(
        mesh_device,
        cluster_axis=0,
        min_bandwidth_gbps=_QUIETBOX_CI_MIN_BANDWIDTH_GBPS,
    )


@pytest.mark.parametrize(
    "device_params,axis_0_min_bandwidth_gbps,axis_1_min_bandwidth_gbps",
    _LINE_PERF_DEVICE_PARAMS,
    indirect=["device_params"],
)
def test_high_bw_all_gather_512k(mesh_device, axis_0_min_bandwidth_gbps, axis_1_min_bandwidth_gbps):
    rank_line, cluster_axis = _rank_line_mesh(mesh_device)
    min_bandwidth_gbps = (axis_0_min_bandwidth_gbps, axis_1_min_bandwidth_gbps)[cluster_axis]

    _run_high_bw_all_gather_test_cases(rank_line, min_bandwidth_gbps, cluster_axis)
    if ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_1D_RING:
        # Cover the Blackhole four-worker region without another device setup. The old 4.5 MB/link heuristic chose
        # eight workers here and measured below this floor; four bank-owning workers sustain more than 60 GB/s.
        if ttnn.device.IsProgramRealtimeProfilerActive():
            _run_high_bw_all_gather_perf(
                rank_line,
                ttnn.bfloat16,
                576,
                ttnn.ROW_MAJOR_LAYOUT,
                1152,
                min_bandwidth_gbps=60.0,
                cluster_axis=cluster_axis,
                rows_per_device=_MEDIUM_BF16_ROWS_PER_DEVICE,
            )
        _run_high_bw_all_gather_accuracy(
            rank_line,
            ttnn.bfloat16,
            576,
            ttnn.ROW_MAJOR_LAYOUT,
            1152,
            cluster_axis=cluster_axis,
            rows_per_device=_MEDIUM_BF16_ROWS_PER_DEVICE,
        )


@pytest.mark.parametrize("device_params", [_FABRIC_2D_LINE_DEVICE_PARAMS], indirect=True)
def test_high_bw_all_gather_512k_fabric_2d_line(mesh_device):
    # QuietBox uses its physical 4x1 line directly. LoudBox uses a physical
    # 1x4 row submesh; both omit a wraparound edge under FABRIC_2D.
    rank_line, cluster_axis = _rank_line_mesh(mesh_device)
    _run_high_bw_all_gather_test_cases(rank_line, min_bandwidth_gbps=45.0, cluster_axis=cluster_axis)


@pytest.mark.parametrize("device_params", [_FABRIC_2D_TORUS_XY_DEVICE_PARAMS], indirect=True)
def test_high_bw_all_gather_ragged_accuracy(mesh_device):
    if os.getenv("TT_METAL_HIGH_BW_ALL_GATHER_RUN_RAGGED_ACCURACY") != "1":
        pytest.skip("set TT_METAL_HIGH_BW_ALL_GATHER_RUN_RAGGED_ACCURACY=1 to run local ragged-slice accuracy")

    rank_line, cluster_axis = _rank_line_mesh(mesh_device)

    selected_formats = set(
        os.getenv(
            "TT_METAL_HIGH_BW_ALL_GATHER_TEST_FORMATS",
            "bf16_1152b_rows,scaled_fp8_704b_rows",
        ).split(",")
    )
    rows_per_device_values = [
        int(value)
        for value in os.getenv(
            "TT_METAL_HIGH_BW_ALL_GATHER_RAGGED_ACCURACY_ROWS",
            "1003,6951",
        ).split(",")
    ]
    for rows_per_device in rows_per_device_values:
        for case_name, dtype, width, layout, expected_page_size in _TEST_CASES:
            if case_name not in selected_formats:
                continue
            _run_high_bw_all_gather_accuracy(
                rank_line,
                dtype,
                width,
                layout,
                expected_page_size,
                cluster_axis=cluster_axis,
                rows_per_device=rows_per_device,
            )


def _token_sweep_sizes():
    return [*range(5_000, 100_001, 5_000), *range(150_000, 500_001, 50_000)]


def _print_high_bw_all_gather_perf_summary(mesh_device, cluster_axis, profile_samples, records):
    axis_size = mesh_device.shape[cluster_axis]
    lines = [
        "",
        "=== HIGH_BW_ALL_GATHER PERF SUMMARY ===",
        (
            f"fabric={str(ttnn.get_fabric_config()).removeprefix('FabricConfig.')} "
            f"mesh={mesh_device.shape[0]}x{mesh_device.shape[1]} cluster_axis={cluster_axis} "
            f"axis_size={axis_size} num_links={_NUM_LINKS} samples={profile_samples}"
        ),
        "Bandwidth is effective receive bandwidth per device; duration is the median device-program runtime.",
        (
            f"{'seq_len':>9}  {'effective':>9}  {'format':<23}  {'layout':<11}  "
            f"{'page_B':>6}  {'duration_ms':>11}  {'BW_GB/s':>8}"
        ),
        f"{'-' * 9}  {'-' * 9}  {'-' * 23}  {'-' * 11}  {'-' * 6}  {'-' * 11}  {'-' * 8}",
    ]
    for record in records:
        lines.append(
            f"{record['global_tokens']:>9,}  {record['effective_global_tokens']:>9,}  "
            f"{record['case']:<23}  {record['layout']:<11}  {record['page_size_bytes']:>6}  "
            f"{record['median_us'] / 1e3:>11.3f}  {record['bandwidth_gbps']:>8.3f}"
        )
    lines.append("=== END HIGH_BW_ALL_GATHER PERF SUMMARY ===")
    print("\n".join(lines), flush=True)


def _run_high_bw_all_gather_token_sweep(
    mesh_device,
    min_bandwidth_gbps,
    cluster_axis,
    *,
    token_sizes=None,
    profile_samples=None,
    require_opt_in=True,
    output_path=None,
):
    if require_opt_in and os.getenv("TT_METAL_HIGH_BW_ALL_GATHER_RUN_TOKEN_SWEEP") != "1":
        pytest.skip("set TT_METAL_HIGH_BW_ALL_GATHER_RUN_TOKEN_SWEEP=1 to run the local token sweep")
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("the local token sweep requires the realtime device profiler")

    output_path = Path(
        os.getenv("TT_METAL_HIGH_BW_ALL_GATHER_TOKEN_SWEEP_CSV", "/tmp/high_bw_all_gather_token_sweep.csv")
    )
    selected_formats = set(
        os.getenv(
            "TT_METAL_HIGH_BW_ALL_GATHER_TOKEN_SWEEP_FORMATS",
            ",".join(case_name for case_name, *_ in _TEST_CASES),
        ).split(",")
    )
    profile_samples = int(os.getenv("TT_METAL_HIGH_BW_ALL_GATHER_TOKEN_SWEEP_SAMPLES", "3"))
    token_sizes = [
        int(value)
        for value in os.getenv(
            "TT_METAL_HIGH_BW_ALL_GATHER_TOKEN_SWEEP_TOKENS",
            ",".join(str(value) for value in _token_sweep_sizes()),
        ).split(",")
    ]
    axis_size = mesh_device.shape[cluster_axis]
    fabric_config = str(ttnn.get_fabric_config()).removeprefix("FabricConfig.")

    for global_tokens in token_sizes:
        assert global_tokens % axis_size == 0
        rows_per_device = global_tokens // axis_size
        for case_name, dtype, width, layout, expected_page_size in _TEST_CASES:
            if case_name not in selected_formats:
                continue

            effective_rows_per_device = (
                math.ceil(rows_per_device / 32) * 32 if layout == ttnn.TILE_LAYOUT else rows_per_device
            )

            bandwidth_gbps, median_ns = _run_high_bw_all_gather_perf(
                mesh_device,
                dtype,
                width,
                layout,
                expected_page_size,
                min_bandwidth_gbps=0.0,
                cluster_axis=cluster_axis,
                rows_per_device=rows_per_device,
                profile_samples=profile_samples,
            )
            record = {
                "fabric_config": fabric_config,
                "cluster_axis": cluster_axis,
                "case": case_name,
                "global_tokens": global_tokens,
                "rows_per_device": rows_per_device,
                "effective_global_tokens": effective_rows_per_device * axis_size,
                "bandwidth_gbps": f"{bandwidth_gbps:.6f}",
                "median_us": f"{median_ns / 1e3:.6f}",
                "required_bandwidth_gbps": f"{min_bandwidth_gbps:.1f}",
                "meets_floor": int(bandwidth_gbps >= min_bandwidth_gbps),
            }
            write_header = not output_path.exists()
            with output_path.open("a", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=record)
                if write_header:
                    writer.writeheader()
                writer.writerow(record)
            print(f"HIGH_BW_ALL_GATHER_TOKEN_SWEEP {record}")

            # Each distinct shape owns a global semaphore. Release cached workloads and dead tensor references so
            # the complete 28-size sweep fits in L1_SMALL and device DRAM.
            mesh_device.clear_program_cache()
            gc.collect()


@pytest.mark.parametrize(
    "device_params,axis_0_min_bandwidth_gbps,axis_1_min_bandwidth_gbps",
    _LINE_PERF_DEVICE_PARAMS,
    indirect=["device_params"],
)
@pytest.mark.timeout(1800)
def test_high_bw_all_gather_token_sweep(mesh_device, axis_0_min_bandwidth_gbps, axis_1_min_bandwidth_gbps):
    rank_line, cluster_axis = _rank_line_mesh(mesh_device)
    min_bandwidth_gbps = (axis_0_min_bandwidth_gbps, axis_1_min_bandwidth_gbps)[cluster_axis]
    _run_high_bw_all_gather_token_sweep(rank_line, min_bandwidth_gbps, cluster_axis)
