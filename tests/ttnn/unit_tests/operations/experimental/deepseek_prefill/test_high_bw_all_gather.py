# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Production-shaped correctness and bandwidth coverage for high_bw_all_gather."""

import math
import statistics

import pytest
import torch

import ttnn
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


_DEVICE_PARAMS = [
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_1D),
        id="fabric_1d_line",
    ),
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_1D_RING),
        id="fabric_1d_ring",
    ),
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_2D),
        id="fabric_2d",
    ),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_X), id="fabric_2d_torus_x_line"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_Y), id="fabric_2d_torus_y_ring"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY), id="fabric_2d_torus_xy_ring"),
]

_AXIS_1_DEVICE_PARAMS = [
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D), id="fabric_2d"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_X), id="fabric_2d_torus_x_ring"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_Y), id="fabric_2d_torus_y_line"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY), id="fabric_2d_torus_xy_ring"),
]

# A line uses one direction for interior ranks and is expected to deliver roughly
# half the effective receive bandwidth of the bidirectional ring schedule.
_PERF_DEVICE_PARAMS = [
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_1D),
        45.0,
        id="fabric_1d_line",
    ),
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_1D_RING),
        90.0,
        id="fabric_1d_ring",
    ),
    pytest.param(
        _device_params(ttnn.FabricConfig.FABRIC_2D),
        45.0,
        id="fabric_2d",
    ),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_X), 45.0, id="fabric_2d_torus_x_line"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_Y), 90.0, id="fabric_2d_torus_y_ring"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY), 90.0, id="fabric_2d_torus_xy_ring"),
]

_AXIS_1_PERF_DEVICE_PARAMS = [
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D), 45.0, id="fabric_2d"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_X), 90.0, id="fabric_2d_torus_x_ring"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_Y), 45.0, id="fabric_2d_torus_y_line"),
    pytest.param(_device_params(ttnn.FabricConfig.FABRIC_2D_TORUS_XY), 90.0, id="fabric_2d_torus_xy_ring"),
]

_FABRIC_2D_LINE_DEVICE_PARAMS = pytest.param(
    _device_params(ttnn.FabricConfig.FABRIC_2D),
    id="fabric_2d_line",
)

_TEST_CASES = [
    pytest.param(ttnn.bfloat16, 576, ttnn.ROW_MAJOR_LAYOUT, 1152, id="bf16_1152b_rows"),
    pytest.param(ttnn.fp8_e4m3, 656, ttnn.ROW_MAJOR_LAYOUT, 704, id="scaled_fp8_704b_rows"),
    pytest.param(ttnn.bfloat16, 576, ttnn.TILE_LAYOUT, 2048, id="bf16_tiles"),
    pytest.param(ttnn.bfloat8_b, 576, ttnn.TILE_LAYOUT, 1088, id="bfloat8_b_tiles"),
]


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
        if not any("/experimental/deepseek_prefill/high_bw_all_gather/" in source for source in sources):
            continue
        assert any(source.endswith("/unicast_writer.cpp") for source in sources)
        runtime_id = record["runtime_id"]
        programs[runtime_id] = max(programs.get(runtime_id, 0.0), record["duration_ns"])
    assert programs, "realtime profiler returned no high_bw_all_gather program"
    return sum(programs.values())


def _assert_exact_all_gather(device_input, persistent_output, mesh_device, dtype, host_input):
    """The collective performs no arithmetic, so compare the gathered device values exactly."""
    if dtype == ttnn.fp8_e4m3:
        expected = torch.cat(list(_fp8_payloads(device_input, mesh_device)), dim=2)
        actual_outputs = _fp8_payloads(persistent_output, mesh_device)
    elif dtype == ttnn.bfloat8_b:
        # Compare the quantized device representation, not the pre-quantization host input.
        expected = ttnn.to_torch(device_input, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
        actual_outputs = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(persistent_output)]
    else:
        expected = host_input
        actual_outputs = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(persistent_output)]
    for actual in actual_outputs:
        assert torch.equal(actual, expected)


def _run_high_bw_all_gather_accuracy(mesh_device, dtype, width, layout, expected_page_size, cluster_axis):
    rows_per_device = 65536
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
    persistent_output = _make_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=torch.bfloat16),
        dtype,
        layout,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    assert ttnn.get_device_tensors(device_input)[0].buffer_aligned_page_size() == expected_page_size

    ttnn.experimental.deepseek_prefill.high_bw_all_gather(
        device_input,
        dim=2,
        output_tensor=persistent_output,
        cluster_axis=cluster_axis,
    )
    ttnn.synchronize_device(mesh_device)

    _assert_exact_all_gather(device_input, persistent_output, mesh_device, dtype, host_input)


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize("dtype,width,layout,expected_page_size", _TEST_CASES)
def test_high_bw_all_gather_512k_accuracy(mesh_device, dtype, width, layout, expected_page_size):
    _run_high_bw_all_gather_accuracy(mesh_device, dtype, width, layout, expected_page_size, cluster_axis=0)


@pytest.mark.parametrize("device_params", _AXIS_1_DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype,width,layout,expected_page_size", _TEST_CASES)
def test_high_bw_all_gather_512k_axis_1_accuracy(mesh_device, dtype, width, layout, expected_page_size):
    _run_high_bw_all_gather_accuracy(mesh_device, dtype, width, layout, expected_page_size, cluster_axis=1)


@pytest.mark.parametrize("device_params", [_FABRIC_2D_LINE_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize("dtype,width,layout,expected_page_size", _TEST_CASES)
def test_high_bw_all_gather_512k_fabric_2d_line_accuracy(mesh_device, dtype, width, layout, expected_page_size):
    # The 8-device parent initializes the complete physical Fabric. Its 4-device slice deliberately omits the
    # physical wraparound edge, exercising the Fabric2D direct-line admission path.
    line_mesh = mesh_device.create_submesh(ttnn.MeshShape(4, 1))
    _run_high_bw_all_gather_accuracy(line_mesh, dtype, width, layout, expected_page_size, cluster_axis=0)


def _run_high_bw_all_gather_perf(
    mesh_device, dtype, width, layout, expected_page_size, min_bandwidth_gbps, cluster_axis
):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.skip("high_bw_all_gather bandwidth test requires the realtime device profiler")

    rows_per_device = 65536
    ring_size = mesh_device.shape[cluster_axis]
    global_shape = (1, 1, rows_per_device * ring_size, width)
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
    persistent_output = _make_tensor(
        mesh_device,
        torch.zeros(global_shape, dtype=torch.bfloat16),
        dtype,
        layout,
        ttnn.ReplicateTensorToMesh(mesh_device),
    )
    page_size = ttnn.get_device_tensors(device_input)[0].buffer_aligned_page_size()
    assert page_size == expected_page_size
    assert ttnn.get_tt_fabric_max_payload_size_bytes() == 14 * 1024

    def run():
        return ttnn.experimental.deepseek_prefill.high_bw_all_gather(
            device_input,
            dim=2,
            output_tensor=persistent_output,
            cluster_axis=cluster_axis,
        )

    run()
    ttnn.synchronize_device(mesh_device)
    run()
    ttnn.synchronize_device(mesh_device)
    _profile_high_bw_all_gather(mesh_device, run)
    durations_ns = [_profile_high_bw_all_gather(mesh_device, run) for _ in range(7)]

    median_ns = statistics.median(durations_ns)
    if layout == ttnn.TILE_LAYOUT:
        local_padded_shape = ttnn.get_device_tensors(device_input)[0].padded_shape
        pages_per_device = math.prod(local_padded_shape) // (32 * 32)
    else:
        pages_per_device = rows_per_device
    bandwidth_gbps = pages_per_device * page_size * (ring_size - 1) / median_ns
    assert bandwidth_gbps >= min_bandwidth_gbps
    print(
        f"HIGH_BW_ALL_GATHER fabric={ttnn.get_fabric_config()} dtype={dtype} "
        f"layout={layout} rows_per_device={rows_per_device} page_size={page_size}B "
        f"median={median_ns / 1e6:.3f}ms effective_receive_bw={bandwidth_gbps:.3f}GB/s "
        f"samples_ms={[round(duration / 1e6, 3) for duration in durations_ns]}"
    )

    _assert_exact_all_gather(device_input, persistent_output, mesh_device, dtype, host_input)


@pytest.mark.parametrize("device_params,min_bandwidth_gbps", _PERF_DEVICE_PARAMS, indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize("dtype,width,layout,expected_page_size", _TEST_CASES)
def test_high_bw_all_gather_512k(mesh_device, dtype, width, layout, expected_page_size, min_bandwidth_gbps):
    _run_high_bw_all_gather_perf(
        mesh_device, dtype, width, layout, expected_page_size, min_bandwidth_gbps, cluster_axis=0
    )


@pytest.mark.parametrize("device_params,min_bandwidth_gbps", _AXIS_1_PERF_DEVICE_PARAMS, indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype,width,layout,expected_page_size", _TEST_CASES)
def test_high_bw_all_gather_512k_axis_1(mesh_device, dtype, width, layout, expected_page_size, min_bandwidth_gbps):
    _run_high_bw_all_gather_perf(
        mesh_device, dtype, width, layout, expected_page_size, min_bandwidth_gbps, cluster_axis=1
    )


@pytest.mark.parametrize("device_params", [_FABRIC_2D_LINE_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize("dtype,width,layout,expected_page_size", _TEST_CASES)
def test_high_bw_all_gather_512k_fabric_2d_line(mesh_device, dtype, width, layout, expected_page_size):
    line_mesh = mesh_device.create_submesh(ttnn.MeshShape(4, 1))
    _run_high_bw_all_gather_perf(
        line_mesh, dtype, width, layout, expected_page_size, min_bandwidth_gbps=45.0, cluster_axis=0
    )
