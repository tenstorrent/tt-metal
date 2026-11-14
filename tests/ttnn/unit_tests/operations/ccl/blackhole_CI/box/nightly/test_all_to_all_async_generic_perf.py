# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced correctness and performance coverage for GLM's production all-to-all reshards.

The complete four-chip QuietBox or eight-chip LoudBox is used as one TP ring.  Per-chip tensor
volume matches production, while the output shard scales with the physical ring size.  Both
reshard directions run in one test so a passing performance result also proves bit-exact data
movement through the same traced command stream.

The folded-view cases open a LoudBox as 4x2, whose 2-wide axis 1 runs North/South on the fabric (not East/West), at
the production payload, so the 2-wide axis takes the compact schedule. They check ttMLA's sparse head <-> sequence
reshards at TP=2 bit-exactly; before the op took the fabric's directions they hung.
"""

import math
import statistics
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

_NUM_LINKS = 2
_MAX_PACKET_PAYLOAD_SIZE = 14 * 1024
_PROFILE_SAMPLES = 3
# Same-shape BF16 TILE qualification references with a 2% relative margin.
_REFERENCE_USEFUL_BANDWIDTH_GBPS = {
    (4, "head_to_sequence"): 31.3,
    (4, "sequence_to_head"): 31.3,
    (8, "head_to_sequence"): 29.6,
    (8, "sequence_to_head"): 33.9,
}
_PERF_MARGIN = 0.02


@dataclass(frozen=True)
class ReshardCase:
    name: str
    local_input_shape: tuple[int, ...]
    in_dim: int
    out_dim: int


def _reshard_cases(ring_size):
    return (
        ReshardCase("head_to_sequence", (1, 16, 640, 576), in_dim=1, out_dim=2),
        ReshardCase("sequence_to_head", (1, 16 * ring_size, 640 // ring_size, 512), in_dim=2, out_dim=1),
    )


def _fabric_router_config():
    config = ttnn.FabricRouterConfig()
    config.max_packet_payload_size_bytes = _MAX_PACKET_PAYLOAD_SIZE
    return config


_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    "fabric_router_config": _fabric_router_config(),
    "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
    "fabric_tensix_config": ttnn.FabricTensixConfig.DISABLED,
    "trace_region_size": 100_000,
    "require_exact_physical_num_devices": True,
}


def _make_input(mesh_device, case):
    ring_size = mesh_device.shape[0]
    global_shape = list(case.local_input_shape)
    global_shape[case.in_dim] *= ring_size
    torch.manual_seed(0)
    host_input = torch.rand(global_shape, dtype=torch.bfloat16)
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig([ttnn.PlacementShard(case.in_dim), ttnn.PlacementShard(case.out_dim)], mesh_device.shape),
    )
    device_input = ttnn.from_torch(
        host_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )
    assert tuple(ttnn.get_device_tensors(device_input)[0].shape) == case.local_input_shape
    return host_input, device_input


def _make_output(mesh_device, case):
    output_shape = list(case.local_input_shape)
    output_shape[case.in_dim] *= mesh_device.shape[0]
    output_shape[case.out_dim] //= mesh_device.shape[0]
    return ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _all_to_all_duration_ns(programs):
    durations = [
        info["duration_ns"]
        for info in programs.values()
        if any("all_to_all_sender_writer.cpp" in source for source in info["kernel_sources"])
    ]
    assert len(durations) == 1, f"expected one generic all-to-all program, found {len(durations)} in {programs}"
    return durations[0]


def _run_reshard(mesh_device, case):
    host_input, device_input = _make_input(mesh_device, case)
    output_buffer = _make_output(mesh_device, case)
    trace_id = None
    trace_capture_ended = False
    trace_output = None
    try:

        def run_once():
            return ttnn.experimental.all_to_all_async_generic(
                device_input,
                in_dim=case.in_dim,
                out_dim=case.out_dim,
                persistent_output_buffer=output_buffer,
                num_links=_NUM_LINKS,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                cluster_axis=0,
            )

        # Compile outside capture, then capture exactly one production collective.
        run_once()
        ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = run_once()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        trace_capture_ended = True
        ttnn.synchronize_device(mesh_device)

        def replay_once():
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)

        # Exclude first-replay jitter, then take a real-time device-profiler median.
        replay_once()
        ttnn.synchronize_device(mesh_device)
        durations_ns = []
        for _ in range(_PROFILE_SAMPLES):
            _, programs = profile_realtime_program_merged(mesh_device, replay_once)
            durations_ns.append(_all_to_all_duration_ns(programs))

        actual = ttnn.to_torch(trace_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=case.out_dim))
        equal, message = comp_equal(host_input, actual)
        assert equal, f"{case.name} traced all-to-all output mismatch: {message}"

        median_ns = statistics.median(durations_ns)
        local_input_bytes = math.prod(case.local_input_shape) * torch.bfloat16.itemsize
        useful_remote_bytes = local_input_bytes * (mesh_device.shape[0] - 1) / mesh_device.shape[0]
        useful_bandwidth_gbps = useful_remote_bytes / median_ns
        reference_bandwidth_gbps = _REFERENCE_USEFUL_BANDWIDTH_GBPS[(mesh_device.shape[0], case.name)]
        minimum_bandwidth_gbps = reference_bandwidth_gbps * (1 - _PERF_MARGIN)
        logger.info(
            "all_to_all_async_generic production proxy: case={} ring_size={} fabric={} topology=ring "
            "payload={}B links={} local_input={} median={:.3f}us useful_bw={:.3f}GB/s "
            "reference={:.3f}GB/s minimum={:.3f}GB/s samples_us={}".format(
                case.name,
                mesh_device.shape[0],
                ttnn.get_fabric_config(),
                ttnn.get_tt_fabric_max_payload_size_bytes(),
                _NUM_LINKS,
                case.local_input_shape,
                median_ns / 1e3,
                useful_bandwidth_gbps,
                reference_bandwidth_gbps,
                minimum_bandwidth_gbps,
                [round(duration / 1e3, 3) for duration in durations_ns],
            )
        )
        assert useful_bandwidth_gbps >= minimum_bandwidth_gbps, (
            f"{case.name} useful bandwidth {useful_bandwidth_gbps:.3f} GB/s is below "
            f"{minimum_bandwidth_gbps:.3f} GB/s (reference {reference_bandwidth_gbps:.3f} GB/s)"
        )
    finally:
        if trace_id is not None:
            try:
                if not trace_capture_ended:
                    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            finally:
                ttnn.release_trace(mesh_device, trace_id)
        ttnn.deallocate(output_buffer)
        ttnn.deallocate(device_input)


@run_for_blackhole("generic all-to-all production perf coverage requires Blackhole fabric")
@pytest.mark.requires_host_iommu
@skip_with_llk_assert("No need to verify LLK asserts for performance tests.")
@skip_with_watcher("Watcher perturbs kernel timing; perf checks are not meaningful with it enabled.")
@pytest.mark.parametrize("mesh_device", [(4, 1), (8, 1)], ids=["quietbox_4x1", "loudbox_8x1"], indirect=True)
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.timeout(0)
def test_all_to_all_async_generic_production_perf_and_accuracy(mesh_device):
    """Verify both GLM reshards on the complete physical Ring and gate their traced device bandwidth."""
    require_realtime_profiler("generic all-to-all production perf checks")
    assert ttnn.get_num_devices() == math.prod(mesh_device.shape)
    assert ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    assert ttnn.get_tt_fabric_max_payload_size_bytes() == _MAX_PACKET_PAYLOAD_SIZE

    for case in _reshard_cases(mesh_device.shape[0]):
        _run_reshard(mesh_device, case)


_FOLDED_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "fabric_router_config": _fabric_router_config(),
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
}


def _shard_2d(mesh_device, x, dims, layout):
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
    )


def _check_shards(out, expected_for):
    for idx, shard in enumerate(ttnn.get_device_tensors(out)):
        expected, got = expected_for(idx), ttnn.to_torch(shard)
        assert tuple(got.shape) == tuple(expected.shape), f"device {idx}: {tuple(got.shape)} != {tuple(expected.shape)}"
        passed, msg = comp_equal(expected, got)
        assert passed, f"device {idx}: {msg}"


def _folded_head_to_sequence(mesh_device, heads_local, rows_local, width, cluster_axis, num_links):
    """Head-sharded [1, H/tp, S/sp, W] -> sequence-sharded [1, H, S/(sp*tp), W] over cluster_axis."""
    rows, cols = mesh_device.shape
    group, other = (cols, rows) if cluster_axis == 1 else (rows, cols)
    torch.manual_seed(0)
    x = torch.randn(1, heads_local * group, rows_local * other, width, dtype=torch.bfloat16)
    tt = _shard_2d(mesh_device, x, (2, 1) if cluster_axis == 1 else (1, 2), ttnn.TILE_LAYOUT)
    out = ttnn.experimental.all_to_all_async_generic(
        tt, in_dim=1, out_dim=2, num_links=num_links, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=cluster_axis
    )
    ttnn.synchronize_device(mesh_device)
    rows_out = rows_local // group

    def expected(idx):
        r, c = divmod(idx, cols)
        g, o = (c, r) if cluster_axis == 1 else (r, c)  # position along / across the collective axis
        start = o * rows_local + g * rows_out
        return x[:, :, start : start + rows_out, :]

    _check_shards(out, expected)


def _folded_sequence_to_head(mesh_device, heads, rows_local, width, num_links):
    """Row-major sequence-sharded [1, H, S/(sp*tp), W] -> head-sharded [1, H/tp, S/sp, W] over axis 1: the return
    after sparse_sdpa."""
    rows, cols = mesh_device.shape
    rows_out = rows_local // cols
    torch.manual_seed(1)
    # dim 0 = position along axis 1, so each chip starts with every head of its own sequence slice.
    x = torch.randn(cols, heads, rows * rows_out, width, dtype=torch.bfloat16)
    tt = _shard_2d(mesh_device, x, (2, 0), ttnn.ROW_MAJOR_LAYOUT)
    out = ttnn.experimental.all_to_all_async_generic(
        tt, in_dim=2, out_dim=1, num_links=num_links, memory_config=ttnn.DRAM_MEMORY_CONFIG, cluster_axis=1
    )
    ttnn.synchronize_device(mesh_device)
    heads_out = heads // cols

    def expected(idx):
        r, g = divmod(idx, cols)
        blocks = [
            x[src : src + 1, g * heads_out : (g + 1) * heads_out, r * rows_out : (r + 1) * rows_out]
            for src in range(cols)
        ]
        return torch.cat(blocks, dim=2)

    _check_shards(out, expected)


@run_for_blackhole("the folded 4x2 view is a LoudBox mesh")
@pytest.mark.parametrize("mesh_device", [(4, 2)], ids=["loudbox_4x2_folded"], indirect=True)
@pytest.mark.parametrize("device_params", [_FOLDED_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "case",
    [
        # Kimi-K3.1's q at TP=2: 48 heads per chip, 1280 rows per SP chip.
        pytest.param(("head_to_sequence", 1, dict(heads_local=48, rows_local=1280, width=576), 1), id="axis1-1link"),
        pytest.param(("head_to_sequence", 1, dict(heads_local=48, rows_local=1280, width=576), 2), id="axis1-2link"),
        # The 4-chip axis 0 runs East/West on the folded view.
        pytest.param(("head_to_sequence", 0, dict(heads_local=16, rows_local=1280, width=576), 2), id="axis0-2link"),
        pytest.param(
            ("sequence_to_head", 1, dict(heads=96, rows_local=1280, width=512), 2), id="axis1-return-row_major"
        ),
    ],
)
def test_all_to_all_async_generic_folded_view(mesh_device, case):
    if mesh_device.get_num_devices() != 8:
        pytest.skip("LoudBox-only: the 4x2 view must cover the whole 8-chip box")
    direction, cluster_axis, shape, num_links = case
    if direction == "head_to_sequence":
        _folded_head_to_sequence(mesh_device, **shape, cluster_axis=cluster_axis, num_links=num_links)
    else:
        _folded_sequence_to_head(mesh_device, **shape, num_links=num_links)
