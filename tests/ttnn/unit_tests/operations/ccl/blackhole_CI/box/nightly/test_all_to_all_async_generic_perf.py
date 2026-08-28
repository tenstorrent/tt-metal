# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced correctness and performance coverage for GLM's production all-to-all reshards.

The complete four-chip QuietBox or eight-chip LoudBox is used as one TP ring.  Per-chip tensor
volume matches production, while the output shard scales with the physical ring size.  Both
reshard directions run in one test so a passing performance result also proves bit-exact data
movement through the same traced command stream.
"""

import math
import statistics
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole, skip_with_llk_assert, skip_with_watcher
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

_NUM_LINKS = 2
_MAX_PACKET_PAYLOAD_SIZE = 14 * 1024
_PROFILE_SAMPLES = 3
# Same-shape BF16 TILE qualification references with a 2% relative margin.
_REFERENCE_USEFUL_BANDWIDTH_GBPS = {
    (4, "head_to_sequence"): 31.3,
    (4, "sequence_to_head"): 31.3,
    (8, "head_to_sequence"): 31.8,
    (8, "sequence_to_head"): 36.7,
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
    # The control plane consolidates torus axes no mesh realizes (a wrapped dimension needs more than
    # 2 devices), so the latched config may be a reduced torus flavor of the requested TORUS_XY.
    assert ttnn.get_fabric_config() in (
        ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
        ttnn.FabricConfig.FABRIC_2D_TORUS_X,
        ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
    )
    assert ttnn.get_tt_fabric_max_payload_size_bytes() == _MAX_PACKET_PAYLOAD_SIZE

    for case in _reshard_cases(mesh_device.shape[0]):
        _run_reshard(mesh_device, case)
