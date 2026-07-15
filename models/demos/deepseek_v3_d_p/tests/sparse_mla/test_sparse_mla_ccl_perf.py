# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Galaxy CCL microbenchmarks and LoudBox proxies for GLM sparse-MLA tensor shapes and placements.
"""

import math
import os
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import glm_hf_config
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_perf import CHUNK_TOKENS, SCENARIOS
from models.demos.deepseek_v3_d_p.tt.tt_ccl import create_global_semaphores
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program


@dataclass(frozen=True)
class CCLTraffic:
    critical_path_bytes: float
    total_network_bytes: float
    theoretical_ns: float
    link_gigabits_per_second_per_direction: float
    num_links: int

    @property
    def roofline_gigabits_per_second(self):
        return self.link_gigabits_per_second_per_direction * self.num_links * 2

    @property
    def roofline_gigabytes_per_second(self):
        return self.roofline_gigabits_per_second / 8


@dataclass(frozen=True)
class WorkloadDimensions:
    chunk_tokens: int
    cache_tokens: int
    num_attention_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    galaxy_sp: int = 8


@dataclass(frozen=True)
class CCLMeasurement:
    records: tuple[dict, ...]
    program_durations_ns: dict[int, float]
    input_description: str
    output_description: str

    @property
    def duration_ns(self) -> float:
        return sum(self.program_durations_ns.values())


def _scenario_id(scenario):
    return f"{scenario}_{SCENARIOS[scenario]['cache'] // 1024}k_cache"


def _query_tokens(workload, sp):
    return workload.chunk_tokens * sp // workload.galaxy_sp


def _kvpe_local_input_shape(workload, mesh_shape):
    sp, _ = mesh_shape
    total_tokens = workload.cache_tokens + workload.chunk_tokens
    return [1, 1, total_tokens // sp, workload.kv_lora_rank + workload.qk_rope_head_dim]


def _head_to_sequence_local_input_shape(workload, mesh_shape):
    sp, tp = mesh_shape
    return [
        1,
        workload.num_attention_heads // tp,
        _query_tokens(workload, sp) // sp,
        workload.kv_lora_rank + workload.qk_rope_head_dim,
    ]


def _sequence_to_head_local_input_shape(workload, mesh_shape):
    sp, tp = mesh_shape
    return [
        1,
        workload.num_attention_heads,
        _query_tokens(workload, sp) // (sp * tp),
        workload.kv_lora_rank,
    ]


def _global_semaphores(mesh_device):
    compute_grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    gather_semaphores = create_global_semaphores(mesh_device, cores, 0)
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, cores, 0)
    return gather_semaphores, barrier_semaphore


def _profile_programs(mesh_device, run_fn):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for sparse MLA CCL perf checks")

    # Drain setup programs before registering the callback so only run_fn contributes records.
    ttnn.synchronize_device(mesh_device)
    result, records = profile_realtime_program(mesh_device, run_fn, collect_all=True)
    program_durations_ns = {}
    for record in records:
        runtime_id = record["runtime_id"]
        if runtime_id:
            program_durations_ns[runtime_id] = max(
                program_durations_ns.get(runtime_id, 0.0), float(record["duration_ns"])
            )
    assert program_durations_ns, "real-time profiler returned no valid program durations"
    return result, tuple(records), program_durations_ns


def _run_reshard(tt_input, partition_dim, **all_gather_kwargs):
    """Run mainline MLA's all-gather followed by mesh_partition as one logical reshard."""
    gathered = ttnn.experimental.all_gather_async(tt_input, **all_gather_kwargs)
    output = ttnn.mesh_partition(
        gathered,
        dim=partition_dim,
        cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(gathered)
    return output


def _all_gather_traffic(
    local_input_bytes,
    participants,
    mesh_devices,
    link_gigabits_per_second_per_direction,
    num_links,
):
    # Independent fabric roofline for the all-gather component used by all three mainline flows.
    critical_path_bytes = local_input_bytes * (participants - 1)
    roofline_gigabytes_per_second = link_gigabits_per_second_per_direction * num_links * 2 / 8
    return CCLTraffic(
        critical_path_bytes=critical_path_bytes,
        total_network_bytes=critical_path_bytes * mesh_devices,
        theoretical_ns=critical_path_bytes / roofline_gigabytes_per_second,
        link_gigabits_per_second_per_direction=link_gigabits_per_second_per_direction,
        num_links=num_links,
    )


def _tensor_description(tensor):
    local_tensor = ttnn.get_device_tensors(tensor)[0]
    return f"{list(local_tensor.shape)} ({local_tensor.dtype}, {local_tensor.layout}, {local_tensor.memory_config()})"


def _report_ccl_perf(op_name, scenario, mesh_shape, measurement, traffic):
    measured_ns = measurement.duration_ns
    assert measured_ns > 0, "real-time profiler measured no device-program duration"
    measured_gigabytes_per_second = traffic.critical_path_bytes / measured_ns
    roofline_utilization = traffic.theoretical_ns / measured_ns
    sp, tp = mesh_shape

    logger.info(
        f"{op_name}/{scenario} [SP{sp}xTP{tp}]: " f"{measurement.input_description} -> {measurement.output_description}"
    )
    logger.info(
        f"theoretical fabric roofline: {traffic.link_gigabits_per_second_per_direction:.1f} "
        f"Gbps/link/direction x "
        f"{traffic.num_links} links x 2 directions = "
        f"{traffic.roofline_gigabits_per_second:.1f} Gbps "
        f"({traffic.roofline_gigabytes_per_second:.1f} GB/s); "
        f"critical-path={traffic.critical_path_bytes / 1e6:.3f} MB, "
        f"total-mesh={traffic.total_network_bytes / 1e6:.3f} MB, "
        f"theoretical={traffic.theoretical_ns / 1e3:.3f} us"
    )
    logger.info(
        f"real-time profiler measured: {measured_ns / 1e3:.3f} us, "
        f"bandwidth={measured_gigabytes_per_second:.3f} GB/s, "
        f"roofline utilization={roofline_utilization:.1%}, "
        f"measured/theoretical={measured_ns / traffic.theoretical_ns:.2f}x"
    )
    for runtime_id, duration_ns in sorted(measurement.program_durations_ns.items()):
        runtime_records = [record for record in measurement.records if record["runtime_id"] == runtime_id]
        chips = sorted(record["chip_id"] for record in runtime_records)
        kernel_sources = sorted(
            {os.path.basename(source) for record in runtime_records for source in record["kernel_sources"]}
        )
        logger.info(
            f"real-time profiler program: runtime_id={runtime_id}, critical={duration_ns / 1e3:.3f} us, "
            f"chips={chips}, kernels={kernel_sources}"
        )


def _run_kvpe_all_gather(mesh_device, workload, topology, num_links):
    """Profile the SP all-gather used for the GLM KVPE prefix."""
    sp, _ = mesh_device.shape
    assert sp == workload.galaxy_sp
    total_tokens = workload.cache_tokens + workload.chunk_tokens
    kvpe_dim = workload.kv_lora_rank + workload.qk_rope_head_dim
    global_shape = [1, 1, total_tokens, kvpe_dim]

    mesh_mapper = ttnn.MeshMapperConfig(
        [
            ttnn.PlacementShard(2),  # Mesh axis 0 (SP): shard tensor dim 2 (tokens).
            ttnn.PlacementReplicate(),  # Mesh axis 1 (TP): replicate each SP token shard.
        ],
        mesh_device.shape,
    )
    tt_input = ttnn.rand(
        global_shape,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,  # Per device: [B, H, Q/SP, D].
    )
    gather_semaphores, barrier_semaphore = _global_semaphores(mesh_device)

    tt_output, records, program_durations_ns = _profile_programs(
        mesh_device,
        lambda: ttnn.experimental.all_gather_async(
            tt_input,
            dim=2,
            multi_device_global_semaphore=gather_semaphores,
            barrier_semaphore=barrier_semaphore,
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            cluster_axis=0,
        ),
    )
    assert list(tt_output.shape) == global_shape
    assert tt_output.tensor_topology().placements() == [
        ttnn.PlacementReplicate(),
        ttnn.PlacementReplicate(),
    ]
    input_description = _tensor_description(tt_input)
    output_description = _tensor_description(tt_output)
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)
    return CCLMeasurement(records, program_durations_ns, input_description, output_description)


def _make_reshard_input(
    mesh_device,
    torch_input,
    input_placements,
    output_placements,
    gather_semaphores,
    barrier_semaphore,
    topology,
    num_links,
):
    input_dims = [placement.dim for placement in input_placements]
    # Distinct shard dimensions can be constructed directly by the host mesh mapper.
    if len(set(input_dims)) == len(input_dims):
        mesh_mapper = ttnn.MeshMapperConfig(input_placements, mesh_device.shape)
        return ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_mapper),
        )

    # mla.py obtains [Shard(seq), Shard(seq)] from the forward mainline reshard. Reproduce that path
    # because the generic host mapper deliberately rejects duplicate shard dimensions.
    source_mapper = ttnn.MeshMapperConfig(output_placements, mesh_device.shape)
    source = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, source_mapper),
    )
    gathered = ttnn.experimental.all_gather_async(
        source,
        dim=1,
        multi_device_global_semaphore=gather_semaphores,
        barrier_semaphore=barrier_semaphore,
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        cluster_axis=1,
    )
    tt_input = ttnn.mesh_partition(
        gathered,
        dim=2,
        cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(gathered)
    ttnn.deallocate(source)
    return tt_input


def _run_glm_reshard(
    mesh_device,
    logical_shape,
    gather_dim,
    partition_dim,
    input_placements,
    output_placements,
    expected_input_shape,
    expected_output_shape,
    topology,
    num_links,
):
    sp, tp = mesh_device.shape
    torch_input = torch.rand(logical_shape, dtype=torch.bfloat16)
    gather_semaphores, barrier_semaphore = _global_semaphores(mesh_device)
    tt_input = _make_reshard_input(
        mesh_device,
        torch_input,
        input_placements,
        output_placements,
        gather_semaphores,
        barrier_semaphore,
        topology,
        num_links,
    )
    # Input shape is shared with the traffic roofline; output shape remains an explicit reshard assertion.
    assert list(ttnn.get_device_tensors(tt_input)[0].shape) == expected_input_shape

    tt_output, records, program_durations_ns = _profile_programs(
        mesh_device,
        lambda: _run_reshard(
            tt_input,
            partition_dim,
            dim=gather_dim,
            multi_device_global_semaphore=gather_semaphores,
            barrier_semaphore=barrier_semaphore,
            num_links=num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            cluster_axis=1,
        ),
    )
    assert list(ttnn.get_device_tensors(tt_output)[0].shape) == expected_output_shape

    # Compose in mesh row-major order explicitly: the generic 2D composer rejects the valid case
    # where both mesh axes shard the same tensor dimension (head -> sequence).
    device_outputs = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(tt_output)]
    mesh_rows = [torch.cat(device_outputs[row * tp : (row + 1) * tp], dim=partition_dim) for row in range(sp)]
    actual = torch.cat(mesh_rows, dim=output_placements[0].dim)
    equal, message = comp_equal(torch_input, actual)
    assert equal, message
    input_description = _tensor_description(tt_input)
    output_description = _tensor_description(tt_output)
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)
    return CCLMeasurement(records, program_durations_ns, input_description, output_description)


def _run_glm_head_to_sequence_reshard(mesh_device, workload, topology, num_links):
    """Run GLM's production head-sharded to sequence-sharded TP redistribution."""
    sp, tp = mesh_device.shape
    query_tokens = _query_tokens(workload, sp)
    heads = workload.num_attention_heads
    q_dim = workload.kv_lora_rank + workload.qk_rope_head_dim
    return _run_glm_reshard(
        mesh_device,
        [1, heads, query_tokens, q_dim],
        gather_dim=1,
        partition_dim=2,
        input_placements=[
            ttnn.PlacementShard(2),  # SP axis shards Q.
            ttnn.PlacementShard(1),  # TP axis shards H.
        ],  # Per device: [B, H/TP, Q/SP, D].
        output_placements=[
            ttnn.PlacementShard(2),  # SP axis shards Q.
            ttnn.PlacementShard(2),  # TP axis also shards Q instead of H.
        ],  # Per device: [B, H, Q/(SP*TP), D].
        expected_input_shape=_head_to_sequence_local_input_shape(workload, mesh_device.shape),
        expected_output_shape=[1, heads, query_tokens // (sp * tp), q_dim],
        topology=topology,
        num_links=num_links,
    )


def _run_glm_sequence_to_head_reshard(mesh_device, workload, topology, num_links):
    """Run GLM's production sequence-sharded to head-sharded TP redistribution."""
    sp, tp = mesh_device.shape
    query_tokens = _query_tokens(workload, sp)
    heads = workload.num_attention_heads
    latent_dim = workload.kv_lora_rank
    return _run_glm_reshard(
        mesh_device,
        [1, heads, query_tokens, latent_dim],
        gather_dim=2,
        partition_dim=1,
        input_placements=[
            ttnn.PlacementShard(2),  # SP axis shards Q.
            ttnn.PlacementShard(2),  # TP axis also shards Q.
        ],  # Per device: [B, H, Q/(SP*TP), D].
        output_placements=[
            ttnn.PlacementShard(2),  # SP axis keeps sharding Q.
            ttnn.PlacementShard(1),  # TP axis switches from Q sharding to H sharding.
        ],  # Per device: [B, H/TP, Q/SP, D].
        expected_input_shape=_sequence_to_head_local_input_shape(workload, mesh_device.shape),
        expected_output_shape=[1, heads // tp, query_tokens // sp, latent_dim],
        topology=topology,
        num_links=num_links,
    )


def _mesh_parameter(collective_axis):
    num_devices = detect_num_devices()
    fabric_2d = {"trace_region_size": 100000, "fabric_config": ttnn.FabricConfig.FABRIC_2D}
    if num_devices == 32:
        system = "galaxy"
        mesh_shape = (8, 4)
        mesh_topology = "mesh-8x4"
        device_params = fabric_2d
    elif num_devices == 8:
        system = "loudbox_proxy"
        if collective_axis == 0:
            # One SP=8 ring approximates each of Galaxy's four concurrent SP rings.
            mesh_shape = (8, 1)
            mesh_topology = "ring"
            device_params = {
                "trace_region_size": 100000,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            }
        else:
            # TP collectives retain TP=4 and Galaxy-local tensor shapes.
            mesh_shape = (2, 4)
            mesh_topology = "mesh-2x4"
            device_params = fabric_2d
    else:
        reason = f"CCL perf supports Galaxy (32 chips) or LoudBox (8), found {num_devices}"
        return pytest.param(
            (1, 1),
            fabric_2d,
            marks=pytest.mark.skip(reason=reason),
            id="unsupported",
        )

    return pytest.param(
        mesh_shape,
        device_params,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh_shape, topology=mesh_topology),
        id=f"{system}_sp{mesh_shape[0]}_tp{mesh_shape[1]}",
    )


def _system_link_gigabits_per_second_per_direction(mesh_shape):
    default = 200.0 if math.prod(mesh_shape) == 32 else 400.0
    return float(os.environ.get("MLA_CCL_LINK_GBPS_PER_DIRECTION", default))


def _workload_dimensions(glm_config, scenario):
    return WorkloadDimensions(
        chunk_tokens=CHUNK_TOKENS,
        cache_tokens=SCENARIOS[scenario]["cache"],
        num_attention_heads=glm_config.num_attention_heads,
        kv_lora_rank=glm_config.kv_lora_rank,
        qk_rope_head_dim=glm_config.qk_rope_head_dim,
    )


def _traffic(
    local_input_shape,
    participants,
    mesh_shape,
    link_gigabits_per_second_per_direction,
    num_links,
):
    local_input_bytes = math.prod(local_input_shape) * torch.bfloat16.itemsize
    return _all_gather_traffic(
        local_input_bytes,
        participants,
        math.prod(mesh_shape),
        link_gigabits_per_second_per_direction,
        num_links,
    )


@pytest.mark.perf
@pytest.mark.timeout(0)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="performance test - run locally")
@pytest.mark.parametrize(
    "scenario",
    tuple(name for name, scenario in SCENARIOS.items() if not scenario["loop"]),
    ids=_scenario_id,
)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [_mesh_parameter(0)],
    indirect=["mesh_device", "device_params"],
)
def test_kvpe_all_gather_perf(mesh_device, device_params, scenario):
    """Profile the SP all-gather used for the GLM KVPE prefix."""
    del device_params
    assert mesh_device.arch() == ttnn.Arch.BLACKHOLE, "bandwidth assumptions apply to Blackhole only"
    workload = _workload_dimensions(glm_hf_config(), scenario)
    mesh_shape = tuple(mesh_device.shape)
    num_links = 2
    topology = ttnn.Topology.Ring if math.prod(mesh_shape) == 8 else ttnn.Topology.Linear
    measurement = _run_kvpe_all_gather(mesh_device, workload, topology, num_links)
    _report_ccl_perf(
        "kvpe_all_gather",
        scenario,
        mesh_shape,
        measurement,
        _traffic(
            _kvpe_local_input_shape(workload, mesh_shape),
            mesh_shape[0],
            mesh_shape,
            _system_link_gigabits_per_second_per_direction(mesh_shape),
            num_links,
        ),
    )


@pytest.mark.perf
@pytest.mark.timeout(0)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="performance test - run locally")
@pytest.mark.parametrize(
    "scenario",
    tuple(name for name, scenario in SCENARIOS.items() if not scenario["loop"]),
    ids=_scenario_id,
)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [_mesh_parameter(1)],
    indirect=["mesh_device", "device_params"],
)
def test_glm_head_to_sequence_reshard_perf(mesh_device, device_params, scenario):
    """Profile GLM's head-sharded to sequence-sharded TP redistribution."""
    del device_params
    assert mesh_device.arch() == ttnn.Arch.BLACKHOLE, "bandwidth assumptions apply to Blackhole only"
    workload = _workload_dimensions(glm_hf_config(), scenario)
    mesh_shape = tuple(mesh_device.shape)
    num_links = 2
    measurement = _run_glm_head_to_sequence_reshard(mesh_device, workload, ttnn.Topology.Linear, num_links)
    _report_ccl_perf(
        "glm_head_to_sequence_reshard",
        scenario,
        mesh_shape,
        measurement,
        _traffic(
            _head_to_sequence_local_input_shape(workload, mesh_shape),
            mesh_shape[1],
            mesh_shape,
            _system_link_gigabits_per_second_per_direction(mesh_shape),
            num_links,
        ),
    )


@pytest.mark.perf
@pytest.mark.timeout(0)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="performance test - run locally")
@pytest.mark.parametrize(
    "scenario",
    tuple(name for name, scenario in SCENARIOS.items() if not scenario["loop"]),
    ids=_scenario_id,
)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [_mesh_parameter(1)],
    indirect=["mesh_device", "device_params"],
)
def test_glm_sequence_to_head_reshard_perf(mesh_device, device_params, scenario):
    """Profile GLM's sequence-sharded to head-sharded TP redistribution."""
    del device_params
    assert mesh_device.arch() == ttnn.Arch.BLACKHOLE, "bandwidth assumptions apply to Blackhole only"
    workload = _workload_dimensions(glm_hf_config(), scenario)
    mesh_shape = tuple(mesh_device.shape)
    num_links = 2
    measurement = _run_glm_sequence_to_head_reshard(mesh_device, workload, ttnn.Topology.Linear, num_links)
    _report_ccl_perf(
        "glm_sequence_to_head_reshard",
        scenario,
        mesh_shape,
        measurement,
        _traffic(
            _sequence_to_head_local_input_shape(workload, mesh_shape),
            mesh_shape[1],
            mesh_shape,
            _system_link_gigabits_per_second_per_direction(mesh_shape),
            num_links,
        ),
    )
