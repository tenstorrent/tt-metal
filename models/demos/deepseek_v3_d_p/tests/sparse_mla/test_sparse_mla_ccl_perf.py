# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Galaxy CCL microbenchmarks and LoudBox proxies for GLM sparse-MLA tensor shapes and placements.

Run the driver tests through ``scripts/run_safe_pytest.sh`` with ``-m perf``.
Each driver launches its single-operation worker under Tracy.
"""

import functools
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from unittest import mock

import pandas as pd
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import glm_hf_config
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_perf import CHUNK_TOKENS, SCENARIOS
from models.demos.deepseek_v3_d_p.tt.tt_ccl import create_global_semaphores
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal

CCL_SCENARIOS = tuple(name for name, scenario in SCENARIOS.items() if not scenario["loop"])

GALAXY_MESH_SHAPE = (8, 4)  # SP=8 x TP=4
LOUD_BOX_ALL_GATHER_MESH_SHAPE = (8, 1)
LOUD_BOX_RESHARD_MESH_SHAPE = (2, 4)
BFLOAT16_BYTES = 2
FABRIC_2D_DEVICE_PARAMS = {"trace_region_size": 100000, "fabric_config": ttnn.FabricConfig.FABRIC_2D}
NUM_LINKS = 2
RESHARD_TOPOLOGY = ttnn.Topology.Linear
LINK_GBPS_PER_DIRECTION = float(os.environ.get("MLA_CCL_LINK_GBPS_PER_DIRECTION", 200.0))
DIRECTIONS_PER_BIDIRECTIONAL_LINK = 2
ROOFLINE_GBPS = LINK_GBPS_PER_DIRECTION * NUM_LINKS * DIRECTIONS_PER_BIDIRECTIONAL_LINK
ROOFLINE_BYTES_PER_SECOND = ROOFLINE_GBPS * 1e9 / 8
GLM_CONFIG = glm_hf_config()
NUM_DEVICES = detect_num_devices()
PM_COLUMNS = (
    "PM IDEAL [ns]",
    "PM COMPUTE [ns]",
    "PM BANDWIDTH [ns]",
    "PM REQ I BW",
    "PM REQ O BW",
    "ETH BW UTIL (%)",
)


@dataclass(frozen=True)
class CCLTraffic:
    critical_path_bytes: float
    total_network_bytes: float
    theoretical_ns: float


@dataclass(frozen=True)
class ProfileParameters:
    mesh: object
    device_params: dict

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return self.mesh.values[0]  # the shape _mesh_param wrapped in its pytest.param


@dataclass(frozen=True)
class CCLOperation:
    runner: Callable
    local_input_shape: Callable
    participants_axis: int
    profile: ProfileParameters
    validates_op_codes: Callable[[set[str]], bool]
    scenario_dependent: bool = False

    def run(self, mesh_device, scenario):
        args = (mesh_device, scenario) if self.scenario_dependent else (mesh_device,)
        return self.runner(*args)

    def input_shape(self, mesh_shape, scenario):
        args = (mesh_shape, scenario) if self.scenario_dependent else (mesh_shape,)
        return self.local_input_shape(*args)


def _mesh_param(mesh_shape, topology, system):
    return pytest.param(
        mesh_shape,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh_shape, topology=topology),
        id=f"{system}_sp{mesh_shape[0]}_tp{mesh_shape[1]}",
    )


if NUM_DEVICES == 32:
    ALL_GATHER_DEVICE_PARAMS = FABRIC_2D_DEVICE_PARAMS
    ALL_GATHER_TOPOLOGY = ttnn.Topology.Linear
    ALL_GATHER_MESH = _mesh_param(GALAXY_MESH_SHAPE, "mesh-8x4", "galaxy")
    RESHARD_MESH = _mesh_param(GALAXY_MESH_SHAPE, "mesh-8x4", "galaxy")
elif NUM_DEVICES == 8:
    ALL_GATHER_DEVICE_PARAMS = {"trace_region_size": 100000, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}
    ALL_GATHER_TOPOLOGY = ttnn.Topology.Ring
    # One SP=8 ring has the same per-ring payload and link depth as each of Galaxy's four concurrent
    # SP rings. TP collectives retain TP=4 and Galaxy-local tensor shapes on the 2x4 mesh.
    ALL_GATHER_MESH = _mesh_param(LOUD_BOX_ALL_GATHER_MESH_SHAPE, "ring", "loudbox_proxy")
    RESHARD_MESH = _mesh_param(LOUD_BOX_RESHARD_MESH_SHAPE, "mesh-2x4", "loudbox_proxy")
else:
    ALL_GATHER_DEVICE_PARAMS = FABRIC_2D_DEVICE_PARAMS
    ALL_GATHER_TOPOLOGY = ttnn.Topology.Linear
    unsupported = pytest.mark.skip(reason=f"CCL perf supports Galaxy (32 chips) or LoudBox (8), found {NUM_DEVICES}")
    ALL_GATHER_MESH = pytest.param((1, 1), marks=unsupported, id="unsupported")
    RESHARD_MESH = pytest.param((1, 1), marks=unsupported, id="unsupported")


def _scenario_id(scenario):
    return f"{scenario}_{SCENARIOS[scenario]['cache'] // 1024}k_cache"


def _query_tokens(sp):
    return CHUNK_TOKENS * sp // GALAXY_MESH_SHAPE[0]


def _placement_strs(placements):
    return [str(placement) for placement in placements]


def _kvpe_local_input_shape(mesh_shape, scenario):
    sp, _ = mesh_shape
    total_tokens = SCENARIOS[scenario]["cache"] + CHUNK_TOKENS
    return [1, 1, total_tokens // sp, GLM_CONFIG.kv_lora_rank + GLM_CONFIG.qk_rope_head_dim]


def _head_to_sequence_local_input_shape(mesh_shape):
    sp, tp = mesh_shape
    return [
        1,
        GLM_CONFIG.num_attention_heads // tp,
        _query_tokens(sp) // sp,
        GLM_CONFIG.kv_lora_rank + GLM_CONFIG.qk_rope_head_dim,
    ]


def _sequence_to_head_local_input_shape(mesh_shape):
    sp, tp = mesh_shape
    return [1, GLM_CONFIG.num_attention_heads, _query_tokens(sp) // (sp * tp), GLM_CONFIG.kv_lora_rank]


def _global_semaphores(mesh_device):
    compute_grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid.x - 1, compute_grid.y - 1))}
    )
    gather_semaphores = create_global_semaphores(mesh_device, cores, 0)
    barrier_semaphore = ttnn.create_global_semaphore(mesh_device, cores, 0)
    return gather_semaphores, barrier_semaphore


def _signposted(func):
    """Wrap a Tracy region; decorated functions must take mesh_device as their first argument."""

    @functools.wraps(func)
    def wrapper(mesh_device, *args, **kwargs):
        from tracy import signpost

        signpost("start")
        try:
            result = func(mesh_device, *args, **kwargs)
        finally:
            ttnn.synchronize_device(mesh_device)
            signpost("stop")
        return result

    return wrapper


@_signposted
def _profile_operation(mesh_device, operation, *args, **kwargs):
    return operation(*args, **kwargs)


@_signposted
def _profile_reshard(mesh_device, tt_input, partition_dim, **all_gather_kwargs):
    """Profile mainline MLA's all-gather followed by mesh_partition as one logical reshard."""
    gathered = ttnn.experimental.all_gather_async(tt_input, **all_gather_kwargs)
    output = ttnn.mesh_partition(
        gathered,
        dim=partition_dim,
        cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(gathered)
    return output


def _all_gather_traffic(local_input_bytes, participants, mesh_devices):
    # Independent fabric roofline for the all-gather component used by all three mainline flows.
    critical_path_bytes = local_input_bytes * (participants - 1)
    return CCLTraffic(
        critical_path_bytes=critical_path_bytes,
        total_network_bytes=critical_path_bytes * mesh_devices,
        theoretical_ns=critical_path_bytes / ROOFLINE_BYTES_PER_SECOND * 1e9,
    )


def _tracy_tensor_description(row, prefix):
    shape = [str(row[f"{prefix}_{dim}_PAD[LOGICAL]"]).split("[")[0] for dim in ("W", "Z", "Y", "X")]
    tensor_format = ", ".join(str(row[f"{prefix}_{field}"]) for field in ("DATATYPE", "LAYOUT", "MEMORY"))
    return f"[{', '.join(shape)}] ({tensor_format})"


def _read_tracy_ccl_rows(subdir):
    from models.tt_transformers.tests.test_utils import merge_device_rows
    from tests.nightly.sdpa_perf_utils import post_process_ops_log

    rows = merge_device_rows(post_process_ops_log(subdir, has_signposts=True))
    assert len(rows), "expected at least one device op between Tracy signposts"
    return rows


def _report_ccl_perf(op_name, scenario, mesh_shape, rows, traffic):
    measured_ns = pd.to_numeric(rows["DEVICE KERNEL DURATION [ns]"], errors="coerce").sum()
    assert measured_ns > 0, "no measured kernel duration between signposts"
    pm_ideal = pd.to_numeric(rows["PM IDEAL [ns]"], errors="coerce")
    modeled = pm_ideal > 1  # Missing custom models currently emit a generic 1 ns placeholder.
    measured_bandwidth = traffic.critical_path_bytes / measured_ns
    apriori_utilization = traffic.theoretical_ns / measured_ns
    sp, tp = mesh_shape

    logger.info(
        f"{op_name}/{scenario} [SP{sp}xTP{tp}]: "
        f"{_tracy_tensor_description(rows.iloc[0], 'INPUT_0')} -> "
        f"{_tracy_tensor_description(rows.iloc[-1], 'OUTPUT_0')}"
    )
    logger.info(
        f"coded a-priori: {LINK_GBPS_PER_DIRECTION:.1f} Gbps/link/direction x "
        f"{NUM_LINKS} links x {DIRECTIONS_PER_BIDIRECTIONAL_LINK} directions = "
        f"{ROOFLINE_GBPS:.1f} Gbps ({ROOFLINE_BYTES_PER_SECOND / 1e9:.1f} GB/s); "
        f"critical-path={traffic.critical_path_bytes / 1e6:.3f} MB, "
        f"total-mesh={traffic.total_network_bytes / 1e6:.3f} MB, "
        f"theoretical={traffic.theoretical_ns / 1e3:.3f} us"
    )
    logger.info(
        f"Tracy measured: {measured_ns / 1e3:.3f} us, bandwidth={measured_bandwidth:.3f} GB/s, "
        f"apriori utilization={apriori_utilization:.1%}, measured/apriori={measured_ns / traffic.theoretical_ns:.2f}x"
    )
    for _, row in rows.iterrows():
        details = ", ".join(f"{column}={row.get(column)}" for column in PM_COLUMNS)
        logger.info(
            f"Tracy component: {row['OP CODE']}, measured={row['DEVICE KERNEL DURATION [ns]'] / 1e3:.3f} us; {details}"
        )

    if modeled.all():
        pm_ideal_ns = pm_ideal.sum()
        modeled_bandwidth = traffic.critical_path_bytes / pm_ideal_ns
        logger.info(
            f"Tracy logical PM: ideal={pm_ideal_ns / 1e3:.3f} us, "
            f"modeled bandwidth={modeled_bandwidth:.3f} GB/s, "
            f"PM/apriori={pm_ideal_ns / traffic.theoretical_ns:.3f}x, "
            f"measured/PM={measured_ns / pm_ideal_ns:.2f}x, PM utilization={pm_ideal_ns / measured_ns:.1%}"
        )
    else:
        known_pm_ns = pm_ideal[modeled].sum()
        logger.info(
            f"Tracy logical PM: incomplete ({modeled.sum()}/{len(rows)} components modeled, "
            f"known subtotal={known_pm_ns / 1e3:.3f} us); coded a-priori model retained"
        )


def _run_kvpe_all_gather(mesh_device, scenario):
    """Profile the SP all-gather used for the GLM KVPE prefix."""
    sp, tp = mesh_device.shape
    assert sp == GALAXY_MESH_SHAPE[0]
    total_tokens = SCENARIOS[scenario]["cache"] + CHUNK_TOKENS
    kvpe_dim = GLM_CONFIG.kv_lora_rank + GLM_CONFIG.qk_rope_head_dim
    global_shape = [1, 1, total_tokens, kvpe_dim]
    # Validate the smaller warm case; checking long would copy the full 500K cache from every device.
    check_values = scenario == "warm"
    mesh_mapper = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(2), ttnn.PlacementReplicate()],
        mesh_device.shape,
    )
    tt_input = ttnn.rand(
        global_shape,
        mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )
    if check_values:
        # TP replicas are identical; one replica per SP row reconstructs the global KVPE tensor.
        device_inputs = ttnn.get_device_tensors(tt_input)
        torch_input = torch.cat(
            [ttnn.to_torch(device_inputs[sp_index * tp]) for sp_index in range(sp)],
            dim=2,
        )
    gather_semaphores, barrier_semaphore = _global_semaphores(mesh_device)

    op_kwargs = {
        "dim": 2,
        "multi_device_global_semaphore": gather_semaphores,
        "barrier_semaphore": barrier_semaphore,
        "num_links": NUM_LINKS,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "topology": ALL_GATHER_TOPOLOGY,
        "cluster_axis": 0,
    }
    tt_output = _profile_operation(mesh_device, ttnn.experimental.all_gather_async, tt_input, **op_kwargs)
    assert list(tt_output.shape) == global_shape
    assert tt_output.tensor_topology().placements() == [
        ttnn.PlacementReplicate(),
        ttnn.PlacementReplicate(),
    ]
    if check_values:
        for device_tensor in ttnn.get_device_tensors(tt_output):
            equal, message = comp_equal(torch_input, ttnn.to_torch(device_tensor))
            assert equal, message
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)


def _make_reshard_input(
    mesh_device,
    torch_input,
    input_placements,
    output_placements,
    gather_semaphores,
    barrier_semaphore,
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
        num_links=NUM_LINKS,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=RESHARD_TOPOLOGY,
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
    expected_input_topology,
    expected_output_topology,
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
    )
    # Input shape is shared with the traffic roofline; output shape remains an explicit reshard assertion.
    assert list(ttnn.get_device_tensors(tt_input)[0].shape) == expected_input_shape
    assert _placement_strs(tt_input.tensor_topology().placements()) == _placement_strs(expected_input_topology)

    tt_output = _profile_reshard(
        mesh_device,
        tt_input,
        partition_dim,
        dim=gather_dim,
        multi_device_global_semaphore=gather_semaphores,
        barrier_semaphore=barrier_semaphore,
        num_links=NUM_LINKS,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=RESHARD_TOPOLOGY,
        cluster_axis=1,
    )
    assert list(ttnn.get_device_tensors(tt_output)[0].shape) == expected_output_shape
    assert _placement_strs(tt_output.tensor_topology().placements()) == _placement_strs(expected_output_topology)

    # Compose in mesh row-major order explicitly: the generic 2D composer rejects the valid case
    # where both mesh axes shard the same tensor dimension (head -> sequence).
    device_outputs = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(tt_output)]
    mesh_rows = [torch.cat(device_outputs[row * tp : (row + 1) * tp], dim=partition_dim) for row in range(sp)]
    actual = torch.cat(mesh_rows, dim=output_placements[0].dim)
    equal, message = comp_equal(torch_input, actual)
    assert equal, message
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)


def _run_glm_head_to_sequence_reshard(mesh_device):
    """Run GLM's production head-sharded to sequence-sharded TP redistribution."""
    sp, tp = mesh_device.shape
    query_tokens = _query_tokens(sp)
    heads = GLM_CONFIG.num_attention_heads
    q_dim = GLM_CONFIG.kv_lora_rank + GLM_CONFIG.qk_rope_head_dim
    _run_glm_reshard(
        mesh_device,
        [1, heads, query_tokens, q_dim],
        gather_dim=1,
        partition_dim=2,
        input_placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(1)],
        output_placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(2)],
        expected_input_shape=_head_to_sequence_local_input_shape(mesh_device.shape),
        expected_output_shape=[1, heads, query_tokens // (sp * tp), q_dim],
        expected_input_topology=[ttnn.PlacementShard(2), ttnn.PlacementShard(1)],
        expected_output_topology=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()],
    )


def _run_glm_sequence_to_head_reshard(mesh_device):
    """Run GLM's production sequence-sharded to head-sharded TP redistribution."""
    sp, tp = mesh_device.shape
    query_tokens = _query_tokens(sp)
    heads = GLM_CONFIG.num_attention_heads
    latent_dim = GLM_CONFIG.kv_lora_rank
    _run_glm_reshard(
        mesh_device,
        [1, heads, query_tokens, latent_dim],
        gather_dim=2,
        partition_dim=1,
        input_placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(2)],
        output_placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(1)],
        expected_input_shape=_sequence_to_head_local_input_shape(mesh_device.shape),
        expected_output_shape=[1, heads // tp, query_tokens // sp, latent_dim],
        expected_input_topology=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()],
        expected_output_topology=[ttnn.PlacementShard(2), ttnn.PlacementReplicate()],
    )


def _valid_kvpe_op_codes(op_codes):
    return any("AllGather" in code for code in op_codes) or {
        "AllBroadcastDeviceOperation",
        "ConcatDeviceOperation",
    }.issubset(op_codes)


def _valid_reshard_op_codes(op_codes):
    return any("AllGather" in code for code in op_codes) and any("MeshPartition" in code for code in op_codes)


CCL_OPERATIONS = {
    "kvpe_all_gather": CCLOperation(
        runner=_run_kvpe_all_gather,
        local_input_shape=_kvpe_local_input_shape,
        participants_axis=0,
        profile=ProfileParameters(ALL_GATHER_MESH, ALL_GATHER_DEVICE_PARAMS),
        validates_op_codes=_valid_kvpe_op_codes,
        scenario_dependent=True,
    ),
    "glm_head_to_sequence_reshard": CCLOperation(
        runner=_run_glm_head_to_sequence_reshard,
        local_input_shape=_head_to_sequence_local_input_shape,
        participants_axis=1,
        profile=ProfileParameters(RESHARD_MESH, FABRIC_2D_DEVICE_PARAMS),
        validates_op_codes=_valid_reshard_op_codes,
    ),
    "glm_sequence_to_head_reshard": CCLOperation(
        runner=_run_glm_sequence_to_head_reshard,
        local_input_shape=_sequence_to_head_local_input_shape,
        participants_axis=1,
        profile=ProfileParameters(RESHARD_MESH, FABRIC_2D_DEVICE_PARAMS),
        validates_op_codes=_valid_reshard_op_codes,
    ),
}
CCL_OPS = tuple(CCL_OPERATIONS)
PROFILE_OP = os.environ.get("MLA_CCL_OP", CCL_OPS[0])
PROFILE_SCENARIO = os.environ.get("MLA_CCL_SCENARIO", CCL_SCENARIOS[0])
PROFILE_IMPL = os.environ.get("MLA_CCL_IMPL") == "1"
PROFILE_OPERATION = CCL_OPERATIONS[PROFILE_OP]
PROFILE_PARAMETERS = PROFILE_OPERATION.profile


def _traffic_for_workload(operation, scenario, mesh_shape):
    participants = mesh_shape[operation.participants_axis]
    local_input_bytes = math.prod(operation.input_shape(mesh_shape, scenario)) * BFLOAT16_BYTES
    return _all_gather_traffic(local_input_bytes, participants, math.prod(mesh_shape))


@pytest.mark.perf
@pytest.mark.timeout(0)
@pytest.mark.skipif(not PROFILE_IMPL, reason="Tracy worker - launched by test_sparse_mla_ccl_perf")
@pytest.mark.parametrize("mesh_device", [PROFILE_PARAMETERS.mesh], indirect=True)
@pytest.mark.parametrize("device_params", [PROFILE_PARAMETERS.device_params], indirect=True)
def test_sparse_mla_ccl_perf_impl(mesh_device, device_params):
    """Execute exactly one signposted CCL operation in the Tracy subprocess."""
    del device_params
    assert PROFILE_OP in CCL_OPS
    assert PROFILE_SCENARIO in CCL_SCENARIOS

    PROFILE_OPERATION.run(mesh_device, PROFILE_SCENARIO)


def _profiler_subdir(op_name, scenario):
    return f"sparse_mla_ccl_perf/{op_name}_{scenario}"


@pytest.mark.perf
@pytest.mark.timeout(0)
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf test - run locally with Tracy")
@pytest.mark.parametrize("scenario", CCL_SCENARIOS, ids=_scenario_id)
@pytest.mark.parametrize("op_name", CCL_OPS)
def test_sparse_mla_ccl_perf(op_name, scenario):
    """Profile one mainline CCL flow and compare measured, built-in PM, and a-priori estimates."""
    from tracy.process_model_log import run_device_profiler

    operation = CCL_OPERATIONS[op_name]
    parameters = operation.profile
    subdir = _profiler_subdir(op_name, scenario)
    command = (
        "pytest -m perf models/demos/deepseek_v3_d_p/tests/sparse_mla/test_sparse_mla_ccl_perf.py"
        "::test_sparse_mla_ccl_perf_impl"
    )
    with mock.patch.dict(
        os.environ,
        {"CI": "false", "MLA_CCL_IMPL": "1", "MLA_CCL_OP": op_name, "MLA_CCL_SCENARIO": scenario},
    ):
        run_device_profiler(
            command,
            subdir,
            device_analysis_types=["device_kernel_duration"],
            op_support_count=64,
        )

    rows = _read_tracy_ccl_rows(subdir)
    op_codes = set(rows["OP CODE"])
    assert operation.validates_op_codes(op_codes), op_codes
    _report_ccl_perf(
        op_name,
        scenario,
        parameters.mesh_shape,
        rows,
        _traffic_for_workload(operation, scenario, parameters.mesh_shape),
    )
