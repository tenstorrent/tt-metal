# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Unscheduled Fabric2D CCL diagnostics for GLM sparse-MLA tensor shapes and placements.

The retained TP redistributions from production ``tt/mla/mla.py`` are expressed as ``CollectivePath``
values and turned into build -> profile -> verify -> report cycles. The SP all-gather is intentionally
owned by the full sparse-MLA production gate rather than a degenerate 8x1 proxy.
"""

import math
import os
from dataclasses import dataclass
from typing import Callable

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import glm_hf_config
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_mesh import detect_num_devices
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_perf import CHUNK_TOKENS, GALAXY_SP, SCENARIOS
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

# Mesh axes. In production the layout is SP (sequence) on axis 0, TP (tensor) on axis 1.
SP_AXIS = 0
TP_AXIS = 1

# Fabric link count per direction. Galaxy 8x4 uses 2 links (see conftest FABRIC_2D_..._8x4 params); the
# LoudBox proxies mirror it. A hardware fact — see external Blackhole fabric docs.
NUM_LINKS = 2

# Per-direction fabric link bandwidth, Gbps. Blackhole-only; sourced from external hardware docs (not
# in-repo). Override with MLA_CCL_LINK_GBPS_PER_DIRECTION for what-if analysis.
_GALAXY_LINK_GBPS_PER_DIRECTION = 200.0
_LOUDBOX_LINK_GBPS_PER_DIRECTION = 400.0


# --------------------------------------------------------------------------------------------------
# Domain values
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Workload:
    """GLM tensor dimensions a collective moves. Model dims come from the single-source reference config."""

    chunk_tokens: int
    cache_tokens: int
    num_attention_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int

    @property
    def kvpe_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


@dataclass(frozen=True)
class CCLTraffic:
    """Topology-aware fabric roofline for one collective."""

    # Bytes crossing the busiest link in one direction. This sets the collective latency, so theoretical
    # time and measured bandwidth use it rather than aggregate network traffic.
    critical_path_bytes: float
    # Bytes moved across every fabric link, including forwarding hops. This is aggregate work for context,
    # not a latency term.
    total_network_bytes: float
    link_gigabits_per_second_per_direction: float
    num_links: int
    sustained_directions: int
    topology: object

    @property
    def roofline_gigabits_per_second(self) -> float:
        return self.link_gigabits_per_second_per_direction * self.num_links * self.sustained_directions

    @property
    def roofline_gigabytes_per_second(self) -> float:
        return self.roofline_gigabits_per_second / 8

    @property
    def theoretical_ns(self) -> float:
        return self.critical_path_bytes / self.roofline_gigabytes_per_second  # bytes / (GB/s) = ns


@dataclass(frozen=True)
class Measurement:
    """Real-time profiler output for one measured region (may span several device programs)."""

    records: tuple
    program_durations_ns: dict
    input_description: str
    output_description: str

    @property
    def duration_ns(self) -> float:
        # Sum across every program dispatched by the measured logical collective.
        return sum(self.program_durations_ns.values())


@dataclass(frozen=True)
class RuntimeSystem:
    """Fabric parameters resolved from the live mesh for a given collective axis."""

    mesh_shape: tuple
    topology: object
    num_links: int
    link_gigabits_per_second_per_direction: float


@dataclass(frozen=True)
class CollectivePath:
    """One production sparse-MLA collective, described declaratively.

    ``collective_axis`` unifies three things: which mesh axis the op runs over (SP/TP), the ``cluster_axis``
    of its collective, and which axis supplies the roofline ``participants`` count. Both retained
    paths are all-to-all reshards. Shapes are ``(workload, mesh_shape) -> list`` builders so a proxy
    and Galaxy share one source.
    """

    name: str
    mla_ref: str  # the production source this mirrors, e.g. "mla.py:1468"
    collective_axis: int
    gather_dim: int
    input_placements: tuple
    layout: object
    logical_shape: Callable
    local_input_shape: Callable
    output_placements: tuple  # tensor placements after the collective (asserted post-op)
    partition_dim: int
    expected_output_shape: Callable
    verify_reshard: bool = False

    def __post_init__(self):
        assert all(
            isinstance(placement, ttnn.PlacementShard) for placement in self.input_placements + self.output_placements
        ), f"{self.name}: reshard placements must all be PlacementShard"
        assert self.input_placements[self.collective_axis].dim == self.gather_dim
        assert self.output_placements[self.collective_axis].dim == self.partition_dim


# --------------------------------------------------------------------------------------------------
# Shape builders (Galaxy-normalized so every proxy profiles the Galaxy per-chip shard)
# --------------------------------------------------------------------------------------------------
def _query_tokens(workload: Workload, sp: int) -> int:
    # Scale the global query count by SP/GALAXY_SP so the per-chip sequence shard (query/sp) equals the
    # Galaxy-local size (chunk/GALAXY_SP) on every proxy. Same idiom as test_sparse_mla_perf._local_cache_tokens.
    return workload.chunk_tokens * sp // GALAXY_SP


def _head_to_sequence_logical_shape(w: Workload, mesh_shape) -> list:
    sp, _ = mesh_shape
    return [1, w.num_attention_heads, _query_tokens(w, sp), w.kvpe_dim]


def _head_to_sequence_local_input_shape(w: Workload, mesh_shape) -> list:
    sp, tp = mesh_shape
    return [1, w.num_attention_heads // tp, _query_tokens(w, sp) // sp, w.kvpe_dim]


def _head_to_sequence_output_shape(w: Workload, mesh_shape) -> list:
    sp, tp = mesh_shape
    return [1, w.num_attention_heads, _query_tokens(w, sp) // (sp * tp), w.kvpe_dim]


def _sequence_to_head_logical_shape(w: Workload, mesh_shape) -> list:
    sp, _ = mesh_shape
    return [1, w.num_attention_heads, _query_tokens(w, sp), w.kv_lora_rank]


def _sequence_to_head_local_input_shape(w: Workload, mesh_shape) -> list:
    sp, tp = mesh_shape
    return [1, w.num_attention_heads, _query_tokens(w, sp) // (sp * tp), w.kv_lora_rank]


def _sequence_to_head_output_shape(w: Workload, mesh_shape) -> list:
    sp, tp = mesh_shape
    return [1, w.num_attention_heads // tp, _query_tokens(w, sp) // sp, w.kv_lora_rank]


# The two retained TP redistribution diagnostics, as data. The redundant 8x1 SP all-gather proxy was
# removed; full sparse-MLA production coverage owns that collective on the real 8x4 Galaxy profile.
GLM_HEAD_TO_SEQUENCE = CollectivePath(
    name="glm_head_to_sequence_reshard",
    mla_ref="mla.py:1481-1489 (_sparse_mla thin-head transpose)",
    collective_axis=TP_AXIS,
    gather_dim=1,
    partition_dim=2,
    input_placements=(ttnn.PlacementShard(2), ttnn.PlacementShard(1)),  # SP shards Q, TP shards H.
    output_placements=(ttnn.PlacementShard(2), ttnn.PlacementShard(2)),  # SP shards Q, TP also shards Q.
    layout=ttnn.TILE_LAYOUT,
    verify_reshard=True,
    logical_shape=_head_to_sequence_logical_shape,
    local_input_shape=_head_to_sequence_local_input_shape,
    expected_output_shape=_head_to_sequence_output_shape,
)

GLM_SEQUENCE_TO_HEAD = CollectivePath(
    name="glm_sequence_to_head_reshard",
    mla_ref="mla.py:1528-1538 (_sparse_mla transpose inverse)",
    collective_axis=TP_AXIS,
    gather_dim=2,
    partition_dim=1,
    input_placements=(ttnn.PlacementShard(2), ttnn.PlacementShard(2)),  # SP shards Q, TP also shards Q.
    output_placements=(ttnn.PlacementShard(2), ttnn.PlacementShard(1)),  # SP shards Q, TP shards H.
    layout=ttnn.TILE_LAYOUT,
    verify_reshard=True,
    logical_shape=_sequence_to_head_logical_shape,
    local_input_shape=_sequence_to_head_local_input_shape,
    expected_output_shape=_sequence_to_head_output_shape,
)


# --------------------------------------------------------------------------------------------------
# System resolution
# --------------------------------------------------------------------------------------------------
def ccl_mesh_param():
    """Stable unwrapped-Fabric2D diagnostic profile selected only by available mesh size."""
    num_devices = detect_num_devices()
    canonical_fabric = {
        "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    }
    fabric_2d = {"trace_region_size": 100000, "fabric_config": ttnn.FabricConfig.FABRIC_2D, **canonical_fabric}
    if num_devices == 32:
        mesh_shape, mesh_topology, device_params = (8, 4), "mesh-8x4", fabric_2d
    elif num_devices == 8:
        mesh_shape, mesh_topology, device_params = (2, 4), "mesh-2x4", fabric_2d
    else:
        reason = f"CCL perf supports Galaxy (32 chips) or LoudBox (8), found {num_devices}"
        return pytest.param((2, 2), fabric_2d, marks=pytest.mark.skip(reason=reason), id="unsupported")

    return pytest.param(
        mesh_shape,
        device_params,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=mesh_shape, topology=mesh_topology),
        id=f"fabric2d-{mesh_shape[0]}x{mesh_shape[1]}",
    )


def resolve_runtime_system(mesh_device, topology) -> RuntimeSystem:
    """Fabric roofline inputs for the live mesh: topology, link count, per-direction bandwidth."""
    mesh_shape = tuple(mesh_device.shape)
    default_gbps = _GALAXY_LINK_GBPS_PER_DIRECTION if math.prod(mesh_shape) == 32 else _LOUDBOX_LINK_GBPS_PER_DIRECTION
    link_gbps = float(os.environ.get("MLA_CCL_LINK_GBPS_PER_DIRECTION", default_gbps))
    return RuntimeSystem(mesh_shape, topology, NUM_LINKS, link_gbps)


# --------------------------------------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------------------------------------
def _profile_programs(mesh_device, run_fn):
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("Real-time profiler must be active for sparse MLA CCL perf checks")

    # Drain setup programs before registering the callback so only run_fn contributes records.
    ttnn.synchronize_device(mesh_device)
    result, records = profile_realtime_program(mesh_device, run_fn, collect_all=True)
    # Receiver delivery is asynchronous: a setup program completed before callback registration can arrive
    # after it and appear in records. A2A dispatches exactly one measured program, so select the newest ID.
    measured_runtime_id = max(record["runtime_id"] for record in records)
    records = tuple(record for record in records if record["runtime_id"] == measured_runtime_id)

    program_durations_ns = {}
    for record in records:
        runtime_id = record["runtime_id"]
        if runtime_id:
            program_durations_ns[runtime_id] = max(
                program_durations_ns.get(runtime_id, 0.0), float(record["duration_ns"])
            )
    assert program_durations_ns, "real-time profiler returned no valid program durations"
    return result, tuple(records), program_durations_ns


def _tensor_description(tensor):
    local_tensor = ttnn.get_device_tensors(tensor)[0]
    return f"{list(local_tensor.shape)} ({local_tensor.dtype}, {local_tensor.layout}, {local_tensor.memory_config()})"


# --------------------------------------------------------------------------------------------------
# Collective execution
# --------------------------------------------------------------------------------------------------
def _reshard(tt_input, output_buffer, path, system):
    """Run the single all-to-all used by sparse MLA to exchange sharded tensor dimensions."""
    return ttnn.experimental.all_to_all_async_generic(
        tt_input,
        in_dim=path.gather_dim,
        out_dim=path.partition_dim,
        persistent_output_buffer=output_buffer,
        num_links=system.num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=system.topology,
        cluster_axis=path.collective_axis,
    )


def _build_reshard_input(mesh_device, path, torch_input, system):
    input_dims = [placement.dim for placement in path.input_placements]
    # Distinct shard dims can be constructed directly by the host mesh mapper.
    if len(set(input_dims)) == len(input_dims):
        mapper = ttnn.create_mesh_mapper(
            mesh_device, ttnn.MeshMapperConfig(list(path.input_placements), mesh_device.shape)
        )
        return ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=path.layout,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    # Duplicate shard dims (e.g. [Shard(seq), Shard(seq)]): the host mapper rejects these, so build the
    # source in the DISTINCT output placement and run the all-to-all inverse. Drain this construction op
    # before profiling so the measurement contains only the production-direction collective.
    cax = path.collective_axis
    source_mapper = ttnn.create_mesh_mapper(
        mesh_device, ttnn.MeshMapperConfig(list(path.output_placements), mesh_device.shape)
    )
    source = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        layout=path.layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=source_mapper,
    )
    tt_input = ttnn.experimental.all_to_all_async_generic(
        source,
        in_dim=path.output_placements[cax].dim,
        out_dim=path.input_placements[cax].dim,
        num_links=system.num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=cax,
        topology=system.topology,
    )
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(source)
    return tt_input


def _assert_reshard_lossless(tt_output, torch_input, path, sp, tp):
    # Compose in mesh row-major order explicitly: the generic 2D composer rejects the valid case where both
    # mesh axes shard the same tensor dimension. Cat the TP shards within each SP row, then cat the SP rows.
    tp_shard_dim = path.output_placements[TP_AXIS].dim
    sp_shard_dim = path.output_placements[SP_AXIS].dim
    device_outputs = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(tt_output)]
    mesh_rows = [torch.cat(device_outputs[row * tp : (row + 1) * tp], dim=tp_shard_dim) for row in range(sp)]
    actual = torch.cat(mesh_rows, dim=sp_shard_dim)
    equal, message = comp_equal(torch_input, actual)  # bit-exact: a reshard is pure data movement
    assert equal, message


def _run_reshard(mesh_device, path, workload, system) -> Measurement:
    sp, tp = mesh_device.shape
    mesh_shape = tuple(mesh_device.shape)
    torch_input = torch.rand(path.logical_shape(workload, mesh_shape), dtype=torch.bfloat16)
    tt_input = _build_reshard_input(mesh_device, path, torch_input, system)
    # Input shape is shared with the traffic roofline; output shape remains an explicit reshard assertion.
    assert list(ttnn.get_device_tensors(tt_input)[0].shape) == path.local_input_shape(workload, mesh_shape)
    # Match MLA: preallocate the exact per-device output once, outside the profiled collective.
    output_buffer = ttnn.empty(
        path.expected_output_shape(workload, mesh_shape),
        device=mesh_device,
        layout=path.layout,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output, records, program_durations_ns = _profile_programs(
        mesh_device,
        lambda: _reshard(tt_input, output_buffer, path, system),
    )
    assert list(ttnn.get_device_tensors(tt_output)[0].shape) == path.expected_output_shape(workload, mesh_shape)
    if path.verify_reshard:
        _assert_reshard_lossless(tt_output, torch_input, path, sp, tp)

    measurement = Measurement(
        records, program_durations_ns, _tensor_description(tt_input), _tensor_description(tt_output)
    )
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)
    return measurement


# --------------------------------------------------------------------------------------------------
# Roofline + reporting
# --------------------------------------------------------------------------------------------------
def collective_roofline(path: CollectivePath, workload: Workload, mesh_device, system: RuntimeSystem) -> CCLTraffic:
    """Independent Linear/Ring fabric roofline for the production collective."""
    mesh_shape = tuple(mesh_device.shape)
    local_input_bytes = math.prod(path.local_input_shape(workload, mesh_shape)) * torch.bfloat16.itemsize
    participants = mesh_shape[path.collective_axis]
    num_devices = math.prod(mesh_shape)
    assert system.topology in (ttnn.Topology.Linear, ttnn.Topology.Ring)
    # Both logical schedules use destination-based routing over the same physical FABRIC_2D mesh.
    # On the busiest midpoint link, left_sources * right_destinations chunks cross in each direction.
    midpoint = participants / 2
    critical_path_bytes = local_input_bytes * midpoint * (participants - midpoint) / participants
    total_network_bytes = local_input_bytes * num_devices * (participants**2 - 1) / (3 * participants)
    sustained_directions = 1
    return CCLTraffic(
        critical_path_bytes=critical_path_bytes,
        total_network_bytes=total_network_bytes,
        link_gigabits_per_second_per_direction=system.link_gigabits_per_second_per_direction,
        num_links=system.num_links,
        sustained_directions=sustained_directions,
        topology=system.topology,
    )


def report(path: CollectivePath, scenario: str, mesh_device, measurement: Measurement, traffic: CCLTraffic):
    measured_ns = measurement.duration_ns
    assert measured_ns > 0, "real-time profiler measured no device-program duration"
    measured_gigabytes_per_second = traffic.critical_path_bytes / measured_ns
    roofline_utilization = traffic.theoretical_ns / measured_ns
    sp, tp = mesh_device.shape

    logger.info(
        f"{path.name}/{scenario} [SP{sp}xTP{tp}]: {measurement.input_description} -> {measurement.output_description}"
    )
    logger.info(
        f"theoretical fabric roofline: {traffic.link_gigabits_per_second_per_direction:.1f} "
        f"Gbps/link/direction x {traffic.num_links} links x {traffic.sustained_directions} direction(s) "
        f"({traffic.topology}) = "
        f"{traffic.roofline_gigabits_per_second:.1f} Gbps ({traffic.roofline_gigabytes_per_second:.1f} GB/s); "
        f"critical-path={traffic.critical_path_bytes / 1e6:.3f} MB, "
        f"total-mesh={traffic.total_network_bytes / 1e6:.3f} MB, "
        f"theoretical={traffic.theoretical_ns / 1e3:.3f} us"
    )
    logger.info(
        f"real-time profiler measured: {measured_ns / 1e3:.3f} us (all_to_all), "
        f"achieved ethernet bandwidth={measured_gigabytes_per_second:.3f} / "
        f"{traffic.roofline_gigabytes_per_second:.3f} GB/s, "
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


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------
# Shared marks for every benchmark: perf-only, no timeout, skipped in CI (run locally on hardware).
pytestmark = [
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.skipif(os.environ.get("CI") == "true", reason="performance test - run locally"),
]

# Single-chunk scenarios only; the looping (cold prefill) scenario is not a single CCL measurement.
_NON_LOOP_SCENARIOS = tuple(name for name, scenario in SCENARIOS.items() if not scenario["loop"])


def _scenario_id(scenario):
    return f"{scenario}_{SCENARIOS[scenario]['cache'] // 1024}k_cache"


def _workload(scenario):
    config = glm_hf_config()
    return Workload(
        chunk_tokens=CHUNK_TOKENS,
        cache_tokens=SCENARIOS[scenario]["cache"],
        num_attention_heads=config.num_attention_heads,
        kv_lora_rank=config.kv_lora_rank,
        qk_rope_head_dim=config.qk_rope_head_dim,
    )


def _run(mesh_device, path, scenario):
    assert mesh_device.arch() == ttnn.Arch.BLACKHOLE, "bandwidth assumptions apply to Blackhole only"
    workload = _workload(scenario)
    topology = per_axis_topology()[path.collective_axis]
    system = resolve_runtime_system(mesh_device, topology)
    measurement = _run_reshard(mesh_device, path, workload, system)
    traffic = collective_roofline(path, workload, mesh_device, system)
    report(path, scenario, mesh_device, measurement, traffic)


@pytest.mark.parametrize("scenario", _NON_LOOP_SCENARIOS, ids=_scenario_id)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [ccl_mesh_param()],
    indirect=["mesh_device", "device_params"],
)
def test_glm_head_to_sequence_reshard_perf(mesh_device, scenario):
    """Profile GLM's head-sharded to sequence-sharded TP redistribution."""
    _run(mesh_device, GLM_HEAD_TO_SEQUENCE, scenario)


@pytest.mark.parametrize("scenario", _NON_LOOP_SCENARIOS, ids=_scenario_id)
@pytest.mark.parametrize(
    "mesh_device,device_params",
    [ccl_mesh_param()],
    indirect=["mesh_device", "device_params"],
)
def test_glm_sequence_to_head_reshard_perf(mesh_device, scenario):
    """Profile GLM's sequence-sharded to head-sharded TP redistribution."""
    _run(mesh_device, GLM_SEQUENCE_TO_HEAD, scenario)
