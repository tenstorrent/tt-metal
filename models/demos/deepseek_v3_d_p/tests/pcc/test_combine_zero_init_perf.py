# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Benchmark test for the combine operation's DRAM zero initialization.

For each topology/link configuration, runs the combine op N times with
DRAM output and init_zeros=True using BOTH the legacy (single-core) and
distributed (multi-core) zero-init paths, then reports device kernel
duration from Tracy side by side. A summary table is printed after all
configurations have been benchmarked.

Requires Tracy env vars for device timing:
  TT_METAL_DEVICE_PROFILER=1
  TT_METAL_PROFILER_MID_RUN_DUMP=1
  TT_METAL_PROFILER_CPP_POST_PROCESS=1
Without them, falls back to wall-clock timing.
"""

import atexit
from dataclasses import dataclass

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.tt.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    create_fabric_router_config,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import assert_output_shape, validate_combine_output
from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time

NUM_ITERATIONS = 10

_all_results: list["BenchResult"] = []


@dataclass
class BenchResult:
    config_label: str
    no_init_ns: int
    legacy_ns: int
    dist_ns: int
    inline_ns: int
    source: str

    @property
    def speedup_dist(self) -> float:
        return self.legacy_ns / self.dist_ns if self.dist_ns > 0 else float("inf")

    @property
    def speedup_inline(self) -> float:
        return self.legacy_ns / self.inline_ns if self.inline_ns > 0 else float("inf")

    @property
    def legacy_overhead_us(self) -> float:
        return (self.legacy_ns - self.no_init_ns) / 1e3

    @property
    def dist_overhead_us(self) -> float:
        return (self.dist_ns - self.no_init_ns) / 1e3

    @property
    def inline_overhead_us(self) -> float:
        return (self.inline_ns - self.no_init_ns) / 1e3


def _get_tracy_duration_ns(device_id: int) -> int | None:
    """Extract device kernel duration (ns) from Tracy profiler data, or None if unavailable."""
    try:
        latest = ttnn.get_latest_programs_perf_data()
    except Exception:
        return None
    if not latest or device_id not in latest:
        return None
    programs = latest[device_id]
    if not programs:
        return None
    duration_ns = None
    for p in programs:
        for key in ("DEVICE KERNEL DURATION [ns]", "DEVICE FW DURATION [ns]"):
            if key in p.program_analyses_results:
                d = p.program_analyses_results[key].duration
                if d is not None:
                    duration_ns = max(duration_ns, d) if duration_ns is not None else d
                break
    return int(duration_ns) if duration_ns is not None else None


def _run_combine_variant(
    mesh_device,
    tt_dispatched_buffer,
    tt_dispatched_metadata,
    tt_expert_token_counts,
    dispatch_group_size,
    num_dispatch_groups,
    experts_per_chip,
    num_experts_per_tok,
    seq_len_per_chip,
    sp_axis,
    num_links,
    topology,
    distributed_zero_init: bool,
    init_zeros: bool = True,
    inline_zero_init: bool = False,
) -> tuple[int, str, ttnn.Tensor]:
    """Run one variant, return (avg_duration_ns, source_label, last_output_tensor)."""
    tt_combine = TtCombineModule(
        mesh_device=mesh_device,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        cluster_axis=sp_axis,
        num_links=num_links,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        init_zeros=init_zeros,
        distributed_zero_init=distributed_zero_init,
        inline_zero_init=inline_zero_init,
    )

    # Warmup
    _ = tt_combine(tt_dispatched_buffer, tt_dispatched_metadata, tt_expert_token_counts)
    ttnn.synchronize_device(mesh_device)

    # Timed iterations
    ttnn.synchronize_device(mesh_device)
    wallclock_start = start_measuring_time()

    output = None
    for _ in range(NUM_ITERATIONS):
        output = tt_combine(tt_dispatched_buffer, tt_dispatched_metadata, tt_expert_token_counts)

    ttnn.synchronize_device(mesh_device)
    wallclock_duration_ns = stop_measuring_time(wallclock_start)

    ttnn.ReadDeviceProfiler(mesh_device)
    device_id = mesh_device.get_device_ids()[0]
    tracy_duration_ns = _get_tracy_duration_ns(device_id)

    if tracy_duration_ns is not None:
        return tracy_duration_ns, "Tracy", output
    else:
        return wallclock_duration_ns // NUM_ITERATIONS, "wall-clock", output


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-1link",
        ),
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="linear"),
            id="linear-2link",
        ),
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            1,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="ring-1link",
        ),
        pytest.param(
            (4, 1),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
            },
            2,
            ttnn.Topology.Ring,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 1), topology="ring"),
            id="ring-2link",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_combine_zero_init_perf(
    mesh_device,
    num_links,
    topology,
):
    """Benchmark combine op: runs both legacy and distributed zero-init, reports side by side."""

    seq_len_per_chip = 3200
    emb_dim = 7168
    num_routed_experts = 64
    num_experts_per_tok = 2
    capacity_factor = 2

    topo_name = "Ring" if topology == ttnn.Topology.Ring else "Linear"
    config_label = f"{topo_name}-{num_links}link"

    num_devices = mesh_device.get_num_devices()
    mesh_config = extract_mesh_config(mesh_device)
    sp_axis = mesh_config.sp_axis
    dispatch_group_size = mesh_config.dispatch_group_size
    num_dispatch_groups = mesh_config.num_dispatch_groups

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, dispatch_group_size, capacity_factor
    )

    x, weights, indices = initialize_test_inputs(
        dispatch_group_size,
        seq_len_per_chip,
        emb_dim,
        num_routed_experts,
        num_experts_per_tok,
        max_dispatched_tokens_per_expert,
        seed=42,
        num_dispatch_groups=num_dispatch_groups,
    )

    expert_offsets, expert_token_counts, _ = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )

    torch_dispatch_module = TorchDispatchModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_dispatch_groups=num_dispatch_groups,
        expert_dispatch_table=expert_dispatch_table,
    )

    dispatched_buffer, dispatched_metadata = torch_dispatch_module(x, weights, indices, expert_offsets)

    mesh_mapper = get_ep_mesh_mapper(mesh_device)

    tt_dispatched_buffer = ttnn.from_torch(
        dispatched_buffer,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    tt_dispatched_metadata = ttnn.from_torch(
        dispatched_metadata,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    tt_expert_token_counts = ttnn.from_torch(
        expert_token_counts,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    common_args = dict(
        mesh_device=mesh_device,
        tt_dispatched_buffer=tt_dispatched_buffer,
        tt_dispatched_metadata=tt_dispatched_metadata,
        tt_expert_token_counts=tt_expert_token_counts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        sp_axis=sp_axis,
        num_links=num_links,
        topology=topology,
    )

    # Compute torch reference for PCC validation
    torch_combine = TorchCombineModule(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
        num_dispatch_groups=num_dispatch_groups,
    )
    torch_output = torch_combine(dispatched_buffer, dispatched_metadata, expert_token_counts)

    # Run no-init baseline (init_zeros=False)
    logger.info(f"[{config_label}] Running no-init baseline...")
    no_init_ns, no_init_src, _ = _run_combine_variant(**common_args, distributed_zero_init=False, init_zeros=False)

    # Run legacy (single-core) variant
    logger.info(f"[{config_label}] Running legacy (single-core) zero-init...")
    legacy_ns, legacy_src, legacy_output = _run_combine_variant(**common_args, distributed_zero_init=False)

    # Run distributed (multi-core) variant
    logger.info(f"[{config_label}] Running distributed (multi-core) zero-init...")
    dist_ns, dist_src, dist_output = _run_combine_variant(**common_args, distributed_zero_init=True)

    # Run inline (sender-core CB reuse) variant
    logger.info(f"[{config_label}] Running inline (sender-core) zero-init...")
    inline_ns, inline_src, inline_output = _run_combine_variant(
        **common_args, distributed_zero_init=False, inline_zero_init=True
    )

    # PCC validation for all zero-init variants
    mesh_composer = get_ep_mesh_composer(mesh_device)

    pcc_errors = []
    for variant_name, tt_output in [
        ("legacy", legacy_output),
        ("distributed", dist_output),
        ("inline", inline_output),
    ]:
        tt_output_torch = ttnn.to_torch(tt_output, mesh_composer=mesh_composer, dtype=torch.bfloat16)
        assert_output_shape(tt_output_torch, num_dispatch_groups, dispatch_group_size, f"{variant_name} combine output")

        pcc_result = validate_combine_output(
            torch_output,
            tt_output_torch,
            indices,
            num_dispatch_groups,
            num_routed_experts,
            verbose=False,
            expert_dispatch_table=expert_dispatch_table,
            expert_token_counts=expert_token_counts,
            experts_per_chip=experts_per_chip,
        )
        if pcc_result.passed:
            logger.info(f"[{config_label}] {variant_name} PCC: PASSED")
        else:
            logger.error(f"[{config_label}] {variant_name} PCC: FAILED")
            pcc_errors.append(variant_name)

    source = legacy_src if legacy_src == "Tracy" else dist_src
    result = BenchResult(
        config_label=config_label,
        no_init_ns=no_init_ns,
        legacy_ns=legacy_ns,
        dist_ns=dist_ns,
        inline_ns=inline_ns,
        source=source,
    )
    _all_results.append(result)

    # Per-config report
    logger.info(f"")
    logger.info(f"{'=' * 90}")
    logger.info(f"  COMBINE ZERO-INIT  {config_label}  (seq=3200 experts=64 topk=2)")
    logger.info(f"{'=' * 90}")
    logger.info(f"  No-init ({no_init_src}):        {no_init_ns:>10} ns  ({no_init_ns/1e3:>10.2f} us)")
    logger.info(
        f"  Legacy  ({legacy_src}):        {legacy_ns:>10} ns  ({legacy_ns/1e3:>10.2f} us)  overhead: {result.legacy_overhead_us:+.2f} us"
    )
    logger.info(
        f"  Distributed ({dist_src}):    {dist_ns:>10} ns  ({dist_ns/1e3:>10.2f} us)  overhead: {result.dist_overhead_us:+.2f} us"
    )
    logger.info(
        f"  Inline ({inline_src}):         {inline_ns:>10} ns  ({inline_ns/1e3:>10.2f} us)  overhead: {result.inline_overhead_us:+.2f} us"
    )
    logger.info(f"  Speedup legacy/dist:           {result.speedup_dist:.2f}x")
    logger.info(f"  Speedup legacy/inline:         {result.speedup_inline:.2f}x")
    logger.info(f"{'=' * 90}")

    assert not pcc_errors, f"PCC validation failed for: {', '.join(pcc_errors)}"


def _print_summary():
    """Print a summary table of all benchmark results collected across test runs."""
    if not _all_results:
        return

    logger.info(f"")
    logger.info(f"{'=' * 140}")
    logger.info(f"  COMBINE ZERO-INIT BENCHMARK SUMMARY  (seq=3200, experts=64, topk=2, {NUM_ITERATIONS} iters)")
    logger.info(f"{'=' * 140}")
    logger.info(
        f"  {'Config':<16} {'Source':<10} {'No-init(us)':>12} {'Legacy(us)':>11} {'Dist(us)':>11} {'Inline(us)':>11}"
        f" {'Lgcy OH(us)':>12} {'Dist OH(us)':>12} {'Inln OH(us)':>12} {'Dist spd':>9} {'Inln spd':>9}"
    )
    logger.info(
        f"  {'-'*16} {'-'*10} {'-'*12} {'-'*11} {'-'*11} {'-'*11}" f" {'-'*12} {'-'*12} {'-'*12} {'-'*9} {'-'*9}"
    )
    for r in _all_results:
        logger.info(
            f"  {r.config_label:<16} {r.source:<10} {r.no_init_ns/1e3:>12.2f} {r.legacy_ns/1e3:>11.2f}"
            f" {r.dist_ns/1e3:>11.2f} {r.inline_ns/1e3:>11.2f}"
            f" {r.legacy_overhead_us:>+12.2f} {r.dist_overhead_us:>+12.2f} {r.inline_overhead_us:>+12.2f}"
            f" {r.speedup_dist:>8.2f}x {r.speedup_inline:>8.2f}x"
        )
    logger.info(f"{'=' * 140}")


atexit.register(_print_summary)
