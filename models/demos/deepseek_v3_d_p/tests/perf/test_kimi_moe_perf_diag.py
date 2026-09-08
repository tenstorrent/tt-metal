# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""Reproduction harness for #55439, the MoE perf gate's 1.10 ms bimodality.

Not a gate -- it asserts nothing about time. It runs the same forward ``test_kimi_moe_perf.py``
measures, N times in one profiler window, and reports per iteration the summed critical path, the
dispatch program's own record, and one chip's program timeline.

What it showed. The dispatch program's record is binary: ~0.7 us when its workers return before the
fabric transfer lands, ~1.12 ms (K2.7) / ~1.55 ms (K3) when they wait for it. The gated total tracks
that 1:1 and nothing else in the forward moves by more than ~45 us, which is the whole flakiness.
The chip timeline is what distinguishes the two: in the async state the record closes in <1 us and
the transfer time reappears as the gap before the next program, so a duration alone cannot tell them
apart. The state is per forward but strongly autocorrelated -- one process measured 4 async and 1
bracketed, another 3 bracketed in a row -- so sampling within a process does not average it out.

Record loss was ruled out here too: every program came back with all 32 chip records, and dropping
any single chip's records costs at most 2.2% of the total, against a 1.10 ms (20%) gap.

Run it as the gate runs, one measured pass:
    KIMI_MOE_DIAG_ITERS=1 pytest models/demos/deepseek_v3_d_p/tests/perf/test_kimi_moe_perf_diag.py -k k2_7
"""

import os
import statistics

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tests.perf.test_kimi_moe_perf import (
    _CASES,
    _DISPATCH_BUFFER_CAPACITY_FACTOR,
    _DISPATCH_KERNEL,
    _FABRIC_PAYLOAD_SIZE,
    _SEQ_LEN_PER_CHIP,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import get_ddr_speed, get_tdp_limit_max
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program, require_realtime_profiler

_ITERS = int(os.environ.get("KIMI_MOE_DIAG_ITERS", "3"))
# Generous: this harness cares about a complete record set, not about finishing fast.
_RECORD_TIMEOUT_S = 30.0


def _kernel_tag(sources):
    return ",".join(sorted({source.rsplit("/", 1)[-1] for source in sources}))


@pytest.mark.skipif(not is_blackhole(), reason="Kimi prefill MoE requires Blackhole")
@pytest.mark.timeout(0)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=_FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant, case", _CASES, indirect=["variant"])
def test_kimi_moe_perf_diag(variant, case, config_only, mesh_device, device_params, num_links, request):
    require_realtime_profiler(f"the {case.label} MoE perf diagnostic")
    topology = per_axis_topology(device_params["fabric_config"])
    logger.info(f"DIAG env: tdp_limit_max={get_tdp_limit_max()} ddr_speed={get_ddr_speed()}")

    collected = {}

    def measure(forward):
        warm = forward()
        ttnn.synchronize_device(mesh_device)
        del warm

        def run_all():
            last = None
            for _ in range(_ITERS):
                last = forward()
            return last

        result, records = profile_realtime_program(
            mesh_device, run_all, collect_all=True, record_timeout_seconds=_RECORD_TIMEOUT_S
        )
        collected["records"] = records
        return result

    run_model(
        variant,
        config_only,
        mesh_device,
        device_params,
        _SEQ_LEN_PER_CHIP,
        case.config.EMB_SIZE,
        case.config.MOE_INTERMEDIATE_SIZE,
        case.config.NUM_ROUTED_EXPERTS,
        case.config.NUM_EXPERTS_PER_TOKEN,
        _DISPATCH_BUFFER_CAPACITY_FACTOR,
        False,
        num_links,
        topology,
        GateComputeMode.DEVICE_FP32,
        request,
        **case.extra,
        measure=measure,
    )

    records = collected.get("records")
    assert records, "real-time profiler returned no records"

    # runtime_id is unique per dispatch, and records arrive in dispatch order, so the programs of
    # iteration k are the k-th slice of equal length.
    per_program = {}
    for record in records:
        runtime_id = record["runtime_id"]
        if not runtime_id:
            continue
        entry = per_program.setdefault(
            runtime_id, {"by_chip": {}, "spans": {}, "kernels": _kernel_tag(record["kernel_sources"])}
        )
        entry["by_chip"][record["chip_id"]] = record["duration_ns"]
        entry["spans"][record["chip_id"]] = (record["start_ns"], record["end_ns"])

    ordered = list(per_program.items())
    counts = sorted(len(entry["by_chip"]) for _, entry in ordered)
    mesh_size = mesh_device.get_num_devices()
    logger.info(
        f"DIAG {case.label}: {len(ordered)} programs, {len(records)} records, chip records per program "
        f"min={counts[0]} max={counts[-1]} short={sum(1 for c in counts if c < mesh_size)} (mesh={mesh_size})"
    )

    if len(ordered) % _ITERS:
        logger.warning(f"DIAG {case.label}: {len(ordered)} programs is not a multiple of {_ITERS} iters")
        return
    per_iter = len(ordered) // _ITERS

    for iteration in range(_ITERS):
        chunk = ordered[iteration * per_iter : (iteration + 1) * per_iter]
        total = sum(max(entry["by_chip"].values()) for _, entry in chunk)
        dispatch = [max(entry["by_chip"].values()) for _, entry in chunk if _DISPATCH_KERNEL in entry["kernels"]]
        dispatch_ns = dispatch[0] if dispatch else 0.0
        logger.info(
            f"DIAG {case.label} iter {iteration}: total {total:,.0f} ns | dispatch {dispatch_ns:,.0f} ns | "
            f"total without dispatch {total - dispatch_ns:,.0f} ns"
        )

    # One chip's timeline for the first iteration. The gap column is what shows a record closing
    # before its work is done: the time leaves the record and turns up in the gap behind it.
    ref_chip = min(chip for _, entry in ordered for chip in entry["spans"])
    timeline = [(entry["spans"][ref_chip], entry["kernels"]) for _, entry in ordered[:per_iter]]
    origin = min(start for (start, _), _ in timeline)
    logger.info(f"DIAG {case.label} chip-{ref_chip} timeline (offset, duration, gap to next):")
    for idx, ((start, end), kernels) in enumerate(timeline):
        gap = timeline[idx + 1][0][0] - end if idx + 1 < len(timeline) else 0
        logger.info(
            f"DIAG   [{idx:>2}] t+{start - origin:>12,.0f} dur {end - start:>10,.0f} gap {gap:>12,.0f}  "
            f"{kernels.split(',')[0]}"
        )

    spread = [max(entry["by_chip"].values()) - statistics.median(entry["by_chip"].values()) for _, entry in ordered]
    logger.info(
        f"DIAG {case.label} per-program chip skew (max - median), worst {max(spread):,.0f} ns, "
        f"summed {sum(spread):,.0f} ns -- how much of this metric is straggler chips"
    )
