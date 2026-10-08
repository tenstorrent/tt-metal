# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device time of one TtMoe forward per MoE block (tt/moe/moe_block.py), per model, on the LoudBox and the Galaxy.

Seeded random weights at the 640-token-per-chip chunk the production prefill feeds (LoudBox 2 x 4: 1280 tokens; Galaxy
8 x 4: 5120). Per case it logs, and writes to ``generated/moe_block_perf/<case>.json``:

  * device ns: the sum of every program's device duration over the forward (the critical path across chips per
    program, the real-time profiler; same metric as test_kimi_moe_perf), and that sum per op (the program's kernel
    directory), so the two blocks' data movement can be compared op by op;
  * host ms: wall clock of the forward + a device sync, median of a few passes (includes the gaps between programs the
    device sum leaves out, and host dispatch).

Kimi-K3 runs 512 routed experts on the LoudBox (the flat expert, which the all-gather block needs, holds <= 64 per chip).

Gating: ``EXPECTED_NS`` holds LoudBox baselines (one BH LoudBox, the values this file was calibrated with); a case
without one only reports. ``MOE_BLOCK_PERF_REPORT_ONLY=1`` reports every case (calibration).
"""

import json
import os
import statistics
import time
from pathlib import Path

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.pcc.test_moe_block import MESHES, MODELS, MOE_BLOCKS, model_case
from models.demos.deepseek_v3_d_p.tests.pcc.test_ttnn_moe import run_model
from models.demos.deepseek_v3_d_p.tt.moe.tt_flat_routed_expert import resolve_routed_expert_impl
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS_PER_CHIP
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged, require_realtime_profiler

HOST_PASSES = 5
RECORD_TIMEOUT_S = 5.0
MARGIN = 0.05
OUT_DIR = Path(os.environ.get("MOE_BLOCK_PERF_DIR", "generated/moe_block_perf"))

# (model, moe_block, mesh id) -> device ns (sum over programs). LoudBox only: the Galaxy cases report until calibrated.
# One BH LoudBox (8 x p150b, fabric 2D, 2 links), 2026-10-08, median of 3 runs (device spread <= 1.0%).
EXPECTED_NS = {
    ("k2_7", "dispatch_combine", "loudbox-2x4"): 5_077_000,
    ("k2_7", "all_gather", "loudbox-2x4"): 4_591_000,
    ("glm_5_3", "dispatch_combine", "loudbox-2x4"): 3_439_000,
    ("glm_5_3", "all_gather", "loudbox-2x4"): 3_225_000,
    ("k3", "dispatch_combine", "loudbox-2x4"): 6_394_000,
    ("k3", "all_gather", "loudbox-2x4"): 5_452_000,
}


def op_of(kernel_sources):
    """A program's op: the directory under ttnn/operations (or tt_metal) of its first kernel, e.g.
    "experimental/deepseek_prefill/moe_ag" or "ccl/reduce_scatter"."""
    for src in sorted(kernel_sources):
        path = src.replace("\\", "/")
        if "/operations/" in path:
            parts = path.split("/operations/", 1)[1].split("/")
            stop = parts.index("device") if "device" in parts else len(parts) - 1
            return "/".join(parts[:stop]) or parts[0]
    return "other"


@pytest.mark.skipif(not is_blackhole(), reason="the all-gather MoE block needs the flat routed expert (Blackhole)")
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params, num_links", MESHES, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("moe_block", MOE_BLOCKS)
@pytest.mark.parametrize("model", MODELS)
def test_moe_block_perf(model, moe_block, mesh_device, device_params, num_links, request):
    from models.demos.deepseek_v3_d_p.tests.conftest import TEST_VARIANTS, _resolve_config_only

    require_realtime_profiler("the MoE block perf comparison")
    rows, cols = tuple(mesh_device.shape)
    mesh_id = "loudbox-2x4" if (rows, cols) == (2, 4) else f"{rows}x{cols}"
    variant_name, cfg, experts, capacity, extra = model_case(model, mesh_device.get_num_devices())
    extra = {k: v for k, v in extra.items() if not k.endswith("_pcc")}
    result = {}

    def measure(forward):
        warm = forward()  # JIT compile + program cache fill: neither is device time
        ttnn.synchronize_device(mesh_device)
        del warm
        host = []
        for _ in range(HOST_PASSES):
            t0 = time.perf_counter()
            out = forward()
            ttnn.synchronize_device(mesh_device)
            host.append(time.perf_counter() - t0)
            del out
        out, records = profile_realtime_program_merged(mesh_device, forward, record_timeout_seconds=RECORD_TIMEOUT_S)
        per_op = {}
        for entry in records.values():
            op = op_of(entry["kernel_sources"])
            n, ns = per_op.get(op, (0, 0.0))
            per_op[op] = (n + 1, ns + entry["duration_ns"])
        result.update(
            device_ns=sum(e["duration_ns"] for e in records.values()),
            programs=len(records),
            host_ms=statistics.median(host) * 1e3,
            host_ms_all=[h * 1e3 for h in host],
            per_op={
                k: {"programs": n, "device_ns": ns} for k, (n, ns) in sorted(per_op.items(), key=lambda kv: -kv[1][1])
            },
        )
        return out

    variant = TEST_VARIANTS[variant_name]
    run_model(
        variant,
        _resolve_config_only(variant.name),
        mesh_device,
        device_params,
        PREFILL_CHUNK_TOKENS_PER_CHIP,
        cfg.EMB_SIZE,
        cfg.MOE_INTERMEDIATE_SIZE,
        experts,
        cfg.NUM_EXPERTS_PER_TOKEN,
        capacity,
        False,  # run_pcc_check: test_moe_block.py owns correctness
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        GateComputeMode.DEVICE_FP32,
        request,
        routed_expert_impl=resolve_routed_expert_impl(cfg),
        moe_block=moe_block,
        measure=measure,
        **extra,
    )
    assert result, "measure() never ran: the forward was not profiled"

    case = f"{model}-{moe_block}-{mesh_id}"
    result.update(
        case=case,
        model=model,
        moe_block=moe_block,
        mesh=[rows, cols],
        routed_experts=experts,
        tokens_per_chip=PREFILL_CHUNK_TOKENS_PER_CHIP,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{case}.json").write_text(json.dumps(result, indent=1))
    lines = [
        f"{op:<55s} {v['programs']:>4d} prog {v['device_ns'] / 1e3:>10.1f} us" for op, v in result["per_op"].items()
    ]
    logger.info(
        f"MOE-BLOCK-PERF {case}: device {result['device_ns'] / 1e6:.3f} ms over {result['programs']} programs, "
        f"host {result['host_ms']:.2f} ms (median of {HOST_PASSES})\n  " + "\n  ".join(lines)
    )

    expected = EXPECTED_NS.get((model, moe_block, mesh_id))
    if expected is None or os.environ.get("MOE_BLOCK_PERF_REPORT_ONLY") == "1":
        logger.warning(f"{case}: report only (no baseline in EXPECTED_NS)")
        return
    lower, upper = expected * (1 - MARGIN), expected * (1 + MARGIN)
    assert lower <= result["device_ns"] <= upper, (
        f"{case}: device {result['device_ns']:,.0f} ns outside [{lower:,.0f}, {upper:,.0f}] "
        f"(expected {expected:,} ns +/- {MARGIN * 100:.0f}%)"
    )
