# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reconcile the saved plan's traffic assumptions with measured B16/32K.

No hardware or network access. Byte counts are useful-traffic estimates, not
physical bus counters; neither the memory floor nor P2 scenario is a forecast.
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run():
    paths = {
        "operators": HERE.parent / "operator-scope-v1/analysis.json",
        "candidate": HERE / "candidate-sweep.json",
        "controls": HERE / "control-drift.json",
        "plan": HERE / "spec-snapshot.json",
    }
    sources = {name: json.loads(path.read_text()) for name, path in paths.items()}
    sweep, plan = sources["candidate"], sources["plan"]
    assert sweep["state"] == "completed" and sweep["replicas"] == 1 and sweep["chips"] == 4
    cell = next(c for c in sweep["cells"] if c["batch_per_replica"] == 16 and c["input_tokens"] == 32768)
    assert cell["status"] == "completed" and len(cell["samples"]) == 3
    assert set(sweep["precision"]["weight_groups"].values()) == {"bfloat8_b"}
    assert sweep["precision"]["kv_cache_dtype"] == "bfloat8_b"
    assert sweep["precision"]["recurrent_dtype"] == "float32"
    control = next(
        c for c in sources["controls"]["cells"] if c["batch_per_replica"] == 16 and c["input_tokens"] == 32768
    )
    assert control["same_output_hash_as_native"] and abs(control["decode_uplift_percent"]) < 3

    bandwidth = 512e9
    weights = sources["operators"]["matmul_totals"]["encoded_weight_bytes"]
    # Sixteen users; 48 layers; 12 local value heads; 128x128 FP32 state,
    # read once and written once. Retain FP32 in both the plan and this run.
    state_bytes = 16 * 48 * 12 * 128 * 128 * 4 * 2
    scenarios = []
    for dtype, bytes_per_tile in (("bfloat4_b", 576), ("bfloat8_b", 1088)):
        assert weights % 1088 == 0
        encoded_weights = weights // 1088 * bytes_per_tile
        for context in (8192, 32768):
            # Sixteen full-attention layers, K and V, one local KV head,
            # 256 elements/head/token. BFP8 has 1088 bytes per 32x32 tile.
            kv_bytes = 16 * context * 16 * 2 * 256 // 1024 * 1088
            useful_bytes = encoded_weights + state_bytes + kv_bytes
            floor_ms = useful_bytes / bandwidth * 1000
            p2_ms = floor_ms / plan["p2_bandwidth_fraction"] + plan["p2_fixed_overhead_ms"]
            scenarios.append(
                dict(
                    weights=dtype,
                    context=context,
                    weights_bytes=encoded_weights,
                    kv_bytes=kv_bytes,
                    recurrent_bytes=state_bytes,
                    useful_bytes=useful_bytes,
                    memory_only_floor_ms=floor_ms,
                    memory_only_tsu_ceiling=1000 / floor_ms,
                    assumed_p2_step_ms=p2_ms,
                    assumed_p2_tsu=1000 / p2_ms,
                )
            )
    current = scenarios[-1]
    measured_ms = cell["summary"]["tpot_ms"]
    result = dict(
        scope="One TP4 replica, batch 16; native decode only, no HTTP/prefill or speculative decoding",
        source_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
        plan_snapshot=plan,
        physical_dram_counters=False,
        combined_compute_memory_roofline_calibrated=False,
        assumptions=dict(
            per_chip_peak_gbs=512,
            weights_read_once=True,
            kv_read_once=True,
            recurrent_state_read_write_once=True,
            current_projection_padding_retained_in_all_scenarios=True,
            excluded=["extra physical traffic", "activation traffic", "compute", "collectives", "program dependencies"],
        ),
        scenarios=scenarios,
        measured=dict(
            step_ms=measured_ms,
            tsu=cell["summary"]["tokens_per_second_per_user"],
            useful_byte_gbs=current["useful_bytes"] / measured_ms / 1e6,
            useful_byte_fraction_of_peak=current["memory_only_floor_ms"] / measured_ms,
            p2_scenario_speedup_required=measured_ms / current["assumed_p2_step_ms"],
            p2_scenario_excess_ms=measured_ms - current["assumed_p2_step_ms"],
            before_after_control_drift_percent=control["decode_uplift_percent"],
            gpqa_qualified=False,
        ),
        native_30_tsu=dict(
            target_step_ms=1000 / 30,
            further_ms_to_remove=measured_ms - 1000 / 30,
            minimum_useful_byte_bandwidth_fraction=current["memory_only_floor_ms"] / (1000 / 30),
            other_work_budget_at_85_percent_ms=1000 / 30 - current["memory_only_floor_ms"] / 0.85,
        ),
        caveats=[
            "Current 128-token timing scans a growing context; modeled traffic uses the initial 32768 tokens.",
            "P2 overhead and bandwidth are assumptions, not established current performance.",
            "Operator inventory predates current fusion; profiled family sums are not additive wall time.",
            "Matched BFP4 scenarios are hypothetical byte substitutions, not qualified precision changes.",
        ],
    )
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"measured": result["measured"], "current_traffic_model": current}, indent=2))


if __name__ == "__main__":
    run()
