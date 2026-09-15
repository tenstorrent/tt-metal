# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Residual-error diagnostics; standalone matmul is a proxy, not an SDPA score dump."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import ttnn

from qualify_full import QUAL, module, verify_diagnostic_source

ANALYZE = module("stress", "experiments/sdpa-l2/stress-analysis/analyze.py")


def metrics(actual, gold):
    result = ANALYZE.statistics(actual, gold)
    a, g = actual.double(), gold.double()
    gain = (a * g).sum() / g.square().sum()
    row_gain = (a * g).sum(-1) / g.square().sum(-1)
    result.update(
        gain=float(gain),
        gain_corrected_l2_pct=float(100 * (a / gain - g).norm() / g.norm()),
        row_gain_min=float(row_gain.min()),
        row_gain_max=float(row_gain.max()),
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--new-worst-only", action="store_true")
    args = parser.parse_args()
    os.environ["TT_SDPA_ACCURACY_DIAG"] = "3"
    verify_diagnostic_source()
    torch.set_num_threads(16)
    records = [json.loads(x) for x in (args.output.parent / "fp32-sub-accurate.jsonl").read_text().splitlines()]
    if args.new_worst_only:
        records = []
    # Include the worst newly failing head at each length, not only the original
    # HiFi2-directed examples. Qualification is completed before this diagnostic.
    qualified = [json.loads(x) for x in (args.output.parent / "qualification.jsonl").read_text().splitlines()]
    for length in (32768, 262144):
        failed = [r for r in qualified if r["status"] == "FAIL" and r["kv_len"] == length]
        if failed:
            worst = max(failed, key=lambda r: max(h["row_max_pct"] for h in r["per_head"]))
            head = max(worst["per_head"], key=lambda h: h["row_max_pct"])["head"]
            records.append(dict(worst, head=head))
    old = {
        r["id"]: r
        for r in map(
            json.loads, (args.output.parents[1] / "qualification-v1/accepted-results.jsonl").read_text().splitlines()
        )
        if r["mode"] == "accurate"
    }
    device = ttnn.open_device(device_id=0)
    device.enable_program_cache()
    try:
        with args.output.open("x") as log:
            for selected in records:
                c, head = old[selected["id"]], selected["head"]
                print("RESIDUAL_START", c["id"], head, flush=True)
                q, k, v = QUAL.inputs(c)
                pos = QUAL.positions(c["q_len"], c["seed"])
                actual, full_hash = ANALYZE.device_original_q(device, q, k, v, pos)
                assert full_hash == selected["full_output_sha256"]
                actual = actual[0, head].double()
                ones, _ = ANALYZE.device_original_q(device, q, k, torch.ones_like(v), pos)
                ones = ones[0, head].double()
                reverse, _ = ANALYZE.device_original_q(device, q, k.flip(-2), v.flip(-2), pos)
                reverse = reverse[0, head].double()
                qs, ks, vs = q[0, head, pos].contiguous(), k[0, head].contiguous(), v[0, head].double()
                del q, k, v
                # Standalone HiFi4 QK tests a different kernel schedule; attribution is
                # supporting evidence only unless it reproduces the SDPA error vector.
                tq = ttnn.from_torch(
                    qs[None, None],
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                tk = ttnn.from_torch(
                    ks.T.contiguous()[None, None],
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ts = ttnn.matmul(
                    tq,
                    tk,
                    dtype=ttnn.float32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        packer_l1_acc=False,
                    ),
                )
                scores = ttnn.to_torch(ts)[0, 0].double()
                for tensor in (tq, tk, ts):
                    ttnn.deallocate(tensor)
                results = {name: [] for name in ("gold", "p10_rne", "p10_trunc", "qk_device", "qk_device_p10")}
                score_error_sq = score_gold_sq = score_max = 0.0
                for start in range(0, len(pos), 32):
                    exact = qs[start : start + 32].double() @ ks.double().T
                    measured = scores[start : start + 32]
                    delta = measured - exact
                    score_error_sq += float(delta.square().sum())
                    score_gold_sq += float(exact.square().sum())
                    score_max = max(score_max, float(delta.abs().max()))
                    p = (exact / math.sqrt(128)).softmax(-1)
                    pm = (measured / math.sqrt(128)).softmax(-1)
                    results["gold"].append(p @ vs)
                    results["qk_device"].append(pm @ vs)
                    for name, weights, rounding in (
                        ("p10_rne", p, True),
                        ("p10_trunc", p, False),
                        ("qk_device_p10", pm, False),
                    ):
                        weights = ANALYZE.quantize(weights, 10, rounding=rounding)
                        results[name].append((weights @ vs) / weights.sum(-1, keepdim=True))
                results = {name: torch.cat(parts) for name, parts in results.items()}
                gold = results.pop("gold")
                error = actual - gold
                models = {}
                for name, value in results.items():
                    e = value.bfloat16().double() - gold
                    models[name] = dict(
                        metrics(value.bfloat16(), gold),
                        error_cosine=float((e * error).sum() / (e.norm() * error.norm())),
                        device_vs_model_l2_pct=float(100 * (actual - value.bfloat16()).norm() / gold.norm()),
                    )
                row = dict(
                    id=c["id"],
                    head=head,
                    distribution=c["distribution"],
                    kv_len=c["kv_len"],
                    device=metrics(actual, gold),
                    models=models,
                    reversed_keys=metrics(reverse, gold),
                    permutation_difference_l2_pct=float(100 * (actual - reverse).norm() / gold.norm()),
                    constant_v_max_abs=float((ones - 1).abs().max()),
                    constant_v_nonexact=int((ones != 1).sum()),
                    standalone_qk_l2_pct=100 * math.sqrt(score_error_sq / score_gold_sq),
                    standalone_qk_max_abs=score_max,
                )
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                print("RESIDUAL_RESULT", json.dumps(row), flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
