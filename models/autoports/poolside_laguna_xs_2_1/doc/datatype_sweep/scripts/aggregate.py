# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Aggregate datatype-sweep candidate results into sweep_results.json + sweep_results.csv.

Reads the per-candidate JSONs written by sweep_run.py (the fresh-build, clean-perf runs), applies
the accuracy gate, marks the selected config, and emits the two sweep tables with every field the
goal requires: config id, dtype policy, compute-fidelity policy, top-1/5/100, TTFT, trace-verified
teacher-forcing decode t/s/u, token-out decode t/s/u, measurement regime, command, hardware, mesh,
pass/fail.

  python aggregate.py <candidate_results_dir> <out_dir> <selected_id> [--order id1,id2,...]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TOP1_GATE = 0.90
TOP5_GATE = 0.98
TOP100_EXPECT = 1.00

HARDWARE = "4x Blackhole p300c (P150x4), KMD 2.10.0, fw 19.11.0"
MESH = "1x4 mesh (MeshShape(1,4)), FABRIC_1D_RING, cluster_axis 1 = TP/EP, 2 ethernet links"
BRANCH = "agentic-research/hous/laguna-xs-2.1"
CMD_TEMPLATE = (
    "cd /tmp && TT_METAL_HOME=<installed-tree> PYTHONPATH=<repo> python "
    "doc/datatype_sweep/scripts/sweep_run.py --spec doc/datatype_sweep/configs/spec_<ID>.json "
    "--out-dir doc/datatype_sweep/candidate_results --perf-tokenout"
)
REGIME = (
    "accuracy = official run_teacher_forcing scoring (TokenAccuracy) vs AIME24[0] chat-template "
    "reference, 192-token prompt + 100 forced tokens, top-100; teacher_decode_tsu = trace-verified "
    "teacher-forcing decode t/s/u (traced decode path, enable_trace=True, wall incl. per-token host "
    "next_input/readback) = the RANKING metric; token_out_decode_tsu = warmed no-readback traced "
    "token-out decode (prompt128/gen128, capture->warm8->measure non-blocking+sync) = post-selection "
    "serving metric. batch-1."
)

DESC = {
    "C0_baseline": "baseline optimized-full-model policy (BFP8 attn/dense/shared, BFP4 experts, BFP8 KV, BF16 CCL, BFP8 LM head; LoFi proj / HiFi2 gate+router)",
    "C1_kv_bf16": "KV-cache dtype BFP8->BF16 (higher precision control / yes-no switch)",
    "C2_attn_bfp4": "attention QKV+O weights BFP8->BFP4 (+LoFi)",
    "C3_shared_bfp4": "shared-expert gate/up/down weights BFP8->BFP4 (+LoFi)",
    "C4_dense_bfp4": "dense-MLP (layer 0) weights BFP8->BFP4 (+LoFi)",
    "C5_attn_shared_bfp4": "attention QKV+O AND shared-expert weights BFP4 (+LoFi) [combined]",
    "C6_attn_bfp8_hifi2": "attention BFP8 weights, LoFi->HiFi2 fidelity (dominant BFP8 projection LoFi-vs-HiFi2)",
    "C7_moe_bfp4_hifi2": "routed-expert BFP4 weights, LoFi->HiFi2 fidelity (dominant BFP4 group LoFi-vs-HiFi2)",
    "C8_ccl_bfp8": "all_reduce CCL payload BF16->BFP8 (yes-no switch)",
    "C9_allproj_hifi2": "all projection groups LoFi->HiFi2 (higher-fidelity control)",
    "C10_lmhead_bfp4": "column-sharded LM-head weight BFP8->BFP4 (largest terminal matmul)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_dir")
    ap.add_argument("out_dir")
    ap.add_argument("selected_id")
    ap.add_argument("--order", default=None)
    args = ap.parse_args()

    cand_dir = Path(args.cand_dir)
    files = {p.stem: p for p in cand_dir.glob("*.json") if not p.stem.startswith("_run")}
    order = args.order.split(",") if args.order else sorted(files)

    configs = []
    for cid in order:
        if cid not in files:
            print("WARN missing", cid)
            continue
        r = json.loads(files[cid].read_text())
        if r.get("status") != "ok":
            configs.append({"id": cid, "status": r.get("status"), "error": r.get("error")})
            continue
        pol = r["policy"]
        top1, top5, top100 = r["top1"], r["top5"], r["top100"]
        passes = (top1 >= TOP1_GATE) and (top5 >= TOP5_GATE)
        configs.append(
            {
                "id": cid,
                "description": DESC.get(cid, ""),
                "status": "ok",
                "dtype_policy": {
                    k: pol[k]
                    for k in (
                        "attn_qkv",
                        "attn_o",
                        "attn_gate",
                        "dense_ff13",
                        "dense_ff2",
                        "moe_ff13",
                        "moe_ff2",
                        "shared_ff13",
                        "shared_ff2",
                        "router",
                        "qk_norm",
                        "lm_head",
                        "kv_cache",
                        "ccl",
                        "activation",
                        "logits",
                    )
                },
                "compute_fidelity_policy": {
                    k: pol[k]
                    for k in (
                        "fid_attn_qkv",
                        "fid_attn_o",
                        "fid_attn_gate",
                        "fid_dense",
                        "fid_shared",
                        "fid_router",
                        "fid_moe",
                    )
                },
                "built_readback": r.get("built_readback"),
                "top1": top1,
                "top5": top5,
                "top100": top100,
                "matches_top1": r["matches_top1"],
                "matches_top5": r["matches_top5"],
                "matches_top100": r["matches_top100"],
                "total": r["total"],
                "k": r["k"],
                "ttft_ms": r.get("ttft_ms"),
                "teacher_decode_tsu": r.get("teacher_decode_tsu"),
                "teacher_decode_ms_tok": r.get("teacher_decode_ms_tok"),
                "token_out_decode_tsu": r.get("token_out_decode_tsu"),
                "token_out_decode_ms_tok": r.get("token_out_decode_ms_tok"),
                "logits_only_decode_tsu": r.get("logits_only_decode_tsu"),
                "measurement_regime": REGIME,
                "command": CMD_TEMPLATE.replace("<ID>", cid),
                "hardware": HARDWARE,
                "mesh": MESH,
                "branch": BRANCH,
                "passes_gate": passes,
                "selected": (cid == args.selected_id),
            }
        )

    out = {
        "acceptance_thresholds": {"top1_min": TOP1_GATE, "top5_min": TOP5_GATE, "top100_expected": TOP100_EXPECT},
        "ranking_metric": "trace-verified teacher-forcing decode t/s/u (teacher_decode_tsu)",
        "selected_id": args.selected_id,
        "hardware": HARDWARE,
        "mesh": MESH,
        "branch": BRANCH,
        "reference": "readiness_aime24_chat.refpt (AIME24[0] chat-template, 192 prompt + 100 gen, top-100)",
        "configs": configs,
    }
    Path(args.out_dir, "sweep_results.json").write_text(json.dumps(out, indent=2))

    # CSV
    cols = [
        "id",
        "passes_gate",
        "selected",
        "top1",
        "top5",
        "top100",
        "ttft_ms",
        "teacher_decode_tsu",
        "teacher_decode_ms_tok",
        "token_out_decode_tsu",
        "attn_qkv",
        "attn_o",
        "shared_ff13",
        "shared_ff2",
        "dense_ff13",
        "moe_ff13",
        "moe_ff2",
        "lm_head",
        "kv_cache",
        "ccl",
        "fid_attn_qkv",
        "fid_shared",
        "fid_moe",
        "description",
    ]
    with open(Path(args.out_dir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in configs:
            if c.get("status") != "ok":
                w.writerow([c["id"], "ERROR"] + [""] * (len(cols) - 2))
                continue
            dp = c["dtype_policy"]
            fp = c["compute_fidelity_policy"]
            w.writerow(
                [
                    c["id"],
                    c["passes_gate"],
                    c["selected"],
                    c["top1"],
                    c["top5"],
                    c["top100"],
                    c["ttft_ms"],
                    c["teacher_decode_tsu"],
                    c["teacher_decode_ms_tok"],
                    c["token_out_decode_tsu"],
                    dp["attn_qkv"],
                    dp["attn_o"],
                    dp["shared_ff13"],
                    dp["shared_ff2"],
                    dp["dense_ff13"],
                    dp["moe_ff13"],
                    dp["moe_ff2"],
                    dp["lm_head"],
                    dp["kv_cache"],
                    dp["ccl"],
                    fp["fid_attn_qkv"],
                    fp["fid_shared"],
                    fp["fid_moe"],
                    c["description"],
                ]
            )
    print("wrote sweep_results.json + sweep_results.csv;", len(configs), "configs; selected", args.selected_id)


if __name__ == "__main__":
    main()
