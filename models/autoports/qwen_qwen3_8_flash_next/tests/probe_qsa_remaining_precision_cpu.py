# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only exact-input analysis of captured QSA pool, norm, RoPE, and scores.

This loads saved device captures without importing TTNN or touching hardware.
Hybrid key controls retain HF keys outside the sampled critical chunks and
must not be interpreted as full-context device acceptance.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa import load_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_projection_precision import compact_metrics


def selection(scores, block):
    top = scores.topk(513)
    return {
        "self_score": float(scores[block]),
        "strictly_higher_blocks": int((scores > scores[block]).sum()),
        "self_selected": bool((top.indices[:512] == block).any()),
        "score_512": float(top.values[511]),
        "score_513": float(top.values[512]),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/home/ttuser/qwen38-flash"))
    args = parser.parse_args()
    torch.set_num_threads(8)

    def load(name):
        return torch.load(args.root / name, weights_only=True, mmap=True, map_location="cpu")

    fixture = load("qsa_context_window_222272.pt")
    captured = load("qsa_indexer_precision_222275.pt")
    window = load("qsa_context_tt_index_prefix_222272.pt")
    projected = load("qsa_projection_sfpu.pt")
    source = load("qsa_cpu_prefill_262144.pt")
    _, hf, rope, _, provenance = load_reference(args.snapshot)
    assert provenance == fixture["provenance"] and source["key"] == fixture["source_key"]
    from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

    stages = {name: [] for name in ("pool", "norm", "rope", "exact_norm_rope")}
    for number, start in enumerate(captured["chunk_starts"]):
        hidden = source["hidden"][:, start : start + 32]
        previous_raw = hf.indexer.index_qk_proj(source["hidden"][:, start - 3 : start])[..., 512:]
        rows = slice(number * 32, (number + 1) * 32)
        history = torch.cat((previous_raw, captured["raw"][:, 0, rows]), 1)
        stages["pool"].append(history.float().unfold(1, 4, 1).mean(-1).bfloat16().unsqueeze(1))
        stages["norm"].append(hf.indexer.k_layernorm(captured["pool"][:, :, rows]))
        positions = torch.arange(start, start + 32).div(4, rounding_mode="floor").reshape(1, -1) * 4
        cos, sin = rope(hidden, positions)
        stages["rope"].append(apply_rotary_pos_emb(captured["norm"][:, :, rows], cos=cos, sin=sin))
        raw = hf.indexer.index_qk_proj(hidden)[..., 512:]
        exact_pool = torch.cat((previous_raw, raw), 1).float().unfold(1, 4, 1).mean(-1).bfloat16()
        exact_norm = hf.indexer.k_layernorm(exact_pool).unsqueeze(1)
        stages["exact_norm_rope"].append(apply_rotary_pos_emb(exact_norm, cos=cos, sin=sin))
    expected = {key: torch.cat(value, 2) for key, value in stages.items()}
    report = {
        "scope": "CPU-only analysis of saved TT exact-input controls; not device acceptance",
        "provenance": provenance,
        "sampled_projection_chunks": len(captured["chunk_starts"]),
        "exact_input_stage_controls": {key: compact_metrics(value, captured[key]) for key, value in expected.items()},
        "precise_rope_on_same_native_norm_input": compact_metrics(expected["rope"], captured["precise_rope"]),
        "exact_raw_vs_exact_pool_rope_equal": torch.equal(captured["exact_raw_rope"], captured["exact_pool_rope"]),
        "exact_pool_vs_exact_norm_rope_equal": torch.equal(captured["exact_pool_rope"], captured["exact_norm_rope"]),
        "score_controls": [],
    }
    keys = torch.cat((window["tt_compressed_prefix"], window["compressed_window"]), 2)[0, 0].float()
    for row in range(32):
        position = fixture["prefix"] + row
        complete = (position + 1) // 4
        query = window["index_query"][0, :, row].float()
        cpu_scores = (query @ keys[:complete].T).relu().sum(0) / math.sqrt(128)
        tt_scores = window["indexer_scores"][row, :complete]
        cpu_ids = set(cpu_scores.topk(512).indices.tolist())
        tt_ids = set(tt_scores.topk(512).indices.tolist())
        item = {
            "position": position,
            "same_operands_cpu_vs_TT_scores": compact_metrics(cpu_scores[None], tt_scores[None]),
            "changed_top512_blocks": sorted(cpu_ids ^ tt_ids),
            "score_error_rms": float((cpu_scores - tt_scores).square().mean().sqrt()),
        }
        if position % 4 == 3:
            item["cpu_self"] = selection(cpu_scores, position // 4)
            item["TT_self"] = selection(tt_scores, position // 4)
        report["score_controls"].append(item)

    assert projected["chunk_starts"] == [fixture["prefix"]]
    q = hf.indexer.q_layernorm(projected["sfpu_tree"][..., :512].reshape(1, 32, 4, 128))
    cos, sin = rope(fixture["hidden"], torch.arange(fixture["prefix"], fixture["prefix"] + 32).reshape(1, -1))
    q = apply_rotary_pos_emb(q.transpose(1, 2), cos=cos, sin=sin)
    report["SFPU_projection_then_CPU_HF_query"] = compact_metrics(fixture["projections"]["index_query"], q)
    report["hybrid_key_controls_with_SFPU_projection_CPU_query"] = {}
    for policy, hybrid in captured["variants"].items():
        scores = (q[0, :, 3].float() @ hybrid[0].float().T).relu().sum(0) / math.sqrt(128)
        report["hybrid_key_controls_with_SFPU_projection_CPU_query"][policy] = selection(scores, 55568)
    report["limitations"] = [
        "Score controls isolate matmul/relu/head-sum/scale jointly; no saved pre-reduction head scores exist.",
        "SFPU projection was measured on only one chunk; CPU query normalization/RoPE are not device predictions.",
        "Hybrid key controls retain exact HF keys outside the 242 sampled chunks.",
    ]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("provenance", "score_controls")}, indent=2
        )
    )
    print(
        json.dumps(
            {
                "score_rows_with_changed_top512": [
                    v["position"] for v in report["score_controls"] if v["changed_top512_blocks"]
                ],
                "score_maximum_absolute_error": max(
                    v["same_operands_cpu_vs_TT_scores"]["maximum_absolute_error"] for v in report["score_controls"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
