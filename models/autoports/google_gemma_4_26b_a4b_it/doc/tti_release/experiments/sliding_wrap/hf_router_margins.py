# SPDX-License-Identifier: Apache-2.0
"""How close are this MoE's expert-selection decisions?

Gemma-4-26B-A4B routes top-8 of 128 experts per token per layer. If the 8th and
9th expert scores are routinely separated by less than an accelerator's
arithmetic noise, expert-set divergence from the reference is structural: it is
not a precision bug and no policy removes it. Nothing in the bringup pipeline
measures this, so it has never been quantified.

Runs the HF reference once over a real prompt+continuation and reports, per
layer, the distribution of the top8/top9 router-score gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

HF_MODEL = "google/gemma-4-26B-A4B-it"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=512)
    args = ap.parse_args()

    run = json.loads(args.tt_run.read_text(encoding="utf-8"))
    seq = run["prompt_token_ids"] + run["generated_token_ids"][: args.limit]
    print(f"sequence {len(seq)} tokens", flush=True)

    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=torch.bfloat16, trust_remote_code=True).eval()

    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            # forward returns (router_probabilities, top_k_weights, top_k_index)
            captured[idx] = output[0].detach().float().reshape(-1, output[0].shape[-1]).clone()

        return hook

    handles = []
    layers = model.model.language_model.layers if hasattr(model.model, "language_model") else model.model.layers
    for idx, layer in enumerate(layers):
        router = None
        for name, module in layer.named_modules():
            if module.__class__.__name__ == "Gemma4TextRouter":
                router = module
                break
        if router is not None:
            handles.append(router.register_forward_hook(make_hook(idx)))
    print(f"hooked {len(handles)} routers", flush=True)

    with torch.no_grad():
        model(input_ids=torch.tensor([seq], dtype=torch.long), use_cache=False)
    for handle in handles:
        handle.remove()

    top_k = model.config.text_config.top_k_experts
    per_layer = {}
    per_token_fragile_1pct: list = []
    per_token_fragile_5pct: list = []
    all_gaps: list[float] = []
    all_rel: list[float] = []
    for idx, probs in sorted(captured.items()):
        top9 = torch.topk(probs, k=top_k + 1, dim=-1).values
        gap = (top9[:, top_k - 1] - top9[:, top_k]).cpu()  # 8th minus 9th probability
        eighth = top9[:, top_k - 1].cpu()
        rel = gap / eighth.clamp_min(1e-12)
        per_layer[str(idx)] = {
            "tokens": int(gap.numel()),
            "gap_median": float(gap.median()),
            "gap_p5": float(gap.sort().values[max(0, int(0.05 * gap.numel()) - 1)]),
            "frac_rel_gap_lt_1pct": float((rel < 0.01).float().mean()),
            "frac_rel_gap_lt_5pct": float((rel < 0.05).float().mean()),
        }
        all_gaps.extend(gap.tolist())
        all_rel.extend(rel.tolist())
        per_token_fragile_1pct.append((rel < 0.01).int())
        per_token_fragile_5pct.append((rel < 0.05).int())

    gaps = torch.tensor(all_gaps)
    rels = torch.tensor(all_rel)
    tokens = len(seq)
    n_moe_layers = len(per_layer)
    summary = {
        "sequence_tokens": tokens,
        "moe_layers": n_moe_layers,
        "top_k_experts": top_k,
        "decisions": int(gaps.numel()),
        "relative_gap_percentiles": {
            f"p{p}": float(rels.sort().values[max(0, int(p / 100 * rels.numel()) - 1)]) for p in (1, 5, 10, 25, 50)
        },
        "frac_decisions_rel_gap_lt_1pct": float((rels < 0.01).float().mean()),
        "frac_decisions_rel_gap_lt_5pct": float((rels < 0.05).float().mean()),
        # A token's expert set is fragile if ANY of its layers is a near-tie.
        "expected_fragile_layers_per_token_1pct": float((rels < 0.01).float().mean()) * n_moe_layers,
        "expected_fragile_layers_per_token_5pct": float((rels < 0.05).float().mean()) * n_moe_layers,
        "per_layer": per_layer,
        # Per generated-token counts, so flips can be correlated with routing ties.
        "prompt_len": len(run["prompt_token_ids"]),
        "fragile_layers_per_position_1pct": torch.stack(per_token_fragile_1pct).sum(0).tolist(),
        "fragile_layers_per_position_5pct": torch.stack(per_token_fragile_5pct).sum(0).tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_layer"}, indent=1))


if __name__ == "__main__":
    main()
