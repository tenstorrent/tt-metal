#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Compare EAGLE3 acceptance rates: FC only at depth 0 (paper) vs FC at every depth (old)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.config import EagleConfig
from models.demos.speculative_deepseek_r1_broad.eagle_engine import EagleEngine
from models.demos.speculative_deepseek_r1_broad.models_base import DeepSeekBaseAdapter
from models.demos.speculative_deepseek_r1_broad.models_draft import Eagle3HiddenStateDraftAdapter


BASE_PRESETS = {
    "distill_llama_8b": "deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B",
    "distill_qwen_1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}
DRAFT_PRESETS = {
    "eagle3_8b": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    "eagle3_r1": "eigen-ai-labs/deepseek-v3.1-eagle3",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Compare FC-at-depth-0 (paper) vs FC-every-depth (old)")
    p.add_argument("--prompt", type=str, default="The capital of France is")
    p.add_argument("--base-model-preset", default="distill_qwen_1_5b", choices=sorted(BASE_PRESETS))
    p.add_argument("--draft-model-preset", default="eagle3_8b", choices=sorted(DRAFT_PRESETS))
    p.add_argument("--base-impl", default="reference", choices=["reference", "adapter", "accelerate"])
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    base_id = BASE_PRESETS[args.base_model_preset]
    draft_id = DRAFT_PRESETS[args.draft_model_preset]

    cfg = EagleConfig(
        top_k=args.top_k,
        depth=args.depth,
        num_steps=args.num_steps,
        max_paths=16,
        verification_mode="batched_single_pass",
        verification_acceptance="argmax",
        random_seed=args.random_seed,
    )

    print("Loading base and draft (once)...")
    base = DeepSeekBaseAdapter(
        base_id,
        device=args.device,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        base_impl=args.base_impl,
    )
    draft = Eagle3HiddenStateDraftAdapter(draft_id, device=args.device, torch_dtype=args.dtype)
    engine = EagleEngine(base=base, draft=draft, cfg=cfg)

    results = {}
    for name, use_fc_every in [("FC only at depth 0 (paper)", False), ("FC at every depth (old)", True)]:
        draft.use_fc_at_every_depth = use_fc_every
        result = engine.generate(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
        results[name] = result.stats
        print(
            f"  {name}: accepted_tokens_percentage={result.stats.accepted_tokens_percentage:.4f} "
            f"selected_path_acceptance_rate={result.stats.selected_path_acceptance_rate:.4f}"
        )

    print()
    print("=" * 60)
    print("EAGLE3 FC mode comparison")
    print("=" * 60)
    print(f"Prompt: {args.prompt!r}")
    print(f"Base: {base_id}, Draft: {draft_id}")
    print(f"depth={args.depth} top_k={args.top_k} max_new_tokens={args.max_new_tokens} seed={args.random_seed}")
    print()
    paper_stats = results["FC only at depth 0 (paper)"]
    old_stats = results["FC at every depth (old)"]
    print(f"{'Metric':<35} {'Paper (FC once)':<18} {'Old (FC every depth)':<18} {'Better':<10}")
    print("-" * 85)
    for label, key in [
        ("Accepted tokens percentage", "accepted_tokens_percentage"),
        ("Selected-path accept / depth", "selected_path_acceptance_rate"),
        ("Acceptance vs round capacity", "acceptance_rate"),
        ("First-token match rate", "first_token_match_rate"),
        ("Rounds-with-any-accept rate", "any_accept_rate"),
    ]:
        a = getattr(paper_stats, key)
        b = getattr(old_stats, key)
        better = "Paper" if a >= b else "Old"
        print(f"{label:<35} {a:<18.4f} {b:<18.4f} {better:<10}")
    print("-" * 85)
    if paper_stats.selected_path_acceptance_rate >= old_stats.selected_path_acceptance_rate:
        print("Conclusion: FC only at depth 0 (paper) is better or equal on acceptance rate (accepted/depth).")
    else:
        print("Conclusion: FC at every depth (old) is better on acceptance rate for this run.")
    print("=" * 60)


if __name__ == "__main__":
    main()
