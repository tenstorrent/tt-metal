#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.config import EagleConfig
from models.demos.speculative_deepseek_r1_broad.models_base import DeepSeekBaseAdapter
from models.demos.speculative_deepseek_r1_broad.config import PathProposal
from models.demos.speculative_deepseek_r1_broad.models_draft import Eagle3HiddenStateDraftAdapter

BASE_MODEL_PRESETS = {
    "r1_0528": "deepseek-ai/DeepSeek-R1-0528",
    "r1": "deepseek-ai/DeepSeek-R1",
    "distill_llama_8b": "deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B",
    "distill_qwen_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "distill_qwen_1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "tiny_gpt2": "sshleifer/tiny-gpt2",
}


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            prompt_items = data
        else:
            prompt_items = data.get("prompts", [])
        prompts = [str(item["prompt"]) for item in prompt_items if isinstance(item, dict) and "prompt" in item]
        if args.num_prompts is not None:
            prompts = prompts[: args.num_prompts]
        if prompts:
            return prompts

    if args.prompt:
        return [args.prompt]
    if args.prompts:
        return args.prompts
    raise SystemExit("Provide --prompt, positional prompts, or --prompts-file.")


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Yuhuili draft sanity check (CPU)")
    p.add_argument("prompts", nargs="*", help="Prompt text. Ignored when --prompt or --prompts-file is set.")
    p.add_argument("--prompt", type=str, help="Single prompt string.")
    p.add_argument("--prompts-file", type=Path, help='JSON with [{"prompt": ...}] or {"prompts": [...]} format.')
    p.add_argument("--num-prompts", type=int, default=None, help="Max prompts loaded from --prompts-file.")
    p.add_argument("--base-model-id", type=str, default="deepseek-ai/DeepSeek-R1-0528")
    p.add_argument(
        "--base-model-preset",
        type=str,
        default=None,
        choices=sorted(BASE_MODEL_PRESETS),
        help="Optional preset for base model id (overrides --base-model-id).",
    )
    p.add_argument("--draft-model-id", type=str, default="yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B")
    p.add_argument("--sample-steps", type=int, default=32, help="Number of decode steps to sample.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for deterministic prompt/step sampling order.")
    p.add_argument("--draft-top-k", type=int, default=1, help="Draft top-k used for one-step proposal diagnostics.")
    p.add_argument("--base-top-k", type=int, default=5, help="Base top-k used for overlap diagnostics.")
    p.add_argument(
        "--diagnose-overlap",
        action="store_true",
        default=False,
        help="Compute draft/base top-k overlap diagnostics.",
    )
    p.add_argument(
        "--print-step-overlap",
        action="store_true",
        default=False,
        help="Print per-step draft/base top-k token ids and decoded text (for manual check-ups).",
    )
    p.add_argument(
        "--print-max-steps",
        type=int,
        default=8,
        help="Maximum number of per-step overlap lines to print when --print-step-overlap is enabled.",
    )
    p.add_argument(
        "--base-impl",
        type=str,
        default="reference",
        choices=["adapter", "reference", "accelerate"],
        help="Base runtime implementation path.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    return p


def main() -> None:
    args = create_parser().parse_args()
    if args.base_model_preset is not None:
        args.base_model_id = BASE_MODEL_PRESETS[args.base_model_preset]
    random.seed(args.seed)
    prompts = _load_prompts(args)

    try:
        base = DeepSeekBaseAdapter(
            args.base_model_id,
            device=args.device,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            base_impl=args.base_impl,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "FP8 quantization" in msg and args.device == "cpu":
            raise SystemExit(
                f"{msg}\nHint: use a GPU server for DeepSeek-R1-0528, or run CPU smoke checks with "
                "--base-model-preset distill_qwen_1_5b (or tiny_gpt2 for fast debugging)."
            )
        raise
        draft = Eagle3HiddenStateDraftAdapter(args.draft_model_id, device=args.device, torch_dtype=args.dtype)

    cfg = EagleConfig(top_k=max(1, args.draft_top_k), depth=1)
    total = 0
    matches = 0
    overlap_total = 0
    overlap_non_empty = 0
    draft_in_base_topk = 0
    printed_steps = 0

    for prompt in prompts:
        prefix = base.encode_prompt(prompt)
        state = base.forward_prefill(prefix)
        for _ in range(max(0, args.sample_steps)):
            base_next = base.decode_state_next_token(state)
            proposal = draft.propose_paths(prefix, cfg, decode_state=state, base_adapter=base)
            paths = proposal.paths if isinstance(proposal, PathProposal) else proposal
            if paths and paths[0]:
                draft_next = int(paths[0][0])
                total += 1
                if draft_next == base_next:
                    matches += 1
            if args.diagnose_overlap and paths:
                draft_tokens = sorted({int(p[0]) for p in paths if p})
                if draft_tokens:
                    overlap_non_empty += 1
                    base_topk_idx = (
                        torch.topk(state.next_token_logits, k=max(1, args.base_top_k), dim=-1)
                        .indices.reshape(-1)
                        .tolist()
                    )
                    base_topk = {int(x) for x in base_topk_idx}
                    overlap = sum(1 for token_id in draft_tokens if token_id in base_topk)
                    overlap_total += overlap
                    if overlap > 0:
                        draft_in_base_topk += 1
                    if args.print_step_overlap and printed_steps < max(0, args.print_max_steps):
                        draft_text = [base.decode_tokens([t]) for t in draft_tokens]
                        base_text = [base.decode_tokens([t]) for t in base_topk_idx]
                        print(
                            f"[step {printed_steps + 1}] "
                            f"draft_ids={draft_tokens} draft_text={draft_text} | "
                            f"base_ids={base_topk_idx} base_text={base_text} | "
                            f"overlap={overlap}"
                        )
                        printed_steps += 1
            # Always advance with base next token to keep one deterministic trajectory.
            prefix.append(base_next)
            state = base.forward_decode(state, base_next)

    agreement = (matches / total) if total > 0 else 0.0
    print("=" * 40)
    print("Yuhuili draft sanity check")
    print(f"Base model id: {args.base_model_id}")
    print(f"Base model preset: {args.base_model_preset or 'custom'}")
    print(f"Base implementation: {args.base_impl}")
    print(f"Draft model id: {args.draft_model_id}")
    print(f"Prompts: {len(prompts)}")
    print(f"Sampled steps: {args.sample_steps}")
    print(f"Compared steps: {total}")
    print(f"Top-1 matches: {matches}")
    print(f"Top-1 agreement rate: {agreement:.4f}")
    if args.diagnose_overlap:
        avg_overlap = (overlap_total / overlap_non_empty) if overlap_non_empty > 0 else 0.0
        any_overlap_rate = (draft_in_base_topk / overlap_non_empty) if overlap_non_empty > 0 else 0.0
        print(f"Draft top-k: {args.draft_top_k}, Base top-k: {args.base_top_k}")
        print(f"Steps with non-empty draft proposals: {overlap_non_empty}")
        print(f"Avg overlap count per step: {avg_overlap:.4f}")
        print(f"Any-overlap rate (draft in base top-k): {any_overlap_rate:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
