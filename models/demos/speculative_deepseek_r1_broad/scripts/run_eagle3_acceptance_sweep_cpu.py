#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.config import EagleConfig
from models.demos.speculative_deepseek_r1_broad.eagle_engine import EagleEngine
from models.demos.speculative_deepseek_r1_broad.models_base import DeepSeekBaseAdapter
from models.demos.speculative_deepseek_r1_broad.models_draft import BaseAsDraftAdapter, Eagle3HiddenStateDraftAdapter, TraditionalDraftAdapter

BASE_MODEL_PRESETS = {
    "r1_0528": "deepseek-ai/DeepSeek-R1-0528",
    "r1": "deepseek-ai/DeepSeek-R1",
    "distill_llama_8b": "deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B",
    "distill_qwen_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "distill_qwen_1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "tiny_gpt2": "sshleifer/tiny-gpt2",
}


@dataclass(frozen=True)
class SweepSummary:
    top_k: int
    depth: int
    num_steps: int
    prompts: int
    max_new_tokens: int
    avg_accepted_tokens: float
    avg_proposed_tokens: float
    avg_rounds_with_paths: float
    avg_accepted_tokens_percentage: float
    avg_any_accept_rate: float
    avg_tokens_per_round: float
    avg_tokens_per_s: float


def _fmt_config(s: SweepSummary) -> str:
    return f"top_k={s.top_k}, depth={s.depth}, num_steps={s.num_steps}"


def _load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompt_items = data if isinstance(data, list) else data.get("prompts", [])
        prompts = [str(item["prompt"]) for item in prompt_items if isinstance(item, dict) and "prompt" in item]
        if args.num_prompts is not None:
            prompts = prompts[: args.num_prompts]
        if prompts:
            return prompts
        raise SystemExit(f"No valid prompts found in {args.prompts_file}")
    if args.prompt:
        return [args.prompt]
    if args.prompts:
        return args.prompts
    raise SystemExit("Provide --prompt, positional prompts, or --prompts-file.")


def _parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    if not out:
        raise ValueError(f"Expected at least one integer in '{value}'")
    return out


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    sep = "-+-".join("-" * widths[i] for i in range(len(headers)))
    body = [" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in rows]
    return "\n".join([line, sep, *body])


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("EAGLE3 acceptance/iteration sweep (CPU)")
    p.add_argument("prompts", nargs="*", help="Prompt text. Ignored when --prompt or --prompts-file is set.")
    p.add_argument("--prompt", type=str, help="Single prompt string.")
    p.add_argument("--prompts-file", type=Path, help='JSON file with [{"prompt": ...}] or {"prompts": [...]}')
    p.add_argument("--num-prompts", type=int, default=None, help="Max prompts loaded from --prompts-file.")
    p.add_argument("--base-model-id", type=str, default="deepseek-ai/DeepSeek-R1-0528")
    p.add_argument(
        "--base-model-preset",
        type=str,
        default=None,
        choices=sorted(BASE_MODEL_PRESETS),
        help="Optional preset for base model id (overrides --base-model-id).",
    )
    p.add_argument("--draft-model-id", type=str, default="jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0")
    p.add_argument("--draft-mode", type=str, default="draft_r1", choices=["eagle3_8b", "yuhuili", "self", "draft_r1"])
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--top-k-list", type=str, default="1,2,4", help="CSV list, e.g. 1,2,4")
    p.add_argument("--depth-list", type=str, default="1,2,3", help="CSV list, e.g. 1,2,3")
    p.add_argument("--num-steps-list", type=str, default="1", help="CSV list, e.g. 1 or 1,2")
    p.add_argument("--max-paths", type=int, default=16)
    p.add_argument(
        "--base-impl",
        type=str,
        default="reference",
        choices=["adapter", "reference", "accelerate"],
        help="Base runtime implementation path.",
    )
    p.add_argument(
        "--verification-mode",
        type=str,
        default="batched_single_pass",
        choices=["batched_single_pass", "cache_per_path", "flattened_tree"],
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON results.")
    return p


def _run_single_config(
    base: DeepSeekBaseAdapter,
    draft: BaseAsDraftAdapter | Eagle3HiddenStateDraftAdapter,
    prompts: list[str],
    *,
    max_new_tokens: int,
    top_k: int,
    depth: int,
    num_steps: int,
    max_paths: int,
    verification_mode: str,
) -> SweepSummary:
    cfg = EagleConfig(
        top_k=top_k,
        depth=depth,
        num_steps=num_steps,
        max_paths=max_paths,
        verification_mode=verification_mode,
        verbose=False,
        log_every_steps=0,
    )
    engine = EagleEngine(base=base, draft=draft, cfg=cfg)

    accepted: list[float] = []
    proposed: list[float] = []
    rounds: list[float] = []
    acc_rate: list[float] = []
    any_acc_rate: list[float] = []
    tps: list[float] = []

    for prompt in prompts:
        result = engine.generate(prompt=prompt, max_new_tokens=max_new_tokens)
        s = result.stats
        accepted.append(float(s.accepted_tokens))
        proposed.append(float(s.proposed_tokens))
        rounds.append(float(s.total_rounds_with_paths))
        acc_rate.append(float(s.accepted_tokens_percentage))
        any_acc_rate.append(float(s.any_accept_rate))
        tps.append(float(s.tokens_per_s))

    avg_rounds = statistics.mean(rounds)
    avg_accepted = statistics.mean(accepted)
    return SweepSummary(
        top_k=top_k,
        depth=depth,
        num_steps=num_steps,
        prompts=len(prompts),
        max_new_tokens=max_new_tokens,
        avg_accepted_tokens=avg_accepted,
        avg_proposed_tokens=statistics.mean(proposed),
        avg_rounds_with_paths=avg_rounds,
        avg_accepted_tokens_percentage=statistics.mean(acc_rate),
        avg_any_accept_rate=statistics.mean(any_acc_rate),
        avg_tokens_per_round=(avg_accepted / avg_rounds) if avg_rounds > 0 else 0.0,
        avg_tokens_per_s=statistics.mean(tps),
    )


def main() -> None:
    args = create_parser().parse_args()
    if args.base_model_preset is not None:
        args.base_model_id = BASE_MODEL_PRESETS[args.base_model_preset]
    prompts = _load_prompts(args)
    top_k_list = _parse_int_csv(args.top_k_list)
    depth_list = _parse_int_csv(args.depth_list)
    num_steps_list = _parse_int_csv(args.num_steps_list)

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

    if args.draft_mode == "self":
        draft: BaseAsDraftAdapter | Eagle3HiddenStateDraftAdapter = BaseAsDraftAdapter(base)
        draft_backend = "base-as-draft adapter"
    elif args.draft_mode == "draft_r1":
        draft = TraditionalDraftAdapter(args.draft_model_id, device=args.device, torch_dtype=args.dtype)
        draft_backend = f"traditional draft ({args.draft_model_id})"
    else:
        draft = Eagle3HiddenStateDraftAdapter(
            args.draft_model_id,
            device=args.device,
            torch_dtype=args.dtype,
        )
        draft_backend = "EAGLE3-DeepSeek-R1-Distill-LLaMA-8B draft head"

    print("=" * 90)
    print("NOTE: Results on small CPU smoke-test bases (for example tiny GPT-2)")
    print("and mismatched draft/base hidden spaces are not reliable for final conclusions.")
    print("Use these numbers for trend/debugging guidance only.")
    print("=" * 90)

    print("=" * 90)
    print("EAGLE3 acceptance/iteration sweep")
    print(f"Base model id: {args.base_model_id}")
    print(f"Base model preset: {args.base_model_preset or 'custom'}")
    print(f"Base implementation: {args.base_impl}")
    print(f"Draft mode: {args.draft_mode}")
    print(f"Draft model id: {args.draft_model_id}")
    print(f"Draft backend: {draft_backend}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Verification mode: {args.verification_mode}")
    print(f"Device: {args.device}, DType: {args.dtype}")
    print("=" * 90)

    summaries: list[SweepSummary] = []
    for num_steps in num_steps_list:
        for top_k in top_k_list:
            for depth in depth_list:
                summaries.append(
                    _run_single_config(
                        base=base,
                        draft=draft,
                        prompts=prompts,
                        max_new_tokens=args.max_new_tokens,
                        top_k=top_k,
                        depth=depth,
                        num_steps=num_steps,
                        max_paths=args.max_paths,
                        verification_mode=args.verification_mode,
                    )
                )

    rows = [
        [
            str(s.top_k),
            str(s.depth),
            str(s.num_steps),
            f"{s.avg_accepted_tokens:.2f}",
            f"{s.avg_rounds_with_paths:.2f}",
            f"{s.avg_tokens_per_round:.3f}",
            f"{s.avg_accepted_tokens_percentage:.4f}",
            f"{s.avg_any_accept_rate:.4f}",
            f"{s.avg_tokens_per_s:.2f}",
        ]
        for s in sorted(summaries, key=lambda x: (x.num_steps, x.top_k, x.depth))
    ]
    table = _format_table(
        [
            "top_k",
            "depth",
            "steps",
            "avg accepted",
            "avg verify iters",
            "accepted/iter",
            "accepted_tokens_%",
            "avg rounds-with-any-accept",
            "avg tok/s",
        ],
        rows,
    )
    print("\nSweep summary")
    print(table)

    sorted_summaries = sorted(summaries, key=lambda x: (x.num_steps, x.top_k, x.depth))
    best_by_accepted = max(sorted_summaries, key=lambda x: x.avg_accepted_tokens)
    best_by_fewest_iters = min(sorted_summaries, key=lambda x: x.avg_rounds_with_paths)
    best_by_accepted_per_iter = max(sorted_summaries, key=lambda x: x.avg_tokens_per_round)
    print("\nBest configs (heuristics)")
    print(
        f"- Max accepted tokens: {_fmt_config(best_by_accepted)} "
        f"(avg_accepted={best_by_accepted.avg_accepted_tokens:.2f})"
    )
    print(
        f"- Min draft-verify iterations: {_fmt_config(best_by_fewest_iters)} "
        f"(avg_verify_iters={best_by_fewest_iters.avg_rounds_with_paths:.2f})"
    )
    print(
        f"- Max accepted/iteration: {_fmt_config(best_by_accepted_per_iter)} "
        f"(accepted_per_iter={best_by_accepted_per_iter.avg_tokens_per_round:.3f})"
    )

    if args.json_out is not None:
        payload = {
            "config": {
                "base_model_id": args.base_model_id,
                "draft_model_id": args.draft_model_id,
                "draft_mode": args.draft_mode,
                "base_impl": args.base_impl,
                "max_new_tokens": args.max_new_tokens,
                "top_k_list": top_k_list,
                "depth_list": depth_list,
                "num_steps_list": num_steps_list,
                "max_paths": args.max_paths,
                "verification_mode": args.verification_mode,
                "device": args.device,
                "dtype": args.dtype,
                "prompts_count": len(prompts),
            },
            "rows": [asdict(s) for s in sorted(summaries, key=lambda x: (x.num_steps, x.top_k, x.depth))],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON results to: {args.json_out}")


if __name__ == "__main__":
    main()
