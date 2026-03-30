#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
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
class BaselineSummary:
    max_new_tokens: int
    prompts: int
    avg_tokens_per_s: float
    median_tokens_per_s: float
    avg_elapsed_s: float


@dataclass(frozen=True)
class EagleSummary:
    max_new_tokens: int
    top_k: int
    depth: int
    num_steps: int
    verification_mode: str
    prompts: int
    avg_tokens_per_s: float
    median_tokens_per_s: float
    avg_accepted_tokens_percentage: float
    avg_acceptance_rate: float
    avg_any_accept_rate: float
    avg_elapsed_s: float
    speedup_vs_baseline: float


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


def _run_baseline_once(base: DeepSeekBaseAdapter, prompt: str, max_new_tokens: int) -> tuple[float, float]:
    prompt_ids = base.encode_prompt(prompt)
    committed = list(prompt_ids)
    decode_state = base.create_decode_state(committed)
    generated = 0
    start = time.perf_counter()
    for step_idx in range(max_new_tokens):
        next_token = base.decode_state_next_token(decode_state)
        committed.append(next_token)
        generated += 1
        if step_idx + 1 < max_new_tokens:
            decode_state = base.advance_decode_state(decode_state, next_token)
    elapsed_s = max(time.perf_counter() - start, 1e-9)
    return generated / elapsed_s, elapsed_s


def _run_baseline(base: DeepSeekBaseAdapter, prompts: list[str], max_new_tokens: int) -> BaselineSummary:
    tps_values: list[float] = []
    elapsed_values: list[float] = []
    for prompt in prompts:
        tps, elapsed_s = _run_baseline_once(base, prompt, max_new_tokens)
        tps_values.append(tps)
        elapsed_values.append(elapsed_s)
    return BaselineSummary(
        max_new_tokens=max_new_tokens,
        prompts=len(prompts),
        avg_tokens_per_s=statistics.mean(tps_values),
        median_tokens_per_s=statistics.median(tps_values),
        avg_elapsed_s=statistics.mean(elapsed_values),
    )


def _run_eagle(
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
    baseline_tps: float,
) -> EagleSummary:
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

    tps_values: list[float] = []
    elapsed_values: list[float] = []
    acceptance_values: list[float] = []
    best_path_values: list[float] = []
    any_accept_values: list[float] = []
    for prompt in prompts:
        result = engine.generate(prompt=prompt, max_new_tokens=max_new_tokens)
        tps_values.append(result.stats.tokens_per_s)
        elapsed_values.append(result.stats.elapsed_s)
        acceptance_values.append(result.stats.accepted_tokens_percentage)
        best_path_values.append(result.stats.selected_path_acceptance_rate)
        any_accept_values.append(result.stats.any_accept_rate)

    avg_tps = statistics.mean(tps_values)
    return EagleSummary(
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        depth=depth,
        num_steps=num_steps,
        verification_mode=verification_mode,
        prompts=len(prompts),
        avg_tokens_per_s=avg_tps,
        median_tokens_per_s=statistics.median(tps_values),
        avg_accepted_tokens_percentage=statistics.mean(acceptance_values),
        avg_acceptance_rate=statistics.mean(best_path_values),
        avg_any_accept_rate=statistics.mean(any_accept_values),
        avg_elapsed_s=statistics.mean(elapsed_values),
        speedup_vs_baseline=(avg_tps / baseline_tps) if baseline_tps > 0 else 0.0,
    )


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
    p = argparse.ArgumentParser("EAGLE3 CPU benchmark sweep with table output")
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
    p.add_argument("--max-new-tokens-list", type=str, default="32,128", help="CSV list, e.g. 32,128")
    p.add_argument("--top-k-list", type=str, default="2,4", help="CSV list, e.g. 2,4")
    p.add_argument("--depth-list", type=str, default="1,2", help="CSV list, e.g. 1,2")
    p.add_argument("--num-steps-list", type=str, default="1,2", help="CSV list, e.g. 1,2")
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
    p.add_argument("--json-out", type=Path, default=None, help="Optional path to write JSON benchmark results.")
    return p


def main() -> None:
    args = create_parser().parse_args()
    if args.base_model_preset is not None:
        args.base_model_id = BASE_MODEL_PRESETS[args.base_model_preset]
    prompts = _load_prompts(args)
    max_new_tokens_list = _parse_int_csv(args.max_new_tokens_list)
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

    print("=" * 80)
    print("EAGLE3 CPU benchmark sweep")
    print(f"Base model id: {args.base_model_id}")
    print(f"Base model preset: {args.base_model_preset or 'custom'}")
    print(f"Base implementation: {args.base_impl}")
    print(f"Draft mode: {args.draft_mode}")
    print(f"Draft model id: {args.draft_model_id}")
    print(f"Draft backend: {draft_backend}")
    print(f"Prompts: {len(prompts)}")
    print(f"Verification mode: {args.verification_mode}")
    print(f"Device: {args.device}, DType: {args.dtype}")
    print("=" * 80)

    baseline_by_tokens: dict[int, BaselineSummary] = {}
    eagle_rows: list[EagleSummary] = []
    for max_new_tokens in max_new_tokens_list:
        baseline_summary = _run_baseline(base, prompts, max_new_tokens)
        baseline_by_tokens[max_new_tokens] = baseline_summary
        for top_k in top_k_list:
            for depth in depth_list:
                for num_steps in num_steps_list:
                    eagle_rows.append(
                        _run_eagle(
                            base=base,
                            draft=draft,
                            prompts=prompts,
                            max_new_tokens=max_new_tokens,
                            top_k=top_k,
                            depth=depth,
                            num_steps=num_steps,
                            max_paths=args.max_paths,
                            verification_mode=args.verification_mode,
                            baseline_tps=baseline_summary.avg_tokens_per_s,
                        )
                    )

    baseline_table_rows = [
        [
            str(item.max_new_tokens),
            str(item.prompts),
            f"{item.avg_tokens_per_s:.2f}",
            f"{item.median_tokens_per_s:.2f}",
            f"{item.avg_elapsed_s:.4f}",
        ]
        for item in sorted(baseline_by_tokens.values(), key=lambda x: x.max_new_tokens)
    ]
    baseline_table = _format_table(
        ["tokens", "prompts", "baseline avg tok/s", "baseline median tok/s", "baseline avg sec"],
        baseline_table_rows,
    )

    eagle_table_rows = [
        [
            str(item.max_new_tokens),
            str(item.top_k),
            str(item.depth),
            str(item.num_steps),
            f"{item.avg_tokens_per_s:.2f}",
            f"{item.median_tokens_per_s:.2f}",
            f"{item.avg_accepted_tokens_percentage:.4f}",
            f"{item.avg_acceptance_rate:.4f}",
            f"{item.avg_any_accept_rate:.4f}",
            f"{item.speedup_vs_baseline:.3f}x",
        ]
        for item in sorted(
            eagle_rows,
            key=lambda x: (x.max_new_tokens, x.top_k, x.depth, x.num_steps),
        )
    ]
    eagle_table = _format_table(
        [
            "tokens",
            "top_k",
            "depth",
            "steps",
            "eagle avg tok/s",
            "eagle median tok/s",
            "accepted_tokens_%",
            "sel_path_accept_rate",
            "avg rounds-with-any-accept",
            "speedup vs base",
        ],
        eagle_table_rows,
    )

    print("\nBaseline summary")
    print(baseline_table)
    print("\nEAGLE sweep summary")
    print(eagle_table)

    if args.json_out is not None:
        payload = {
            "config": {
                "base_model_id": args.base_model_id,
                "draft_model_id": args.draft_model_id,
                "draft_mode": args.draft_mode,
                "base_impl": args.base_impl,
                "verification_mode": args.verification_mode,
                "device": args.device,
                "dtype": args.dtype,
                "max_new_tokens_list": max_new_tokens_list,
                "top_k_list": top_k_list,
                "depth_list": depth_list,
                "num_steps_list": num_steps_list,
                "max_paths": args.max_paths,
                "prompts_count": len(prompts),
            },
            "baseline": [asdict(item) for item in sorted(baseline_by_tokens.values(), key=lambda x: x.max_new_tokens)],
            "eagle": [
                asdict(item)
                for item in sorted(
                    eagle_rows,
                    key=lambda x: (x.max_new_tokens, x.top_k, x.depth, x.num_steps),
                )
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote JSON results to: {args.json_out}")


if __name__ == "__main__":
    main()
