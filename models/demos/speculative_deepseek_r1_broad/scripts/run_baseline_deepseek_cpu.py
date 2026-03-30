# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.distributed_utils import init_distributed, is_main_process
from models.demos.speculative_deepseek_r1_broad.models_base import DeepSeekBaseAdapter

BASE_MODEL_PRESETS = {
    "r1_0528": "deepseek-ai/DeepSeek-R1-0528",
    "r1": "deepseek-ai/DeepSeek-R1",
    "distill_llama_8b": "deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B",
    "distill_qwen_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "distill_qwen_1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "tiny_gpt2": "sshleifer/tiny-gpt2",
}


def _postprocess_generation_text(text: str, *, strip_think_tags: bool) -> str:
    if not strip_think_tags:
        return text
    cleaned = text
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def _print_raw_tokens(base: DeepSeekBaseAdapter, token_ids: list[int], *, max_tokens: int) -> None:
    shown = token_ids[:max_tokens]
    pieces = [base.decode_tokens([token_id]) for token_id in shown]
    print(f"Raw token ids (first {len(shown)}): {shown}")
    print(f"Raw token text (first {len(shown)}): {pieces}")
    if len(token_ids) > max_tokens:
        print(f"... truncated {len(token_ids) - max_tokens} tokens")


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
        if not prompts:
            raise SystemExit(f"No valid prompts found in {args.prompts_file}")
        return prompts

    if args.prompt:
        return [args.prompt]
    if args.prompts:
        return args.prompts
    raise SystemExit("Provide --prompt, positional prompts, or --prompts-file.")


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("DeepSeek-R1 CPU baseline generation")
    p.add_argument("prompts", nargs="*", help="Prompt text. Ignored when --prompt or --prompts-file is set.")
    p.add_argument("--prompt", type=str, help="Single prompt string.")
    p.add_argument("--prompts-file", type=Path, help="JSON file with [{\"prompt\": ...}] or {\"prompts\": [...]} format.")
    p.add_argument("--num-prompts", type=int, default=None, help="Max prompts loaded from --prompts-file.")
    p.add_argument("--model-id", type=str, default="deepseek-ai/DeepSeek-R1-0528")
    p.add_argument(
        "--base-model-preset",
        type=str,
        default=None,
        choices=sorted(BASE_MODEL_PRESETS),
        help="Optional preset for model id (overrides --model-id).",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--base-impl",
        type=str,
        default="reference",
        choices=["adapter", "reference", "accelerate"],
        help="Base runtime implementation path.",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument("--tp-size", type=int, default=1, help="Tensor parallel world size.")
    p.add_argument("--tp-backend", type=str, default="deepspeed", choices=["deepspeed"], help="TP backend.")
    p.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed launch (torchrun).")
    p.add_argument("--trust-remote-code", action="store_true", default=False)
    p.add_argument(
        "--strip-think-tags",
        action="store_true",
        default=True,
        help="Clean '<think>...</think>' style segments from displayed generation text.",
    )
    p.add_argument(
        "--no-strip-think-tags",
        dest="strip_think_tags",
        action="store_false",
        help="Disable think-tag cleanup and print raw generated text.",
    )
    p.add_argument(
        "--print-raw-tokens",
        action="store_true",
        default=False,
        help="Print raw generated token ids and token-by-token decoded pieces.",
    )
    p.add_argument(
        "--raw-token-max",
        type=int,
        default=64,
        help="Maximum number of raw tokens to print when --print-raw-tokens is enabled.",
    )
    p.add_argument("--verbose", action="store_true", default=False, help="Enable detailed runtime logs.")
    return p


def main() -> None:
    args = create_parser().parse_args()
    if args.base_model_preset is not None:
        args.model_id = BASE_MODEL_PRESETS[args.base_model_preset]
    dist_ctx = init_distributed(args.tp_size, args.local_rank)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    prompts = _load_prompts(args)

    base = DeepSeekBaseAdapter(
        args.model_id,
        device=args.device,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        base_impl=args.base_impl,
        tp_size=args.tp_size,
        tp_backend=args.tp_backend,
        local_rank=dist_ctx.local_rank,
    )
    if is_main_process(dist_ctx):
        print("=" * 40)
        print("Baseline run configuration")
        print(f"Base model id: {args.model_id}")
        print(f"Base model preset: {args.base_model_preset or 'custom'}")
        print(f"Base implementation: {args.base_impl}")
        print(f"Strip think tags: {args.strip_think_tags}")
        print(f"Print raw tokens: {args.print_raw_tokens}")
        print(f"Device: {args.device}, DType: {args.dtype}")
        print(f"TP size: {args.tp_size}, TP backend: {args.tp_backend}, Rank: {dist_ctx.rank}")
        print("=" * 40)

    for idx, prompt in enumerate(prompts):
        prompt_ids = base.encode_prompt(prompt)
        committed = list(prompt_ids)
        decode_state = base.create_decode_state(committed)
        generated: list[int] = []

        start = time.perf_counter()
        for step_idx in range(args.max_new_tokens):
            next_token = base.decode_state_next_token(decode_state)
            committed.append(next_token)
            generated.append(next_token)
            if step_idx + 1 < args.max_new_tokens:
                decode_state = base.advance_decode_state(decode_state, next_token)
        elapsed_s = max(time.perf_counter() - start, 1e-9)

        if is_main_process(dist_ctx):
            print("-" * 30)
            print(f"Prompt[{idx + 1}]: {prompt}")
            raw_text = base.decode_tokens(generated)
            display_text = _postprocess_generation_text(raw_text, strip_think_tags=args.strip_think_tags)
            print(f"Generation[{idx + 1}]: {display_text}")
            if args.print_raw_tokens:
                _print_raw_tokens(base, generated, max_tokens=max(0, args.raw_token_max))
            print(f"Tokens generated: {len(generated)}")
            print(f"Elapsed seconds: {elapsed_s:.4f}")
            print(f"Tokens/sec: {len(generated) / elapsed_s:.2f}")
            print("-" * 30)


if __name__ == "__main__":
    main()

