# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.config import EagleConfig
from models.demos.speculative_deepseek_r1_broad.distributed_utils import init_distributed, is_main_process
from models.demos.speculative_deepseek_r1_broad.eagle_engine import EagleEngine
from models.demos.speculative_deepseek_r1_broad.models_base import DeepSeekBaseAdapter
from models.demos.speculative_deepseek_r1_broad.models_draft import (
    BaseAsDraftAdapter,
    Eagle3HiddenStateDraftAdapter,
    TraditionalDraftAdapter,
)

BASE_MODEL_PRESETS = {
    "r1_0528": "deepseek-ai/DeepSeek-R1-0528",
    "r1": "deepseek-ai/DeepSeek-R1",
    "distill_llama_8b": "deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B",
    "distill_qwen_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "distill_qwen_1_5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "tiny_gpt2": "sshleifer/tiny-gpt2",
}

# EAGLE3-8B (yuhuili) draft expects LLaMA-based base hidden states; Qwen base is wrong.
EAGLE3_8B_QWEN_BASE_IDS = frozenset({
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
})


def _is_qwen_base(base_model_id: str) -> bool:
    return (
        base_model_id in EAGLE3_8B_QWEN_BASE_IDS
        or "qwen" in base_model_id.lower()
    )

DRAFT_MODEL_PRESETS = {
    "eagle3_r1": "eigen-ai-labs/deepseek-v3.1-eagle3",
    "eagle3_8b": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    "r1_draft_0_6b": "jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0",
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
    p = argparse.ArgumentParser("EAGLE3 PyTorch CPU baseline for DeepSeek-R1")
    p.add_argument("prompts", nargs="*", help="Prompt text. Ignored when --prompt or --prompts-file is set.")
    p.add_argument("--prompt", type=str, help="Single prompt string.")
    p.add_argument("--prompts-file", type=Path, help="JSON file with [{\"prompt\": ...}] or {\"prompts\": [...]} format.")
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
    p.add_argument(
        "--draft-model-preset",
        type=str,
        default=None,
        choices=sorted(DRAFT_MODEL_PRESETS),
        help="Named preset for draft model id (overrides --draft-model-id).",
    )
    p.add_argument(
        "--draft-mode",
        type=str,
        default="draft_r1",
        choices=["self", "eagle3_8b", "yuhuili", "draft_r1"],
        help="draft_r1: traditional 0.6B draft, eagle3_8b: EAGLE3 hidden-state head, self: base-as-draft. "
        "For record + NextN MTP without loading a full base, use scripts/run_nextn_mtp_from_record_cpu.py.",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--base-impl",
        type=str,
        default="reference",
        choices=["adapter", "reference", "accelerate"],
        help="Base runtime implementation path.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Draft branches per step; with --draft-branching temperature_top_p, 0 = full nucleus.",
    )
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=2)
    p.add_argument(
        "--max-paths",
        type=int,
        default=16,
        help="Beam cap (0 = no cap; expensive if draft is wide).",
    )
    p.add_argument(
        "--verification-mode",
        type=str,
        default="batched_single_pass",
        choices=["batched_single_pass", "cache_per_path", "flattened_tree"],
        help="Path verification: batched_single_pass (recommended), cache_per_path, or flattened_tree (single forward with tree attention mask).",
    )
    p.add_argument(
        "--verification-acceptance",
        type=str,
        default="argmax",
        choices=["argmax", "probabilistic"],
        help="Verification acceptance rule: argmax match or probabilistic accept/reject from base distribution.",
    )
    p.add_argument(
        "--per-path-verification",
        action="store_true",
        default=False,
        help="Use one forward per path (for backends that do not batch correctly). Default: one batched forward per round.",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed used for probabilistic verification and draft nucleus sampling.",
    )
    p.add_argument(
        "--draft-branching",
        type=str,
        default="top_k",
        choices=["top_k", "temperature_top_p"],
        help="Draft branches: top_k (default) or temperature_top_p nucleus sampling (see models_draft).",
    )
    p.add_argument(
        "--draft-temperature",
        type=float,
        default=0.6,
        help="With temperature_top_p: softmax temperature for draft branching (<=0 → argmax).",
    )
    p.add_argument(
        "--draft-top-p",
        type=float,
        default=0.95,
        help="With temperature_top_p: nucleus cumulative probability mass.",
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
    p.add_argument(
        "--verbose-shapes",
        action="store_true",
        default=False,
        help="Print per-round shape/cache diagnostics (requires --verbose).",
    )
    p.add_argument("--log-every-steps", type=int, default=0, help="Log every N generated tokens (0 = every event).")
    p.add_argument(
        "--debug-proposal-rounds",
        type=int,
        default=0,
        help="Print speculative (draft) and base tokens per round for first N rounds (0 = off). Use with --verbose. "
        "Example: --debug-proposal-rounds 10 --verbose",
    )
    p.add_argument(
        "--print-speculated-tokens",
        action="store_true",
        default=False,
        help="Print speculated token IDs every round (stdout). Use to compare rounds 5 vs 6 between FC and no-FC.",
    )
    p.add_argument(
        "--no-eagle3-fc-every-depth",
        action="store_true",
        default=False,
        help="[EAGLE3 only] Use FC only at depth 0 (paper). Default: FC at every depth (matches released checkpoints).",
    )
    return p


def main() -> None:
    args = create_parser().parse_args()
    if args.base_model_preset is not None:
        args.base_model_id = BASE_MODEL_PRESETS[args.base_model_preset]
    if args.draft_model_preset is not None:
        args.draft_model_id = DRAFT_MODEL_PRESETS[args.draft_model_preset]
    # EAGLE3 mode requires an EAGLE3 head checkpoint (fc, norm, lm_head, d2t), not the traditional 0.6B model
    if args.draft_mode in ("eagle3_8b", "yuhuili") and args.draft_model_id == "jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0":
        args.draft_model_id = DRAFT_MODEL_PRESETS["eagle3_8b"]
    # EAGLE3-8B draft expects LLaMA-based base (same hidden space as Distill-LLaMA-8B). Qwen base is a mismatch.
    if args.draft_mode in ("eagle3_8b", "yuhuili") and _is_qwen_base(args.base_model_id):
        raise SystemExit(
            f"EAGLE3-8B draft requires a LLaMA-based base model, not Qwen.\n"
            f"Current base: {args.base_model_id}\n"
            "Use: --base-model-preset distill_llama_8b\n"
            "Or:  --base-model-id deepseek-ai/DeepSeek-R1-Distill-LLaMA-8B"
        )
    dist_ctx = init_distributed(args.tp_size, args.local_rank)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    prompts = _load_prompts(args)

    eagle_cfg = EagleConfig(
        top_k=args.top_k,
        depth=args.depth,
        num_steps=args.num_steps,
        max_paths=args.max_paths,
        verification_mode=args.verification_mode,
        verification_acceptance=args.verification_acceptance,
        verification_per_path_forward=getattr(args, "per_path_verification", False),
        verbose=args.verbose,
        verbose_shapes=args.verbose_shapes,
        log_every_steps=args.log_every_steps,
        random_seed=args.random_seed,
        draft_branching=args.draft_branching,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        debug_proposal_rounds=getattr(args, "debug_proposal_rounds", 0),
        print_speculated_tokens_per_round=getattr(args, "print_speculated_tokens", False),
    )
    try:
        base = DeepSeekBaseAdapter(
            args.base_model_id,
            device=args.device,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            base_impl=args.base_impl,
            tp_size=args.tp_size,
            tp_backend=args.tp_backend,
            local_rank=dist_ctx.local_rank,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "FP8 quantization" in msg and args.device == "cpu":
            hint = (
                "For EAGLE3-8B use: --base-model-preset distill_llama_8b. "
                "Other CPU options: --base-model-preset distill_qwen_1_5b or tiny_gpt2."
            )
            raise SystemExit(f"{msg}\nHint: use a GPU server for DeepSeek-R1-0528, or run CPU smoke tests with {hint}")
        raise
    if args.draft_mode == "self":
        draft = BaseAsDraftAdapter(base)
        draft_backend = "base-as-draft adapter"
    elif args.draft_mode in ("eagle3_8b", "yuhuili"):
        draft = Eagle3HiddenStateDraftAdapter(
            args.draft_model_id,
            device=args.device,
            torch_dtype=args.dtype,
        )
        if getattr(args, "no_eagle3_fc_every_depth", False):
            draft.use_fc_at_every_depth = False
        draft_backend = "EAGLE3-DeepSeek-R1-Distill-LLaMA-8B draft head"
    elif args.draft_mode == "draft_r1":
        draft = TraditionalDraftAdapter(
            args.draft_model_id,
            device=args.device,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        draft_backend = f"traditional draft ({args.draft_model_id})"
    else:
        raise SystemExit(f"Unsupported draft mode: {args.draft_mode}")
    engine = EagleEngine(base=base, draft=draft, cfg=eagle_cfg)

    if is_main_process(dist_ctx):
        print("=" * 40)
        print("Run configuration")
        print(f"Base model id: {args.base_model_id}")
        print(f"Base model preset: {args.base_model_preset or 'custom'}")
        print(f"Base implementation: {args.base_impl}")
        print(f"Draft mode: {args.draft_mode}")
        print(f"Draft model id: {args.draft_model_id}")
        print(f"Draft backend: {draft_backend}")
        if args.draft_mode in ("eagle3_8b", "yuhuili") and getattr(args, "no_eagle3_fc_every_depth", False):
            print("EAGLE3 FC mode: only at depth 0 (paper)")
        print(f"Verification mode: {args.verification_mode}")
        print(f"Verification: {'one forward per path' if getattr(args, 'per_path_verification', False) else 'one batched forward per round (default)'}")
        print(f"Verification acceptance: {args.verification_acceptance}")
        print(f"Depth: {args.depth}, Max new tokens: {args.max_new_tokens}")
        print("(Defaults: depth=2, max_new_tokens=32)")
        if args.random_seed is not None:
            print(f"Random seed: {args.random_seed}")
        print(f"Strip think tags: {args.strip_think_tags}")
        print(f"Print raw tokens: {args.print_raw_tokens}")
        print(f"Device: {args.device}, DType: {args.dtype}")
        print(f"TP size: {args.tp_size}, TP backend: {args.tp_backend}, Rank: {dist_ctx.rank}")
        print("=" * 40)
        print("Metric formulas:")
        print("  accepted_tokens_percentage = accepted_draft_tokens / proposed_draft_tokens")
        print("  acceptance_rate = accepted_draft_tokens / (depth * rounds_with_proposals)")
        print("  selected_path_acceptance_rate = accepted_on_chosen_path / min(depth, path_len) slots")
        print("  rounds_with_any_accept_rate = rounds_with_any_accept / rounds_with_proposals")
        print("=" * 40)
        if args.device == "cpu" and "8b" in args.base_model_id.lower():
            per_path = getattr(args, "per_path_verification", False)
            print(
                "Note: 8B base on CPU is slow. "
                + ("One forward per path. " if per_path else "Default: one batched forward per round. ")
                + "Use --verbose to see progress; --max-new-tokens 8 for a quicker run."
            )
        print()

    for idx, prompt in enumerate(prompts):
        if is_main_process(dist_ctx):
            print(f"Generating for prompt {idx + 1}/{len(prompts)} (max_new_tokens={args.max_new_tokens})...", flush=True)
        result = engine.generate(prompt=prompt, max_new_tokens=args.max_new_tokens)
        if is_main_process(dist_ctx):
            print("-" * 30)
            print(f"Prompt[{idx + 1}]: {prompt}")
            display_text = _postprocess_generation_text(result.generated_text, strip_think_tags=args.strip_think_tags)
            print(f"Generation[{idx + 1}]: {display_text}")
            if args.print_raw_tokens:
                _print_raw_tokens(base, result.generated_token_ids, max_tokens=max(0, args.raw_token_max))
            print(f"Generated tokens: {result.stats.generated_tokens}")
            print(f"Proposed draft tokens: {result.stats.proposed_tokens}")
            print(f"Accepted draft tokens: {result.stats.accepted_tokens}")
            print(f"Accepted tokens percentage: {result.stats.accepted_tokens_percentage:.4f}")
            print(f"Acceptance rate: {result.stats.acceptance_rate:.4f}")
            print(f"Selected-path acceptance rate: {result.stats.selected_path_acceptance_rate:.4f}")
            print(f"First-token match rate: {result.stats.first_token_match_rate:.4f}")
            print(f"Second-token match rate: {result.stats.second_token_match_rate:.4f}")
            print(f"Rounds-with-any-accept rate: {result.stats.any_accept_rate:.4f}")
            print(f"Rounds with proposals: {result.stats.total_rounds_with_paths}")
            print(f"Elapsed seconds: {result.stats.elapsed_s:.4f}")
            print(f"Tokens/sec: {result.stats.tokens_per_s:.2f}")
            print("-" * 30)


if __name__ == "__main__":
    main()

