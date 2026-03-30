#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Replay speculative decoding from a collected base trace or MTP reference on CPU.

Workflow (trace from collect_base_trace_gpu):
  1. GPU: collect_base_trace_gpu.py --base-model-preset r1_0528 --out trace.pt
  2. CPU: run_eagle3_from_trace_cpu.py --trace trace.pt --draft-mode draft_r1

Alternatively, use an MTP reference .pt from DeepSeek v3 tests (test_generate_mtp_reference_io):
  - File format: hidden_states [num_steps, batch, hidden_size], next_tokens [num_steps, batch],
    start_tokens [batch], optional logits [num_steps, batch, vocab], optional metadata.
    Usually at $DEEPSEEK_V3_CACHE/test_io_cache/mtp_full_model_seq{N}.pt
  - run_eagle3_from_trace_cpu.py --trace /path/to/mtp_full_model_seq128.pt --draft-mode draft_r1
  - Use --batch-index 0 (default) for the first sequence in the batch.
  - Tokenizer Hub id for MTP files defaults to ``lmsys/DeepSeek-R1-NextN`` (not the full R1 repo);
    for NextN MTP draft + record base use ``run_nextn_mtp_from_record_cpu.py`` instead.

The trace/reference contains per-step greedy tokens and hidden states. Draft tokens are
proposed by the draft model and verified against the recorded trajectory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.default_paths import DEFAULT_HF_HOME
from models.demos.speculative_deepseek_r1_broad.hf_cache import bootstrap_hf_env_before_transformers, set_hf_home

bootstrap_hf_env_before_transformers(sys.argv, default_hf_home=DEFAULT_HF_HOME)

from models.demos.speculative_deepseek_r1_broad.config import EagleConfig
from models.demos.speculative_deepseek_r1_broad.eagle_engine import EagleEngine
from models.demos.speculative_deepseek_r1_broad.models_draft import (
    TraditionalDraftAdapter,
    Eagle3HiddenStateDraftAdapter,
)
from models.demos.speculative_deepseek_r1_broad.trace_replay_base import (
    TraceReplayBaseAdapter,
    format_mtp_prefix_banner,
    load_trace_or_mtp_reference,
)

DRAFT_MODEL_PRESETS = {
    "eagle3_r1": "eigen-ai-labs/deepseek-v3.1-eagle3",
    "eagle3_8b": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
    "r1_draft_0_6b": "jukofyork/DeepSeek-R1-DRAFT-0.6B-v3.0",
}


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Replay speculative decoding from collected base trace (CPU)")
    p.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Path to trace file (collect_base_trace_gpu format) or MTP reference .pt (hidden_states, next_tokens, start_tokens).",
    )
    p.add_argument(
        "--batch-index",
        type=int,
        default=0,
        help="Batch index to use when trace is an MTP reference file (default 0).",
    )
    p.add_argument(
        "--draft-mode",
        type=str,
        default="draft_r1",
        choices=["eagle3_8b", "draft_r1"],
        help="draft_r1: traditional 0.6B draft model, eagle3_8b: EAGLE3 hidden-state head.",
    )
    p.add_argument("--draft-model-id", type=str, default=None, help="Override draft model HF id.")
    p.add_argument(
        "--draft-model-preset",
        type=str,
        default=None,
        choices=sorted(DRAFT_MODEL_PRESETS),
        help="Named preset for draft model id.",
    )
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--num-steps", type=int, default=1)
    p.add_argument("--max-paths", type=int, default=16)
    p.add_argument(
        "--verification-acceptance",
        type=str,
        default="argmax",
        choices=["argmax", "probabilistic"],
    )
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--verbose-shapes", action="store_true", default=False)
    p.add_argument("--log-every-steps", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    p.add_argument(
        "--log-base-confidence",
        action="store_true",
        default=False,
        help="Print [base_conf] lines when MTP reference includes logits.",
    )
    p.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="Directory for HF_HOME / hub cache (draft + tokenizer Hub downloads).",
    )
    return p


def main() -> None:
    args = create_parser().parse_args()
    if args.hf_home is not None:
        set_hf_home(args.hf_home)
        print(f"HF_HOME={args.hf_home}")
    trace = load_trace_or_mtp_reference(args.trace, batch_index=args.batch_index)
    mtp_banner = format_mtp_prefix_banner(trace, batch_index=args.batch_index)
    if mtp_banner:
        print(mtp_banner, flush=True)
    base = TraceReplayBaseAdapter(trace)
    if trace.step_next_logits is not None:
        print(f"Trace step_next_logits: shape={tuple(trace.step_next_logits.shape)}")
    elif args.log_base_confidence:
        print("Trace has no logits; --log-base-confidence will only warn (verification uses one-hot logits).")

    draft_model_id = args.draft_model_id
    if args.draft_model_preset is not None:
        draft_model_id = DRAFT_MODEL_PRESETS[args.draft_model_preset]
    if draft_model_id is None:
        draft_model_id = DRAFT_MODEL_PRESETS.get(
            "r1_draft_0_6b" if args.draft_mode == "draft_r1" else "eagle3_8b"
        )

    if args.draft_mode == "draft_r1":
        draft = TraditionalDraftAdapter(
            draft_model_id, device=args.device, torch_dtype=args.dtype, trust_remote_code=True,
        )
        draft_backend = f"traditional draft ({draft_model_id})"
    elif args.draft_mode == "eagle3_8b":
        draft = Eagle3HiddenStateDraftAdapter(
            draft_model_id, device=args.device, torch_dtype=args.dtype,
        )
        draft_backend = f"EAGLE3 draft head ({draft_model_id})"
    else:
        raise SystemExit(f"Unsupported draft mode: {args.draft_mode}")

    cfg = EagleConfig(
        top_k=args.top_k,
        depth=args.depth,
        num_steps=args.num_steps,
        max_paths=args.max_paths,
        verification_mode="cache_per_path",
        verification_acceptance=args.verification_acceptance,
        verbose=args.verbose,
        verbose_shapes=args.verbose_shapes,
        log_every_steps=args.log_every_steps,
        random_seed=args.random_seed,
        log_base_confidence=args.log_base_confidence,
    )
    engine = EagleEngine(base=base, draft=draft, cfg=cfg)
    if trace.prompt:
        result = engine.generate(prompt=trace.prompt, max_new_tokens=args.max_new_tokens)
    else:
        result = engine.generate(prefix_token_ids=trace.prompt_token_ids, max_new_tokens=args.max_new_tokens)

    print("=" * 60)
    print("Speculative decoding replay from trace")
    print(f"Trace file: {args.trace}")
    print(f"Trace model id: {trace.model_id}")
    print(f"Draft mode: {args.draft_mode}")
    print(f"Draft backend: {draft_backend}")
    print(f"Verification acceptance: {args.verification_acceptance}")
    prompt_display = trace.prompt or base.decode_tokens(trace.prompt_token_ids)
    print(f"Prompt: {prompt_display}")
    print(f"Generated text: {result.generated_text}")
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
    print("=" * 60)
    print("NOTE: Replay verifies against recorded greedy trajectory.")
    print("Divergent branches are conservatively rejected.")


if __name__ == "__main__":
    main()
