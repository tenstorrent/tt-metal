#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Record base + NextN MTP draft (CPU). Self-contained bundle: imports ``specfr`` only.

**Base** = ``specfr.trace_replay_base.TraceReplayBaseAdapter`` — replays an MTP ``.pt``.

**Draft** (default) = ``specfr.nextn_sglang_cpu_draft.NextNSglangCPUDraftAdapter`` — full MTP layer.

Optional: ``--sglang-draft-structure``. For more demos, see
``models/demos/speculative_deepseek_r1_broad/scripts/`` in the tt-metal tree.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# This folder contains the ``specfr`` package (sibling of this script).
_BUNDLE_ROOT = Path(__file__).resolve().parent
if str(_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_ROOT))

from specfr.default_paths import (
    DEFAULT_HF_HOME,
    DEFAULT_MTP_RECORD_PATH,
    NEXTN_HF_REPO_ID,
)
from specfr.hf_cache import (
    bootstrap_hf_env_before_transformers,
    ensure_nextn_snapshot_has_modeling_deepseek,
    set_hf_home,
)

bootstrap_hf_env_before_transformers(sys.argv, default_hf_home=DEFAULT_HF_HOME)

from huggingface_hub import snapshot_download

from specfr.config import EagleConfig
from specfr.eagle_engine import (
    BaseNextLogitBucketStats,
    EagleEngine,
    format_draft_depth_pmax_summary,
)
from specfr.local_hf_snapshot import verify_record_dims_vs_snapshot
from specfr.nextn_sglang_cpu_draft import NextNSglangCPUDraftAdapter
from specfr.nextn_sglang_structure_draft import NextNSglangStructureDraftAdapter
from specfr.trace_replay_base import (
    TraceReplayBaseAdapter,
    format_mtp_prefix_banner,
    load_trace_or_mtp_reference,
)


def _fmt_next_logit_bucket(title: str, b: BaseNextLogitBucketStats) -> str:
    return f"{title}: n={b.count} mean_p_max={b.mean_p_max:.6f} mean_entropy={b.mean_entropy:.6f}"


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NextN MTP draft from record (CPU replay base)")
    p.add_argument("--hf-home", type=Path, default=None, help=f"HF root (default {DEFAULT_HF_HOME})")
    p.add_argument(
        "--record",
        type=Path,
        default=DEFAULT_MTP_RECORD_PATH,
        help=f"MTP reference .pt (replay base). Default: {DEFAULT_MTP_RECORD_PATH}",
    )
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument(
        "--embed-head-aux-safetensors",
        type=Path,
        default=None,
        help="Optional .safetensors with embed_tokens + shared_head.head.",
    )
    p.add_argument(
        "--sglang-draft-structure",
        action="store_true",
        help="Use NextNSglangStructureDraftAdapter (loads full NextN HF weights via from_pretrained).",
    )
    p.add_argument(
        "--sglang-cpu-no-batch-beams",
        action="store_true",
        help="Legacy per-beam forward (disable batched beam decoding).",
    )
    p.add_argument("--depth", type=int, default=2)
    p.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Branch budget per beam per draft step. With temperature_top_p: 0 expands full nucleus.",
    )
    p.add_argument("--max-paths", type=int, default=16, help="0 = no beam cap.")
    p.add_argument("--draft-mtp-greedy", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=1)
    p.add_argument(
        "--verification-acceptance",
        type=str,
        default="argmax",
        choices=["argmax", "probabilistic"],
    )
    p.add_argument(
        "--verification-mode",
        type=str,
        default="cache_per_path",
        choices=["cache_per_path", "batched_single_pass", "flattened_tree"],
    )
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument(
        "--draft-branching",
        type=str,
        default="top_k",
        choices=["top_k", "temperature_top_p"],
    )
    p.add_argument("--draft-temperature", type=float, default=0.6)
    p.add_argument("--draft-top-p", type=float, default=0.95)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--verbose-shapes", action="store_true")
    p.add_argument("--log-every-steps", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
    )
    p.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        default=True,
    )
    p.add_argument("--skip-alignment-check", action="store_true")
    p.add_argument(
        "--log-base-confidence",
        action="store_true",
        help="Print [base_conf] lines when the record has step_next_logits.",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument(
        "--log-round-replay-detail",
        action="store_true",
        help="Per-round record greedy vs draft vs verify (trace replay).",
    )
    return p


def main() -> None:
    args = create_parser().parse_args()
    if args.quiet:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    if args.hf_home is not None:
        hp = set_hf_home(args.hf_home)
    else:
        hp = set_hf_home(DEFAULT_HF_HOME)

    record_path = Path(args.record).expanduser().resolve()
    if not record_path.is_file():
        raise SystemExit(f"Record not found: {record_path}")

    hub_cache = hp / "hub"
    if not args.quiet:
        print(f"Downloading NextN snapshot: {NEXTN_HF_REPO_ID} -> {hub_cache}", flush=True)
    nextn_dir = Path(snapshot_download(repo_id=NEXTN_HF_REPO_ID, cache_dir=str(hub_cache)))

    nextn_fusion_file = nextn_dir / "nextn_layer_parameters.safetensors"
    if not nextn_fusion_file.is_file():
        raise SystemExit(f"Fusion weights not found: {nextn_fusion_file}")

    ensure_nextn_snapshot_has_modeling_deepseek(nextn_dir, hub_cache=hub_cache)

    trace = load_trace_or_mtp_reference(record_path, batch_index=args.batch_index)
    if not args.quiet:
        mtp_banner = format_mtp_prefix_banner(trace, batch_index=args.batch_index)
        if mtp_banner:
            print(mtp_banner, flush=True)

    base = TraceReplayBaseAdapter(
        trace,
        tokenizer_local_dir=nextn_dir if nextn_dir.is_dir() else None,
        tokenizer_trust_remote_code=args.trust_remote_code,
    )

    if not args.skip_alignment_check:
        all_ids = list(trace.prompt_token_ids) + list(trace.step_next_tokens)
        max_tid = max(all_ids) if all_ids else 0
        rec_h = int(trace.step_last_hidden.shape[-1])
        verify_record_dims_vs_snapshot(
            record_hidden_size=rec_h,
            record_max_token_id=max_tid,
            snapshot_dir=nextn_dir,
        )

    embed_aux = args.embed_head_aux_safetensors

    if args.sglang_draft_structure:
        draft = NextNSglangStructureDraftAdapter(
            device=args.device,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        draft.bind_from_nextn_paths(
            nextn_safetensors=nextn_fusion_file,
            embed_head_aux_safetensors=embed_aux,
            nextn_config_dir=nextn_dir,
        )
        if not args.quiet:
            print("Draft mode: sglang-draft-structure (NextNSglangStructureDraftAdapter — HF model)", flush=True)
    else:
        draft = NextNSglangCPUDraftAdapter(
            device=args.device,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        draft.bind_from_nextn_paths(
            nextn_safetensors=nextn_fusion_file,
            embed_head_aux_safetensors=embed_aux,
            nextn_config_dir=nextn_dir,
        )
        if not args.quiet:
            print("Draft mode: full MTP layer (NextNSglangCPUDraftAdapter — MLA + MoE, CPU)", flush=True)

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = len(trace.step_next_tokens)
        if not args.quiet:
            print(f"Using full record decode length: max_new_tokens={max_new_tokens}", flush=True)

    cfg = EagleConfig(
        depth=args.depth,
        top_k=args.top_k,
        num_steps=args.num_steps,
        max_paths=args.max_paths,
        verification_mode=args.verification_mode,
        verification_acceptance=args.verification_acceptance,
        verbose=args.verbose,
        verbose_shapes=args.verbose_shapes,
        log_every_steps=args.log_every_steps,
        random_seed=args.random_seed,
        draft_mtp_greedy=args.draft_mtp_greedy,
        draft_branching=args.draft_branching,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_sglang_cpu_batch_beams=not args.sglang_cpu_no_batch_beams,
        log_base_confidence=args.log_base_confidence,
        log_round_replay_detail=args.log_round_replay_detail,
    )
    engine = EagleEngine(base=base, draft=draft, cfg=cfg)
    result = engine.generate(prefix_token_ids=trace.prompt_token_ids, max_new_tokens=max_new_tokens)

    st = result.stats

    if args.quiet:
        extra_parts = []
        for j, rate in enumerate(st.additional_token_match_rates):
            label = f"t{j + 3}_match"
            extra_parts.append(f"{label}={rate:.4f}")
        extra_str = " ".join(extra_parts)
        if extra_str:
            extra_str = " " + extra_str

        bucket_parts = []
        for L, b in enumerate(st.base_next_by_accept_len):
            bucket_parts.append(f"L{L}:n={b.count},pm={b.mean_p_max:.4f},ent={b.mean_entropy:.4f}")
        bucket_line = " ".join(bucket_parts)

        print(
            f"acceptance_rate={st.acceptance_rate:.4f} "
            f"accepted_pct={st.accepted_tokens_percentage:.4f} "
            f"ft_match={st.first_token_match_rate:.4f} "
            f"st_match={st.second_token_match_rate:.4f}"
            f"{extra_str} "
            f"spec_rounds={st.total_rounds_with_paths} "
            f"multi_accept_saves={st.speculation_rounds_saved_by_multi_accept} "
            f"elapsed_s={st.elapsed_s:.2f} "
            f"uniq_draft_avg={st.avg_unique_draft_tokens_per_round:.2f} "
            f"uniq_draft_max={st.max_unique_draft_tokens_per_round} "
            f"{bucket_line}\n"
            f"{result.generated_text}",
            flush=True,
        )
    else:
        print("=" * 60)
        print(f"Record: {record_path}")
        print(f"NextN:  {nextn_dir}")
        print(f"depth={args.depth} top_k={args.top_k} max_paths={args.max_paths}")

        prefix_text = base.decode_tokens(list(trace.prompt_token_ids))
        print(f"Prefix ({len(trace.prompt_token_ids)} tokens): {prefix_text!r}")
        print(f"Generated ({len(result.generated_token_ids)} tokens): {result.generated_text!r}")

        print(f"\nAcceptance rate:           {st.acceptance_rate:.4f}")
        print(f"First-token match rate:    {st.first_token_match_rate:.4f}")
        print(f"Second-token match rate:   {st.second_token_match_rate:.4f}")
        ordinals = ["Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"]
        for j, rate in enumerate(st.additional_token_match_rates):
            label = ordinals[j] if j < len(ordinals) else f"Pos-{j + 3}"
            print(f"{label}-token match rate:  {rate:.4f}")
        print(f"Speculation rounds:        {st.total_rounds_with_paths}")
        print(f"Multi-accept saves:        {st.speculation_rounds_saved_by_multi_accept}")
        print(f"Bonus tokens committed:    {st.bonus_tokens_committed}")
        print(
            f"Unique draft tokens/round: avg={st.avg_unique_draft_tokens_per_round:.2f} "
            f"max={st.max_unique_draft_tokens_per_round}"
        )
        print(f"Elapsed:                   {st.elapsed_s:.2f}s")

        print("\nBase softmax at round start, by accept_len:")
        for L, b in enumerate(st.base_next_by_accept_len):
            print(f"  {_fmt_next_logit_bucket(f'accept_len=={L}', b)}")

        batch_note = (
            f"sglang_cpu_batch_beams={'on' if cfg.draft_sglang_cpu_batch_beams else 'off'}"
            if not args.sglang_draft_structure
            else ""
        )
        draft_note = ""
        if args.draft_branching != "top_k":
            draft_note = (
                f"draft_branching={args.draft_branching} T={args.draft_temperature} top_p={args.draft_top_p}"
            )
        if batch_note or draft_note:
            print(f"\n{batch_note}  {draft_note}".strip())

        print()
        print(format_draft_depth_pmax_summary(args.depth, st.draft_depth_mean_pmax_by_accept_len))
        print("=" * 60)


if __name__ == "__main__":
    main()
