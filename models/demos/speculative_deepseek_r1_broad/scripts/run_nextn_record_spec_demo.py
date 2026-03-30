#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""**Record base** + **NextN MTP draft** (from Hugging Face), verbose speculative rounds.

**NextN** — ``lmsys/DeepSeek-R1-NextN`` is downloaded via ``huggingface_hub``. Embed + lm head
either live inside ``nextn_layer_parameters.safetensors`` or you pass
``--embed-head-aux-safetensors`` (small file; see ``load_nextn_mtp_auxiliary_safetensors``).
This tree does **not** load a full DeepSeek checkpoint.

**Record** — Defaults to this workspace's MTP capture path (see ``default_paths.py``).

Uses **full MTP CPU** draft (:class:`NextNSglangCPUDraftAdapter`). For each draft+verify cycle
(default depth 2, top_k 4): prints draft confidence per slot,
base/record softmax stats at the matching trace position, and how many draft tokens were
accepted (0…depth). Stops after 128 new tokens unless overridden.

Requires ``logits`` in the .pt for meaningful ``base_p_*``; otherwise verification uses
synthetic one-hot logits.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.default_paths import DEFAULT_HF_HOME, DEFAULT_MTP_RECORD_PATH, NEXTN_HF_REPO_ID
from models.demos.speculative_deepseek_r1_broad.hf_cache import (
    bootstrap_hf_env_before_transformers,
    ensure_nextn_snapshot_has_modeling_deepseek,
    set_hf_home,
)

bootstrap_hf_env_before_transformers(sys.argv, default_hf_home=DEFAULT_HF_HOME)

from huggingface_hub import snapshot_download

from models.demos.speculative_deepseek_r1_broad.config import EagleConfig, PathProposal
from models.demos.speculative_deepseek_r1_broad.eagle_engine import (
    draft_per_depth_max_p_max,
    finalize_draft_depth_means,
    format_draft_depth_pmax_summary,
)
from models.demos.speculative_deepseek_r1_broad.local_hf_snapshot import verify_record_dims_vs_snapshot
from models.demos.speculative_deepseek_r1_broad.nextn_sglang_cpu_draft import NextNSglangCPUDraftAdapter
from models.demos.speculative_deepseek_r1_broad.trace_replay_base import (
    TraceReplayBaseAdapter,
    format_mtp_prefix_banner,
    load_trace_or_mtp_reference,
)


def _ensure_nextn_snapshot(hub_cache: Path) -> tuple[Path, Path]:
    """Download ``DeepSeek-R1-NextN`` if needed; return (snapshot_dir, fusion_safetensors)."""
    print(f"Loading NextN from Hugging Face ({NEXTN_HF_REPO_ID}) into cache_dir={hub_cache} …", flush=True)
    # cache_dir required: HF_HUB_CACHE is fixed at huggingface_hub import time, before set_hf_home in main().
    nextn_dir = Path(snapshot_download(repo_id=NEXTN_HF_REPO_ID, cache_dir=str(hub_cache)))
    ensure_nextn_snapshot_has_modeling_deepseek(nextn_dir, hub_cache=hub_cache)
    nextn_file = nextn_dir / "nextn_layer_parameters.safetensors"
    if not nextn_file.is_file():
        raise SystemExit(f"Missing NextN fusion weights in snapshot: {nextn_file}")
    return nextn_dir, nextn_file


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record base + NextN draft (HF) — verbose speculative-decoding demo",
    )
    p.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help=(
            f"HF cache root for this process. Default: {DEFAULT_HF_HOME} (ignores shell HF_HOME)."
        ),
    )
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
        help="Optional embed+head .safetensors if not present in NextN fusion file",
    )
    p.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        default=True,
        help="Disable trust_remote_code for tokenizer load (default: on)",
    )
    p.add_argument("--depth", type=int, default=2)
    p.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="With temperature_top_p: 0 = full nucleus; else cap on sampled branches.",
    )
    p.add_argument(
        "--max-paths",
        type=int,
        default=16,
        help="Beam cap (0 = none). Default 16 matches depth-2 top-4^2 for top_k mode.",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=1, help="Inner speculative rounds per outer step (usually 1).")
    p.add_argument(
        "--verification-acceptance",
        type=str,
        default="argmax",
        choices=["argmax", "probabilistic"],
    )
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument(
        "--draft-branching",
        type=str,
        default="top_k",
        choices=["top_k", "temperature_top_p"],
        help="Draft branches: top_k (default) or temperature+top-p nucleus sampling.",
    )
    p.add_argument("--draft-temperature", type=float, default=0.6)
    p.add_argument("--draft-top-p", type=float, default=0.95)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Draft (full MTP CPU) weight/compute dtype.",
    )
    p.add_argument("--skip-alignment-check", action="store_true")
    p.add_argument(
        "--draft-extend-min-p-max",
        type=float,
        default=None,
        help="MTP draft: if max drafted-token prob at a new step is below this, stop extending.",
    )
    p.add_argument(
        "--base-skip-spec-p-max",
        type=float,
        default=None,
        help="If base next-token p_max is below this, skip drafting and take one base token.",
    )
    return p


def main() -> None:
    args = create_parser().parse_args()
    if args.hf_home is not None:
        hp = set_hf_home(args.hf_home)
    else:
        hp = set_hf_home(DEFAULT_HF_HOME)
    print(
        f"HF_HOME={hp} HF_HUB_CACHE={hp / 'hub'} (use --hf-home to override)",
        flush=True,
    )

    record_path = Path(args.record).expanduser().resolve()
    if not record_path.is_file():
        raise SystemExit(f"Record not found: {record_path}")

    embed_aux: Path | None = None
    if args.embed_head_aux_safetensors is not None:
        embed_aux = Path(args.embed_head_aux_safetensors).expanduser().resolve()
        if not embed_aux.is_file():
            raise SystemExit(f"embed_head_aux_safetensors not found: {embed_aux}")

    nextn_dir, nextn_file = _ensure_nextn_snapshot(hp / "hub")
    trace = load_trace_or_mtp_reference(record_path, batch_index=args.batch_index)
    mtp_banner = format_mtp_prefix_banner(trace, batch_index=args.batch_index)
    if mtp_banner:
        print(mtp_banner, flush=True)
    base = TraceReplayBaseAdapter(
        trace,
        tokenizer_local_dir=nextn_dir,
        tokenizer_trust_remote_code=args.trust_remote_code,
    )

    if trace.step_next_logits is None:
        print(
            "\n*** WARNING: record has no `logits` tensor — base_p_record / base_p_max are from "
            "synthetic one-hot logits (not meaningful). Regenerate the .pt with logits for real stats.\n",
            flush=True,
        )

    if not args.skip_alignment_check:
        all_ids = list(trace.prompt_token_ids) + list(trace.step_next_tokens)
        max_tid = max(all_ids) if all_ids else 0
        verify_record_dims_vs_snapshot(
            record_hidden_size=int(trace.step_last_hidden.shape[-1]),
            record_max_token_id=max_tid,
            snapshot_dir=nextn_dir,
        )
        print("Record vs NextN config.json: OK", flush=True)

    draft = NextNSglangCPUDraftAdapter(
        device=args.device,
        torch_dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        draft.bind_from_nextn_paths(
            nextn_safetensors=nextn_file,
            embed_head_aux_safetensors=embed_aux,
            nextn_config_dir=nextn_dir,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(
            f"{exc}\n"
            "If fusion lacks embed/head, pass --embed-head-aux-safetensors (see "
            "load_nextn_mtp_auxiliary_safetensors)."
        ) from exc

    cfg = EagleConfig(
        depth=args.depth,
        top_k=args.top_k,
        num_steps=args.num_steps,
        max_paths=args.max_paths,
        verification_mode="cache_per_path",
        verification_acceptance=args.verification_acceptance,
        verbose=False,
        verbose_shapes=False,
        log_every_steps=0,
        random_seed=args.random_seed,
        draft_mtp_greedy=False,
        log_base_confidence=False,
        draft_branching=args.draft_branching,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_extend_min_p_max=args.draft_extend_min_p_max,
        base_skip_speculation_p_max=args.base_skip_spec_p_max,
    )

    import random

    rng = random.Random(cfg.random_seed) if cfg.random_seed is not None else None
    draft_gen = torch.Generator(device="cpu")
    if cfg.random_seed is not None:
        draft_gen.manual_seed(int(cfg.random_seed))
    committed = list(trace.prompt_token_ids)
    decode_state = base.forward_prefill(committed)
    generated: list[int] = []
    max_new = min(args.max_new_tokens, len(trace.step_next_tokens))
    spec_round = 0
    verify_unique_token_sum = 0
    verify_unique_token_max = 0
    verify_rounds = 0
    total_proposed_tokens = 0
    t0 = time.perf_counter()

    print(
        f"\nStarting: max_new_tokens={max_new} depth={cfg.depth} top_k={cfg.top_k} "
        f"max_paths={cfg.max_paths} draft_branching={cfg.draft_branching} "
        f"draft_temperature={cfg.draft_temperature} draft_top_p={cfg.draft_top_p} "
        f"verify={cfg.verification_acceptance}\n",
        flush=True,
    )

    depth_i = max(1, int(cfg.depth))
    n_accept_buckets = depth_i + 1
    accept_len_counts = [0] * n_accept_buckets
    draft_d_sums = [[0.0] * depth_i for _ in range(n_accept_buckets)]
    draft_d_cnts = [[0] * depth_i for _ in range(n_accept_buckets)]

    while len(generated) < max_new:
        made_progress = False
        for _ in range(max(1, cfg.num_steps)):
            if len(generated) >= max_new:
                break

            base_skip_thr = getattr(cfg, "base_skip_speculation_p_max", None)
            if base_skip_thr is not None:
                p1 = F.softmax(decode_state.next_token_logits.float().reshape(-1), dim=-1)
                bpm = float(p1.max().item())
                if bpm < base_skip_thr:
                    fb = base.decode_state_next_token(decode_state)
                    meta_sk = decode_state.past_key_values if isinstance(decode_state.past_key_values, dict) else {}
                    tp_sk = int(meta_sk.get("pos", 0))
                    print("-" * 72, flush=True)
                    print(
                        f"BASE_SKIP_SPEC | trace_pos={tp_sk} | base_p_max={bpm:.6f} < {base_skip_thr} | "
                        f"token={fb}",
                        flush=True,
                    )
                    if fb < 0:
                        break
                    committed.append(fb)
                    generated.append(fb)
                    made_progress = True
                    if len(generated) < max_new:
                        decode_state = base.forward_decode(decode_state, fb)
                    break

            proposal = draft.propose_paths(
                committed,
                cfg,
                decode_state=decode_state,
                base_adapter=base,
                draft_torch_generator=draft_gen,
            )
            if isinstance(proposal, PathProposal):
                paths = proposal.paths
                draft_probs_per_path = proposal.draft_probs_per_path
            else:
                paths = proposal
                draft_probs_per_path = None
            if not paths:
                break

            spec_round += 1
            verify_rounds += 1
            total_proposed_tokens += sum(len(p) for p in paths)
            meta0 = decode_state.past_key_values if isinstance(decode_state.past_key_values, dict) else {}
            trace_pos = int(meta0.get("pos", 0))
            unique_spec_tokens_this_round = len({int(t) for p in paths for t in p})
            verify_unique_token_sum += unique_spec_tokens_this_round
            verify_unique_token_max = max(verify_unique_token_max, unique_spec_tokens_this_round)

            use_draft_probs = (
                cfg.verification_acceptance == "probabilistic" and draft_probs_per_path is not None
            )
            verification = base.verify_paths_from_decode_state(
                decode_state,
                paths,
                acceptance_mode=cfg.verification_acceptance,
                rng=rng,
                draft_probs_per_path=draft_probs_per_path if use_draft_probs else None,
                return_base_argmax=True,
            )

            best_idx = -1
            best_len = -1
            for path_idx, accepted_len in enumerate(verification.accepted_prefix_lengths):
                if accepted_len > best_len:
                    best_len = accepted_len
                    best_idx = path_idx
            if best_len >= 0 and verification.base_argmax_pos0 is not None:
                for path_idx, accepted_len in enumerate(verification.accepted_prefix_lengths):
                    if accepted_len == best_len and len(verification.proposed_paths[path_idx]) > 0:
                        if verification.proposed_paths[path_idx][0] == verification.base_argmax_pos0:
                            best_idx = path_idx
                            break

            bucket = min(best_len, depth_i) if best_len >= 0 else 0
            accept_len_counts[bucket] += 1

            per_depth = draft_per_depth_max_p_max(paths, draft_probs_per_path, depth_i)
            if 0 <= best_len < n_accept_buckets:
                for d in range(depth_i):
                    v = per_depth[d]
                    if not math.isnan(v):
                        draft_d_sums[best_len][d] += v
                        draft_d_cnts[best_len][d] += 1
            elif best_len >= n_accept_buckets:
                for d in range(depth_i):
                    v = per_depth[d]
                    if not math.isnan(v):
                        draft_d_sums[-1][d] += v
                        draft_d_cnts[-1][d] += 1

            sel_path = verification.proposed_paths[best_idx] if best_idx >= 0 else []
            sel_probs = (
                draft_probs_per_path[best_idx]
                if draft_probs_per_path is not None and best_idx >= 0 and best_idx < len(draft_probs_per_path)
                else None
            )

            print("=" * 72, flush=True)
            print(
                f"SPEC_ROUND {spec_round} | trace_pos={trace_pos} | "
                f"generated={len(generated)}/{max_new} | chosen_path_idx={best_idx}",
                flush=True,
            )
            print(f"  accepted_token_count={best_len} (depth={cfg.depth})", flush=True)
            print(
                f"  unique_spec_tokens_passed_to_verify={unique_spec_tokens_this_round}",
                flush=True,
            )
            for k in range(cfg.depth):
                dt = int(sel_path[k]) if k < len(sel_path) else -1
                if sel_probs is not None and k < len(sel_probs):
                    dq = float(sel_probs[k])
                else:
                    dq = float("nan")
                st = base.softmax_stats_at_trace_pos(trace_pos + k)
                if st is not None:
                    print(
                        f"  depth[{k}]: draft_tok={dt} draft_p={dq:.6e} | "
                        f"record_greedy_tok={st.record_greedy_token_id} "
                        f"base_p_record={st.p_record_greedy:.6e} base_p_max={st.p_max:.6e} "
                        f"base_argmax_tok={st.argmax_token_id}",
                        flush=True,
                    )
                else:
                    print(
                        f"  depth[{k}]: draft_tok={dt} draft_p={dq:.6e} | record_pos OOB",
                        flush=True,
                    )
            print(f"  all_paths_accept_lens={verification.accepted_prefix_lengths}", flush=True)

            if best_idx >= 0 and best_len > 0:
                accepted = verification.proposed_paths[best_idx][:best_len]
                for token_id in accepted:
                    if len(generated) >= max_new:
                        break
                    committed.append(token_id)
                    generated.append(token_id)
                    made_progress = True
                    if len(generated) < max_new:
                        decode_state = base.forward_decode(decode_state, token_id)
                if len(accepted) > 1:
                    break
                continue

            break

        if made_progress:
            continue

        fb = base.decode_state_next_token(decode_state)
        meta = decode_state.past_key_values if isinstance(decode_state.past_key_values, dict) else {}
        tp = int(meta.get("pos", 0))
        print("-" * 72, flush=True)
        print(
            f"FALLBACK | trace_pos={tp} | take record token {fb}",
            flush=True,
        )
        if fb < 0:
            break
        committed.append(fb)
        generated.append(fb)
        if len(generated) < max_new:
            decode_state = base.forward_decode(decode_state, fb)

    elapsed = time.perf_counter() - t0
    text = base.decode_tokens(generated)
    dd_by_accept = tuple(
        finalize_draft_depth_means(draft_d_sums[i], draft_d_cnts[i]) for i in range(n_accept_buckets)
    )
    accepted_total = sum(accept_len_counts[k] * k for k in range(n_accept_buckets))
    print("\n" + "=" * 72, flush=True)
    print(f"DONE: generated {len(generated)} tokens in {elapsed:.2f}s ({len(generated)/elapsed:.2f} tok/s)", flush=True)
    print(f"Decoded: {text[:200]!r}{'...' if len(text) > 200 else ''}", flush=True)
    print(f"\nVerification rounds: {verify_rounds}", flush=True)
    print(f"Total proposed draft token slots: {total_proposed_tokens}", flush=True)
    print(f"Total accepted draft tokens: {accepted_total}", flush=True)
    print(f"Accept-length histogram (per verification round):", flush=True)
    for k in range(n_accept_buckets):
        print(f"  accepted_len=={k}: {accept_len_counts[k]} rounds", flush=True)
    avg_unique = (verify_unique_token_sum / verify_rounds) if verify_rounds > 0 else 0.0
    print(
        f"Speculative diversity at verification: "
        f"avg_unique_spec_tokens_per_round={avg_unique:.2f} "
        f"max_unique_spec_tokens_per_round={verify_unique_token_max}",
        flush=True,
    )
    print(format_draft_depth_pmax_summary(cfg.depth, dd_by_accept), flush=True)


if __name__ == "__main__":
    main()
