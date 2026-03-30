#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Case study: **record base** + **full NextN Hugging Face stack** as draft.

This does **not** replace ``run_nextn_mtp_from_record_cpu.py``. That script’s default uses
:class:`NextNSglangCPUDraftAdapter` (full MTP layer on CPU).

**Default draft here (SGLang ``DeepseekModelNextN`` order):** :class:`NextNSglangStructureDraftAdapter`
— ``eh_proj`` / ``enorm`` / ``hnorm`` on ``decode_state.last_hidden_state`` (trace) for the first
draft step, then **draft post-norm hidden** for deeper steps; then the same Hub
``DeepseekV3DecoderLayer`` + ``model.norm`` + ``lm_head`` as SGLang.

**Legacy:** pass ``--ids-only-draft`` for :class:`NextNFullHuggingfaceDraftAdapter` only:
``AutoModelForCausalLM`` on **committed token ids** (no fusion on record hidden).

**Caveats**

- **Memory / device:** the NextN checkpoint is a large MoE-class block. **CPU is supported**
  (defaults: ``--device cpu --dtype float32``) but needs **enough RAM**; use
  ``--draft-mtp-greedy``, ``--depth 1``, ``--max-new-tokens 8``, and ``-q`` for lighter runs.
  On GPU, prefer ``--dtype bfloat16``.
- **FP8 / dtypes:** On **CPU/MPS** the adapter strips ``quantization_config`` and **block-dequantizes** FP8
  checkpoint weights to ``--dtype`` so matmuls do not mix float32 activations with FP8 parameters.
- **MoE vs config:** Single-layer NextN checkpoints use **MoE** tensors; the adapter sets ``first_k_dense_replace=0``
  when needed so those weights load (large RAM).
- **Embed / head / final norm:** If ``DEFAULT_EMBED_HEAD_AUX_PATH`` exists it is used automatically; override with
  ``--embed-head-aux-safetensors``. Final RMS norm is copied from ``shared_head.norm`` in the NextN shard when present.
- **Record hidden (default):** fusion mats condition on ``decode_state.last_hidden_state`` like
  SGLang; use ``--ids-only-draft`` to disable and match the old ids-only experiment.
- **How long it runs:** ``--max-new-tokens`` counts **new** tokens after the prompt (same as
  ``run_nextn_mtp_from_record_cpu.py``). Omit it to decode **all** ``len(trace.step_next_tokens)``
  steps. Printed ``generated_text`` is **decode(new_tokens_only)** — not prompt+completion; decode
  ``prompt_token_ids + generated_token_ids`` yourself for the full string.
- **Fusion-parity flags:** this script accepts the same optional Eagle controls as the MTP script
  (``--verbose``, ``--random-seed``, ``--log-base-confidence``, ``--base-skip-spec-p-max``, …).
  ``--draft-extend-min-p-max`` is ignored here (case study uses full HF draft adapters only).
- **transformers dynamic modules:** set ``HF_MODULES_CACHE`` to a disk with space *before*
  running if ``/home`` is full (this script sets it under ``HF_HOME`` / ``--hf-home`` when possible).

See ``case_studies/nextn_full_layer_draft_from_record.md`` for design notes.

Example (CPU — defaults, keep work small)::

  export PYTHONPATH=/path/to/tt-metal:$PYTHONPATH
  python models/demos/speculative_deepseek_r1_broad/scripts/case_study_nextn_full_layer_draft_from_record_cpu.py \\
    --draft-mtp-greedy --depth 1 --max-new-tokens 8 -q

Example (GPU)::

  python models/demos/speculative_deepseek_r1_broad/scripts/case_study_nextn_full_layer_draft_from_record_cpu.py \\
    --device cuda --dtype bfloat16 --draft-mtp-greedy --depth 2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.demos.speculative_deepseek_r1_broad.default_paths import (
    DEFAULT_EMBED_HEAD_AUX_PATH,
    DEFAULT_HF_HOME,
    DEFAULT_MTP_RECORD_PATH,
    NEXTN_HF_REPO_ID,
)
from models.demos.speculative_deepseek_r1_broad.hf_cache import (
    bootstrap_hf_env_before_transformers,
    ensure_nextn_snapshot_has_modeling_deepseek,
    set_hf_home,
)

bootstrap_hf_env_before_transformers(sys.argv, default_hf_home=DEFAULT_HF_HOME)

from models.demos.speculative_deepseek_r1_broad.config import EagleConfig
from models.demos.speculative_deepseek_r1_broad.eagle_engine import EagleEngine
from models.demos.speculative_deepseek_r1_broad.local_hf_snapshot import verify_record_dims_vs_snapshot
from models.demos.speculative_deepseek_r1_broad.trace_replay_base import (
    TraceReplayBaseAdapter,
    format_mtp_prefix_banner,
    load_trace_or_mtp_reference,
)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Case study: full HF NextN draft + record replay base")
    p.add_argument("--hf-home", type=Path, default=None, help=f"HF root (default {DEFAULT_HF_HOME})")
    p.add_argument(
        "--record",
        type=Path,
        default=DEFAULT_MTP_RECORD_PATH,
        help=f"MTP reference .pt (replay base). Default: {DEFAULT_MTP_RECORD_PATH}",
    )
    p.add_argument("--batch-index", type=int, default=0)
    p.add_argument(
        "--nextn-repo-id",
        type=str,
        default=NEXTN_HF_REPO_ID,
        help="Hub id or local snapshot directory for full NextN model",
    )
    p.add_argument(
        "--nextn-local-only",
        action="store_true",
        help="Pass local_files_only=True (``--nextn-repo-id`` must be a snapshot directory)",
    )
    p.add_argument("--depth", type=int, default=2)
    p.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="With temperature_top_p: 0 expands full nucleus; else max sampled branches.",
    )
    p.add_argument("--max-paths", type=int, default=16, help="0 = no beam cap.")
    p.add_argument(
        "--ids-only-draft",
        action="store_true",
        help=(
            "Legacy draft: embed → decoder → norm → lm_head on token ids only (ignores record "
            "last_hidden; not SGLang MTP). Default uses eh_proj/enorm/hnorm + same stack (see "
            "NextNSglangStructureDraftAdapter)."
        ),
    )
    p.add_argument("--draft-mtp-greedy", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--num-steps", type=int, default=1)
    p.add_argument(
        "--verification-mode",
        type=str,
        default="cache_per_path",
        choices=["cache_per_path", "batched_single_pass", "flattened_tree"],
        help="Trace replay: see run_nextn_mtp_from_record_cpu.py doc (flattened_tree is replay-equivalent).",
    )
    p.add_argument("--verification-acceptance", type=str, default="argmax", choices=["argmax", "probabilistic"])
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu (default) or cuda. NextN MoE draft on CPU needs substantial RAM.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="float32 recommended on CPU; bfloat16/float16 on GPU.",
    )
    p.add_argument(
        "--no-trust-remote-code",
        dest="trust_remote_code",
        action="store_false",
        default=True,
        help="Disable trust_remote_code for tokenizer / model load",
    )
    p.add_argument("--skip-alignment-check", action="store_true")
    p.add_argument(
        "--embed-head-aux-safetensors",
        type=Path,
        default=None,
        help=(
            "Optional .safetensors with embed_tokens + shared_head.head (see materialize_nextn_embed_head_aux_from_r1_shards.py). "
            f"If omitted and {DEFAULT_EMBED_HEAD_AUX_PATH} exists, that path is used."
        ),
    )
    p.add_argument(
        "--decoder-layer0-override-safetensors",
        type=Path,
        default=None,
        help=(
            "Optional .safetensors of model.layers.0.* remapped from a main R1 decoder layer — see "
            "materialize_r1_decoder_layer_as_nextn_layer0.py (default layer index 60, not 61)."
        ),
    )
    # Same optional engine / logging knobs as run_nextn_mtp_from_record_cpu.py
    p.add_argument("--random-seed", type=int, default=None)
    p.add_argument(
        "--draft-branching",
        type=str,
        default="top_k",
        choices=["top_k", "temperature_top_p"],
        help="Draft candidate selection (see run_nextn_mtp_from_record_cpu.py).",
    )
    p.add_argument("--draft-temperature", type=float, default=0.6)
    p.add_argument("--draft-top-p", type=float, default=0.95)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--verbose-shapes", action="store_true", default=False)
    p.add_argument("--log-every-steps", type=int, default=0)
    p.add_argument(
        "--log-base-confidence",
        action="store_true",
        default=False,
        help="Print [base_conf] lines when the record has step_next_logits.",
    )
    p.add_argument(
        "--log-round-replay-detail",
        action="store_true",
        help="Per-round record greedy vs draft vs verify (trace replay).",
    )
    p.add_argument(
        "--draft-extend-min-p-max",
        type=float,
        default=None,
        help="Ignored for full HF NextN draft (MTP adapter only). Passed for CLI parity with fusion script.",
    )
    p.add_argument(
        "--base-skip-spec-p-max",
        type=float,
        default=None,
        help="If base next-token p_max < this, skip drafting for that round (see EagleConfig).",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    return p


def main() -> None:
    args = create_parser().parse_args()
    if args.quiet:
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    if args.hf_home is not None:
        hp = set_hf_home(args.hf_home)
    else:
        hp = set_hf_home(DEFAULT_HF_HOME)

    # Lazy imports that pull in transformers (bootstrap_hf_env_before_transformers already ran at startup).
    from huggingface_hub import snapshot_download

    from models.demos.speculative_deepseek_r1_broad.nextn_full_layer_draft import NextNFullHuggingfaceDraftAdapter
    from models.demos.speculative_deepseek_r1_broad.nextn_sglang_structure_draft import NextNSglangStructureDraftAdapter

    record_path = Path(args.record).expanduser().resolve()
    if not record_path.is_file():
        raise SystemExit(f"Record not found: {record_path}")

    hub_cache = hp / "hub"
    if not args.nextn_local_only:
        if not args.quiet:
            print(f"Downloading NextN snapshot for full draft: {args.nextn_repo_id} -> {hub_cache}", flush=True)
        nextn_dir = Path(snapshot_download(repo_id=args.nextn_repo_id, cache_dir=str(hub_cache)))
        model_path = str(nextn_dir)
    else:
        model_path = str(Path(args.nextn_repo_id).expanduser().resolve())
        nextn_dir = Path(model_path)
        if not nextn_dir.is_dir():
            raise SystemExit(f"--nextn-local-only: not a directory: {nextn_dir}")

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

    all_ids = list(trace.prompt_token_ids) + list(trace.step_next_tokens)
    max_tid = max(all_ids) if all_ids else 0
    rec_h = int(trace.step_last_hidden.shape[-1])
    if not args.skip_alignment_check:
        verify_record_dims_vs_snapshot(
            record_hidden_size=rec_h,
            record_max_token_id=max_tid,
            snapshot_dir=nextn_dir,
        )

    embed_aux = args.embed_head_aux_safetensors
    if embed_aux is None and DEFAULT_EMBED_HEAD_AUX_PATH.is_file():
        embed_aux = DEFAULT_EMBED_HEAD_AUX_PATH
        if not args.quiet:
            print(f"Using default embed/head aux: {embed_aux}", flush=True)
    elif embed_aux is not None:
        embed_aux = Path(embed_aux).expanduser().resolve()

    layer0_override = args.decoder_layer0_override_safetensors
    if layer0_override is not None:
        layer0_override = Path(layer0_override).expanduser().resolve()
        if not layer0_override.is_file():
            raise SystemExit(f"--decoder-layer0-override-safetensors not found: {layer0_override}")
        if not args.quiet:
            print(f"Decoder layer0 override: {layer0_override}", flush=True)

    if args.ids_only_draft:
        draft = NextNFullHuggingfaceDraftAdapter(
            model_id_or_path=model_path,
            device=args.device,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.nextn_local_only,
            embed_head_aux_safetensors=embed_aux,
            decoder_layer0_override_safetensors=layer0_override,
        )
        if not args.quiet:
            print("Draft mode: ids-only (legacy NextNFullHuggingfaceDraftAdapter)", flush=True)
    else:
        nextn_fusion_file = nextn_dir / "nextn_layer_parameters.safetensors"
        if not nextn_fusion_file.is_file():
            raise SystemExit(
                f"SGLang-order draft needs fusion weights: missing {nextn_fusion_file} "
                "(use --ids-only-draft if you only have partial snapshot files)."
            )
        draft = NextNSglangStructureDraftAdapter(
            device=args.device,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.nextn_local_only,
        )
        draft.bind_from_nextn_paths(
            nextn_safetensors=nextn_fusion_file,
            embed_head_aux_safetensors=embed_aux,
            nextn_config_dir=nextn_dir,
            decoder_layer0_override_safetensors=layer0_override,
        )
        if not args.quiet:
            print(
                "Draft mode: SGLang order (eh_proj/enorm/hnorm + record last_hidden → layer0 → norm → lm_head)",
                flush=True,
            )

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = len(trace.step_next_tokens)
        if not args.quiet:
            print(f"Using full record decode length: max_new_tokens={max_new_tokens}", flush=True)
    elif not args.quiet:
        print(
            f"max_new_tokens={max_new_tokens} (new tokens only; record has {len(trace.step_next_tokens)} steps). "
            "Omit --max-new-tokens to match full trace.",
            flush=True,
        )

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
        log_base_confidence=args.log_base_confidence,
        log_round_replay_detail=args.log_round_replay_detail,
        draft_extend_min_p_max=args.draft_extend_min_p_max,
        base_skip_speculation_p_max=args.base_skip_spec_p_max,
    )
    engine = EagleEngine(base=base, draft=draft, cfg=cfg)
    result = engine.generate(prefix_token_ids=trace.prompt_token_ids, max_new_tokens=max_new_tokens)

    if args.quiet:
        print(
            f"acceptance_rate={result.stats.acceptance_rate:.4f} "
            f"accepted_pct={result.stats.accepted_tokens_percentage:.4f}\n{result.generated_text}",
            flush=True,
        )
    else:
        full_ids = list(trace.prompt_token_ids) + list(result.generated_token_ids)
        full_text = base.decode_tokens(full_ids)
        print("=" * 60)
        print(
            "CASE STUDY: NextN draft + record replay base (default=SGLang fusion+stack; "
            "--ids-only-draft for legacy)"
        )
        print(f"Record: {record_path}")
        print(f"NextN model path: {model_path}")
        print(f"Device/dtype: {args.device} {args.dtype}")
        print(
            f"Eagle: depth={args.depth} top_k={args.top_k} max_paths={args.max_paths} "
            f"num_steps={args.num_steps} greedy={args.draft_mtp_greedy} verify={args.verification_mode}"
        )
        print(f"New tokens only ({len(result.generated_token_ids)} ids): {result.generated_text!r}")
        print(f"Prompt + new (full string): {full_text!r}")
        print(f"Stats: {result.stats}")
        if not args.ids_only_draft:
            print(
                "\nNote on acceptance: default draft uses **SGLang order** (fusion → decoder → model.norm → lm_head). "
                "Verification still uses **full R1** logits from the record. That distribution differs from "
                "**NextNSglangCPUDraftAdapter** (manual full MTP on CPU), so **low acceptance here "
                "does not imply a bad load**. For alignment checks vs the same .pt, run "
                "`run_nextn_mtp_from_record_cpu.py` without `--sglang-draft-structure`.\n"
                "A long “layers were not sharded” line during load (if any) is from HF/Accelerate checkpoint "
                "grouping, not a failed load; NextNFullHuggingfaceDraftAdapter now suppresses it."
            )
        print("=" * 60)


if __name__ == "__main__":
    main()
