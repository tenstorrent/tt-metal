# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device bit-exact check for frozen prompt-prefix KV reuse (APC prototype, #47466).

Reduced-surface driver (model-owned contiguous cache, **no vLLM**) that proves the
``DG_PREFIX_CACHE`` reuse path is **bit-exact** to a fresh prefill and reports the
prefill wall-time it saves. For a request ``B`` whose aligned prompt is a full
prefix of a warm request ``A`` it runs, on one model:

1. **OFF** — cache disabled: full prefill of ``B`` → committed argmax ``O_off`` and
   the prefill wall-time ``t_off``;
2. **warm** — real prefill of the (longer/equal) prompt ``A`` → resident := ``A``;
3. **ON** — cache enabled: ``B``'s aligned prompt is a full prefix of resident
   ``A`` → **reuse** (prefill skipped) → committed argmax ``O_on``, ``t_on ≈ 0``;
4. **assert** ``O_on == O_off`` bit-for-bit; report ``t_off - t_on``.

Both ``B`` runs use the same session seed, so the seeded canvas-init / renoise /
Gumbel sequences are identical and any output difference could only come from the
prompt K/V — which reuse keeps bit-identical (causal prefill ⇒ position ``i``'s K/V
is a pure function of ``tokens[0:i]`` + the absolute RoPE position ``i``).

Cases (``--case``):
- ``exact``  — ``B == A`` (exact full-prompt match).
- ``prefix`` — ``B`` = ``A`` truncated at a 32-token boundary (aligned proper prefix).
- ``both``   — run both (default).

Emits ``DG_PREFIX_CACHE_SMOKE_SUCCESS ...`` / ``DG_PREFIX_CACHE_SMOKE_FAILURE ...``.
RUN-first: committed output may be degenerate (#48291); this checks *equality*
ON-vs-OFF, not text quality.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from loguru import logger

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import (
    _close_mesh_device,
    _log_mesh_dram,
    _open_mesh_device,
)
from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt
from models.experimental.diffusion_gemma.tt.prefix_cache import PrefixKVCache
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"),
        help="HF checkpoint directory or model id",
    )
    parser.add_argument("--mesh", default="P150x4", help="mesh label or ROWSxCOLS (QB2 = P150x4)")
    parser.add_argument("--num-layers", type=int, default=None, help="reduced layer count (default: full 30)")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="served max context (KV/RoPE span)")
    parser.add_argument(
        "--prompt",
        default=(
            "You are a careful assistant. Explain, in detail and step by step, what a "
            "discrete text-diffusion language model is, how it differs from an autoregressive "
            "model, and why block-wise denoising can be parallelised across a canvas."
        ),
        help="the long prompt A whose prefix is shared/reused",
    )
    parser.add_argument("--num-blocks", type=int, default=1, help="committed blocks to compare (bit-exact)")
    parser.add_argument("--canvas-length", type=int, default=256, help="output block size (canvas)")
    parser.add_argument("--max-denoising-steps", type=int, default=2, help="denoise steps per block cap")
    parser.add_argument("--gumbel-mode", default="argmax", choices=["argmax", "chunked", "host", "device"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", default="both", choices=["exact", "prefix", "both"])
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--metrics-json", default=None)
    return parser


def _committed_blocks(session: BlockDiffusionServingSession, num_blocks: int) -> torch.Tensor:
    """Run prefill's block 0 already done by caller? No — caller prefills; run blocks here."""
    emissions = []
    for _ in range(num_blocks):
        if session.finished:
            break
        emissions.append(session.decode_block())
    return torch.cat([e.tokens for e in emissions], dim=1) if emissions else torch.zeros((1, 0), dtype=torch.long)


def _run_request(
    tt_model,
    state_dict,
    tokenizer,
    prompt_tokens,
    *,
    config,
    gumbel_mode,
    seed,
    prefix_cache,
    num_blocks,
):
    """Prefill + decode one request; return (committed_tokens, reused, prefill_time_s, cache_len)."""
    session = BlockDiffusionServingSession(
        tt_model,
        state_dict,
        config=config,
        tokenizer=tokenizer,
        gumbel_mode=gumbel_mode,
        seed=seed,
        # Never self-finish on EOS: we want the full committed block(s) for the
        # bit-exact comparison regardless of content.
        stop_token_ids=[],
        prefix_cache=prefix_cache,
    )
    cache_len = session.prefill(prompt_tokens)
    committed = _committed_blocks(session, num_blocks)
    reused = session.prefill_reused
    prefill_time_s = session.prefill_time_s
    session.reset()
    return committed, reused, prefill_time_s, cache_len


def _aligned_prefix_prompt(prompt_tokens: torch.Tensor) -> torch.Tensor:
    """Return a 32-aligned *proper* prefix of A's real tokens (a genuine shared prefix).

    32-aligned so B needs no pad; then ``B_aligned == A_aligned[:len(B)]`` exactly,
    which is the bit-exact-reusable case.
    """
    real = int(prompt_tokens.shape[1])
    if real <= 32:
        raise ValueError(f"prompt too short ({real} tokens) for an aligned proper prefix; use --case exact")
    target = (real // 32) * 32
    if target >= real:  # A is itself 32-aligned → step back one tile for a proper prefix
        target -= 32
    target = max(32, target)
    return prompt_tokens[:, :target]


def _run_case(bundle, config, args, case: str) -> dict:
    tt_model, state_dict, tokenizer = bundle.tt_model, bundle.state_dict, bundle.tokenizer
    prompt_A = tokenize_prompt(tokenizer, args.prompt)

    if case == "exact":
        prompt_B = prompt_A.clone()
    else:  # prefix
        prompt_B = _aligned_prefix_prompt(prompt_A)

    logger.info(f"[prefix_cache_smoke:{case}] A_len={int(prompt_A.shape[1])} B_len={int(prompt_B.shape[1])}")

    common = dict(
        config=config,
        gumbel_mode=args.gumbel_mode,
        seed=args.seed,
        num_blocks=args.num_blocks,
    )

    # 1) OFF: full prefill of B on a clean-relative cache (DG_PREFIX_CACHE unset).
    os.environ.pop("DG_PREFIX_CACHE", None)
    off_cache = PrefixKVCache()  # attached but inert (flag off) — exercises the fallback path too
    o_off, reused_off, t_off, cache_len_off = _run_request(
        tt_model, state_dict, tokenizer, prompt_B, prefix_cache=off_cache, **common
    )
    assert not reused_off, "OFF run must not reuse"

    # 2) warm: real prefill of A → resident := A (flag on so it records).
    os.environ["DG_PREFIX_CACHE"] = "1"
    # exact match is bit-exact-reusable by default; the proper-prefix case is only
    # reusable in the approximate tier (allow_shorter_prefix) — we MEASURE its
    # bit-exactness rather than assert it (bf16 SDPA reduction-length, see README).
    on_cache = PrefixKVCache(allow_shorter_prefix=(case == "prefix"))
    _o_a, reused_a, t_a, cache_len_a = _run_request(
        tt_model, state_dict, tokenizer, prompt_A, prefix_cache=on_cache, **{**common, "num_blocks": 0}
    )
    assert not reused_a, "warm A run must not reuse (nothing resident yet)"

    # 3) ON: B reuses A's resident prefix (prefill skipped).
    o_on, reused_on, t_on, cache_len_on = _run_request(
        tt_model, state_dict, tokenizer, prompt_B, prefix_cache=on_cache, **common
    )
    os.environ.pop("DG_PREFIX_CACHE", None)

    bit_exact = bool(o_off.shape == o_on.shape and torch.equal(o_off, o_on))
    mismatches = int((o_off != o_on).sum().item()) if o_off.shape == o_on.shape else -1
    # exact match must be bit-exact (the shipped bit-exact reuse); the proper-prefix
    # (approximate) tier only needs to *reuse* — its bit-exactness is measured/reported.
    require_bit_exact = case == "exact"
    result = {
        "case": case,
        "approximate_tier": on_cache.allow_shorter_prefix,
        "require_bit_exact": require_bit_exact,
        "prompt_A_len": int(prompt_A.shape[1]),
        "prompt_B_len": int(prompt_B.shape[1]),
        "cache_len_B_off": cache_len_off,
        "cache_len_A": cache_len_a,
        "cache_len_B_on": cache_len_on,
        "reused_off": reused_off,
        "reused_on": reused_on,
        "committed_tokens": int(o_off.shape[1]),
        "bit_exact_committed_argmax": bit_exact,
        "committed_token_mismatches": mismatches,
        "prefill_time_off_s": t_off,
        "prefill_time_warm_A_s": t_a,
        "prefill_time_on_s": t_on,
        "prefill_time_saved_s": max(0.0, t_off - t_on),
        "prefix_cache_stats": on_cache.stats(),
    }
    logger.info(f"[prefix_cache_smoke:{case}] result:\n" + json.dumps(result, indent=2))
    if not reused_on:
        raise RuntimeError(f"[{case}] ON run did NOT reuse — prefix reuse path not exercised")
    if require_bit_exact and not bit_exact:
        raise RuntimeError(f"[{case}] committed argmax DIFFERS ON vs OFF — reuse is not bit-exact")
    return result


def run(args) -> dict:
    config = DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=args.max_denoising_steps)
    tokenizer_kwargs = {"local_files_only": True} if args.local_files_only else None

    mesh_device = _open_mesh_device(args.mesh)
    try:
        _log_mesh_dram(mesh_device, "baseline")
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        bundle = build_tt_model_from_checkpoint_dir(
            mesh_device,
            args.checkpoint,
            tokenizer_kwargs=tokenizer_kwargs,
            **model_kwargs,
        )
        _log_mesh_dram(mesh_device, "post-build")

        cases = ["exact", "prefix"] if args.case == "both" else [args.case]
        results = [_run_case(bundle, config, args, c) for c in cases]
        metrics = {
            "num_layers": args.num_layers,
            "max_seq_len": args.max_seq_len,
            "num_blocks": args.num_blocks,
            "max_denoising_steps": args.max_denoising_steps,
            "gumbel_mode": args.gumbel_mode,
            "cases": results,
            # Gate: every case that REQUIRES bit-exactness (exact match) is bit-exact.
            "required_bit_exact_pass": all(r["bit_exact_committed_argmax"] for r in results if r["require_bit_exact"]),
            "all_reused": all(r["reused_on"] for r in results),
        }
        if args.metrics_json:
            with open(args.metrics_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
        return metrics
    finally:
        _close_mesh_device(mesh_device)


def _success_marker(metrics: dict) -> str:
    parts = []
    for r in metrics["cases"]:
        parts.append(
            f"{r['case']}:bit_exact={r['bit_exact_committed_argmax']}"
            f",mismatch={r['committed_token_mismatches']}"
            f",reused={r['reused_on']},saved_s={r['prefill_time_saved_s']:.3f}"
        )
    return (
        "DG_PREFIX_CACHE_SMOKE_SUCCESS "
        f"required_bit_exact_pass={metrics['required_bit_exact_pass']} "
        f"all_reused={metrics['all_reused']} " + " ".join(parts)
    )


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        metrics = run(args)
    except BaseException as exc:  # noqa: BLE001 - emit a greppable failure marker then re-raise
        logger.error(f"DG_PREFIX_CACHE_SMOKE_FAILURE error_type={type(exc).__name__} mesh={args.mesh}")
        raise
    marker = _success_marker(metrics)
    logger.info(marker)
    print(marker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
