# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Load-once multi-config serving throughput sweep (session 9, path to 100 t/s).

The checkpoint load (safetensors read + remap + upload, ~15-20 min) dominates a
single ``serving_smoke`` run, so measuring N configs as N separate processes wastes
N loads. This driver loads the full 30L model ONCE and runs a list of serving
configs in sequence, each with a fresh :class:`BlockDiffusionServingSession` (so the
traced controller re-captures its per-budget traces) and an explicit
``controller.release()`` between configs (so only one budget's traces occupy the
trace region at a time — max ~2 GB @12).

All ``DG_*`` flags are read live from ``os.environ`` per block, so this driver
toggles them per config (tuned/untuned, traced/eager) between sessions in one process.

Each config: {label, env (dict of DG_* overrides), steps, blocks}. Steady-state
throughput is the mean over blocks[1:] (block 0 pays prefill + trace capture).

Emits one greppable ``SWEEP_RESULT <json>`` line per config and dumps a combined
JSON. Device hygiene: a trace-capture FATAL poisons the device, so a failing config
aborts the remaining sweep (the caller must ``tt-smi -r`` before retrying).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
from loguru import logger

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import (
    _close_mesh_device,
    _log_mesh_dram,
    _open_mesh_device,
)
from models.experimental.diffusion_gemma.demo.serving_smoke import _DeviceGenLike
from models.experimental.diffusion_gemma.tt.generate import decode_generation, tokenize_prompt
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession


# The sweep matrix. env overrides are applied on top of a cleared DG_* baseline.
def default_configs():
    TRACED = {"DG_SPARSE_MOE": "1", "DG_DEDUP_ARGMAX": "1", "DG_SPARSE_MOE_TUNED": "1", "DG_DENOISE_TRACED": "1"}
    UNTUNED = {"DG_SPARSE_MOE": "1", "DG_DEDUP_ARGMAX": "1", "DG_DENOISE_TRACED": "1"}  # tuned OFF
    EAGER = {"DG_SPARSE_MOE": "1", "DG_DEDUP_ARGMAX": "1", "DG_SPARSE_MOE_TUNED": "1"}  # traced OFF
    # Multi-step trace batching (whole-block window = one replay/block); takes precedence over
    # DG_DENOISE_TRACED. Quality-safe: bit-exact to the single-step traced path at the same K.
    MULTISTEP = {
        "DG_SPARSE_MOE": "1",
        "DG_DEDUP_ARGMAX": "1",
        "DG_SPARSE_MOE_TUNED": "1",
        "DG_DENOISE_TRACED_MULTISTEP": "1",
    }
    return [
        # Anchor first: reproduces the session-8 verified 58.29 @12 in THIS harness,
        # so the low-step points below inherit that trust. Then descend toward 100 t/s.
        {"label": "traced_tuned_s12", "env": TRACED, "steps": 12, "blocks": 3},
        {"label": "traced_tuned_s8", "env": TRACED, "steps": 8, "blocks": 3},
        {"label": "traced_tuned_s6", "env": TRACED, "steps": 6, "blocks": 3},
        {"label": "traced_tuned_s5", "env": TRACED, "steps": 5, "blocks": 3},
        {"label": "traced_tuned_s4", "env": TRACED, "steps": 4, "blocks": 3},
        {"label": "traced_tuned_s3", "env": TRACED, "steps": 3, "blocks": 3},
        # Multi-step batching at the SAME budgets as the single-step points above — the
        # committed_sha must match single-step at each K (quality-safe) and the block latency
        # shows how much the per-replay dispatch bubbles flatten toward 100 at a higher K.
        # Ordered SMALLEST-window-first: a whole-block trace-region overflow FATALs and poisons
        # the device, so run the smallest (lowest-risk) capture first to keep the earlier results.
        {"label": "multistep_tuned_s6", "env": MULTISTEP, "steps": 6, "blocks": 3},
        {"label": "multistep_tuned_s8", "env": MULTISTEP, "steps": 8, "blocks": 3},
        {"label": "multistep_tuned_s12", "env": MULTISTEP, "steps": 12, "blocks": 3},
        {"label": "traced_untuned_s12", "env": UNTUNED, "steps": 12, "blocks": 3},
        {"label": "eager_tuned_s12", "env": EAGER, "steps": 12, "blocks": 3},
    ]


_DG_FLAGS = (
    "DG_SPARSE_MOE",
    "DG_SPARSE_MOE_TUNED",
    "DG_DEDUP_ARGMAX",
    "DG_DENOISE_TRACED",
    "DG_DENOISE_TRACED_MULTISTEP",
    "DG_DENOISE_MULTISTEP_GROUP",
    "DG_DENOISE_DEVICE_LOOP",
    "DG_COMMIT_BATCHED",
)


def _apply_env(env: dict) -> None:
    for k in _DG_FLAGS:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v


def _release_controller(session) -> None:
    fn = getattr(session, "_logits_fn", None)
    # Both the single-step (DG_DENOISE_TRACED) and multi-step (DG_DENOISE_TRACED_MULTISTEP)
    # controllers cache their captured traces on the logits fn under distinct attributes; free
    # whichever is present so only one budget's traces occupy the trace region at a time.
    for attr in ("_traced_denoise_controller", "_traced_denoise_multistep_controller"):
        controller = getattr(fn, attr, None)
        if controller is not None:
            try:
                controller.release()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[sweep] {attr}.release failed: {exc}")
            try:
                delattr(fn, attr)
            except Exception:  # noqa: BLE001
                pass


def run_config(bundle, mesh_device, prompt_tokens, cfg_spec, args) -> dict:
    _apply_env(cfg_spec["env"])
    steps = cfg_spec["steps"]
    blocks = cfg_spec["blocks"]
    config = DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=steps)
    session = BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=config,
        tokenizer=bundle.tokenizer,
        gumbel_mode="argmax",
        seed=args.seed,
        stop_token_ids=[],  # disable EOS halt (RUN-first degenerate output)
    )
    try:
        t0 = time.perf_counter()
        session.prefill(prompt_tokens)
        first = session.decode_block()
        ttft_s = time.perf_counter() - t0
        emissions = [first]
        for _ in range(1, blocks):
            emissions.append(session.decode_block())

        block_latencies = [e.latency_s for e in emissions]
        steady = block_latencies[1:] if len(block_latencies) > 1 else block_latencies
        mean_block = sum(steady) / len(steady)
        tps = args.canvas_length / mean_block if mean_block > 0 else 0.0
        committed = torch.cat([e.tokens for e in emissions], dim=1)
        # Hash the committed token ids so multi-step-vs-single-step equivalence at a given K is a
        # hard byte-for-byte check (not just a text-head eyeball). Same seed + budget ⇒ same ids.
        committed_sha = hashlib.sha256(committed.to(torch.int64).cpu().numpy().tobytes()).hexdigest()[:16]
        text = decode_generation(
            bundle.tokenizer,
            prompt_tokens,
            _DeviceGenLike(committed, session.cache_len, session.next_pos),
            skip_prompt=True,
            skip_special_tokens=True,
        )
        text_str = text[0] if text else ""
        result = {
            "label": cfg_spec["label"],
            "env": cfg_spec["env"],
            "steps": steps,
            "blocks": blocks,
            "ttft_s": ttft_s,
            "per_block_latency_s": block_latencies,
            "steady_block_latency_s": mean_block,
            "tokens_per_block_per_s": tps,
            "denoise_steps_per_block": [e.num_denoise_steps for e in emissions],
            "halted_per_block": [e.halted for e in emissions],
            "committed_sha": committed_sha,
            "text_chars": len(text_str),
            "text_head": text_str[:220],
        }
    finally:
        _release_controller(session)
        session.reset()
    logger.info(f"[sweep] {cfg_spec['label']}: {tps:.2f} t/s (block {mean_block:.3f}s, steps {steps})")
    print("SWEEP_RESULT " + json.dumps(result))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    p.add_argument("--mesh", default="P150x4")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    p.add_argument("--canvas-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None, help="comma-separated labels to run (default: all)")
    p.add_argument("--out", default=None, help="combined results JSON path")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configs = default_configs()
    if args.only:
        want = set(args.only.split(","))
        configs = [c for c in configs if c["label"] in want]

    mesh_device = _open_mesh_device(args.mesh)
    results = []
    try:
        _log_mesh_dram(mesh_device, "baseline")
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        t_load = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(mesh_device, args.checkpoint, **model_kwargs)
        logger.info(f"[sweep] model load took {time.perf_counter() - t_load:.1f}s")
        _log_mesh_dram(mesh_device, "post-build")
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        logger.info(f"[sweep] prompt_len={int(prompt_tokens.shape[1])}")

        for cfg_spec in configs:
            try:
                results.append(run_config(bundle, mesh_device, prompt_tokens, cfg_spec, args))
            except BaseException as exc:  # noqa: BLE001
                logger.error(f"DG_SWEEP_CONFIG_FAILURE label={cfg_spec['label']} err={type(exc).__name__}: {exc}")
                # A trace-capture FATAL poisons the device; abort the rest of the sweep.
                raise
    finally:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        _close_mesh_device(mesh_device)
    print("DG_SWEEP_DONE configs=" + str(len(results)))
    for r in results:
        print(f"  {r['label']}: {r['tokens_per_block_per_s']:.2f} t/s  block={r['steady_block_latency_s']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
