# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""dg-09 — traced serving decode: eager-vs-traced + early-halt distribution + context/DRAM.

Drives the SERVING path (:class:`~models.experimental.diffusion_gemma.tt.serving.BlockDiffusionServingSession`
— the exact block-emission core the vLLM adapter ``tt/generator_vllm.py`` delegates to) with an
EXPLICIT ``denoise_block_fn`` per config, so it validates the new trace wiring the vLLM adapter
uses (``session.denoise_block_fn`` = ``select_traced_denoise_block_fn()`` when ``enable_trace``),
not just the env-flag path. Loads the model ONCE per ``--max-seq-len`` and reports **per-block**
serving metrics (never ``1000/mean_tpot_ms``):

TASK 2 — eager vs traced at a fixed context (the trace win the vLLM path was missing):
  * ``eager``  = the eager ``denoise_block`` (5 host readbacks/step) — the current serving default.
  * ``traced`` = ``traced_denoise_block`` (Metal capture on block 0, ``execute_trace``-replay/block).
  * ``early_halt`` = ``traced_early_halt_block`` (traced + data-dependent early-halt, dg-08).
  Reports TTFT (prefill+block0), steady per-block latency, tokens/block/s, and DRAM used/free.
  Correctness: the committed-argmax sha of eager / traced / early_halt must match on a prompt that
  runs the full budget in all three (byte-identical → traced serving == the generator's committed
  argmax; eager serving unchanged).

TASK 3 — average early-halt steps/block over a prompt set (``early_halt`` config): the realized
  ``denoise_steps_per_block`` distribution + average. HONEST: early-halt is prompt-dependent —
  simple prompts converge and halt < 48; the harder ones run the full 48 under #48291. Reports the
  ACTUAL per-prompt steps.

    DG_SPARSE_MOE=1 DG_DEDUP_ARGMAX=1 DG_SPARSE_MOE_TUNED=1 DG_TRACE_REGION_SIZE=10737418240 \
      DG_CKPT=... python -u -m models.experimental.diffusion_gemma.doc.vllm_integration.bench_vllm_traced \
        --max-seq-len 4096 --max-denoise-steps 48 --blocks 2

Markers: RESULT_VLLM_TRACED <json>. *** DEVICE-OWNERSHIP: run only when QB2 is free. ***
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _open_mesh_device
from models.experimental.diffusion_gemma.tt.denoise_loop import denoise_block as eager_denoise_block
from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession
from models.experimental.diffusion_gemma.tt.traced_denoise import traced_denoise_block, traced_early_halt_block

# The three serving decode loops, selected via the session's explicit denoise_block_fn (the vLLM
# adapter's enable_trace wiring). None would fall back to the env-gated dispatcher.
DENOISE_FNS = {
    "eager": eager_denoise_block,
    "traced": traced_denoise_block,
    "early_halt": traced_early_halt_block,
}

# Representative prompt set for the early-halt distribution (mix of simple/short and harder/longer).
DEFAULT_PROMPTS = [
    "Hello, how are you?",
    "What is the capital of France?",
    "Explain what a diffusion language model is in one sentence.",
    "Write a short poem about the ocean.",
    "List three uses of a hammer.",
]


def _dram_gib(mesh_device):
    ttnn.synchronize_device(mesh_device)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    gib = 2**30
    return {
        "used_gib": view.num_banks * view.total_bytes_allocated_per_bank / gib,
        "free_gib": view.num_banks * view.total_bytes_free_per_bank / gib,
        "total_gib": view.num_banks * view.total_bytes_per_bank / gib,
    }


def _release_controllers(session) -> None:
    fn = getattr(session, "_logits_fn", None)
    for attr in (
        "_traced_denoise_controller",
        "_traced_denoise_multistep_controller",
        "_traced_early_halt_controller",
    ):
        controller = getattr(fn, attr, None)
        if controller is not None:
            try:
                controller.release()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[bench_vllm] {attr}.release failed: {exc}")
            try:
                delattr(fn, attr)
            except Exception:  # noqa: BLE001
                pass


def _committed_sha(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.to(torch.int64).cpu().numpy().tobytes()).hexdigest()[:16]


def run_session(bundle, args, *, denoise_block_fn, prompt: str, blocks: int) -> dict:
    """Fresh serving session (vLLM-adapter contract) → prefill+block0 (TTFT) then `blocks-1` more."""
    config = DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=args.max_denoise_steps)
    session = BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=config,
        tokenizer=bundle.tokenizer,
        gumbel_mode="argmax",
        seed=args.seed,
        stop_token_ids=[],  # vLLM owns the stop decision; keep emitting blocks
        denoise_block_fn=denoise_block_fn,  # explicit selection = the vLLM enable_trace wiring
    )
    try:
        prompt_tokens = tokenize_prompt(bundle.tokenizer, prompt)
        prompt_len = int(prompt_tokens.shape[1])
        t0 = time.perf_counter()
        session.prefill(prompt_tokens)
        emissions = [session.decode_block()]
        ttft_s = time.perf_counter() - t0
        for _ in range(1, blocks):
            emissions.append(session.decode_block())
        block_latencies = [e.latency_s for e in emissions]
        steady = block_latencies[1:] if len(block_latencies) > 1 else block_latencies
        mean_block = sum(steady) / len(steady)
        tps = args.canvas_length / mean_block if mean_block > 0 else 0.0
        committed = torch.cat([e.tokens for e in emissions], dim=1)
        return {
            "prompt": prompt,
            "prompt_len": prompt_len,
            "prompt_aligned_256": bool(prompt_len % args.canvas_length == 0),
            "blocks": len(emissions),
            "ttft_s": ttft_s,
            "per_block_latency_s": block_latencies,
            "steady_block_latency_s": mean_block,
            "tokens_per_block_per_s": tps,
            "denoise_steps_per_block": [e.num_denoise_steps for e in emissions],
            "halted_per_block": [bool(e.halted) for e in emissions],
            "committed_sha": _committed_sha(committed),
        }
    finally:
        _release_controllers(session)
        session.reset()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    p.add_argument("--mesh", default="P150x4")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--canvas-length", type=int, default=256)
    p.add_argument("--max-denoise-steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--primary-prompt", default="Explain what a diffusion language model is in one sentence.")
    p.add_argument("--halt-prompts", default=None, help="comma-separated; default = a built-in mix")
    p.add_argument("--configs", default="eager,traced,early_halt")
    p.add_argument("--out", default=None)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    configs = [c for c in args.configs.split(",") if c.strip()]
    halt_prompts = [s for s in args.halt_prompts.split("||")] if args.halt_prompts else DEFAULT_PROMPTS
    mesh_device = _open_mesh_device(args.mesh)
    out = {"config": {k: v for k, v in vars(args).items()}, "dram": {}, "fixed_context": {}, "early_halt_distribution": {}}
    try:
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        t_load = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(mesh_device, args.checkpoint, **model_kwargs)
        logger.info(f"[bench_vllm] model load {time.perf_counter() - t_load:.1f}s; max_seq_len={args.max_seq_len}")
        out["dram"] = _dram_gib(mesh_device)
        logger.info(f"[bench_vllm] post-build DRAM: {out['dram']}")

        # -------- TASK 2: eager vs traced vs early_halt at the fixed context (primary prompt) --------
        for cfg in configs:
            r = run_session(bundle, args, denoise_block_fn=DENOISE_FNS[cfg], prompt=args.primary_prompt, blocks=args.blocks)
            out["fixed_context"][cfg] = r
            logger.info(
                f"[bench_vllm] {cfg}: {r['tokens_per_block_per_s']:.2f} t/s ttft={r['ttft_s']:.1f}s "
                f"block={r['steady_block_latency_s']:.3f}s steps={r['denoise_steps_per_block']} "
                f"halted={r['halted_per_block']} sha={r['committed_sha']}"
            )
            print("VLLM_FIXED " + json.dumps(r))

        # -------- TASK 3: early-halt steps distribution over the prompt set --------
        dist = []
        if "early_halt" in configs:
            for prompt in halt_prompts:
                r = run_session(bundle, args, denoise_block_fn=DENOISE_FNS["early_halt"], prompt=prompt, blocks=1)
                dist.append(r)
                logger.info(
                    f"[bench_vllm] early_halt prompt={prompt[:40]!r} steps={r['denoise_steps_per_block']} "
                    f"halted={r['halted_per_block']}"
                )
                print("VLLM_HALT " + json.dumps(r))
        steps = [s for r in dist for s in r["denoise_steps_per_block"]]
        out["early_halt_distribution"] = {
            "per_prompt": [
                {"prompt": r["prompt"], "steps": r["denoise_steps_per_block"], "halted": r["halted_per_block"]}
                for r in dist
            ],
            "all_block_steps": steps,
            "avg_steps": (sum(steps) / len(steps)) if steps else None,
            "min_steps": min(steps) if steps else None,
            "max_steps": max(steps) if steps else None,
            "median_steps": statistics.median(steps) if steps else None,
            "n_blocks_halted_lt_budget": sum(1 for s in steps if s < args.max_denoise_steps),
            "n_blocks_total": len(steps),
        }
    finally:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        _close_mesh_device(mesh_device)
    print("RESULT_VLLM_TRACED " + json.dumps({"dram": out["dram"], "fixed_context": {k: {
        "tps": v["tokens_per_block_per_s"], "ttft_s": v["ttft_s"], "block_s": v["steady_block_latency_s"],
        "steps": v["denoise_steps_per_block"], "halted": v["halted_per_block"], "sha": v["committed_sha"],
    } for k, v in out["fixed_context"].items()}, "early_halt_distribution": out["early_halt_distribution"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
