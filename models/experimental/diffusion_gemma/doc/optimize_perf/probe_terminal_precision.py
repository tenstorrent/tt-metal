# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LEVER 2 assessment — does float32 terminal precision make early-halt fire?

The HF StableAndConfident early-halt fires when BOTH (a) the clean-argmax canvas is
stable for ``stable_steps_to_halt`` steps AND (b) mean per-position entropy of the
temperature-scaled logits < ``entropy_stop_threshold`` (0.005 nats). Under #48291 the
device (bf16/MoE/TP=4) argmax decisions are degenerate and early-halt is a device-
confirmed no-op (runs the full 48 steps). This probe measures whether raising the
TERMINAL entropy computation from bf16 to float32 (the reference computes entropy in
fp32) closes the confidence gap enough to make the halt fire on a COHERENT prompt.

It runs the EAGER ADAPTIVE denoise path (real StableAndConfident halt) at K=48 twice:
  * bf16 terminal (the shipped path)
  * fp32 terminal (token_entropy upcasts logits->fp32; entropy feeds BOTH the accept
    mask and the halt, so this is the fully fp32-faithful terminal decision path)
and reports per-step (temperature, entropy_mean, argmax-stable) + halted/num_steps for
each, so we can see how far entropy sits above 0.005 and whether fp32 helps.

It ALSO captures one step's raw logits and computes the torch fp32 ground-truth entropy
so we can tell "device-fp32 entropy is accurate but logits are diffuse (#48291-gated)"
apart from "bf16 terminal computation error was the culprit".

Run on QB2 (device free + healthy):
    DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
      python models/experimental/diffusion_gemma/doc/optimize_perf/probe_terminal_precision.py

Markers (grep):
  RESULT_L2_STEP mode=.. step=.. T=.. entropy_mean=.. stable=..
  RESULT_L2_GROUNDTRUTH step=.. dev_bf16=.. dev_fp32=.. torch_fp32=..
  RESULT_L2_SUMMARY mode=.. num_steps=.. halted=.. min_entropy=.. min_stable_entropy=..
  RESULT_L2_VERDICT ..
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import (
    _close_mesh_device,
    _open_mesh_device,
)
from models.experimental.diffusion_gemma.tt import denoise_loop as DL
from models.experimental.diffusion_gemma.tt import sampling as TS
from models.experimental.diffusion_gemma.tt.generate import decode_generation, tokenize_prompt
from models.experimental.diffusion_gemma.demo.serving_smoke import _DeviceGenLike
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")
ENTROPY_STOP_THRESHOLD = 0.005  # config.entropy_stop_threshold (verified against generation_config)

_ORIG_TOKEN_ENTROPY = TS.token_entropy
# capture hook state
_CAPTURE = {"want": False, "dev_bf16": None, "dev_fp32": None, "torch_fp32": None, "step": None}
_STEP = {"i": 0}


def _torch_entropy_mean(logits_torch: torch.Tensor, temperature: float) -> float:
    """Reference fp32 entropy mean of softmax(logits/T): H = logsumexp(z) - E[z]."""
    z = (logits_torch.float() / temperature) if temperature != 1.0 else logits_torch.float()
    lse = torch.logsumexp(z, dim=-1)
    p = torch.softmax(z, dim=-1)
    ez = (p * z).sum(dim=-1)
    return (lse - ez).mean().item()


def _patched_token_entropy(logits, temperature: float = 1.0):
    """token_entropy with an optional fp32 upcast + a one-shot ground-truth capture."""
    fp32 = os.environ.get("DG_TERMINAL_FP32", "0") == "1"
    # one-shot ground-truth capture on the requested step (mid-trajectory, converged-ish)
    if _CAPTURE["want"] and _CAPTURE["step"] == _STEP["i"]:
        logits_t = DL._to_host_torch(logits).float()  # [1,1,S,V] one shard (replicated)
        _CAPTURE["torch_fp32"] = _torch_entropy_mean(logits_t, temperature)
        ent_bf16 = _ORIG_TOKEN_ENTROPY(logits, temperature=temperature)
        _CAPTURE["dev_bf16"] = DL._to_host_torch(ent_bf16).float().mean().item()
        ent_bf16.deallocate(True)
        logits_fp32 = ttnn.typecast(logits, ttnn.float32)
        ent_fp32 = _ORIG_TOKEN_ENTROPY(logits_fp32, temperature=temperature)
        _CAPTURE["dev_fp32"] = DL._to_host_torch(ent_fp32).float().mean().item()
        logits_fp32.deallocate(True)
        _CAPTURE["want"] = False  # one-shot
        # fall through to return the real entropy for the run's chosen mode
    if fp32:
        logits_fp32 = ttnn.typecast(logits, ttnn.float32)
        out = _ORIG_TOKEN_ENTROPY(logits_fp32, temperature=temperature)
        logits_fp32.deallocate(True)
        return out
    return _ORIG_TOKEN_ENTROPY(logits, temperature=temperature)


def _run_block(bundle, mesh_device, prompt_tokens, args, mode: str, capture_step: int | None):
    os.environ["DG_TERMINAL_FP32"] = "1" if mode == "fp32" else "0"
    _STEP["i"] = 0
    if capture_step is not None:
        _CAPTURE.update(want=True, step=capture_step, dev_bf16=None, dev_fp32=None, torch_fp32=None)
    config = DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=args.steps)
    session = BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=config,
        tokenizer=bundle.tokenizer,
        gumbel_mode="argmax",
        seed=args.seed,
        stop_token_ids=[],
    )
    try:
        session.prefill(prompt_tokens)
        t0 = time.perf_counter()
        emission = session.decode_block()
        latency = time.perf_counter() - t0
        text = decode_generation(
            bundle.tokenizer,
            prompt_tokens,
            _DeviceGenLike(emission.tokens, session.cache_len, session.next_pos),
            skip_prompt=True,
            skip_special_tokens=True,
        )
        text_str = text[0] if text else ""
    finally:
        session.reset()
    return emission, latency, text_str


def _instrument_trajectory():
    """Wrap denoise_block to keep the last trajectory's per_step records."""
    orig = DL.denoise_block
    holder = {}

    def wrapped(*a, **k):
        traj = orig(*a, **k)
        holder["traj"] = traj
        return traj

    DL.denoise_block = wrapped
    return holder, orig


def _report(mode, traj):
    per_step = traj.per_step
    prev_argmax = None
    min_ent = float("inf")
    min_stable_ent = float("inf")
    stable_confident_step = None
    for rec in per_step:
        stable = prev_argmax is not None and torch.equal(rec.argmax, prev_argmax)
        print(
            f"RESULT_L2_STEP mode={mode} step={rec.step} T={rec.temperature:.4f} "
            f"entropy_mean={rec.entropy_mean:.6f} stable={stable}",
            flush=True,
        )
        min_ent = min(min_ent, rec.entropy_mean)
        if stable:
            min_stable_ent = min(min_stable_ent, rec.entropy_mean)
            if rec.entropy_mean < ENTROPY_STOP_THRESHOLD and stable_confident_step is None:
                stable_confident_step = rec.step
        prev_argmax = rec.argmax
    print(
        f"RESULT_L2_SUMMARY mode={mode} num_steps={traj.num_steps} halted={traj.halted} "
        f"min_entropy={min_ent:.6f} min_stable_entropy={min_stable_ent if min_stable_ent < float('inf') else 'NA'} "
        f"first_stable_confident_step={stable_confident_step}",
        flush=True,
    )
    return {
        "mode": mode,
        "num_steps": traj.num_steps,
        "halted": bool(traj.halted),
        "min_entropy": min_ent,
        "min_stable_entropy": (min_stable_ent if min_stable_ent < float("inf") else None),
        "first_stable_confident_step": stable_confident_step,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="P150x4")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--capture-step", type=int, default=40, help="step index for the torch ground-truth capture")
    ap.add_argument(
        "--out",
        default="/home/zni/tt-metal/models/experimental/diffusion_gemma/doc/optimize_perf/artifacts/lever1/terminal_precision.json",
    )
    args = ap.parse_args(argv)

    # engage flags for the model-faithful stacked path (sparse MoE + dedup + tuned), eager (no traced).
    os.environ["DG_SPARSE_MOE"] = "1"
    os.environ["DG_DEDUP_ARGMAX"] = "1"
    os.environ["DG_SPARSE_MOE_TUNED"] = "1"
    for k in ("DG_DENOISE_TRACED", "DG_DENOISE_TRACED_MULTISTEP", "DG_DENOISE_DEVICE_LOOP"):
        os.environ.pop(k, None)

    TS.token_entropy = _patched_token_entropy  # patch the sampling module (denoise_loop uses TS.token_entropy)
    holder, orig_db = _instrument_trajectory()

    mesh_device = _open_mesh_device(args.mesh)
    out = {"prompt": args.prompt, "steps": args.steps, "runs": {}}
    try:
        t_load = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(
            mesh_device, CKPT, max_seq_len=args.max_seq_len, create_kv_cache=True
        )
        logger.info(f"[L2] model load took {time.perf_counter() - t_load:.1f}s")
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        logger.info(f"[L2] prompt_len={int(prompt_tokens.shape[1])}")

        # ---- bf16 terminal (shipped), with the one-shot ground-truth capture ----
        emission, lat, text = _run_block(bundle, mesh_device, prompt_tokens, args, "bf16", args.capture_step)
        summ_bf16 = _report("bf16", holder["traj"])
        summ_bf16.update(latency_s=lat, text_head=text[:200])
        out["runs"]["bf16"] = summ_bf16
        gt = {
            "step": args.capture_step,
            "dev_bf16": _CAPTURE["dev_bf16"],
            "dev_fp32": _CAPTURE["dev_fp32"],
            "torch_fp32": _CAPTURE["torch_fp32"],
        }
        out["groundtruth"] = gt
        print(
            f"RESULT_L2_GROUNDTRUTH step={gt['step']} dev_bf16={gt['dev_bf16']} "
            f"dev_fp32={gt['dev_fp32']} torch_fp32={gt['torch_fp32']}",
            flush=True,
        )

        # ---- fp32 terminal (entropy feeds accept + halt) ----
        emission2, lat2, text2 = _run_block(bundle, mesh_device, prompt_tokens, args, "fp32", None)
        summ_fp32 = _report("fp32", holder["traj"])
        summ_fp32.update(latency_s=lat2, text_head=text2[:200])
        out["runs"]["fp32"] = summ_fp32

        # ---- verdict ----
        halt_bf16 = summ_bf16["halted"]
        halt_fp32 = summ_fp32["halted"]
        verdict = (
            f"bf16 halted={halt_bf16} (steps={summ_bf16['num_steps']}); "
            f"fp32 halted={halt_fp32} (steps={summ_fp32['num_steps']}); "
            f"min_entropy bf16={summ_bf16['min_entropy']:.4f} fp32={summ_fp32['min_entropy']:.4f} "
            f"vs threshold {ENTROPY_STOP_THRESHOLD}"
        )
        out["verdict"] = verdict
        print("RESULT_L2_VERDICT " + verdict, flush=True)
    finally:
        DL.denoise_block = orig_db
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        _close_mesh_device(mesh_device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
