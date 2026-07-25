# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Eager denoise-step throughput sweep for the dg-07 datatype sweep (#47475 / #47465).

Loads the full 30L model ONCE (honouring the DG_EXPERTS_BFP8 / DG_EXPERTS_DTYPE knob at build
time) and measures the ordinary eager serving loop at several step counts. This script remains
an eager datatype diagnostic; the only supported Metal trace path is model-lifetime up-front
capture through the vLLM wrapper.

Run once with no knob (bf16 reference) and once with DG_EXPERTS_BFP8=1 (bfp8 experts) to get an
apples-to-apples same-harness comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

from loguru import logger
import torch

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.serving_smoke import _DeviceGenLike
from models.experimental.diffusion_gemma.demo.text_demo import (
    _close_mesh_device,
    _log_mesh_dram,
    _open_mesh_device,
)
from models.experimental.diffusion_gemma.tt.generate import decode_generation, tokenize_prompt
from models.experimental.diffusion_gemma.tt.precision_build import dg_experts_dtype_override
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession


def _configs(steps_list):
    return [{"label": f"eager_tuned_s{s}", "steps": s, "blocks": 3} for s in steps_list]


def _run_config(bundle, prompt_tokens, cfg_spec, args) -> dict:
    steps = cfg_spec["steps"]
    blocks = cfg_spec["blocks"]
    session = BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=steps),
        tokenizer=bundle.tokenizer,
        gumbel_mode="argmax",
        seed=args.seed,
        stop_token_ids=[],
    )
    try:
        started = time.perf_counter()
        session.prefill(prompt_tokens)
        emissions = [session.decode_block()]
        ttft_s = time.perf_counter() - started
        for _ in range(1, blocks):
            emissions.append(session.decode_block())

        block_latencies = [emission.latency_s for emission in emissions]
        steady = block_latencies[1:] if len(block_latencies) > 1 else block_latencies
        mean_block = sum(steady) / len(steady)
        committed = torch.cat([emission.tokens for emission in emissions], dim=1)
        committed_sha = hashlib.sha256(committed.to(torch.int64).cpu().numpy().tobytes()).hexdigest()[:16]
        text = decode_generation(
            bundle.tokenizer,
            prompt_tokens,
            _DeviceGenLike(committed, session.cache_len, session.next_pos),
            skip_prompt=True,
            skip_special_tokens=True,
        )
        return {
            "label": cfg_spec["label"],
            "mode": "eager",
            "steps": steps,
            "blocks": blocks,
            "ttft_s": ttft_s,
            "per_block_latency_s": block_latencies,
            "steady_block_latency_s": mean_block,
            "tokens_per_block_per_s": args.canvas_length / mean_block,
            "denoise_steps_per_block": [emission.num_denoise_steps for emission in emissions],
            "halted_per_block": [emission.halted for emission in emissions],
            "committed_sha": committed_sha,
            "text_head": (text[0] if text else "")[:220],
        }
    finally:
        session.reset()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    p.add_argument("--mesh", default="P150x4")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    p.add_argument("--canvas-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", default="48,24,12", help="comma-separated denoise step counts")
    p.add_argument("--out-dir", default=os.environ.get("DG_DTPERF_OUT", "/tmp/dg_dtperf"))
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    os.environ["DG_SPARSE_MOE"] = "1"
    os.environ["DG_DEDUP_ARGMAX"] = "1"
    os.environ["DG_SPARSE_MOE_TUNED"] = "1"
    # Explicit "0": up-front capture is default ON, so unsetting no longer disables it.
    os.environ["DG_UPFRONT_CAPTURE"] = "0"
    os.makedirs(args.out_dir, exist_ok=True)
    steps_list = [int(x) for x in args.steps.split(",")]
    configs = _configs(steps_list)

    mesh_device = _open_mesh_device(args.mesh)
    results = []
    try:
        _log_mesh_dram(mesh_device, "baseline")
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        t_load = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(mesh_device, args.checkpoint, **model_kwargs)
        logger.info(f"[dtperf] load {time.perf_counter() - t_load:.1f}s experts_override={dg_experts_dtype_override()}")
        _log_mesh_dram(mesh_device, "post-build")
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        logger.info(f"[dtperf] prompt_len={int(prompt_tokens.shape[1])}")

        for cfg_spec in configs:
            try:
                r = _run_config(bundle, prompt_tokens, cfg_spec, args)
            except BaseException as exc:  # noqa: BLE001
                logger.error(f"DG_DTPERF_CONFIG_FAILURE label={cfg_spec['label']} err={type(exc).__name__}: {exc}")
                raise
            r["experts_override"] = str(dg_experts_dtype_override())
            with open(os.path.join(args.out_dir, f"{cfg_spec['label']}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2)
            results.append(r)
    finally:
        with open(os.path.join(args.out_dir, "combined.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        _close_mesh_device(mesh_device)
    print("DG_DTPERF_DONE configs=" + str(len(results)))
    for r in results:
        print(
            f"  {r['label']}: {r['tokens_per_block_per_s']:.2f} t/s  block={r['steady_block_latency_s']:.3f}s"
            f"  steps={r['denoise_steps_per_block']}  sha={r['committed_sha']}  experts={r['experts_override']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
