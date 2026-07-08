# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced denoise-step throughput sweep for the dg-07 datatype sweep (#47475 / #47465).

Loads the full 30L model ONCE (honouring the DG_EXPERTS_BFP8 / DG_EXPERTS_DTYPE knob at build
time) and measures the TRACED single-step serving denoise loop at several step counts. Ranking
is by TRACED per-block latency (never eager). Reuses the tested per-config runner from
``sweep_serving.run_config`` (steady = mean of blocks[1:], committed_sha, controller release).

Run once with no knob (bf16 reference) and once with DG_EXPERTS_BFP8=1 (bfp8 experts) to get an
apples-to-apples same-harness comparison. Requires a large trace region for the s48 point
(48 single-step traces at 30L ~= 8 GB): run with DG_TRACE_REGION_SIZE=10737418240.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from loguru import logger

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.demo.text_demo import (
    _close_mesh_device,
    _log_mesh_dram,
    _open_mesh_device,
)
from models.experimental.diffusion_gemma.doc.optimize_perf.sweep_serving import run_config
from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt
from models.experimental.diffusion_gemma.tt.precision_build import dg_experts_dtype_override


def _configs(steps_list):
    traced = {"DG_SPARSE_MOE": "1", "DG_DEDUP_ARGMAX": "1", "DG_SPARSE_MOE_TUNED": "1", "DG_DENOISE_TRACED": "1"}
    return [{"label": f"traced_tuned_s{s}", "env": traced, "steps": s, "blocks": 3} for s in steps_list]


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
                r = run_config(bundle, mesh_device, prompt_tokens, cfg_spec, args)
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
