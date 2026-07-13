# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""STEP-1 ablation: measure the serialized replay cost of the MoE dispatch-build DEPENDENT chain.

``sparse_moe.build_capacity_dispatch`` is a ~18-op dependent chain (topk -> scatter -> cumsum ->
gather -> where -> scatter x2 -> slice x2) run once per layer per denoise step. In the traced
denoise loop, independent ops overlap but a dependent chain serializes. This harness measures the
chain's contribution to the per-block latency by comparing:

  * BASELINE: the real per-call chain (``DG_MOE_DISPATCH_ABLATE`` unset).
  * ABLATE:   the chain skipped, disp/comb replaced by a persistent constant built once
              (``DG_MOE_DISPATCH_ABLATE=1``). Output is WRONG (fixed routing) — latency-only.

The block-latency delta (baseline - ablate) is the chain's serialized replay cost across
30 layers x K steps. If it is large, the chain is a real fusion target; if ~0, it overlaps and is a
dead end. Loads the full 30L model ONCE (like sweep_serving) and runs both configs in one process.

Run (from repo root, venv tt-diffusion-gemma):
  DG_NORM_FULLCANVAS=1 DG_ROPE_FULLCANVAS=1 DG_SDPA_FULLCANVAS=1 \
  DG_TRACE_REGION_SIZE=10737418240 MESH_DEVICE=P150x4 TT_METAL_HOME=/home/zni/tt-metal \
  python models/experimental/diffusion_gemma/doc/optimize_perf/bench_dispatch_ablation.py --steps 16,6
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
from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt
from models.experimental.diffusion_gemma.doc.optimize_perf.sweep_serving import run_config

TRACED = {"DG_SPARSE_MOE": "1", "DG_DEDUP_ARGMAX": "1", "DG_SPARSE_MOE_TUNED": "1", "DG_DENOISE_TRACED": "1"}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    p.add_argument("--mesh", default="P150x4")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    p.add_argument("--canvas-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", default="16", help="comma-separated step budgets to sweep (default 16)")
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument(
        "--flag",
        default="DG_MOE_DISPATCH_ABLATE",
        help="DG_* env var toggled between the baseline (unset) and variant (=1) config. "
        "DG_MOE_DISPATCH_ABLATE (default, latency ceiling) or DG_MOE_DISPATCH_FUSED (real bit-identical fusion).",
    )
    p.add_argument("--out", default=None, help="combined results JSON path")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    steps_list = [int(s) for s in str(args.steps).split(",") if s.strip()]

    mesh_device = _open_mesh_device(args.mesh)
    results = []
    try:
        _log_mesh_dram(mesh_device, "baseline")
        t_load = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(
            mesh_device, args.checkpoint, max_seq_len=args.max_seq_len, create_kv_cache=True
        )
        logger.info(f"[ablate] model load took {time.perf_counter() - t_load:.1f}s")
        _log_mesh_dram(mesh_device, "post-build")
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        logger.info(f"[ablate] prompt_len={int(prompt_tokens.shape[1])}")

        flag = args.flag
        for steps in steps_list:
            # BASELINE (flag unset) then VARIANT (flag=1). The flag is not in sweep_serving._DG_FLAGS,
            # so run_config's _apply_env leaves whatever we set here intact.
            os.environ.pop(flag, None)
            base = run_config(
                bundle,
                mesh_device,
                prompt_tokens,
                {"label": f"baseline_s{steps}", "env": TRACED, "steps": steps, "blocks": args.blocks},
                args,
            )
            os.environ[flag] = "1"
            var = run_config(
                bundle,
                mesh_device,
                prompt_tokens,
                {"label": f"{flag}_s{steps}", "env": TRACED, "steps": steps, "blocks": args.blocks},
                args,
            )
            os.environ.pop(flag, None)

            b_block = base["steady_block_latency_s"]
            v_block = var["steady_block_latency_s"]
            delta_block = b_block - v_block
            # 30 MoE layers x K steps chain invocations per block.
            per_layer_per_step_ms = delta_block / (30 * steps) * 1e3 if steps else 0.0
            bit_identical = base["committed_sha"] == var["committed_sha"]
            verdict = {
                "flag": flag,
                "steps": steps,
                "baseline_block_s": b_block,
                "variant_block_s": v_block,
                "delta_block_s": delta_block,
                "delta_pct_of_block": (delta_block / b_block * 100.0) if b_block else 0.0,
                "per_layer_per_step_ms": per_layer_per_step_ms,
                "baseline_tps": base["tokens_per_block_per_s"],
                "variant_tps": var["tokens_per_block_per_s"],
                "baseline_sha": base["committed_sha"],
                "variant_sha": var["committed_sha"],
                "bit_identical": bit_identical,
                "baseline_per_block_s": base["per_block_latency_s"],
                "variant_per_block_s": var["per_block_latency_s"],
            }
            results.append(verdict)
            logger.info("DG_DISPATCH_VERDICT " + json.dumps(verdict, sort_keys=True))
            print("DG_DISPATCH_VERDICT " + json.dumps(verdict))
    finally:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        _close_mesh_device(mesh_device)
    print("DG_DISPATCH_DONE configs=" + str(len(results)))
    for r in results:
        print(
            f"  [{r['flag']}] s{r['steps']}: baseline {r['baseline_block_s']:.3f}s  variant {r['variant_block_s']:.3f}s  "
            f"delta {r['delta_block_s']:.3f}s ({r['delta_pct_of_block']:.1f}%)  "
            f"bit_identical={r['bit_identical']} (base {r['baseline_sha']} vs {r['variant_sha']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
