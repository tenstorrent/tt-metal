#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DiT Euler-step timing (stage 07): eager ``FlowTransformer.forward`` vs the traced ``DiTStepTrace.step`` for the
200-frame window (T = 689, S_pad 768), per weight dtype, plus the PCC of the traced output against the eager one.

    with_hw_lock timeout 1200 $MM3_PY $MM3_MODEL_DIR/scripts/dit_step_timing.py --weight-dtype bfp8 [--layers 36]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.tt.flow_transformer import BATCH, FlowTransformer
from models.common.utility_functions import comp_pcc

MODEL_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("TT_DIT_CACHE_DIR", str(MODEL_DIR / "generated" / "tt_dit_cache"))
DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight-dtype", default="bfp8", choices=sorted(DTYPES))
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--latents", type=int, default=689)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--trace-region", type=int, default=200_000_000)
    ap.add_argument("--fidelity", default="hifi2")
    ap.add_argument("--silu-mode", default="matmul", choices=["matmul", "multiply", "unary"])
    ap.add_argument("--matmul-configs", default="swept", choices=["swept", "default"])
    args = ap.parse_args()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=args.trace_region)
    mesh.enable_program_cache()
    out = {"weight_dtype": args.weight_dtype, "layers": args.layers, "latents": args.latents}
    try:
        model = FlowTransformer.from_pretrained(
            mesh, weight_dtype=DTYPES[args.weight_dtype], num_layers=args.layers, fidelity=args.fidelity
        )
        model.silu_mode = args.silu_mode
        model.matmul_config_policy = args.matmul_configs
        out.update({"fidelity": args.fidelity, "silu_mode": args.silu_mode, "matmul_configs": args.matmul_configs})
        g = torch.Generator().manual_seed(0)
        t = args.latents
        latents = torch.randn(1, 128, t, generator=g).expand(BATCH, -1, -1).contiguous()
        cond = torch.cat([torch.randn(1, t, 2048, generator=g) * 0.4, torch.zeros(1, t, 2048)], dim=0)
        timestep = torch.full((BATCH,), 0.5)
        cond_proj = model.prepare_condition(cond)
        # eager
        for i in range(args.iters + 2):
            if i == 2:
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
            v_eager = model(latents, timestep, cond_proj=cond_proj)
        out["eager_ms"] = (time.perf_counter() - t0) / args.iters * 1e3
        logger.info(f"eager forward: {out['eager_ms']:.1f} ms")
        # traced
        tr = model.traced_step(t)
        tr.set_condition(cond)
        for i in range(args.iters + 2):
            if i == 2:
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
            v_traced = tr.step(latents, timestep)
        out["traced_ms"] = (time.perf_counter() - t0) / args.iters * 1e3
        # device-only replay time (no host writes / reads)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            ttnn.execute_trace(mesh, tr.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        out["replay_only_ms"] = (time.perf_counter() - t0) / args.iters * 1e3
        out["traced_vs_eager_pcc"] = float(comp_pcc(v_eager, v_traced, 0.0)[1])
        out["traced_vs_eager_max_abs"] = float((v_eager - v_traced).abs().max())
        logger.info(
            f"traced step: {out['traced_ms']:.1f} ms (replay only {out['replay_only_ms']:.1f} ms), PCC vs eager {out['traced_vs_eager_pcc']:.6f}"
        )
        ttnn.deallocate(cond_proj)
        model.release()
    finally:
        ttnn.close_mesh_device(mesh)
    d = MODEL_DIR / "doc" / "optimize" / "dit"
    d.mkdir(parents=True, exist_ok=True)
    p = (
        d
        / f"step_timing_{args.weight_dtype}_{args.fidelity}_{args.matmul_configs}_{args.silu_mode}_l{args.layers}_T{args.latents}.json"
    )
    p.write_text(json.dumps(out, indent=2) + "\n")
    logger.info(f"wrote {p}")


if __name__ == "__main__":
    main()
