#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Can the TTNN vocoder forward be traced, and what is its device-only time? (stage 07 decision evidence)

Captures ``TTVocoder.forward_from_device`` for one window shape over a persistent input buffer, replays it and
compares the replayed waveform with the eager one and the host fp32 reference.

    with_hw_lock timeout 1800 $MM3_PY $MM3_MODEL_DIR/scripts/vocoder_trace_probe.py [--dtype fp32|bf16] [--latents 689]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.autoports.minimaxai_minimax_music3.reference import hf_llm as R
from models.autoports.minimaxai_minimax_music3.reference import vocoder_ref as V
from models.autoports.minimaxai_minimax_music3.tt.vocoder import TTVocoder
from models.common.utility_functions import comp_pcc

MODEL_DIR = Path(__file__).resolve().parents[1]
DTYPES = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    ap.add_argument("--latents", type=int, default=689)
    ap.add_argument("--trace-region", type=int, default=400_000_000)
    args = ap.parse_args()
    torch.set_num_threads(12)
    root = R.reference_dir()
    lat = [t.float() for t in torch.load(root / "chunks.pt")["latents"]]
    x = lat[0] if args.latents == 689 else torch.randn(1, 128, args.latents, generator=torch.Generator().manual_seed(0))
    host = V.load_vocoder(dtype=torch.float32)
    with torch.no_grad():
        ref = host(x)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=args.trace_region)
    mesh.enable_program_cache()
    res = {"dtype": args.dtype, "latents": args.latents}
    try:
        voc = TTVocoder.from_pretrained(mesh, dtype=DTYPES[args.dtype])
        inp = voc.input_buffer(x)
        t0 = time.perf_counter()
        out = voc.forward_from_device(inp)
        wav_eager = voc.read_waveform(out, x.shape[0])
        res["eager_s"] = time.perf_counter() - t0
        res["eager_pcc"] = float(comp_pcc(ref, wav_eager, 0.0)[1])
        logger.info(f"eager {res['eager_s']:.2f} s, PCC {res['eager_pcc']:.6f}")
        ttnn.deallocate(out)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = voc.forward_from_device(inp)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        res["capture_s"] = time.perf_counter() - t0
        logger.info(f"captured in {res['capture_s']:.1f} s")
        for i in range(3):
            voc.write_input(inp, x)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            wav = voc.read_waveform(out, x.shape[0])
            dt = time.perf_counter() - t0
            res.setdefault("traced_s", []).append(dt)
            ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        res["replay_only_s"] = (time.perf_counter() - t0) / 3
        res["traced_pcc"] = float(comp_pcc(ref, wav, 0.0)[1])
        res["traced_vs_eager_max_abs"] = float((wav - wav_eager).abs().max())
        logger.info(json.dumps(res))
        ttnn.release_trace(mesh, tid)
        voc.release()
    finally:
        ttnn.close_mesh_device(mesh)
    d = MODEL_DIR / "doc" / "optimize" / "vocoder"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"trace_probe_{args.dtype}_L{args.latents}.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
