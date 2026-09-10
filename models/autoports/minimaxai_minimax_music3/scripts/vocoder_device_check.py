#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTNN vocoder vs the host fp32 reference on the golden windows (stage 07): wav PCC, log-mel distance, time per window.

    with_hw_lock timeout 1800 $MM3_PY $MM3_MODEL_DIR/scripts/vocoder_device_check.py [--dtype fp32|bf16] [--latents 64]
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
from models.autoports.minimaxai_minimax_music3.tt.audio_metrics import log_mel_distance
from models.common.utility_functions import comp_pcc

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC = MODEL_DIR / "doc" / "optimize"
DTYPES = {"fp32": ttnn.float32, "bf16": ttnn.bfloat16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    ap.add_argument(
        "--latents", type=int, default=None, help="use a random latent of this length instead of the golden"
    )
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--repeat", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from models.autoports.minimaxai_minimax_music3.tt.vocoder import TTVocoder

    root = R.reference_dir()
    if args.latents:
        lat = [torch.randn(1, 128, args.latents, generator=torch.Generator().manual_seed(0))]
    else:
        lat = [t.float() for t in torch.load(root / "chunks.pt")["latents"]]
    host = V.load_vocoder(dtype=torch.float32)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=20_000_000)
    mesh.enable_program_cache()
    results = {"dtype": args.dtype, "windows": []}
    try:
        voc = TTVocoder.from_pretrained(mesh, dtype=DTYPES[args.dtype])
        logger.info(f"TTVocoder loaded in {voc.load_seconds:.1f} s")
        for k, x in enumerate(lat):
            t0 = time.perf_counter()
            with torch.no_grad():
                ref = host(x)
            host_s = time.perf_counter() - t0
            times = []
            for _ in range(args.repeat):
                t0 = time.perf_counter()
                out = voc(x)
                times.append(time.perf_counter() - t0)
            assert out.shape == ref.shape, (out.shape, ref.shape)
            pcc = float(comp_pcc(ref, out, 0.0)[1])
            lm = log_mel_distance(out[0], ref[0], V.SAMPLING_RATE)
            err = float((out - ref).abs().max())
            row = {
                "window": k,
                "latents": int(x.shape[-1]),
                "samples": int(out.shape[-1]),
                "wav_pcc": pcc,
                "max_abs_err": err,
                "log_mel_rms_db": lm["rms_db"],
                "host_fp32_s": host_s,
                "device_s": times,
                "device_dispatch_s": voc.timings.get("dispatch_s"),
            }
            results["windows"].append(row)
            logger.info(json.dumps(row))
        voc.release()
    finally:
        ttnn.close_mesh_device(mesh)
    out_dir = DOC / "vocoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"device_check_{args.dtype}{'_L%d' % args.latents if args.latents else ''}.json"
    path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
