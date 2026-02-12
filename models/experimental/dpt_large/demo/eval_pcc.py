# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from models.common.utility_functions import comp_pcc
from ..tt.config import DPTLargeConfig
from ..tt.fallback import DPTFallbackPipeline


def _collect_images(args) -> list[str]:
    if args.image:
        return [args.image]
    if args.images_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = sorted(str(p) for p in Path(args.images_dir).iterdir() if p.suffix.lower() in exts)
        if args.limit is not None:
            imgs = imgs[: int(args.limit)]
        return imgs
    raise ValueError("Provide either --image or --images-dir.")


def _fps_from_ms(latency_ms: float) -> float:
    return 1000.0 / latency_ms if latency_ms > 0 else 0.0


def time_pipeline(pipeline, images: list[str], warmup: int, repeat: int):
    for _ in range(max(0, int(warmup))):
        _ = pipeline.forward(images[0], normalize=True)

    timings_ms: list[float] = []
    last = None
    for _ in range(max(1, int(repeat))):
        start = time.perf_counter()
        for img in images:
            last = pipeline.forward(img, normalize=True)
        end = time.perf_counter()
        timings_ms.append((end - start) * 1000.0)

    total_ms_mean = float(np.mean(timings_ms))
    per_image_ms = total_ms_mean / max(1, len(images))
    return {
        "repeat_total_ms": timings_ms,
        "total_ms_mean": total_ms_mean,
        "total_ms_std": float(np.std(timings_ms)),
        "per_image_ms": per_image_ms,
        "fps": _fps_from_ms(per_image_ms),
        "last_output": last,
    }


def main():
    parser = argparse.ArgumentParser("DPT-Large TTNN PCC + FPS evaluator")
    parser.add_argument("--image", type=str, default=None, help="Single image path.")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory of images (jpg/jpeg/png/bmp).")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images from --images-dir.")

    parser.add_argument("--device", type=str, default="cpu", help="cpu|wormhole_n300|wormhole_n150|blackhole")
    parser.add_argument("--tt-run", action="store_true", help="Run TT pipeline and compare to CPU reference.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained HF weights (downloads).")
    parser.add_argument(
        "--no-pretrained", dest="pretrained", action="store_false", help="Use random init (no download)."
    )
    parser.set_defaults(pretrained=True)

    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--dump-json", type=str, default=None, help="Write a JSON summary.")
    args = parser.parse_args()

    images = _collect_images(args)
    if len(images) == 0:
        raise SystemExit(
            "No input images found. Supported extensions: .jpg/.jpeg/.png/.bmp "
            "(provide --image or a non-empty --images-dir)."
        )

    if args.tt_run and args.device == "cpu":
        raise SystemExit("--tt-run requires --device != cpu")

    # Always build a CPU reference pipeline (the thing we compare to for PCC).
    cfg_cpu = DPTLargeConfig(
        image_size=int(args.image_size),
        patch_size=16,
        device="cpu",
        allow_cpu_fallback=True,
        enable_tt_device=False,
        tt_device_reassembly=False,
        tt_device_fusion=False,
        tt_perf_encoder=False,
        tt_perf_neck=False,
    )
    cpu = DPTFallbackPipeline(config=cfg_cpu, pretrained=bool(args.pretrained), device="cpu")
    cpu_stats = time_pipeline(cpu, images, warmup=args.warmup, repeat=args.repeat)

    result: dict[str, Any] = {
        "num_images": len(images),
        "image_size": int(args.image_size),
        "pretrained": bool(args.pretrained),
        "cpu": {k: v for k, v in cpu_stats.items() if k != "last_output"},
    }

    if not args.tt_run:
        if args.dump_json:
            out = Path(args.dump_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    from ..tt.pipeline import DPTTTPipeline

    # Force TT path (do not silently fall back to CPU when --tt-run is requested).
    cfg_tt = DPTLargeConfig(
        image_size=int(args.image_size),
        patch_size=16,
        device=str(args.device),
        allow_cpu_fallback=False,
        enable_tt_device=True,
        # Keep reassembly on host; run fusion/head on TT device.
        tt_device_reassembly=False,
        tt_device_fusion=True,
        tt_perf_encoder=True,
        tt_perf_neck=False,
    )

    with DPTTTPipeline(config=cfg_tt, pretrained=bool(args.pretrained), device="cpu") as tt:
        tt_stats = time_pipeline(tt, images, warmup=args.warmup, repeat=args.repeat)

        # PCC: compare per-image TT vs CPU, then summarize.
        pccs: list[float] = []
        pcc_pass_flags: list[bool] = []
        for img in images:
            depth_cpu = cpu.forward(img, normalize=True)
            depth_tt = tt.forward(img, normalize=True)
            passed, pcc = comp_pcc(depth_cpu, depth_tt, pcc=0.99)
            pccs.append(float(pcc))
            pcc_pass_flags.append(bool(passed))

    result["tt"] = {k: v for k, v in tt_stats.items() if k != "last_output"}
    result["pcc"] = {
        "mean": float(np.nanmean(pccs)) if len(pccs) else float("nan"),
        "min": float(np.nanmin(pccs)) if len(pccs) else float("nan"),
        "per_image": pccs,
        "threshold": 0.99,
        "all_pass": all(pcc_pass_flags) if len(pcc_pass_flags) else False,
    }

    if args.dump_json:
        out = Path(args.dump_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
