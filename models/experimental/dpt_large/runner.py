# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from .config import DPTLargeConfig
from .eval_utils import dump_pcc_report, evaluate_tt_vs_cpu, zero_shot_eval
from .pipeline import DPTTTPipeline


def _collect_images(args) -> List[str]:
    if args.image:
        return [args.image]
    if args.images_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted(str(p) for p in Path(args.images_dir).iterdir() if p.suffix.lower() in exts)
    raise ValueError("Provide either --image or --images-dir.")


def _save_depth_color(depth: np.ndarray, path: str):
    depth_min = depth.min()
    depth_max = depth.max()
    norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    img = Image.fromarray((norm.squeeze() * 255).astype(np.uint8))
    img.save(path)


def main():
    parser = argparse.ArgumentParser("DPT-Large TTNN runner")
    parser.add_argument("--image", type=str, default=None, help="Path to a single image.")
    parser.add_argument("--images-dir", type=str, default=None, help="Directory of images.")
    parser.add_argument("--device", type=str, default="cpu", help="wormhole_n300|wormhole_n150|blackhole|cpu")
    parser.add_argument("--tt-run", action="store_true", help="Run TT pipeline.")
    parser.add_argument("--fallback-run", action="store_true", help="Run CPU fallback.")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--dump-depth", type=str, default=None)
    parser.add_argument("--dump-depth-color", type=str, default=None)
    parser.add_argument("--dump-perf", type=str, default=None)
    parser.add_argument("--dump-perf-header", type=str, default=None)
    parser.add_argument("--pcc-eval", action="store_true")
    parser.add_argument("--zero-shot-eval", action="store_true")
    parser.add_argument("--nyu-root", type=str, default=None)
    parser.add_argument("--kitti-root", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--tt-fast-reassembly", action="store_true", help="Use device-native reassembly path")
    parser.add_argument("--tt-fast-head", action="store_true", help="Use device-native fusion head path")
    parser.add_argument("--tt-perf-encoder", action="store_true", help="Enable encoder perf config (L1/sharding/fused)")
    parser.add_argument("--tt-perf-neck", action="store_true", help="Enable aggressive device neck/head perf config")
    parser.add_argument(
        "--tt-safe-attn-programs",
        action="store_true",
        help="Force default attention program configs to avoid shard-height errors",
    )
    args = parser.parse_args()

    images = _collect_images(args)

    config = DPTLargeConfig(
        image_size=args.height,
        patch_size=16,
        device=args.device,
        enable_tt_device=args.tt_run and args.device != "cpu",
        tt_device_reassembly=args.tt_fast_reassembly,
        tt_device_fusion=args.tt_fast_head,
        tt_perf_encoder=args.tt_perf_encoder,
        tt_perf_neck=args.tt_perf_neck,
        tt_force_default_attention_programs=args.tt_safe_attn_programs,
    )
    use_tt = args.tt_run and not args.fallback_run

    # Pipelines (TT path shares weights with CPU fallback).
    tt_pipeline = DPTTTPipeline(config=config, device="cpu")
    cpu_pipeline = tt_pipeline.fallback

    outputs = []
    # Warmup around the selected pipeline only.
    for _ in range(args.warmup):
        if use_tt:
            _ = tt_pipeline.forward(images[0], normalize=True)
        else:
            _ = cpu_pipeline.run_depth_cpu(images[0], normalize=True)

    timings = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        for img in images:
            if use_tt:
                depth = tt_pipeline.forward(img, normalize=True)
            else:
                depth = cpu_pipeline.run_depth_cpu(img, normalize=True)
            outputs.append(depth)
        end = time.perf_counter()
        timings.append((end - start) * 1000)

    depth = outputs[-1]

    if args.dump_depth:
        Path(args.dump_depth).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_depth, depth)
    if args.dump_depth_color:
        Path(args.dump_depth_color).parent.mkdir(parents=True, exist_ok=True)
        _save_depth_color(depth, args.dump_depth_color)

    if args.dump_perf:
        latency_ms_mean = float(np.mean(timings))
        latency_ms_std = float(np.std(timings))
        fps = 1000.0 / latency_ms_mean if latency_ms_mean > 0 else 0.0
        perf = dict(
            latency_ms_mean=latency_ms_mean,
            latency_ms_std=latency_ms_std,
            fps=fps,
            device=args.device,
            dtype="bfloat16",
            input_h=args.height,
            input_w=args.width,
            batch_size=1,
            modules=["backbone", "reassembly", "fusion_head"],
        )
        # Attach last per-stage timing breakdown from the TT pipeline when available.
        if use_tt and getattr(tt_pipeline, "last_perf", None) is not None:
            perf["stage_breakdown_ms"] = tt_pipeline.last_perf
        Path(args.dump_perf).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_perf).write_text(json.dumps(perf, indent=2))

    if args.dump_perf_header:
        # Collect active flags
        active_flags = []
        if args.tt_perf_encoder:
            active_flags.append("tt-perf-encoder")
        if args.tt_fast_reassembly:
            active_flags.append("tt-fast-reassembly")
        if args.tt_fast_head:
            active_flags.append("tt-fast-head")
        if args.tt_safe_attn_programs:
            active_flags.append("tt-safe-attn-programs")

        header = dict(
            model=dict(
                name="dpt-large",
                type="depth_estimation",
                backbone="vit-large",
                num_layers=24,
                hidden_size=1024,
                num_heads=16,
                intermediate_size=4096,
            ),
            input_shape=[1, 3, args.height, args.width],
            patch_size=16,
            num_tokens=(args.height // 16) * (args.width // 16) + 1,
            device=args.device,
            dtype="bfloat16",
            flags=active_flags,
            latency_ms=perf.get("total_ms", 0),
            fps=perf.get("fps", 0),
            stage_breakdown_ms=perf.get("stage_breakdown_ms", {}),
        )
        Path(args.dump_perf_header).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_perf_header).write_text(json.dumps(header, indent=2))

    if args.pcc_eval:
        pccs, mean_pcc = evaluate_tt_vs_cpu(images, cpu_pipeline, tt_pipeline)
        print(f"PCC vs CPU (mean over {len(pccs)} images): {mean_pcc:.4f}")
        if args.dump_perf:
            dump_pcc_report(Path(args.dump_perf).with_suffix(".pcc.json"), pccs, mean_pcc)

    if args.zero_shot_eval:
        zero_shot_eval(args.nyu_root, tt_pipeline)
        zero_shot_eval(args.kitti_root, tt_pipeline)


if __name__ == "__main__":
    main()
