# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..tt.config import DPTLargeConfig
from ..tt.fallback import DPTFallbackPipeline


def _collect_images(args) -> list[str]:
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
    parser.add_argument("--device", type=str, default="cpu", help="cpu|wormhole_n300|wormhole_n150|blackhole")
    parser.add_argument("--tt-run", action="store_true", help="Run TT pipeline (requires --device != cpu).")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--dump-depth", type=str, default=None)
    parser.add_argument("--dump-depth-color", type=str, default=None)
    parser.add_argument("--dump-perf", type=str, default=None)
    parser.add_argument("--dump-perf-header", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    images = _collect_images(args)
    if len(images) == 0:
        raise SystemExit(
            "No input images found. Supported extensions: .jpg/.jpeg/.png/.bmp "
            "(provide --image or a non-empty --images-dir)."
        )

    if args.tt_run and args.device == "cpu":
        raise SystemExit("--tt-run requires --device != cpu")
    if not args.tt_run and args.device != "cpu":
        raise SystemExit("--device must be 'cpu' unless --tt-run is set")

    use_tt = bool(args.tt_run)
    config = DPTLargeConfig(
        image_size=args.height,
        patch_size=16,
        device=args.device,
        allow_cpu_fallback=not use_tt,
        enable_tt_device=use_tt,
        tt_device_reassembly=use_tt,
        tt_device_fusion=use_tt,
        tt_perf_encoder=use_tt,
        tt_perf_neck=use_tt,
    )

    tt_pipeline = None
    cpu_pipeline = None
    try:
        if use_tt:
            from ..tt.pipeline import DPTTTPipeline

            # Pipeline (TT path shares weights with CPU fallback for preprocessing).
            tt_pipeline = DPTTTPipeline(config=config, device="cpu")
            cpu_pipeline = tt_pipeline.fallback
        else:
            cpu_pipeline = DPTFallbackPipeline(config=config, device="cpu")

        # Warmup around the selected pipeline only.
        for _ in range(args.warmup):
            if use_tt:
                assert tt_pipeline is not None
                _ = tt_pipeline.forward(images[0], normalize=True)
            else:
                assert cpu_pipeline is not None
                _ = cpu_pipeline.run_depth_cpu(images[0], normalize=True)

        timings = []
        depth = None
        for _ in range(args.repeat):
            start = time.perf_counter()
            for img in images:
                if use_tt:
                    assert tt_pipeline is not None
                    depth = tt_pipeline.forward(img, normalize=True)
                else:
                    assert cpu_pipeline is not None
                    depth = cpu_pipeline.run_depth_cpu(img, normalize=True)
            end = time.perf_counter()
            timings.append((end - start) * 1000)

        if depth is None:
            raise RuntimeError("No images were processed (check --image/--images-dir input).")

        latency_ms_mean = float(np.mean(timings))
        latency_ms_std = float(np.std(timings))
        fps = 1000.0 / latency_ms_mean if latency_ms_mean > 0 else 0.0
        perf = dict(
            mode="tt" if use_tt else "cpu",
            latency_ms_mean=latency_ms_mean,
            latency_ms_std=latency_ms_std,
            total_ms=latency_ms_mean,
            fps=fps,
            device=args.device,
            dtype="bfloat16",
            input_h=args.height,
            input_w=args.width,
            batch_size=1,
            modules=["backbone", "reassembly", "fusion_head"] if use_tt else ["cpu_fallback"],
        )
        # Attach last per-stage timing breakdown from the TT pipeline when available.
        if use_tt and tt_pipeline is not None and getattr(tt_pipeline, "last_perf", None) is not None:
            perf["stage_breakdown_ms"] = tt_pipeline.last_perf

        if args.dump_depth:
            Path(args.dump_depth).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.dump_depth, depth)
        if args.dump_depth_color:
            Path(args.dump_depth_color).parent.mkdir(parents=True, exist_ok=True)
            _save_depth_color(depth, args.dump_depth_color)

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
            mode=perf.get("mode", "unknown"),
            latency_ms=perf.get("total_ms", 0),
            fps=perf.get("fps", 0),
            stage_breakdown_ms=perf.get("stage_breakdown_ms", {}),
        )

        if args.dump_perf:
            Path(args.dump_perf).parent.mkdir(parents=True, exist_ok=True)
            Path(args.dump_perf).write_text(json.dumps(perf, indent=2))

        if args.dump_perf_header:
            header_path = Path(args.dump_perf_header)
        elif args.dump_perf:
            header_path = Path(args.dump_perf).with_name(Path(args.dump_perf).stem + "_header.json")
        else:
            header_path = None

        if header_path is not None:
            header_path.parent.mkdir(parents=True, exist_ok=True)
            header_path.write_text(json.dumps(header, indent=2))
    finally:
        if tt_pipeline is not None and hasattr(tt_pipeline, "close"):
            tt_pipeline.close()


if __name__ == "__main__":
    main()
