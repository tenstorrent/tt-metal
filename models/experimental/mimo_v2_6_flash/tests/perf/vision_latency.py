# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Latency of the TTNN ViT (tt/vision) on one Galaxy row vs the fp32 / bf16 torch reference on the host CPU.

Cases (inputs from vLLM's processor, repo sample images): ``448`` (448x448 image, 784 patches), ``1024`` (1024x1024,
4096), ``max`` (a 4096x2048 image, the processor's 8.4 MP limit: 32768 patches in one sequence), ``video32``
(32 frames of 640x360: 16 frame pairs of 880 patches) and ``video32_720p`` (32 frames of 1280x720, 56320 patches,
several passes). Device time is the encoder call end to end (host inputs, device, readback of [n, 4096] bf16),
the median of ``--repeats`` calls after a first (compiling) call; the CPU reference runs once per dtype with
``--threads`` torch threads.

Run on one row, as a row engine does:
    TT_VISIBLE_DEVICES=<row chips> python models/experimental/mimo_v2_6_flash/tests/perf/vision_latency.py \
        [--cases 448,1024,max,video32] [--cpu fp32,bf16] [--threads 16]
"""

import argparse
import os
import statistics
import time

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference import vision as ref
from models.experimental.mimo_v2_6_flash.reference import vision_inputs as vin
from models.experimental.mimo_v2_6_flash.reference.checkpoint import default_model_path
from models.experimental.mimo_v2_6_flash.tests.common import pcc
from models.experimental.mimo_v2_6_flash.tt.common import fabric_config
from models.experimental.mimo_v2_6_flash.tt.vision.encoder import MiMoVisionEncoder


def case_inputs(path, case):
    if case == "448":
        return "image", vin.process_images(path, [vin.sample_image(size=(448, 448))])
    if case == "1024":
        return "image", vin.process_images(path, [vin.sample_image("demo.jpeg", size=(1024, 1024))])
    if case == "max":
        return "image", vin.process_images(path, [vin.sample_image("demo.jpeg", size=(4096, 2048))])
    if case == "video32":
        return "video", vin.process_video(path, vin.sample_video(32, (640, 360)))
    if case == "video32_720p":
        return "video", vin.process_video(path, vin.sample_video(32, (1280, 720)))
    raise ValueError(case)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="448,1024,max,video32")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu", default="fp32,bf16", help="reference dtypes to time on the host ('' for none)")
    p.add_argument("--cpu-cases", default="448,1024,video32", help="cases the CPU reference runs on")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--weights", default="bf16", choices=["bf16", "bfp8"])
    p.add_argument("--numerics", default="accurate", help="tt/vision/vit.py NUMERICS preset(s), comma-separated")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    path = default_model_path()
    cases = [c for c in a.cases.split(",") if c]
    cpu_dtypes = [d for d in a.cpu.split(",") if d]
    cpu_cases = set(a.cpu_cases.split(","))

    presets = [n for n in a.numerics.split(",") if n]
    inputs = {c: case_inputs(path, c) for c in cases}
    results = {c: {} for c in cases}

    def fields(c):
        kind, d = inputs[c]
        if kind == "image":
            return kind, d["pixel_values"], d["image_grid_thw"]
        return kind, d["pixel_values_videos"], d["video_grid_thw"]

    ttnn.set_fabric_config(**fabric_config())
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=65536, trace_region_size=0)
    try:
        kwargs = {} if a.max_rows is None else {"max_rows": a.max_rows}
        weight_dtype = ttnn.bfloat16 if a.weights == "bf16" else ttnn.bfloat8_b
        for preset in presets:
            t0 = time.time()
            enc = MiMoVisionEncoder(mesh, path, weight_dtype=weight_dtype, numerics=preset, **kwargs)
            logger.info(f"{preset}: encoder ready in {time.time() - t0:.1f} s")
            for c in cases:
                kind, pv, grid = fields(c)
                fn = enc.embed_image if kind == "image" else enc.embed_video
                t0 = time.time()
                out = fn(pv, grid)
                first = time.time() - t0
                times = []
                for _ in range(a.repeats):
                    t0 = time.time()
                    out = fn(pv, grid)
                    times.append(time.time() - t0)
                r = results[c]
                r.update(grid=ref.as_grid_list(grid), patches=pv.shape[0], tokens=out.shape[0])
                r[preset] = {"first_s": first, "ms": 1000 * statistics.median(times), "out": out}
                logger.info(f"{preset} {c}: {r['grid']} first {first:.2f} s, median {r[preset]['ms']:.1f} ms")
            del enc
    finally:
        ttnn.close_mesh_device(mesh)

    if cpu_dtypes:
        sd = ref.load_vision_state_dict(path)
        cfg = ref.VisionConfig.from_model_path(path)
        for dtype_name in cpu_dtypes:
            dtype = torch.float32 if dtype_name == "fp32" else torch.bfloat16
            model = ref.MiMoVisionReference(cfg, {k: v.to(dtype) for k, v in sd.items()})
            for c in cases:
                if c not in cpu_cases:
                    continue
                _, pv, grid = fields(c)
                t0 = time.time()
                with torch.no_grad():
                    expected = model(pv.to(dtype), grid)
                r = results[c]
                r[f"cpu_{dtype_name}_s"] = time.time() - t0
                if dtype_name == "fp32":
                    for preset in presets:
                        r[preset]["pcc"] = pcc(r[preset]["out"].float(), expected)
                logger.info(f"{c}: CPU {dtype_name} {r[f'cpu_{dtype_name}_s']:.1f} s")

    cols = ["case", "grid", "patches", "tokens"]
    cols += [f"{p} (ms)" for p in presets] + [f"{p} first call (s)" for p in presets]
    cols += [f"CPU {d} (s)" for d in cpu_dtypes] + [f"{p} PCC vs fp32" for p in presets if "fp32" in cpu_dtypes]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for c in cases:
        r = results[c]
        row = [c, str(r["grid"]), str(r["patches"]), str(r["tokens"])]
        row += [f"{r[p]['ms']:.1f}" for p in presets] + [f"{r[p]['first_s']:.2f}" for p in presets]
        row += [f"{r[f'cpu_{d}_s']:.1f}" if f"cpu_{d}_s" in r else "not measured" for d in cpu_dtypes]
        if "fp32" in cpu_dtypes:
            row += [f"{r[p]['pcc']:.4f}" if "pcc" in r[p] else "not measured" for p in presets]
        lines.append("| " + " | ".join(row) + " |")
    print("\n".join(lines), flush=True)
    print(f"host threads for the CPU reference: {a.threads} (of {os.cpu_count()} cores)", flush=True)


if __name__ == "__main__":
    main()
