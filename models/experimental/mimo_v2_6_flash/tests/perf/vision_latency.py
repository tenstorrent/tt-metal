# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Latency of the TTNN ViT (tt/vision) on one Galaxy row vs the fp32 / bf16 torch reference on the host CPU.

Cases (inputs from vLLM's processor, repo sample images): ``448`` (448x448 image, 784 patches), ``640x480`` (1200),
``1024`` (1024x1024, 4096), ``max`` (a 4096x2048 image, the processor's 8.4 MP limit: 32768 patches in one sequence), ``video32``
(32 frames of 640x360: 16 frame pairs of 880 patches) and ``video32_720p`` (32 frames of 1280x720, 56320 patches,
several passes). Device time is the encoder call end to end (host inputs, device, readback of [n, 4096] bf16),
the median of ``--repeats`` calls after a first (compiling) call; the CPU reference runs once per dtype with
``--threads`` torch threads. Buckets up to ``--trace-max-rows`` replay traces (captured before the cases run; their
persistent DRAM and trace region per chip are logged); ``--trace-max-rows 0`` runs every pass eagerly.

Run on one row, as a row engine does:
    TT_VISIBLE_DEVICES=<row chips> python models/experimental/mimo_v2_6_flash/tests/perf/vision_latency.py \
        [--cases 448,1024,max,video32] [--cpu fp32,bf16] [--threads 16] \
        [--numerics accurate,accurate+exact,accurate+bfp8] [--warm] [--trace-max-rows 4096]
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
from models.experimental.mimo_v2_6_flash.tt.vision.vit import ROW_BUCKETS, TRACE_MAX_ROWS


def case_inputs(path, case):
    if case == "448":
        return "image", vin.process_images(path, [vin.sample_image(size=(448, 448))])
    if case == "640x480":
        return "image", vin.process_images(path, [vin.sample_image(size=(640, 480))])
    if case == "1024":
        return "image", vin.process_images(path, [vin.sample_image("demo.jpeg", size=(1024, 1024))])
    if case == "max":
        return "image", vin.process_images(path, [vin.sample_image("demo.jpeg", size=(4096, 2048))])
    if case == "video32":
        return "video", vin.process_video(path, vin.sample_video(32, (640, 360)))
    if case == "video32_720p":
        return "video", vin.process_video(path, vin.sample_video(32, (1280, 720)))
    raise ValueError(case)


def dram_mb(mesh):
    """DRAM allocated per chip (MB); allocation is host bookkeeping, so no device sync is needed."""
    mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return mv.total_bytes_allocated_per_bank * mv.num_banks / 1e6


class PeakTracker:
    """Approximate per-chip DRAM peak of a call: the allocation right after the ops that hold the most temporaries
    (softmax of the window attention, SDPA, the all-reduces)."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.peak = 0.0
        self.active = False
        for owner, name in ((ttnn, "softmax"), (ttnn.transformer, "scaled_dot_product_attention")):
            setattr(owner, name, self._wrap(getattr(owner, name)))
        from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan

        MeshPlan.all_reduce = self._wrap(MeshPlan.all_reduce)

    def _wrap(self, fn):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            if self.active:
                self.peak = max(self.peak, dram_mb(self.mesh))
            return out

        return wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", default="448,1024,max,video32")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cpu", default="fp32,bf16", help="reference dtypes to time on the host ('' for none)")
    p.add_argument("--cpu-cases", default="448,1024,video32", help="cases the CPU reference runs on")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument(
        "--numerics",
        default="balanced",
        help="configurations, comma-separated: a tt/vision/vit.py NUMERICS preset, optionally with '+bfp8' (bfloat8_b "
        "weights) and / or '+exact' (no row buckets: each pass padded to a multiple of 128), e.g. accurate+bfp8",
    )
    p.add_argument("--warm", action="store_true", help="warm() every bucket first and report its time and programs")
    p.add_argument("--trace-max-rows", type=int, default=TRACE_MAX_ROWS, help="largest traced bucket (0: no traces)")
    p.add_argument("--trace-region-mb", type=int, default=256)
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
    trace_region = a.trace_region_mb << 20 if a.trace_max_rows else 0
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8), l1_small_size=65536, trace_region_size=trace_region)
    try:
        kwargs = {} if a.max_rows is None else {"max_rows": a.max_rows}
        tracker = PeakTracker(mesh)
        for preset in presets:
            base_mb = dram_mb(mesh)
            t0 = time.time()
            numerics, *flags = preset.split("+")
            weight_dtype = ttnn.bfloat8_b if "bfp8" in flags else ttnn.bfloat16
            buckets = None if "exact" in flags else ROW_BUCKETS
            enc = MiMoVisionEncoder(
                mesh,
                path,
                weight_dtype=weight_dtype,
                numerics=numerics,
                buckets=buckets,
                trace_max_rows=a.trace_max_rows,
                **kwargs,
            )
            weights_mb = dram_mb(mesh) - base_mb
            logger.info(f"{preset}: encoder ready in {time.time() - t0:.1f} s, weights {weights_mb:.1f} MB per chip")
            if enc.trace_buckets:
                t0 = time.time()
                persistent = enc.vit.prepare_traces(enc.trace_buckets)
                sizes = enc.capture_traces()
                logger.info(
                    f"{preset}: {len(sizes)} traces ({list(sizes)}) in {time.time() - t0:.1f} s: persistent buffers "
                    f"{persistent / 1e6:.1f} MB per chip, trace region {sum(sizes.values()) / 2**20:.1f} MiB per "
                    f"chip ({', '.join(f'{r}: {b / 2**20:.1f}' for r, b in sizes.items())})"
                )
            if a.warm and buckets:
                programs, before = mesh.num_program_cache_entries(), dram_mb(mesh)
                t0 = time.time()
                per_bucket = enc.warm()
                logger.info(
                    f"{preset}: warm() {time.time() - t0:.1f} s for {len(per_bucket)} buckets "
                    f"({', '.join(f'{r}: {s:.2f} s' for r, s in per_bucket.items())}); "
                    f"{mesh.num_program_cache_entries() - programs} programs, {dram_mb(mesh) - before:.2f} MB DRAM"
                )
            for c in cases:
                kind, pv, grid = fields(c)
                fn = enc.embed_image if kind == "image" else enc.embed_video
                t0 = time.time()
                idle_mb = dram_mb(mesh)
                tracker.peak, tracker.active = idle_mb, True
                out = fn(pv, grid)
                tracker.active = False
                first = time.time() - t0
                left_mb = dram_mb(mesh) - idle_mb
                times = []
                for _ in range(a.repeats):
                    t0 = time.time()
                    out = fn(pv, grid)
                    times.append(time.time() - t0)
                r = results[c]
                r.update(grid=ref.as_grid_list(grid), patches=pv.shape[0], tokens=out.shape[0])
                r[preset] = {
                    "first_s": first,
                    "ms": 1000 * statistics.median(times),
                    "out": out,
                    "act_mb": tracker.peak - idle_mb,
                }
                logger.info(
                    f"{preset} {c}: {r['grid']} first {first:.2f} s, median {r[preset]['ms']:.1f} ms, "
                    f"activation peak ~{r[preset]['act_mb']:.0f} MB per chip, left allocated {left_mb:.1f} MB"
                )
            enc.release_traces()
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
    cols += [f"{p} activations (MB/chip)" for p in presets]
    cols += [f"CPU {d} (s)" for d in cpu_dtypes] + [f"{p} PCC vs fp32" for p in presets if "fp32" in cpu_dtypes]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for c in cases:
        r = results[c]
        row = [c, str(r["grid"]), str(r["patches"]), str(r["tokens"])]
        row += [f"{r[p]['ms']:.1f}" for p in presets] + [f"{r[p]['first_s']:.2f}" for p in presets]
        row += [f"{r[p]['act_mb']:.0f}" for p in presets]
        row += [f"{r[f'cpu_{d}_s']:.1f}" if f"cpu_{d}_s" in r else "not measured" for d in cpu_dtypes]
        if "fp32" in cpu_dtypes:
            row += [f"{r[p]['pcc']:.4f}" if "pcc" in r[p] else "not measured" for p in presets]
        lines.append("| " + " | ".join(row) + " |")
    print("\n".join(lines), flush=True)
    print(f"host threads for the CPU reference: {a.threads} (of {os.cpu_count()} cores)", flush=True)


if __name__ == "__main__":
    main()
