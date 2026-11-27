# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: where the first request at a new size spends its extra host time. Warms 1024x1024 only;
then per size of KLEIN_SIZES (HxW, comma-separated) profiles the first and the second eager request
with cProfile and prints the functions whose own time grew most. KLEIN_OUT_DIR gets the .prof files.
"""

import cProfile
import os
import pstats
import time
from pathlib import Path

import pytest

import ttnn

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from .device_params import line_params_flux2_klein


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(2, 4), line_params_flux2_klein], [(2, 2), line_params_flux2_klein], [(4, 8), line_params_flux2_klein]],
    ids=["2x4", "2x2", "4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_klein_first_run(*, mesh_device: ttnn.MeshDevice, model_location_generator) -> None:
    out_dir = Path(os.environ.get("KLEIN_OUT_DIR", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [tuple(int(v) for v in s.split("x")) for s in os.environ.get("KLEIN_SIZES", "1024x1184").split(",")]
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-klein-4B"),
        resolutions=sorted({(1024, 1024), (2048, 2048), *((w, h) for h, w in sizes)}),
        trace_warmup=False,
    )

    def call(h: int, w: int) -> None:
        pipeline(prompts=["a lighthouse on a cliff, golden hour"], num_inference_steps=2, seed=0, width=w, height=h)
        ttnn.synchronize_device(mesh_device)

    call(1024, 1024)
    lines = []
    for h, w in sizes:
        stats, walls = [], []
        for run in ("first", "second"):
            prof = cProfile.Profile()
            t = time.perf_counter()
            prof.enable()
            call(h, w)
            prof.disable()
            walls.append(time.perf_counter() - t)
            prof.dump_stats(out_dir / f"{h}x{w}_{run}.prof")
            stats.append(pstats.Stats(prof).stats)
        own = [{f"{k[0].split('tt-metal/')[-1]}:{k[1]}({k[2]})": v[2] for k, v in s.items()} for s in stats]
        grew = sorted(((own[0].get(k, 0) - own[1].get(k, 0), k) for k in own[0]), reverse=True)[:25]
        lines.append(f"{h}x{w}: first {walls[0]:.3f} s, second {walls[1]:.3f} s")
        lines += [f"  +{d * 1e3:7.1f} ms own  {k}" for d, k in grew if d > 0.002]
        print("\n".join(lines[-26:]), flush=True)
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n")
