# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: FLUX.2 klein 4B request times, to compare two commits on the same machine.

The pipeline warms every size of KLEIN_SIZES (HxW, comma-separated, default 1024x1024,1536x1536),
traced and untraced. Per size, traced and untraced, text to image and (KLEIN_EDIT=1) an edit of one
reference image: 2 unmeasured and KLEIN_RUNS (default 5) timed requests of 4 steps. Prints the median
wall time and the median per pipeline section.
"""

import os
import statistics
import time
from collections import defaultdict

import pytest
from PIL import Image

import ttnn
from models.tt_dit.pipelines.events import SectionEnd, SectionStart

from ....pipelines.flux2.pipeline_flux2 import Flux2Pipeline
from .device_params import line_params_flux2_klein


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(2, 4), line_params_flux2_klein], [(2, 2), line_params_flux2_klein], [(4, 8), line_params_flux2_klein]],
    ids=["2x4", "2x2", "4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_klein_timing(*, mesh_device: ttnn.MeshDevice, model_location_generator) -> None:
    sizes = [
        tuple(int(v) for v in s.split("x")) for s in os.environ.get("KLEIN_SIZES", "1024x1024,1536x1536").split(",")
    ]
    runs = int(os.environ.get("KLEIN_RUNS", "5"))
    edit = os.environ.get("KLEIN_EDIT") == "1"
    h0, w0 = sizes[0]
    pipeline = Flux2Pipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=model_location_generator("black-forest-labs/FLUX.2-klein-4B"),
        width=w0,
        height=h0,
        resolutions=sorted({(w, h) for h, w in sizes}),
        trace_warmup=True,
    )
    reference = Image.effect_noise((1536, 1152), 64).convert("RGB")

    starts: dict[str, float] = {}
    section_s: dict[str, float] = defaultdict(float)

    def on_event(event: object) -> None:
        if isinstance(event, SectionStart):
            starts[event.name] = time.perf_counter()
        elif isinstance(event, SectionEnd) and event.name in starts:
            section_s[event.name] += time.perf_counter() - starts.pop(event.name)

    lines = []
    for h, w in sizes:
        for traced in (True, False):
            for with_ref in (False, True) if edit else (False,):
                walls, sections = [], defaultdict(list)
                for i in range(2 + runs):
                    section_s.clear()
                    t = time.perf_counter()
                    pipeline(
                        prompts=["a lighthouse on a cliff, golden hour"],
                        num_inference_steps=4,
                        seed=0,
                        width=w,
                        height=h,
                        images=[reference] if with_ref else None,
                        traced=traced,
                        on_event=on_event,
                    )
                    if i >= 2:
                        walls.append(time.perf_counter() - t)
                        for k, v in section_s.items():
                            sections[k].append(v)
                case = f"{h}x{w} {'traced' if traced else 'untraced'} {'edit' if with_ref else 't2i'}"
                lines.append(
                    f"TIMING {case}: {statistics.median(walls) * 1e3:.0f} ms median "
                    f"(runs {[round(v * 1e3) for v in walls]}); "
                    + ", ".join(f"{k} {statistics.median(v) * 1e3:.0f}" for k, v in sections.items())
                )
                print(lines[-1], flush=True)
    print("\n".join(lines))
