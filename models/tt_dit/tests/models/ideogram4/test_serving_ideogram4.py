# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ideogram 4 serving: after ``warmup`` a request compiles nothing, captures nothing and keeps no
device allocation, at every warmed size, preset and prompt length.

The pipeline warms the four 2K sizes, then serves every size x preset x {short, longest-allowed}
prompt with the kept step traces. Around every request it asserts:
  * no new program-cache entries,
  * no new trace capture (the pipeline's capture count and the live-trace count are unchanged),
  * DRAM allocated is the same before and after the request.
It also checks that the traced image matches an eager run of the same request, that an unwarmed
size and an over-length prompt raise, and reports warmup time, per-request timings, DRAM and the
trace region in use with every trace resident.

Env: IDEOGRAM4_WEIGHTS (fp8 checkpoint dir or HF repo id), IDEOGRAM_SERVE_SIZES (default the 4 2K
sizes, WxH comma-separated), IDEOGRAM_SERVE_PRESETS (default all 3), IDEOGRAM_SERVE_OUT (report dir).
"""

import gc
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from loguru import logger

import ttnn

from ....pipelines.events import SectionTimer
from ....pipelines.ideogram4.pipeline_ideogram4 import MAX_TEXT_TOKENS, Ideogram4Pipeline, required_trace_region_size
from ....pipelines.ideogram4.sampler import PRESETS
from ....utils.test import ring_params
from ....utils.tracing import Tracer

_NEEDS_WEIGHTS = pytest.mark.skipif(
    not os.environ.get("IDEOGRAM4_WEIGHTS"), reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)"
)
# The trace region the pipeline publishes for its four 2K serving sizes; the test reports the measured use.
_SERVE_PIPE = {**ring_params, "l1_small_size": 32768, "trace_region_size": required_trace_region_size((4, 8))}

DEFAULT_SIZES = "2048x2048,2048x1152,1152x2048,1360x2048"
SEED = 123
MIN_PSNR = 40.0

SHORT_PROMPT = "A red apple on a wooden table."
_FILLER = (
    "a sunlit harbor town with pastel houses, fishing boats, gulls over the water, a lighthouse on the cliff, "
    "market stalls with fruit and flowers, children chasing a dog along the pier, clouds over distant hills, "
)


def _longest_prompt(pipe: Ideogram4Pipeline) -> str:
    """The longest prefix of a repeated description that still fits MAX_TEXT_TOKENS."""
    filler = _FILLER * 64
    assert pipe.count_text_tokens(filler) > MAX_TEXT_TOKENS
    lo, hi = 0, len(filler)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pipe.count_text_tokens(filler[:mid]) <= MAX_TEXT_TOKENS:
            lo = mid
        else:
            hi = mid - 1
    return filler[:lo]


def _dram_allocated(dev) -> int:
    gc.collect()
    ttnn.synchronize_device(dev)
    return ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_bytes_allocated_per_bank


def _psnr(a, b) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mse = float(((x - y) ** 2).mean())
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


@_NEEDS_WEIGHTS
@pytest.mark.parametrize(
    ("mesh_device", "device_params"), [((4, 8), _SERVE_PIPE)], indirect=True, ids=["bh_galaxy_sp8tp4"]
)
def test_serving(*, mesh_device, expect_error) -> None:
    sizes = [
        tuple(int(v) for v in s.split("x")) for s in os.environ.get("IDEOGRAM_SERVE_SIZES", DEFAULT_SIZES).split(",")
    ]
    presets = os.environ.get("IDEOGRAM_SERVE_PRESETS", ",".join(PRESETS)).split(",")
    out = Path(os.environ.get("IDEOGRAM_SERVE_OUT", "ideogram4_serving"))
    (out / "images").mkdir(parents=True, exist_ok=True)

    # Warmed below, by hand, to time it and to measure DRAM before it.
    pipe = Ideogram4Pipeline.create_pipeline(
        mesh_device=mesh_device, width=sizes[0][0], height=sizes[0][1], run_warmup=False
    )
    dev = pipe.mesh_device  # sync the pipeline's submesh, never the parent mesh
    prompts = {"short": SHORT_PROMPT, "longest": _longest_prompt(pipe)}
    tokens = {k: pipe.count_text_tokens(v) for k, v in prompts.items()}
    logger.info(f"SERVE prompt tokens {tokens}")

    dram_before_warmup = _dram_allocated(dev)
    start = time.perf_counter()
    pipe.warmup(sizes=sizes)
    warmup_s = time.perf_counter() - start
    assert pipe.warmed_sizes == tuple(sizes)
    assert pipe.num_trace_captures == len(sizes)

    report = {
        "sizes": [f"{w}x{h}" for w, h in sizes],
        "presets": presets,
        "text_tokens": tokens,
        "warmup_s": round(warmup_s, 1),
        "dram_per_bank_before_warmup": dram_before_warmup,
        "dram_per_bank_after_warmup": _dram_allocated(dev),
        "dram_bank_size": ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).total_bytes_per_bank,
        "dram_banks": ttnn.get_memory_view(dev, ttnn.BufferType.DRAM).num_banks,
        "trace_region_used_per_bank": ttnn.get_memory_view(dev, ttnn.BufferType.TRACE).total_bytes_allocated_per_bank,
        "trace_region_banks": ttnn.get_memory_view(dev, ttnn.BufferType.TRACE).num_banks,
        "trace_region_size": _SERVE_PIPE["trace_region_size"],
        "requests": [],
    }
    logger.info(f"SERVE warmup {json.dumps({k: v for k, v in report.items() if k != 'requests'})}")

    programs = dev.num_program_cache_entries()
    captures = pipe.num_trace_captures
    traces_live = Tracer._traces_live[dev.id()]
    failures = []

    def serve(prompt_name: str, width: int, height: int, preset: str, *, traced: bool = True):
        dram = _dram_allocated(dev)
        timer = SectionTimer(dev, sync_steps=False)
        start = time.perf_counter()
        image = pipe(
            prompts=[prompts[prompt_name]],
            preset=preset,
            seed=SEED,
            width=width,
            height=height,
            traced=traced,
            on_event=timer,
        )[0]
        ttnn.synchronize_device(dev)
        wall_s = time.perf_counter() - start
        row = {
            "size": f"{width}x{height}",
            "preset": preset,
            "prompt": prompt_name,
            "traced": traced,
            "wall_s": round(wall_s, 3),
            "encoder_ms": round(timer.durations["encoder"] * 1000, 1),
            "step_ms": round((timer.starts["vae"] - timer.starts["denoising"]) * 1000 / len(timer.step_ms()), 1),
            "vae_ms": round(timer.durations["vae"] * 1000, 1),
            "new_programs": dev.num_program_cache_entries() - programs,
            "new_captures": pipe.num_trace_captures - captures,
            "traces_live_delta": Tracer._traces_live[dev.id()] - traces_live,
            "dram_delta_per_bank": _dram_allocated(dev) - dram,
        }
        logger.info(f"SERVE {json.dumps(row)}")
        report["requests"].append(row)
        for key in ("new_programs", "new_captures", "traces_live_delta", "dram_delta_per_bank"):
            if row[key]:
                failures.append(f"{row['size']} {preset} {prompt_name} traced={traced}: {key} = {row[key]}")
        if float(np.asarray(image, dtype=np.float32).std()) < 10:
            failures.append(f"{row['size']} {preset} {prompt_name}: flat image")
        return image

    for width, height in sizes:
        for preset in presets:
            for prompt_name in prompts:
                image = serve(prompt_name, width, height, preset)
                image.save(out / "images" / f"{width}x{height}__{preset}__{prompt_name}.png")

    # The kept trace computes what an eager run of the same request computes.
    width, height = sizes[0]
    traced_image = serve("longest", width, height, presets[0])
    eager_image = serve("longest", width, height, presets[0], traced=False)
    psnr = _psnr(traced_image, eager_image)
    report["traced_vs_eager_psnr"] = psnr if math.isfinite(psnr) else "inf"
    if psnr < MIN_PSNR:
        failures.append(f"traced vs eager PSNR {psnr:.1f} dB < {MIN_PSNR}")

    # Neither error touches the device.
    with expect_error(ValueError, "was not warmed up"):
        pipe(prompts=[SHORT_PROMPT], width=1024, height=1024)
    with expect_error(ValueError, "Ideogram 4 supports at most"):
        pipe(prompts=[_FILLER * 64], width=width, height=height)

    (out / "report.json").write_text(json.dumps(report, indent=2))
    assert not failures, "\n".join(failures)
