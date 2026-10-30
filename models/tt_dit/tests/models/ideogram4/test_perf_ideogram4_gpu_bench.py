# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4 perf, matched to the GPU benchmark in spacelike-ai/gpu-benchmark (ideogram.py).
#
# Same configuration as the GPU run:
#   - 2048x2048, sampler presets V4_QUALITY_48 and V4_DEFAULT_20, seed 123
#   - prompt = the JSON caption of the Python example in ideogram-oss/ideogram4 docs/prompting.md
#     (commit 990fe1c), serialized compactly with ensure_ascii=False; no magic prompt
#   - initial noise drawn on the CPU from the seed in (1, image tokens, latent channels) float32,
#     which is exactly how this pipeline draws it, so the same seed gives the same noise
#   - one warmup call, then N timed reps, medians reported
#
# Writes one JSONL record per preset in the GPU benchmark's schema (wall_s, denoising_s, step_ms,
# text_encoder_ms, decoder_ms, reps) plus the last image, so rows compare one to one with
# results/<gpu>/ideogram.jsonl there.
#
# Knobs (env), so the same test serves the smoke-to-full ladder:
#   IDEOGRAM4_WEIGHTS          fp8 checkpoint dir (required)
#   IDEOGRAM_BENCH_SIZE        square side in px                       (default 2048)
#   IDEOGRAM_BENCH_PRESETS     comma list                              (default V4_QUALITY_48,V4_DEFAULT_20)
#   IDEOGRAM_BENCH_REPS        timed reps per preset                   (default 3)
#   IDEOGRAM_BENCH_TRACED      0 = untraced, 1 = traced denoise step    (default 0)
#   IDEOGRAM_BENCH_SYNC_STEPS  1 = device sync at every step end       (default 0; see below)
#   IDEOGRAM_BENCH_OUT         output dir                              (default ./ideogram_bench_out)
#
# Timing: the step events fire at enqueue time, so per-step host times are dispatch times. The
# denoise time is therefore measured from the start of denoising to the start of decode: the
# pipeline reads the latent back and synchronizes in between, so that window covers execution.
# step_ms = denoise time / steps. A device sync at every step end (IDEOGRAM_BENCH_SYNC_STEPS=1)
# HANGS the Galaxy pipeline on the second step (tt-metal main, 2026-10-28), so it is off by default.
# =============================================================================

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import pytest
import ttnn
from loguru import logger

from ....pipelines.events import SectionEnd, SectionStart
from ....pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from ....utils.test import ring_params

_NEEDS_WEIGHTS = pytest.mark.skipif(
    not os.environ.get("IDEOGRAM4_WEIGHTS"), reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)"
)

_RING_PIPE = {**ring_params, "l1_small_size": 32768, "trace_region_size": 60000000}

SEED = 123
WARMUP_PRESET = "V4_TURBO_12"  # presets differ only outside the transformers; any one warms the program cache

# inputs/ideogram4.json of spacelike-ai/gpu-benchmark: the docs/prompting.md example caption, keys in
# the model's order. Do not reorder: the model sees this exact byte string.
CAPTION = {
    "high_level_description": "A golden retriever riding a skateboard down a sunny sidewalk.",
    "style_description": {
        "aesthetics": "warm, playful, vibrant",
        "lighting": "bright afternoon sunlight, long soft shadows",
        "photo": "shallow depth of field, eye-level, 85mm lens",
        "medium": "photograph",
        "color_palette": ["#F5C542", "#87CEEB", "#4A4A4A", "#FFFFFF", "#2E8B57"],
    },
    "compositional_deconstruction": {
        "background": (
            "A sun-drenched suburban sidewalk lined with green hedges and a white picket fence. "
            "Dappled light filters through overhead trees."
        ),
        "elements": [
            {
                "type": "obj",
                "bbox": [200, 300, 800, 900],
                "desc": (
                    "A golden retriever with a fluffy coat, standing on a red skateboard with all four paws. "
                    "Its tongue is out and ears are flapping in the wind."
                ),
            },
            {
                "type": "obj",
                "bbox": [250, 750, 750, 950],
                "desc": "A worn red skateboard with black wheels rolling along the concrete sidewalk.",
            },
        ],
    },
}


def caption() -> str:
    return json.dumps(CAPTION, ensure_ascii=False, separators=(",", ":"))


class SectionTimer:
    """Wall-clock section times from the pipeline's events. Optionally synchronizes the mesh at the
    end of every denoise step so per-step times measure execution, not dispatch."""

    def __init__(self, mesh_device, sync_steps: bool) -> None:
        self.mesh_device = mesh_device
        self.sync_steps = sync_steps
        self.starts: dict[str, float] = {}
        self.durations: dict[str, float] = {}

    def __call__(self, event) -> None:
        if isinstance(event, SectionStart):
            self.starts[event.name] = time.perf_counter()
        elif isinstance(event, SectionEnd):
            if self.sync_steps and event.name.startswith("denoising_step_"):
                ttnn.synchronize_device(self.mesh_device)
            self.durations[event.name] = time.perf_counter() - self.starts[event.name]

    def step_ms(self) -> list[float]:
        steps = sorted(
            (int(k.rsplit("_", 1)[1]), v) for k, v in self.durations.items() if k.startswith("denoising_step_")
        )
        return [v * 1000 for _, v in steps]


def _git_rev() -> str | None:
    if os.environ.get("IDEOGRAM_BENCH_TT_METAL_REV"):  # the job's tree copy has no .git; pass it in
        return os.environ["IDEOGRAM_BENCH_TT_METAL_REV"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance only
        return None


def _generate(pipe, mesh_device, *, preset: str, size: int, traced: bool, sync_steps: bool):
    # The pipeline runs on its own full-mesh SUBMESH (pipe.mesh_device). Sync that, never the parent
    # mesh: a parent sync leaves the parent CQ in use, and closing the submesh at teardown then throws
    # "cq ID 0 is in use by parent mesh". (mesh_device is kept in the signature for callers.)
    dev = pipe.mesh_device
    timer = SectionTimer(dev, sync_steps)
    ttnn.synchronize_device(dev)
    start = time.perf_counter()
    images = pipe(prompts=[caption()], preset=preset, seed=SEED, height=size, width=size, traced=traced, on_event=timer)
    ttnn.synchronize_device(dev)
    wall_s = time.perf_counter() - start
    host_step_ms = timer.step_ms()
    # Synchronized window: denoising start -> decode start (latent readback + sync in between).
    denoising_s = timer.starts["vae"] - timer.starts["denoising"]
    stats = {
        "wall_s": wall_s,
        "text_encoder_ms": timer.durations["encoder"] * 1000,
        "denoising_s": denoising_s,
        # With step sync these are execution times; without it, per-step dispatch times, so the
        # mean step is taken from the synchronized denoising window instead.
        "step_ms": host_step_ms if timer.sync_steps else [denoising_s * 1000 / len(host_step_ms)] * len(host_step_ms),
        "host_step_ms": host_step_ms,
        "decoder_ms": timer.durations["vae"] * 1000,
    }
    return stats, images[0]


@_NEEDS_WEIGHTS
@pytest.mark.parametrize(
    ("mesh_device", "device_params"), [((4, 8), _RING_PIPE)], indirect=True, ids=["bh_galaxy_sp8tp4"]
)
def test_gpu_bench_match(*, mesh_device) -> None:
    size = int(os.environ.get("IDEOGRAM_BENCH_SIZE", "2048"))
    presets = os.environ.get("IDEOGRAM_BENCH_PRESETS", "V4_QUALITY_48,V4_DEFAULT_20").split(",")
    reps = int(os.environ.get("IDEOGRAM_BENCH_REPS", "3"))
    traced = os.environ.get("IDEOGRAM_BENCH_TRACED", "0") == "1"
    sync_steps = os.environ.get("IDEOGRAM_BENCH_SYNC_STEPS", "0") == "1"
    out_dir = Path(os.environ.get("IDEOGRAM_BENCH_OUT", "ideogram_bench_out"))
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    weights_dir = os.environ["IDEOGRAM4_WEIGHTS"]
    mode = "traced" if traced else "untraced"

    start = time.perf_counter()
    pipe = Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device, height=size, width=size)
    load_s = time.perf_counter() - start

    prompt_tokens = pipe.count_text_tokens(caption())
    logger.info(f"load {load_s:.1f}s | caption {len(caption())} chars, {prompt_tokens} tokens incl. chat template")

    # Warmup: compiles every op for this resolution (and the step trace, in traced mode).
    warm, _ = _generate(pipe, mesh_device, preset=WARMUP_PRESET, size=size, traced=traced, sync_steps=sync_steps)
    logger.info(f"warmup {WARMUP_PRESET}: {warm['wall_s']:.2f}s")

    failures = []
    for preset in presets:
        rid = f"ideogram4/fp8/{size}x{size}/{preset}"
        record = {
            "id": rid,
            "device": "Blackhole Galaxy 4x8, TP4 x SP8 ring",
            "weights": weights_dir,
            "revision": Path(weights_dir).name,
            "tt_metal_rev": _git_rev(),
            "preset": preset,
            "seed": SEED,
            "compile": mode,
            "sync_steps": sync_steps,
            "prompt_tokens": prompt_tokens,
            "load_s": load_s,
        }
        try:
            runs = []
            image = None
            for r in range(reps):
                stats, image = _generate(
                    pipe, mesh_device, preset=preset, size=size, traced=traced, sync_steps=sync_steps
                )
                runs.append(stats)
                logger.info(
                    f"{rid} rep {r}: wall {stats['wall_s']:.2f}s | encoder {stats['text_encoder_ms']:.0f}ms | "
                    f"denoise {stats['denoising_s']:.2f}s ({len(stats['step_ms'])} steps, first "
                    f"{stats['step_ms'][0]:.0f}ms, median {statistics.median(stats['step_ms']):.0f}ms) | "
                    f"vae {stats['decoder_ms']:.0f}ms"
                )
            image.save(out_dir / "images" / f"{rid.replace('/', '__')}__{mode}.png")
            record.update(
                steps=len(runs[0]["step_ms"]),
                wall_s=statistics.median(r["wall_s"] for r in runs),
                denoising_s=statistics.median(r["denoising_s"] for r in runs),
                step_ms=statistics.median(statistics.fmean(r["step_ms"]) for r in runs),
                text_encoder_ms=statistics.median(r["text_encoder_ms"] for r in runs),
                decoder_ms=statistics.median(r["decoder_ms"] for r in runs),
                reps=runs,
            )
            logger.info(
                f"RESULT {rid} {mode}: wall {record['wall_s']:.2f}s | denoise {record['denoising_s']:.2f}s | "
                f"step {record['step_ms']:.0f}ms | encoder {record['text_encoder_ms']:.0f}ms | "
                f"vae {record['decoder_ms']:.0f}ms"
            )
        except Exception as error:  # noqa: BLE001 - one failed preset must not drop the others
            logger.exception(f"{rid} failed")
            record["error"] = f"{type(error).__name__}: {error}"
            failures.append(rid)
        with (out_dir / "ideogram.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")

    assert not failures, f"failed presets: {failures}"
