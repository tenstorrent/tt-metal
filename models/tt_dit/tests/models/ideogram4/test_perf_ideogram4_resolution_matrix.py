# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4 perf over the API's output sizes: size tier x aspect ratio x rendering speed.
#
# The sizes are the 14 of the tt-media-server Ideogram 4 API (tt-inference-server PR 5466,
# docs/api/generation/models/ideogram-ai/ideogram-4.yaml): `size` 1K / 2K x `aspect_ratio`, and the
# rendering speeds turbo / default / quality map to V4_TURBO_12 / V4_DEFAULT_20 / V4_QUALITY_48.
#
# One pipeline load serves every size (height and width are per call). Order is size-outer:
#   - one untimed warmup per size (the first call at a new shape compiles it)
#   - per preset, IDEOGRAM_MATRIX_DISCARD untimed reps (the first calls after a warmup run slow),
#     then IDEOGRAM_MATRIX_REPS timed reps; the record carries the medians
# Every call uses the same caption file and seed, so only the size and preset change.
#
# Outputs under IDEOGRAM_MATRIX_OUT:
#   runs.jsonl     one row per call (warmup, discard and timed), with timings
#   results.jsonl  one row per (size, preset): medians over the timed reps
#   images/<tier>_<ratio>__<preset>.png   last timed image of each config
#   run.json       node, power limit, tree revision, caption, load time
#
# Knobs (env):
#   IDEOGRAM4_WEIGHTS        fp8 checkpoint dir (required)
#   IDEOGRAM_MATRIX_CAPTION  model-ready caption file (required)
#   IDEOGRAM_MATRIX_TIERS    comma list of 1K,2K                    (default 1K,2K)
#   IDEOGRAM_MATRIX_RATIOS   comma list of the API's aspect ratios  (default all 7)
#   IDEOGRAM_MATRIX_PRESETS  comma list                             (default V4_TURBO_12,V4_DEFAULT_20,V4_QUALITY_48)
#   IDEOGRAM_MATRIX_REPS     timed reps per config                  (default 5)
#   IDEOGRAM_MATRIX_DISCARD  untimed reps per config before those   (default 1)
#   IDEOGRAM_MATRIX_OUT      output dir                             (default ./ideogram_matrix_out)
#   IDEOGRAM_MATRIX_TRACED   1 = traced denoise step, 0 = eager dispatch (default 0: serving runs eager)
#   IDEOGRAM_BATCH_TDP       power-limit string the job script read from tt-smi (provenance only)
#
# Timing is the GPU-bench test's (see test_perf_ideogram4_gpu_bench.py). Eager at 2048x2048 hung in the
# sliced VAE decode on 2026-10-28; a hang here shows as the pytest timeout.
# =============================================================================

import json
import os
import socket
import statistics
import time
from pathlib import Path

import pytest
import ttnn
from loguru import logger

from ....pipelines.ideogram4.pipeline_ideogram4 import Ideogram4Pipeline
from .test_perf_ideogram4_gpu_bench import _RING_PIPE, SectionTimer, _git_rev

_REQUIRED = ("IDEOGRAM4_WEIGHTS", "IDEOGRAM_MATRIX_CAPTION")
_NEEDS_INPUTS = pytest.mark.skipif(
    not all(os.environ.get(k) for k in _REQUIRED), reason=f"needs env {', '.join(_REQUIRED)}"
)

SEED = 123
WARMUP_PRESET = "V4_TURBO_12"

# (tier, aspect ratio) -> (width, height), the API's table.
SIZES = {
    ("1K", "1:1"): (1024, 1024),
    ("1K", "16:9"): (1280, 720),
    ("1K", "9:16"): (720, 1280),
    ("1K", "4:3"): (1152, 864),
    ("1K", "3:4"): (864, 1152),
    ("1K", "3:2"): (1248, 832),
    ("1K", "2:3"): (832, 1248),
    ("2K", "1:1"): (2048, 2048),
    ("2K", "16:9"): (2048, 1152),
    ("2K", "9:16"): (1152, 2048),
    ("2K", "4:3"): (2048, 1536),
    ("2K", "3:4"): (1536, 2048),
    ("2K", "3:2"): (2048, 1360),
    ("2K", "2:3"): (1360, 2048),
}
RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]


@_NEEDS_INPUTS
@pytest.mark.parametrize(
    ("mesh_device", "device_params"), [((4, 8), _RING_PIPE)], indirect=True, ids=["bh_galaxy_sp8tp4"]
)
def test_resolution_matrix(*, mesh_device) -> None:
    caption_path = Path(os.environ["IDEOGRAM_MATRIX_CAPTION"])
    tiers = os.environ.get("IDEOGRAM_MATRIX_TIERS", "1K,2K").split(",")
    ratios = os.environ.get("IDEOGRAM_MATRIX_RATIOS", ",".join(RATIOS)).split(",")
    presets = os.environ.get("IDEOGRAM_MATRIX_PRESETS", "V4_TURBO_12,V4_DEFAULT_20,V4_QUALITY_48").split(",")
    reps = int(os.environ.get("IDEOGRAM_MATRIX_REPS", "5"))
    discard = int(os.environ.get("IDEOGRAM_MATRIX_DISCARD", "1"))
    out_dir = Path(os.environ.get("IDEOGRAM_MATRIX_OUT", "ideogram_matrix_out"))
    traced = os.environ.get("IDEOGRAM_MATRIX_TRACED", "0") == "1"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    sizes = [(tier, ratio, *SIZES[(tier, ratio)]) for tier in tiers for ratio in ratios]
    text = caption_path.read_text(encoding="utf-8")

    t0 = time.perf_counter()
    pipe = Ideogram4Pipeline.create_pipeline(mesh_device=mesh_device, height=sizes[0][3], width=sizes[0][2])
    load_s = time.perf_counter() - t0
    dev = pipe.mesh_device  # sync the pipeline's submesh, never the parent mesh (see the GPU-bench test)
    tokens = pipe.count_text_tokens(text)

    run_info = {
        "node": socket.gethostname(),
        "tdp": os.environ.get("IDEOGRAM_BATCH_TDP"),
        "tt_metal_rev": _git_rev(),
        "weights": os.environ["IDEOGRAM4_WEIGHTS"],
        "caption": str(caption_path),
        "caption_chars": len(text),
        "text_tokens": tokens,
        "seed": SEED,
        "sizes": [f"{t} {r} {w}x{h}" for t, r, w, h in sizes],
        "presets": presets,
        "reps": reps,
        "discard": discard,
        "mode": "traced" if traced else "eager",
        "load_s": round(load_s, 1),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out_dir / "run.json").write_text(json.dumps(run_info, indent=2))
    logger.info(f"load {load_s:.1f}s | {run_info['mode']} | {len(sizes)} sizes x {presets} | {reps} reps + {discard} discarded")

    def generate(preset: str, width: int, height: int):
        timer = SectionTimer(dev, sync_steps=False)
        ttnn.synchronize_device(dev)
        start = time.perf_counter()
        image = pipe(
            prompts=[text], preset=preset, seed=SEED, height=height, width=width, traced=traced, on_event=timer
        )[0]
        ttnn.synchronize_device(dev)
        wall_s = time.perf_counter() - start
        steps = len(timer.step_ms())
        denoising_s = timer.starts["vae"] - timer.starts["denoising"]
        return image, {
            "wall_s": round(wall_s, 3),
            "text_encoder_ms": round(timer.durations["encoder"] * 1000, 1),
            "denoising_s": round(denoising_s, 3),
            "steps": steps,
            "mean_step_ms": round(denoising_s * 1000 / steps, 1),
            "decoder_ms": round(timer.durations["vae"] * 1000, 1),
        }

    def log_run(row: dict) -> None:
        with (out_dir / "runs.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    failures = []
    for tier, ratio, width, height in sizes:
        base = {"tier": tier, "aspect_ratio": ratio, "width": width, "height": height}
        try:
            _, warm = generate(WARMUP_PRESET, width, height)
            log_run({**base, "preset": WARMUP_PRESET, "kind": "warmup", **warm})
            logger.info(f"{tier} {ratio} {width}x{height} warmup: {warm['wall_s']:.1f}s")
        except Exception as error:  # noqa: BLE001 - one bad size must not drop the rest
            logger.exception(f"{tier} {ratio} {width}x{height} warmup failed")
            failures.append(f"{tier} {ratio} warmup")
            for preset in presets:
                with (out_dir / "results.jsonl").open("a") as f:
                    f.write(json.dumps({**base, "preset": preset, "error": f"warmup: {type(error).__name__}: {error}"}) + "\n")
            continue

        for preset in presets:
            record = {**base, "preset": preset, "node": run_info["node"], "tdp": run_info["tdp"],
                      "tt_metal_rev": run_info["tt_metal_rev"], "mode": run_info["mode"], "text_tokens": tokens}
            try:
                runs = []
                for i in range(discard + reps):
                    image, stats = generate(preset, width, height)
                    kind = "discard" if i < discard else "timed"
                    log_run({**base, "preset": preset, "kind": kind, "rep": i - discard, **stats})
                    if kind == "timed":
                        runs.append(stats)
                image.save(out_dir / "images" / f"{tier}_{ratio.replace(':', 'x')}__{preset}.png")
                for key in ("wall_s", "text_encoder_ms", "denoising_s", "mean_step_ms", "decoder_ms"):
                    record[key] = round(statistics.median(r[key] for r in runs), 3)
                record.update(steps=runs[0]["steps"], reps=len(runs),
                              wall_s_min=min(r["wall_s"] for r in runs), wall_s_max=max(r["wall_s"] for r in runs))
                logger.info(
                    f"RESULT {tier} {ratio} {width}x{height} {preset}: wall {record['wall_s']:.2f}s | "
                    f"step {record['mean_step_ms']:.0f}ms | encoder {record['text_encoder_ms']:.0f}ms | "
                    f"vae {record['decoder_ms']:.0f}ms"
                )
            except Exception as error:  # noqa: BLE001
                logger.exception(f"{tier} {ratio} {preset} failed")
                record["error"] = f"{type(error).__name__}: {error}"
                failures.append(f"{tier} {ratio} {preset}")
            with (out_dir / "results.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")

    assert not failures, f"failed: {failures}"
