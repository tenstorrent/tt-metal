# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-demand perf dev-harness for ``BriaFiboPipeline`` (NOT a CI perf gate).

Builds the pipeline, does one warmup pass + N measured passes driving the pipeline's own stage methods
(``_encode`` / ``_prepare`` / ``_denoise`` / ``_decode_latents``), times each with boundary device syncs,
and logs a per-stage wall-clock breakdown (seconds + %, denoise it/s, images/s). Approach B from
``docs/superpowers/specs/2026-07-09-fibo-perf-harness-design.md``: measures the real code path; no
tracing / on_event / CI-assert / device-op profiling (those are documented follow-ups).

Run explicitly (not collected into CI perf):
  HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \\
    python_env/bin/python -m pytest \\
    models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py -v -s
"""

import os
from time import perf_counter

import numpy as np
import pytest
from huggingface_hub import snapshot_download
from loguru import logger

import ttnn

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")

STAGES = ("encode", "prepare", "denoise", "decode")


def _fibo_local():
    try:
        return snapshot_download(FIBO_PATH, local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO not cached: {e}")


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 50000000}],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "height, width, num_inference_steps, num_measured_runs",
    [(1024, 1024, 30, 3)],
)
def test_fibo_pipeline_perf_breakdown(*, mesh_device, height, width, num_inference_steps, num_measured_runs):
    """Print a per-stage wall-clock breakdown of one 1024x1024/30-step generation on the 2x2 mesh.

    Sanity-asserts the produced image is valid + non-degenerate (proves the timed path really ran);
    it does NOT assert on timing (dev instrument, not a regression gate). Use ``-s`` to see the log.
    Runtime ~ (1 warmup + num_measured_runs) full generations (~30s each) + ~44s model build.
    """
    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig

    ckpt = _fibo_local()
    pipe = BriaFiboPipeline(
        device=mesh_device,
        config=BriaFiboPipelineConfig.default(
            mesh_shape=mesh_device.shape, checkpoint_name=ckpt, height=height, width=width
        ),
    )
    submesh = pipe._submesh
    prompt = "a luxury sports car"
    negative_prompt = ""
    guidance_scale = 5.0
    seed = 0

    def run_once(record: dict):
        """One full generation driving the stage methods; records per-stage seconds into ``record``."""

        def time_stage(name, fn):
            ttnn.synchronize_device(submesh)  # drain enqueued work so t0 is a real boundary
            t0 = perf_counter()
            result = fn()
            ttnn.synchronize_device(submesh)  # wait for this stage's device work to finish
            record[name] = perf_counter() - t0
            return result

        encoded = time_stage("encode", lambda: pipe._encode(prompt, negative_prompt))
        cond_branch, uncond_branch, timesteps, latent, ssl = time_stage(
            "prepare",
            lambda: pipe._prepare(
                encoded,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                seed=seed,
                latents=None,
            ),
        )
        latent = time_stage(
            "denoise",
            lambda: pipe._denoise(cond_branch, uncond_branch, timesteps, latent, ssl, guidance_scale),
        )
        image = time_stage(
            "decode",
            lambda: pipe._decode_latents(
                latent, height=height, width=width, output_type="pil", force_device_decode=True
            ),
        )
        return image

    # 1 warmup pass (absorbs op compilation), then N measured passes.
    logger.info("perf harness: warmup run...")
    run_once({})

    runs = []
    image = None
    for r in range(num_measured_runs):
        logger.info(f"perf harness: measured run {r + 1}/{num_measured_runs}...")
        record = {}
        image = run_once(record)
        runs.append(record)

    # Sanity: the last image is a valid, non-degenerate frame (proves the timed path really ran).
    arr = np.asarray(image[0])
    assert arr.shape == (height, width, 3), f"unexpected image shape {arr.shape}"
    assert arr.std() > 1.0, f"image looks degenerate (std={arr.std():.4f})"

    # Aggregate across measured runs and print the breakdown.
    avg = {s: sum(run[s] for run in runs) / len(runs) for s in STAGES}
    lo = {s: min(run[s] for run in runs) for s in STAGES}
    hi = {s: max(run[s] for run in runs) for s in STAGES}
    total = sum(avg[s] for s in STAGES)

    lines = [
        f"\nFIBO perf breakdown — {width}x{height}, {num_inference_steps} steps, "
        f"avg of {num_measured_runs} runs (after 1 warmup)"
    ]
    for s in STAGES:
        pct = 100.0 * avg[s] / total if total else 0.0
        extra = ""
        if s == "prepare":
            extra = "RoPE recompute + 92-tensor upload"
        elif s == "denoise" and avg[s]:
            extra = f"-> {num_inference_steps / avg[s]:.2f} it/s"
        lines.append(f"  {s:<9} {avg[s]:7.2f} s  ({pct:4.1f}%)  [min {lo[s]:6.2f} / max {hi[s]:6.2f}]  {extra}")
    lines.append("  " + "-" * 62)
    images_per_s = 1.0 / total if total else 0.0
    lines.append(f"  {'total':<9} {total:7.2f} s             -> {images_per_s:.4f} images/s")
    logger.info("\n".join(lines))
