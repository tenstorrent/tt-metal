# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-demand perf dev-harness for ``BriaFiboPipeline`` (NOT a CI perf gate).

Builds the pipeline, does one warmup pass + N measured passes driving the pipeline's own stage methods
(``_encode`` / ``_prepare`` / ``_denoise`` / ``_decode_latents``), times each with boundary device syncs,
and logs a per-stage wall-clock breakdown (seconds + %, denoise it/s, images/s). Approach B from
``docs/superpowers/specs/2026-07-09-fibo-perf-harness-design.md``: measures the real code path; no
tracing / on_event / CI-assert / device-op profiling (those are documented follow-ups).

Two tests share one ``_perf_breakdown`` helper:
* ``test_fibo_pipeline_perf_breakdown`` -- a short free-text prompt (gs=1.0 -> no-CFG gate).
* ``test_fibo_pipeline_perf_breakdown_json`` -- FIBO's intended structured-JSON prompt, read from the
  committed ``fibo_vlm_prompt.json`` (a real VLM text->JSON caption), at production gs=5.0 (CFG on).

The helper honors the CFG gate (``guidance_scale > 1``): at gs<=1 it skips the uncond branch, so the
measured cost reflects what ``BriaFiboPipeline.__call__`` actually does at that guidance_scale.

Run explicitly (not collected into CI perf):
  HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \\
    python_env/bin/python -m pytest \\
    models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py -v -s
"""

import os
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from huggingface_hub import snapshot_download
from loguru import logger

import ttnn

FIBO_PATH = os.environ.get("FIBO_PATH", "briaai/FIBO")

STAGES = ("encode", "prepare", "denoise", "decode")

# FIBO's intended input is a structured JSON caption (VLM text->JSON output); this committed fixture is
# a real one captured by the VLM->image e2e test (test_vlm_pipeline.py).
_JSON_PROMPT_PATH = Path(__file__).parent / "fibo_vlm_prompt.json"

# trace_region_size holds BOTH resident denoise traces (cond + uncond). Each is ~70 MB at 1024² (the
# 4096-token spatial sequence dominates, so prompt length barely matters), so ~200 MB gives headroom.
_DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 200000000}


def _fibo_local():
    try:
        return snapshot_download(FIBO_PATH, local_files_only=True)
    except Exception as e:
        pytest.skip(f"FIBO not cached: {e}")


def _build_pipe(mesh_device, height, width):
    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig

    ckpt = _fibo_local()
    return BriaFiboPipeline(
        device=mesh_device,
        config=BriaFiboPipelineConfig.default(
            mesh_shape=mesh_device.shape, checkpoint_name=ckpt, height=height, width=width
        ),
    )


def _perf_breakdown(
    pipe,
    *,
    label,
    prompt,
    guidance_scale,
    seed,
    height,
    width,
    num_inference_steps,
    num_measured_runs,
    negative_prompt="",
    traced=False,
):
    """Time one generation's stages (1 warmup + N measured), assert the image is valid, log a breakdown.

    Drives the pipeline's own stage methods so it measures the real code path. Honors the CFG gate: at
    ``guidance_scale <= 1`` the uncond branch is skipped (single cond forward/step), matching ``__call__``.
    """
    submesh = pipe._submesh
    do_cfg = guidance_scale > 1

    def run_once(record: dict):
        def time_stage(name, fn):
            ttnn.synchronize_device(submesh)  # drain enqueued work so t0 is a real boundary
            t0 = perf_counter()
            result = fn()
            ttnn.synchronize_device(submesh)  # wait for this stage's device work to finish
            record[name] = perf_counter() - t0
            return result

        encoded = time_stage("encode", lambda: pipe._encode(prompt, negative_prompt, do_cfg=do_cfg))
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
            lambda: pipe._denoise(cond_branch, uncond_branch, timesteps, latent, ssl, guidance_scale, traced=traced),
        )
        image = time_stage(
            "decode",
            lambda: pipe._decode_latents(
                latent, height=height, width=width, output_type="pil", force_device_decode=True
            ),
        )
        return image

    # 1 warmup pass (absorbs op compilation), then N measured passes.
    logger.info(f"perf harness [{label}]: warmup run...")
    run_once({})

    runs = []
    image = None
    for r in range(num_measured_runs):
        logger.info(f"perf harness [{label}]: measured run {r + 1}/{num_measured_runs}...")
        record = {}
        image = run_once(record)
        runs.append(record)

    # Save the last produced image with a timestamped name (runs don't overwrite) so the output can be
    # eyeballed for correctness. Saved BEFORE the asserts so even a degenerate frame lands on disk.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path.cwd() / f"fibo_perf_{label}_{width}x{height}_{num_inference_steps}steps_gs{guidance_scale}_{ts}.png"
    image[0].save(out_path)
    logger.info(f"saved last image -> {out_path}")

    # Sanity: the last image is a valid, non-degenerate frame (proves the timed path really ran).
    arr = np.asarray(image[0])
    assert arr.shape == (height, width, 3), f"unexpected image shape {arr.shape}"
    assert arr.std() > 1.0, f"image looks degenerate (std={arr.std():.4f})"
    assert np.unique(arr).size > 16, f"image looks degenerate ({np.unique(arr).size} unique values)"

    # Aggregate across measured runs and print the breakdown.
    avg = {s: sum(run[s] for run in runs) / len(runs) for s in STAGES}
    lo = {s: min(run[s] for run in runs) for s in STAGES}
    hi = {s: max(run[s] for run in runs) for s in STAGES}
    total = sum(avg[s] for s in STAGES)

    cfg_note = "CFG on (2 fwd/step)" if do_cfg else "no-CFG gate (1 fwd/step)"
    trace_note = "traced" if traced else "untraced"
    lines = [
        f"\nFIBO perf breakdown [{label}] — {width}x{height}, {num_inference_steps} steps, "
        f"gs={guidance_scale} [{cfg_note}, {trace_note}], avg of {num_measured_runs} runs (after 1 warmup)"
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

    # Free the resident denoise traces so a subsequent build/test starts with a clean trace region.
    if traced:
        pipe.release_traces()


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"])
@pytest.mark.parametrize("height, width, num_inference_steps, num_measured_runs", [(1024, 1024, 30, 3)])
@pytest.mark.parametrize("traced", [False, True], ids=["untraced", "traced"])
def test_fibo_pipeline_perf_breakdown(*, mesh_device, height, width, num_inference_steps, num_measured_runs, traced):
    """Per-stage wall-clock breakdown for a short free-text prompt on the 2x2 mesh (gs=5.0 -> CFG on).

    Runs both untraced and traced (parametrized) so the denoise it/s delta from tracing is visible in
    one command. Sanity-asserts the produced image is valid + non-degenerate (proves the timed path
    really ran); it does NOT assert on timing (dev instrument, not a regression gate). Use ``-s`` to
    see the log. Runtime ~ (1 warmup + num_measured_runs) generations + ~44s model build, per param.
    """
    pipe = _build_pipe(mesh_device, height, width)
    _perf_breakdown(
        pipe,
        label="text",
        prompt="a luxury sports car",
        guidance_scale=5.0,
        seed=0,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_measured_runs=num_measured_runs,
        traced=traced,
    )


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"])
@pytest.mark.parametrize("height, width, num_inference_steps, num_measured_runs", [(1024, 1024, 30, 3)])
def test_fibo_pipeline_perf_breakdown_json(*, mesh_device, height, width, num_inference_steps, num_measured_runs):
    """Per-stage wall-clock breakdown for FIBO's intended structured-JSON prompt (gs=5.0, production CFG).

    Reads the committed ``fibo_vlm_prompt.json`` (a real VLM text->JSON caption) and feeds it to the
    pipeline as the raw prompt string -- the same handoff the VLM->image e2e test uses. This is the
    realistic production input (a longer prompt -> more prompt tokens than the free-text case). Sanity-only
    asserts; use ``-s`` to see the breakdown.
    """
    if not _JSON_PROMPT_PATH.is_file():
        pytest.skip(f"JSON prompt fixture missing: {_JSON_PROMPT_PATH}")
    json_prompt = _JSON_PROMPT_PATH.read_text().strip()  # drop the fixture's trailing newline

    pipe = _build_pipe(mesh_device, height, width)
    _perf_breakdown(
        pipe,
        label="json",
        prompt=json_prompt,
        guidance_scale=5.0,
        seed=0,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_measured_runs=num_measured_runs,
    )
