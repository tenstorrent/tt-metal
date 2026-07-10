# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-demand perf dev-harness for ``BriaFiboPipeline`` (NOT a CI perf gate).

Builds the pipeline, does one warmup pass + N measured passes driving the pipeline's own stage methods
(``_encode`` / ``_prepare`` / ``_denoise`` / ``_decode_latents``), times each with boundary device syncs,
and logs a per-stage wall-clock breakdown (seconds + %, denoise it/s, images/s). Approach B from
``docs/superpowers/specs/2026-07-09-fibo-perf-harness-design.md``: measures the real code path (host
wall-clock, no ``on_event`` / CI-assert).

Three tests share one ``_perf_breakdown`` helper:
* ``test_fibo_pipeline_perf_breakdown`` -- a short free-text prompt, both untraced and traced (gs=5.0, CFG on).
* ``test_fibo_pipeline_perf_breakdown_json`` -- FIBO's intended structured-JSON prompt, read from the
  committed ``fibo_vlm_prompt.json`` (a real VLM text->JSON caption), at production gs=5.0 (CFG on).
* ``test_fibo_pipeline_device_profile`` -- a single-step, single-run untraced pass, Tracy-signposted, for
  a device-op perf CSV (see the device-profile command below).

The helper honors the CFG gate (``guidance_scale > 1``): at gs<=1 it skips the uncond branch, so the
measured cost reflects what ``BriaFiboPipeline.__call__`` actually does at that guidance_scale.

Run the wall-clock breakdown (not collected into CI perf):
  HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \\
    python_env/bin/python -m pytest \\
    models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py -v -s

Run the device-op profile (needs a Tracy-enabled build -- the default ``./build_metal.sh``) and turn the
CSV into a per-op report. See ``test_fibo_pipeline_device_profile`` below for why each extra flag is
required (buffer size, mid-run dump, timeout) -- without them the post-process aborts with
"Device data missing".
  TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=24000 \\
    HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \\
    python_env/bin/python -m tracy -r -p -v --dump-device-data-mid-run -m pytest \\
    models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_device_profile \\
    --timeout=1800
  pip install tt-perf-report
  tt-perf-report generated/profiler/reports/.../ops_perf_results_*.csv   # auto-focuses the "fibo profile" signpost region
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

# The device-op profile runs untraced, so it needs no trace region -- drop it to free DRAM for the
# enlarged on-device profiler marker buffer (TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT).
_PROFILE_DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}


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
    signpost_label=None,
    check_image=True,
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
            _read_profiler()  # profile mode: flush this stage's markers so the per-core buffer can't overflow
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

    # Device-profile instrumentation, all no-ops unless signpost_label is set (so the plain-pytest
    # wall-clock tests are unaffected). Two pieces:
    #  * _read_profiler() flushes the on-device profiler after every stage so no per-core marker buffer
    #    overflows across a generation. A full FIBO generation (SmolLM3 encode + denoise + VAE decode) emits
    #    far more than the 12000-marker/core buffer holds; without frequent flushes markers are dropped and
    #    the ops-report post-process aborts with "Device data missing". Needs the driver's
    #    --dump-device-data-mid-run (sets TT_METAL_PROFILER_MID_RUN_DUMP=1) for the mid-run reads to take.
    #  * _signpost() brackets the measured passes so tt-perf-report focuses on the post-warmup region.
    def _read_profiler():
        # No-op unless profiling under --dump-device-data-mid-run (TT_METAL_PROFILER_MID_RUN_DUMP=1); then
        # flush this stage's markers so the per-core buffer can't overflow (see the block above). Gating on
        # the env var (not just signpost_label) keeps the plain-pytest wall-clock tests untouched and matches
        # test_transformer.py::_flush_profiler.
        if not signpost_label or os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP") != "1":
            return
        try:
            ttnn.ReadDeviceProfiler(submesh)
        except Exception as e:  # a real flush failure would silently drop markers -> surface it
            logger.warning(f"ReadDeviceProfiler flush failed ({e}); device-op markers may drop")

    def _signpost(message):
        if not signpost_label:
            return
        try:
            from tracy import signpost
        except Exception:
            return
        signpost(header=signpost_label, message=message)

    # 1 warmup pass (absorbs op compilation), then N measured passes.
    logger.info(f"perf harness [{label}]: warmup run...")
    run_once({})

    runs = []
    image = None
    _signpost("measured passes begin")
    for r in range(num_measured_runs):
        logger.info(f"perf harness [{label}]: measured run {r + 1}/{num_measured_runs}...")
        record = {}
        image = run_once(record)
        runs.append(record)
    _signpost("measured passes end")

    # Save the last produced image with a timestamped name (runs don't overwrite) so the output can be
    # eyeballed for correctness. Saved BEFORE the asserts so even a degenerate frame lands on disk.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path.cwd() / f"fibo_perf_{label}_{width}x{height}_{num_inference_steps}steps_gs{guidance_scale}_{ts}.png"
    image[0].save(out_path)
    logger.info(f"saved last image -> {out_path}")

    arr = np.asarray(image[0])
    assert arr.shape == (height, width, 3), f"unexpected image shape {arr.shape}"
    if check_image:
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


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_PROFILE_DEVICE_PARAMS], indirect=["device_params"])
@pytest.mark.parametrize("height, width, num_inference_steps", [(1024, 1024, 1)])
def test_fibo_pipeline_device_profile(*, mesh_device, height, width, num_inference_steps):
    """Single-step, single-run untraced pass that emits a Tracy signpost, for a device-op perf CSV.

    This is the one-command device profile of the full pipeline (encode -> prepare -> denoise -> decode).
    Run it under the Tracy driver (needs a Tracy-enabled build, i.e. the default ``./build_metal.sh``) to
    record a per-op CSV, then render it with tt-perf-report. FULL COMMAND (verified to pass, ~8 min):

      TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=24000 \\
        HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \\
        python_env/bin/python -m tracy -r -p -v --dump-device-data-mid-run -m pytest \\
        models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_device_profile \\
        --timeout=1800
      pip install tt-perf-report
      tt-perf-report generated/profiler/reports/.../ops_perf_results_*.csv   # auto-focuses the "fibo profile" signpost region

    Why each non-obvious flag is REQUIRED (a full pipeline emits far more markers than one op, so the
    profiler defaults are too small; without these the run drops markers and the post-process aborts with
    "Device data missing: Op N not present in cpp_device_perf_report.csv"):
      * ``TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=24000`` -- enlarges the on-device profiler marker buffer
        (default 1000 programs/core). One 1024^2 pipeline pass dispatches ~15k programs (with a hotspot
        core); at the default the per-core buffer overflows ("markers were dropped"), a dropped op then has
        no device data, and post-processing asserts. The buffer scales linearly with this count (see
        ``tt_metal/impl/profiler/profiler_state_manager.cpp``).
      * ``--dump-device-data-mid-run`` -- sets ``TT_METAL_PROFILER_MID_RUN_DUMP=1`` so the per-stage
        ``ttnn.ReadDeviceProfiler()`` flushes in ``_perf_breakdown`` actually drain markers between stages
        (otherwise the reads defer to the end of the run and the buffer still overflows within a stage).
      * ``--timeout=1800`` -- raises pytest-timeout above its 300s default; reading the enlarged buffer off
        all 4 devices per stage is slow and the whole run takes ~8 min.
      * ``_PROFILE_DEVICE_PARAMS`` (this test's ``device_params``) drops the 200 MB ``trace_region_size``
        (this profile is untraced), freeing DRAM for the enlarged marker buffer.

    Untraced on purpose: the device profiler + ttnn metal-trace only compose for a single trace-execution
    step (more steps -> "Device data mismatch"). The one warmup pass compiles every op (its one-time cost
    is excluded from the CSV by the signpost); the single measured step then captures each unique op once
    per branch (encode -> prepare -> denoise -> decode) at production gs=5.0 (CFG on -> cond + uncond). The
    signpost ("fibo profile") bounds the region tt-perf-report focuses on to that measured step.

    NOTE on interpreting the mix: at 1 step the one-time stages dominate (VAE decode ``Conv3d`` ~52%);
    at production step counts the denoise transformer dominates instead. For the per-forward denoise cost,
    profile the DIT alone via ``test_transformer.py::test_fibo_transformer_mesh_profile``.

    Runs fine under plain pytest too (the signpost is a no-op), but produces no CSV without the driver.
    """
    pipe = _build_pipe(mesh_device, height, width)
    _perf_breakdown(
        pipe,
        label="profile",
        prompt="a luxury sports car",
        guidance_scale=5.0,
        seed=0,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_measured_runs=1,
        traced=False,
        signpost_label="fibo profile",
        check_image=False,
    )
