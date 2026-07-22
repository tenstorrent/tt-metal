# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""On-demand perf dev-harness for ``BriaFiboPipeline``.

Builds the pipeline, does one warmup pass + N measured passes driving the pipeline's own stage methods
(``_encode`` / ``_prepare`` / ``_denoise`` / ``_decode_latents``), times each with boundary device syncs,
and logs a per-stage wall-clock breakdown (seconds + %, denoise it/s, images/s).

Three tests share one ``_perf_breakdown`` helper:
* ``test_fibo_pipeline_perf_breakdown`` -- a short free-text prompt, both untraced and traced (gs=5.0, CFG on).
* ``test_fibo_pipeline_perf_breakdown_json`` -- FIBO's intended structured-JSON prompt, read from the
  committed ``fibo_vlm_prompt.json`` (a real VLM text->JSON caption), at production gs=5.0 (CFG on).
* ``test_fibo_pipeline_device_profile`` -- a single-step, single-run untraced pass, Tracy-signposted, for
  a device-op perf CSV (see the device-profile command below).

Run the wall-clock breakdown:
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
  tt-perf-report generated/profiler/reports/.../ops_perf_results_*.csv --ignore-signposts   # WHOLE run (see below)

WARNING: this whole-pipeline report is NOT focused on the measured pass. The test emits a "fibo profile"
signpost, but at full-pipeline trace volume the host tracy-capture drops it (it never lands in the CSV), so
tt-perf-report analyzes the entire file -- pipe __init__ allocation run + warmup + measured, mixed together.
For focused per-stage reports (signpost survives, region bracketed exactly), use the per-component profiles.

Per-component device-op profiles (``test_fibo_{encode,denoise,decode}_device_profile``) live at the bottom
of this file. Each builds ONLY one component (encoder / transformer / VAE decoder) with synthetic inputs at
the real 1024^2 shapes and runs a single warmup + signposted measured forward -- no pipeline, no allocation
run -- so each yields a small, focused ``tt-perf-report`` for just that stage. See the section comment above
those tests for their run command.
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


def _num_links(mesh_device):
    """Ethernet links available per hop for CCL on this mesh (matches the pipeline's shape-driven pick).

    The 4x8 Blackhole Galaxy exposes only 2 channels per hop (num_links=4 -> fabric 'link index out of
    bounds' fatal); the 2x2 dev mesh supports 4. Unknown shapes fall back to the safe minimum of 1.
    """
    return {(2, 2): 4, (4, 8): 2}.get(tuple(mesh_device.shape), 1)


def _build_pipe(mesh_device, height, width, *, run_allocation_pass=True):
    from models.tt_dit.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline, BriaFiboPipelineConfig

    ckpt = _fibo_local()
    return BriaFiboPipeline(
        device=mesh_device,
        config=BriaFiboPipelineConfig.default(
            mesh_shape=mesh_device.shape, checkpoint_name=ckpt, height=height, width=width
        ),
        run_allocation_pass=run_allocation_pass,
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


# Run the 4x8 Galaxy wall-clock breakdown (-k "mesh_device1" selects the 4x8 mesh; append
# ' and traced' or ' and untraced' to -k for a single variant). Use -s to print the per-stage table:
#   HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
#     python_env/bin/python -m pytest \
#     models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_perf_breakdown \
#     -k "mesh_device1" -v -s --timeout=1800
@pytest.mark.parametrize("mesh_device", [(2, 2), (4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"])
@pytest.mark.parametrize("height, width, num_inference_steps, num_measured_runs", [(1024, 1024, 30, 3)])
@pytest.mark.parametrize("traced", [False, True], ids=["untraced", "traced"])
def test_fibo_pipeline_perf_breakdown(*, mesh_device, height, width, num_inference_steps, num_measured_runs, traced):
    """Per-stage wall-clock breakdown for a short free-text prompt on the mesh (2x2 or 4x8; gs=5.0 -> CFG on).

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
        negative_prompt="white background",
        guidance_scale=5.0,
        seed=0,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_measured_runs=num_measured_runs,
        traced=traced,
    )


# Run the 4x8 Galaxy wall-clock breakdown for the JSON prompt (-k "mesh_device1" selects the 4x8 mesh;
# append ' and traced' or ' and untraced' to -k for a single variant):
#   HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
#     python_env/bin/python -m pytest \
#     models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_perf_breakdown_json \
#     -k "mesh_device1" -v -s --timeout=1800
@pytest.mark.parametrize("mesh_device", [(2, 2), (4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"])
@pytest.mark.parametrize("height, width, num_inference_steps, num_measured_runs", [(1024, 1024, 30, 3)])
@pytest.mark.parametrize("traced", [False, True], ids=["untraced", "traced"])
def test_fibo_pipeline_perf_breakdown_json(
    *, mesh_device, height, width, num_inference_steps, num_measured_runs, traced
):
    """Per-stage wall-clock breakdown for FIBO's intended structured-JSON prompt (gs=5.0, production CFG).

    Reads the committed ``fibo_vlm_prompt.json`` (a real VLM text->JSON caption) and feeds
    it to the pipeline as the raw prompt string. This is the realistic production input
    (a longer prompt -> more prompt tokens than the free-text case).
    """

    if not _JSON_PROMPT_PATH.is_file():
        pytest.skip(f"JSON prompt fixture missing: {_JSON_PROMPT_PATH}")
    json_prompt = _JSON_PROMPT_PATH.read_text().strip()

    pipe = _build_pipe(mesh_device, height, width)
    _perf_breakdown(
        pipe,
        label="json",
        prompt=json_prompt,
        negative_prompt="blurry, low quality, distorted, watermark, text",
        guidance_scale=5.0,
        seed=0,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_measured_runs=num_measured_runs,
        traced=traced,
    )


# Run the 4x8 encode-only wall-clock perf (-k "mesh_device1" selects the 4x8 mesh). Use -s to print the timing:
#   HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
#     python_env/bin/python -m pytest \
#     models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_encode_perf \
#     -k "mesh_device1" -v -s --timeout=1800
@pytest.mark.parametrize("mesh_device", [(2, 2), (4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=["device_params"])
def test_fibo_encode_perf(*, mesh_device):
    """Encode-only wall-clock perf: time the pipeline's ``_encode`` (positive + negative prompt), no perf-report.

    Builds the full ``BriaFiboPipeline`` and times ONLY ``pipe._encode(prompt, negative_prompt, do_cfg=True)``
    -- the same encode the pipeline runs under CFG, encoding both prompts SEQUENTIALLY and returning
    ``(cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states)``. On the 4x8 Galaxy the encoder
    runs SP=8 (axis 1) x TP=4 (axis 0) on the whole mesh: the token sequence (padded to the fixed 1024 bucket)
    is sharded across the SP axis (all-gather K/V per attention layer) and Q/K/V/O are tensor-parallel on the
    TP axis. The encoder forward runs UNTRACED (the encoder trace was removed); the encode is dominated by
    host op-dispatch. The hidden-state readback reads only the SP shards from one TP row via
    get_device_tensors (~0.6 s) instead of the mesh composer over all 32 devices (~10 s). Positive prompt is FIBO's intended structured-JSON caption (the committed
    ``fibo_vlm_prompt.json``, ~833 tokens); negative is a short free-text string. 1 warmup (absorbs op
    compilation) + N measured passes with boundary device syncs; logs encode seconds (avg/min/max) and the
    produced output shapes. Plain pytest -- no Tracy, no signposts, no device-op CSV. Use ``-s`` to see the
    timing. Runtime ~ (1 warmup + N) encodes + ~44s model build, per mesh.
    """
    if not _JSON_PROMPT_PATH.is_file():
        pytest.skip(f"JSON prompt fixture missing: {_JSON_PROMPT_PATH}")
    prompt = _JSON_PROMPT_PATH.read_text().strip()  # FIBO's intended structured-JSON caption
    negative_prompt = "blurry, low quality, distorted, watermark, text"
    num_measured_runs = 3

    # encode is spatial-size independent; 1024x1024 just satisfies _build_pipe's DiT/VAE build.
    # run_allocation_pass=False: this test only calls _encode (never captures a denoise trace), so we
    # skip the __init__ full-generation warmup and its on-device VAE decode (conv3d).
    pipe = _build_pipe(mesh_device, 1024, 1024, run_allocation_pass=False)
    submesh = pipe._submesh

    def encode_once():
        ttnn.synchronize_device(submesh)  # drain enqueued work so t0 is a real boundary
        t0 = perf_counter()
        encoded = pipe._encode(prompt, negative_prompt, do_cfg=True)
        ttnn.synchronize_device(submesh)  # wait for this encode's device work to finish
        return perf_counter() - t0, encoded

    logger.info("encode perf: warmup run...")
    encode_once()  # warmup: compile/populate the program cache

    times = []
    encoded = None
    for r in range(num_measured_runs):
        logger.info(f"encode perf: measured run {r + 1}/{num_measured_runs}...")
        dt, encoded = encode_once()
        times.append(dt)

    # Produce/verify the outputs -- the same 4-tuple the pipeline's _encode returns under CFG.
    cond_embeds, cond_hidden_states, uncond_embeds, uncond_hidden_states = encoded
    assert cond_embeds is not None and len(cond_hidden_states) > 0, "positive branch produced no output"
    assert uncond_embeds is not None and len(uncond_hidden_states) > 0, "negative branch produced no output"

    avg, lo, hi = sum(times) / len(times), min(times), max(times)
    logger.info(
        f"\nFIBO encode perf -- avg of {num_measured_runs} runs (after 1 warmup)"
        f"\n  encode (pos+neg)  {avg:6.3f} s  [min {lo:.3f} / max {hi:.3f}]"
        f"\n  cond_embeds   {list(cond_embeds.shape)}  ({len(cond_hidden_states)} hidden-state layers)"
        f"\n  uncond_embeds {list(uncond_embeds.shape)}  ({len(uncond_hidden_states)} hidden-state layers)"
    )


# Run the 4x8 whole-pipeline device-op CSV (Tracy build required; -k "mesh_device1" selects the 4x8 mesh):
#   TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=24000 \
#     HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
#     python_env/bin/python -m tracy -r -p -v --dump-device-data-mid-run -m pytest \
#     models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_pipeline_device_profile \
#     -k "mesh_device1" --timeout=1800
#   tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv --ignore-signposts
@pytest.mark.parametrize("mesh_device", [(2, 2), (4, 8)], indirect=["mesh_device"])
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
      tt-perf-report generated/profiler/reports/.../ops_perf_results_*.csv --ignore-signposts

    IMPORTANT -- this is a WHOLE-RUN report, NOT a focused measured region. This test emits a "fibo profile"
    signpost around the measured pass, but at full-pipeline trace volume (~66k CSV rows / ~40 GB device log)
    the host tracy-capture DROPS the signpost message -- it never lands in the CSV, so tt-perf-report reports
    "No signposts found" and analyzes the entire file: the pipe ``__init__`` allocation generation + the
    warmup pass + the measured pass, all mixed (~16.7k merged ops). Verified 2026-07-10. For FOCUSED
    per-stage reports -- where the signpost DOES survive (small capture) and tt-perf-report brackets exactly
    the measured forward -- use the per-component profiles at the bottom of this file
    (``test_fibo_{encode,denoise,decode}_device_profile``). Keep this test as a whole-pipeline op-mix
    overview only.

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
    step (more steps -> "Device data mismatch"). The one warmup pass compiles every op and the single
    measured step captures each unique op once per branch (encode -> prepare -> denoise -> decode) at
    production gs=5.0 (CFG on -> cond + uncond). NOTE: since the signpost does not survive (see IMPORTANT
    above), the warmup pass's ops are NOT excluded -- they appear in the report alongside the measured ones.

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


# ---------------------------------------------------------------------------------------------------
# Per-component device-op profiles.
#
# Each test below builds ONLY one pipeline component (text encoder / transformer / VAE decoder) and runs
# it once with synthetic inputs at the SAME shapes the real pipeline uses at 1024^2. Nothing else runs --
# no pipeline, no __init__ allocation generation -- so the Tracy device-op capture stays small and each
# report is focused on just that component. (Profiling the whole pipeline to isolate one part emits ~15k+
# programs/pass and a multi-GB device log that overwhelms the host capture.) One forward captures every
# unique device op, so a per-op report needs a single warmup + single measured pass, bracketed by a
# per-stage Tracy signpost ("fibo encode" / "fibo denoise" / "fibo decode").
#
# WORKFLOW (Tracy build required) -- two steps, e.g. for denoise:
#
#   1. Record the raw device-op CSV. This captures the WHOLE process: model build + warmup forward +
#      measured forward, with one row PER DEVICE (the 2x2 mesh -> x4 rows/op).
#        TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=6000 \
#          HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
#          python_env/bin/python -m tracy -r -p -v --dump-device-data-mid-run -m pytest \
#          models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_denoise_device_profile \
#          --timeout=1800
#      -> generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv
#
#   2. Render + save ONLY the measured forward's ops into a per-stage folder. Passing the SAME name to
#      --start-signpost and --end-signpost brackets the region between the "measured begin" (1st instance)
#      and "measured end" (2nd instance) signposts; --csv writes the per-op table (a *_stacked.csv grouped
#      summary lands alongside it). tt-perf-report does NOT create the output folder, so mkdir it first:
#        mkdir -p denoise_report
#        tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
#          --start-signpost "fibo denoise" --end-signpost "fibo denoise" \
#          --csv denoise_report/ops.csv
#
#   Per stage -- swap the ::test, the signpost name, and the folder together:
#     encode   ::test_fibo_encode_device_profile    "fibo encode"    mkdir -p encode_report  --csv encode_report/ops.csv
#     denoise  ::test_fibo_denoise_device_profile   "fibo denoise"   mkdir -p denoise_report --csv denoise_report/ops.csv
#     decode   ::test_fibo_decode_device_profile    "fibo decode"    mkdir -p decode_report  --csv decode_report/ops.csv
#
# WHY the saved report is much smaller than the raw CSV (and still complete): the raw CSV holds every op
# once per device (2x2 -> x4, which tt-perf-report merges) AND the model build + the compile-warmup forward
# that run BEFORE the signpost. The signpost keeps ONLY the measured forward. E.g. denoise: ~26k raw rows
# -> /4 devices -> ~6.5k logical ops -> signpost -> 2934 measured-forward ops (build ~690 + warmup ~2934
# excluded). So you get exactly the one cache-warm forward's ops, deduped across the mesh -- nothing from
# the measured pass is dropped. (Without the --start/--end-signpost flags, tt-perf-report defaults to the
# last signpost; passing them explicitly is unambiguous. --ignore-signposts analyzes the entire raw file.)
# ---------------------------------------------------------------------------------------------------


def _profile_forward(mesh_device, header, forward):
    """Profile ONE component forward for a focused device-op report: 1 warmup + 1 measured pass.

    ``forward`` is a zero-arg callable that (re)builds its device inputs and runs the component once,
    returning its output (rebuilding per call so warmup and measured see identical fresh device state).
    Under ``python -m tracy`` the signpost brackets the region tt-perf-report focuses on, and (with
    ``--dump-device-data-mid-run``) the mid-run flush drains the warmup's markers so the measured pass's
    per-core marker buffer can't overflow. Everything here is a no-op under plain pytest, so the test
    still runs (producing no CSV) without the Tracy driver.
    """

    def _signpost(message):
        try:
            from tracy import signpost
        except Exception:
            return
        signpost(header=header, message=message)

    def _flush():
        # No-op unless under the driver's --dump-device-data-mid-run (TT_METAL_PROFILER_MID_RUN_DUMP=1).
        if os.environ.get("TT_METAL_PROFILER_MID_RUN_DUMP") != "1":
            return
        try:
            ttnn.ReadDeviceProfiler(mesh_device)
        except Exception as e:  # a real flush failure would silently drop markers -> surface it
            logger.warning(f"ReadDeviceProfiler flush failed ({e}); device-op markers may drop")

    forward()  # warmup: compile/populate the program cache (its one-time cost is before the signpost)
    ttnn.synchronize_device(mesh_device)
    _flush()  # drop the warmup's markers so they don't share the measured pass's buffer

    _signpost("measured begin")
    out = forward()
    ttnn.synchronize_device(mesh_device)
    _signpost("measured end")
    return out


# Run the 4x8 encode per-op report (Tracy build required; -k "mesh_device1" selects the 4x8 mesh):
#   TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=6000 \
#     HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
#     python_env/bin/python -m tracy -r -p -v --dump-device-data-mid-run -m pytest \
#     models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_encode_device_profile \
#     -k "mesh_device1" --timeout=1800
#   mkdir -p encode_report_4x8
#   tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
#     --start-signpost "fibo encode" --end-signpost "fibo encode" --csv encode_report_4x8/ops.csv
@pytest.mark.parametrize("mesh_device", [(2, 2), (4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_PROFILE_DEVICE_PARAMS], indirect=["device_params"])
def test_fibo_encode_device_profile(*, mesh_device):
    """Device-op profile of ONLY the SmolLM3 text encoder (encode stage), tensor-parallel on tp_axis (as in the pipeline).

    Builds just the encoder and runs one free-text prompt through it (its input is a string; the token
    count is the "shape"). 1 warmup + 1 signposted ("fibo encode") measured forward. See the section
    comment above for the Tracy command (swap in ::test_fibo_encode_device_profile and the "fibo encode"
    signpost); encode is light, so a small TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT suffices.
    """
    from models.tt_dit.parallel.config import EncoderParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.pipelines.bria_fibo.text_encoder import SmolLM3TextEncoderWrapper

    ckpt = _fibo_local()
    ccl = CCLManager(mesh_device, num_links=_num_links(mesh_device), topology=ttnn.Topology.Linear)
    # tp on axis 0, sp on axis 1 (SP mandatory) -- matches the pipeline's encoder_parallel_config and
    # test_fibo_wrapper_encode / tests/encoders/smollm3::test_smollm3_encoder_full_mesh.
    encoder = SmolLM3TextEncoderWrapper(
        ckpt,
        device=mesh_device,
        ccl_manager=ccl,
        parallel_config=EncoderParallelConfig.from_tuples(tp=(mesh_device.shape[0], 0), sp=(mesh_device.shape[1], 1)),
    )

    prompt_embeds, hidden_states = _profile_forward(
        mesh_device, "fibo encode", lambda: encoder.encode_prompt("a luxury sports car")
    )
    assert prompt_embeds is not None and len(hidden_states) > 0


# Run the 4x8 denoise per-op report (Tracy build required; -k "mesh_device1" selects the 4x8 mesh):
#   TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=6000 \
#     HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
#     python_env/bin/python -m tracy -r -p -v --dump-device-data-mid-run -m pytest \
#     models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_denoise_device_profile \
#     -k "mesh_device1" --timeout=1800
#   mkdir -p denoise_report_4x8
#   tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
#     --start-signpost "fibo denoise" --end-signpost "fibo denoise" --csv denoise_report_4x8/ops.csv
@pytest.mark.parametrize("mesh_device", [(2, 2), (4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_PROFILE_DEVICE_PARAMS], indirect=["device_params"])
@pytest.mark.parametrize("height, width", [(1024, 1024)])
def test_fibo_denoise_device_profile(*, mesh_device, height, width):
    """Device-op profile of ONLY the BriaFiboTransformer (denoise stage) at production sizes, 2x2 (sp=2, tp=2).

    Builds the full 46-block transformer and runs ONE forward at production spatial sequence 4096
    (1024^2 / 16^2 -> 64x64 patches) with synthetic inputs -- one forward captures every unique device op,
    so the report is small and focused. 1 warmup + 1 signposted ("fibo denoise") measured forward. (CFG
    runs this same forward twice per step; one forward is sufficient for a per-op report.) See the section
    comment above for the Tracy command; the full-depth forward needs the enlarged marker buffer +
    ``--dump-device-data-mid-run`` (e.g. TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=6000).
    """
    import torch

    from models.tt_dit.blocks.attention import Attention
    from models.tt_dit.models.transformers.transformer_bria_fibo import BriaFiboCheckpoint
    from models.tt_dit.parallel.config import DiTParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.utils.tensor import bf16_tensor

    sp_axis, tp_axis = 0, 1
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    ccl = CCLManager(mesh_device, num_links=_num_links(mesh_device), topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(sp_factor, sp_axis), tp=(tp_factor, tp_axis))

    checkpoint = BriaFiboCheckpoint(_fibo_local())
    model = checkpoint.build(ccl_manager=ccl, parallel_config=parallel_config)

    latent_h, latent_w = height // 16, width // 16
    spatial_seq_len = latent_h * latent_w  # 4096 at 1024^2
    prompt_seq_len = 128
    num_blocks = checkpoint._config.num_layers + checkpoint._config.num_single_layers  # 46

    torch.manual_seed(0)
    spatial = torch.randn(1, spatial_seq_len, checkpoint.in_channels).to(torch.bfloat16)
    prompt = torch.randn(1, prompt_seq_len, checkpoint.joint_attention_dim).to(torch.bfloat16)
    text_encoder_layers = [
        torch.randn(1, prompt_seq_len, checkpoint.text_encoder_dim).to(torch.bfloat16) for _ in range(num_blocks)
    ]
    timestep = torch.full((1,), 500.0).to(torch.bfloat16)

    # RoPE ids, Flux-style (txt = zeros, img = pixel grid; random ids are fine -- values don't change op shapes).
    text_ids = torch.zeros(prompt_seq_len, 3).to(torch.bfloat16)
    image_ids = torch.randint(height * width, (spatial_seq_len, 3)).to(torch.bfloat16)
    ids = torch.cat((text_ids, image_ids), dim=0).to(torch.bfloat16)
    rope_cos, rope_sin = checkpoint.pos_embed.forward(ids)

    # Pad the spatial sequence + spatial RoPE to the sp factor (host), then shard on the sp axis.
    spatial_padded = Attention.pad_spatial_sequence(spatial, sp_factor=sp_factor)
    rope_cos_sp = Attention.pad_spatial_sequence(rope_cos[prompt_seq_len:], sp_factor=sp_factor)
    rope_sin_sp = Attention.pad_spatial_sequence(rope_sin[prompt_seq_len:], sp_factor=sp_factor)
    padded_spatial_seq_len = spatial_padded.shape[1]

    def forward():
        return model.forward(
            spatial=bf16_tensor(spatial_padded, device=mesh_device, mesh_axis=sp_axis, shard_dim=1),
            prompt=bf16_tensor(prompt, device=mesh_device),
            timestep=bf16_tensor(timestep.unsqueeze(-1), device=mesh_device),
            text_encoder_layers=[bf16_tensor(t, device=mesh_device) for t in text_encoder_layers],
            spatial_rope=(
                bf16_tensor(rope_cos_sp, device=mesh_device, mesh_axis=sp_axis, shard_dim=0),
                bf16_tensor(rope_sin_sp, device=mesh_device, mesh_axis=sp_axis, shard_dim=0),
            ),
            prompt_rope=(
                bf16_tensor(rope_cos[:prompt_seq_len], device=mesh_device),
                bf16_tensor(rope_sin[:prompt_seq_len], device=mesh_device),
            ),
            spatial_sequence_length=padded_spatial_seq_len,
            prompt_sequence_length=prompt_seq_len,
        )

    out = _profile_forward(mesh_device, "fibo denoise", forward)
    assert out is not None


# Run the 4x8 decode per-op report (Tracy build required; -k "mesh_device1" selects the 4x8 mesh):
#   TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=6000 \
#     HF_HUB_OFFLINE=1 TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole \
#     python_env/bin/python -m tracy -r -p -v --dump-device-data-mid-run -m pytest \
#     models/tt_dit/tests/models/bria_fibo/test_performance_bria_fibo.py::test_fibo_decode_device_profile \
#     -k "mesh_device1" --timeout=1800
#   mkdir -p decode_report_4x8
#   tt-perf-report generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv \
#     --start-signpost "fibo decode" --end-signpost "fibo decode" --csv decode_report_4x8/ops.csv
@pytest.mark.parametrize("mesh_device", [(2, 2), (4, 8)], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_PROFILE_DEVICE_PARAMS], indirect=["device_params"])
@pytest.mark.parametrize("height, width", [(1024, 1024)])
def test_fibo_decode_device_profile(*, mesh_device, height, width):
    """Device-op profile of ONLY the on-device Wan VAE decoder (decode stage) at production 1024^2, 2x2 hw-parallel.

    Builds just the VAE decoder (as the pipeline does: height/width parallel over the 2x2 mesh) and decodes
    ONE synthetic production-shape latent (1, 48, 1, 64, 64) -> 1024x1024 -- small, focused report. 1 warmup
    + 1 signposted ("fibo decode") measured decode. See the section comment above for the Tracy command.
    """
    import torch

    from models.tt_dit.models.vae.vae_wan2_1 import WanVAEDecoderAdapter
    from models.tt_dit.parallel.config import VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager

    sp_axis, tp_axis = 0, 1
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    tp_factor = tuple(mesh_device.shape)[tp_axis]

    ccl = CCLManager(mesh_device, num_links=_num_links(mesh_device), topology=ttnn.Topology.Linear)
    # Height on the tp axis, width on the sp axis (matches the pipeline's vae_parallel_config).
    parallel_config = VaeHWParallelConfig.from_tuples(height=(tp_factor, tp_axis), width=(sp_factor, sp_axis))
    vae = WanVAEDecoderAdapter(
        checkpoint_name=_fibo_local(),
        parallel_config=parallel_config,
        ccl_manager=ccl,
        height=height,
        width=width,
        num_frames=1,
        vae_t_chunk_size=None,  # full-T single pass (T=1)
    )

    latent_h, latent_w = height // 16, width // 16
    torch.manual_seed(0)
    z = torch.randn(1, vae.config.z_dim, 1, latent_h, latent_w)  # BCTHW, z_dim=48

    out = _profile_forward(mesh_device, "fibo decode", lambda: vae.decode(z, output_type="pt"))
    assert out is not None
