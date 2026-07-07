# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Stage-scoped per-op capture for the LTX distilled pipeline.

Generalizes the single-op AllGather scoped capture (``test_ccl_allgather_scoped``) to WHOLE stages.
``LTX_PROFILE_STAGES=vae,audio,upsample`` selects which decode-tail stage(s) to run in isolation and
DRAINS the device profiler (``ttnn.ReadDeviceProfiler``) at each stage boundary, so the chosen
stages' perf-counter markers land in the CSV instead of being dropped when a dense block overflows
the marker DRAM buffer. Pick a stage group that FITS the buffer — see the segmentation map in
``tt_metal/tools/profiler/PERF_COUNTERS.md``.

Why the decode tail: audio / VAE / upsample decode run on a synthetic latent of the exact served
shape and never touch the 22B transformer forward, so the capture is dominated by the profiled
stage. ``audio`` alone builds an audio-only pipeline (no transformer/VAE at all). The dense denoise
blocks (``s1``/``stage2``) need the transformer forward and overflow the counter buffer even alone;
capture them from the full gen (drained per stage) or the ``test_transformer_ltx`` block harnesses —
this harness skips them with a message rather than half-run them.

Run under (prewarmed on the broker):
    LTX_PROFILE_STAGES=vae,audio LTX_FAST=1 \
      python -m tracy -p -r --profiler-capture-perf-counters fpu,instrn \
      -m pytest models/tt_dit/tests/models/ltx/test_ltx_stage_scoped.py::test_stage_scoped \
      -k bh_2x4sp1tp0 -s
"""

import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.vae_ltx import upsample_latent
from models.tt_dit.pipelines.ltx.pipeline_ltx import SPATIAL_COMPRESSION, TEMPORAL_COMPRESSION
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.utils.ltx import default_ltx_checkpoint, default_ltx_gemma
from models.tt_dit.utils.patchifiers import AudioLatentShape, VideoPixelShape
from models.tt_dit.utils.test import line_params

# Selector token -> canonical stage name (matches ltx_stage_bottlenecks / ltx_profile_all).
_SELECTOR_STAGE = {
    "audio": "Audio decode",
    "vae": "VAE decode",
    "upsample": "Latent upsample",
    "s1": "Stage 1 denoise",
    "stage2": "Stage 2 denoise",
}
# Stages that need a transformer forward; this harness does not isolate them (they overflow anyway).
_DENOISE = {"s1", "stage2"}


def _selected():
    raw = os.environ.get("LTX_PROFILE_STAGES", "vae,audio")
    sel = [s.strip().lower() for s in raw.split(",") if s.strip()]
    bad = [s for s in sel if s not in _SELECTOR_STAGE]
    if bad:
        pytest.fail(f"unknown LTX_PROFILE_STAGES token(s) {bad}; choose from {sorted(_SELECTOR_STAGE)}")
    return sel


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, device_params",
    [[(2, 4), (2, 4), 1, 0, 2, line_params]],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_stage_scoped(mesh_device, mesh_shape, sp_axis, tp_axis, num_links, device_params):
    """Run the selected LTX decode-tail stage(s) in isolation, draining the profiler at each stage
    boundary so a scoped perf-counter CSV covers exactly those stages."""
    selected = _selected()
    logger.info(f"LTX stage-scoped capture: stages={[_SELECTOR_STAGE[s] for s in selected]}")

    denoise = [s for s in selected if s in _DENOISE]
    if denoise:
        pytest.skip(
            f"stage(s) {denoise} need a transformer forward and overflow the counter buffer alone; "
            "capture denoise from the full gen (drained per stage) or the test_transformer_ltx block "
            "harness. This harness scopes the decode tail (vae/audio/upsample)."
        )

    parent = mesh_device
    mesh = parent.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))

    # audio alone never touches the transformer or VAE -> build audio-only (skips the 22B push).
    audio_only = set(selected) <= {"audio"}

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh,
        checkpoint_name=default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors"),
        gemma_path=default_ltx_gemma(),  # lazy shell; the decode tail never encodes prompts
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=True,
        topology=ttnn.Topology.Linear,
        is_fsdp=False,
        run_warmup=False,
        traced=False,
        audio_only=audio_only,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    latent_frames = (num_frames - 1) // TEMPORAL_COMPRESSION + 1
    ic = pipeline.in_channels

    def drain(tag):
        # Empty the profiler DRAM marker buffer at the stage boundary so the NEXT stage's counter
        # markers start from zero and never collide with a prior stage's overflow.
        ttnn.ReadDeviceProfiler(mesh)
        logger.info(f"[stage-scoped] drained profiler at boundary: {tag}")

    # Run each selected stage in canonical order, draining before and after so its markers are
    # isolated. A short warm pass (untimed) precedes the profiled pass so counters reflect steady
    # state, then a final drain flushes the profiled markers to the CSV.
    def run_stage(sel):
        if sel == "audio":
            als = AudioLatentShape.from_video_pixel_shape(
                VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
            )
            fix = os.path.join(os.path.dirname(__file__), "fixtures", "girl_audio_latent.npy")
            latent = (
                torch.from_numpy(np.load(fix)).float() if os.path.exists(fix) else torch.randn(1, als.frames, ic) * 0.5
            )
            pipeline.decode_audio(latent, num_frames, fps=24.0)  # warm
            drain("audio-warm")
            pipeline.decode_audio(latent, num_frames, fps=24.0)  # profiled
        elif sel == "vae":
            pipeline._prepare_vae()
            latent_h, latent_w = height // SPATIAL_COMPRESSION, width // SPATIAL_COMPRESSION
            latent = torch.randn(1, latent_frames * latent_h * latent_w, ic)
            pipeline.decode_latents(latent, latent_frames, latent_h, latent_w, output_type="float")  # warm
            drain("vae-warm")
            pipeline.decode_latents(latent, latent_frames, latent_h, latent_w, output_type="float")  # profiled
        elif sel == "upsample":
            pipeline._prepare_upsampler()
            s1_lh, s1_lw = (height // 2) // SPATIAL_COMPRESSION, (width // 2) // SPATIAL_COMPRESSION
            s1_spatial = torch.randn(1, ic, latent_frames, s1_lh, s1_lw)
            stats = pipeline._vae_per_channel_stats()
            upsample_latent(pipeline.upsampler, s1_spatial, *stats)  # warm
            drain("upsample-warm")
            upsample_latent(pipeline.upsampler, s1_spatial, *stats)  # profiled

    for sel in [s for s in ("upsample", "vae", "audio") if s in selected]:
        drain(f"pre-{sel}")
        run_stage(sel)
        drain(f"post-{sel}")  # flush this stage's profiled markers before the next stage

    pipeline.release_traces()
    if audio_only:
        pipeline.release_audio_submesh()
    logger.info(f"LTX stage-scoped capture done: {[_SELECTOR_STAGE[s] for s in selected]}")
