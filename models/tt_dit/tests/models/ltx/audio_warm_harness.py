"""Persistent warm dev harness for LTX audio decode (Step 0).

ONE process: build the audio_only pipeline once (pays the ~317s one-time device-side
program/mesh-workload assembly ONCE), then loop many warm decodes — eager AND traced —
reporting per-stage (mel-VAE / vocoder+BWE) warm walls. This is the fast-iteration tool
the plan's Step 0 calls for: after the single cold prime, each extra decode is ~1s, so
eager-vs-traced (Phase 1b) is answered in one ~6-min process instead of two ~14-min runs.

Env knobs:
  EAGER_REPS   (default 5)  warm eager decodes to average
  TRACED_REPS  (default 5)  warm traced decodes to average (0 to skip the traced leg)
  LTX_VOC_TRACE / LTX_BWE_TRACE / LTX_VAE_TRACE  passed through to the traced leg
  WARMUP_DECODES (default 1) untimed decodes after a mode switch to settle steady state
  NUM_FRAMES / HEIGHT / WIDTH  clip shape (defaults match test_audio_decode_girl)

Quality is NOT checked here (no torch oracle) — that stays test_audio_decode_girl's job
(LTX_TRACED=0, conv1d-vs-torch PCC>0.95). This harness is timing/iteration only.

Run via the broker against the worktree env, e.g.:
  python -m pytest models/tt_dit/tests/models/ltx/audio_warm_harness.py -k bh_4x8sp1tp0 -s
"""
import os
import time

import numpy as np
import pytest
import torch

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.utils.ltx import default_ltx_checkpoint, default_ltx_gemma
from models.tt_dit.utils.patchifiers import AudioLatentShape, VideoPixelShape
from models.tt_dit.utils.test import line_params

line_trace_params = {**line_params, "trace_region_size": 300_000_000}


def _stage_decode(pipeline, latent, num_frames):
    """One decode, returning (mel_ms, voc_ms, total_ms). Mirrors decode_audio's stage split
    but measured here so it works regardless of LTX_TIME_STAGES."""
    z = pipeline.tt_audio_decoder.z_channels
    audio_spatial = latent.reshape(1, latent.shape[1], z, latent.shape[2] // z).permute(0, 2, 1, 3).float()
    ttnn.synchronize_device(pipeline.mesh_device)
    t0 = time.perf_counter()
    mel = pipeline._decode_mel(audio_spatial)
    ttnn.synchronize_device(pipeline.mesh_device)
    t1 = time.perf_counter()
    wav = pipeline.tt_vocoder_with_bwe(mel).squeeze(0).float()
    ttnn.synchronize_device(pipeline.mesh_device)
    t2 = time.perf_counter()
    return (t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t0) * 1000, wav


def _avg_leg(pipeline, latent, num_frames, reps, warmups, label):
    for _ in range(warmups):
        _stage_decode(pipeline, latent, num_frames)
    mels, vocs, tots = [], [], []
    wav = None
    for _ in range(reps):
        m, v, t, wav = _stage_decode(pipeline, latent, num_frames)
        mels.append(m)
        vocs.append(v)
        tots.append(t)

    def _med(xs):
        return sorted(xs)[len(xs) // 2]

    print(
        f"\nWARM_HARNESS {label} reps={reps} "
        f"mel_vae={_med(mels):.1f}ms vocoder+bwe={_med(vocs):.1f}ms total={_med(tots):.1f}ms "
        f"(min_total={min(tots):.1f} max_total={max(tots):.1f})",
        flush=True,
    )
    return _med(tots), wav


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, line_trace_params, ttnn.Topology.Linear, False],
        [(4, 8), (4, 8), 1, 0, 2, False, line_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0", "bh_4x8sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_warm_harness(
    mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp
):
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))
    eager_reps = int(os.environ.get("EAGER_REPS", "5"))
    traced_reps = int(os.environ.get("TRACED_REPS", "5"))
    warmups = int(os.environ.get("WARMUP_DECODES", "1"))

    import models.tt_dit.layers.audio_ops as _ao

    _ao._USE_CONV1D_DEPTHWISE = True  # conv1d path (the PCC-validated one)

    # Build pipeline EAGER first (traced=False). The single cold decode below pays the
    # one-time device-side assembly; both legs reuse the same warm pipeline + kernel cache.
    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors"),
        gemma_path=default_ltx_gemma(),
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=False,
        traced=False,
        audio_only=True,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    vps = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    _lat = os.environ.get("AUDIO_LATENT") or os.path.join(
        os.path.dirname(__file__), "fixtures", "girl_audio_latent.npy"
    )
    if os.path.exists(_lat):
        latent = (torch.from_numpy(np.load(_lat)) if _lat.endswith(".npy") else torch.load(_lat)).float()
    else:
        torch.manual_seed(0)
        latent = torch.randn(1, als.frames, pipeline.in_channels, dtype=torch.float32) * 0.5

    # COLD: one-time device-side program/mesh-workload assembly (~317s on 4x8, NOT JIT).
    t0 = time.perf_counter()
    pipeline.decode_audio(latent, num_frames, fps=24.0)
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"\nWARM_HARNESS cold={cold_ms:.1f}ms", flush=True)

    # ---- EAGER leg ----
    pipeline._traced = False
    eager_total, _ = _avg_leg(pipeline, latent, num_frames, eager_reps, warmups, "EAGER")

    # ---- TRACED leg ---- (in-process switch: release any traces, flip flag, re-prime flags)
    if traced_reps > 0:
        pipeline.release_traces()
        pipeline._traced = True
        # _prepare_audio_decoder re-reads self._traced + env each call, so the next decode
        # lazily captures the vocoder/VAE trace per LTX_VOC_TRACE/LTX_BWE_TRACE/LTX_VAE_TRACE.
        # Use extra warmups so the lazy capture cost isn't counted in the timed reps.
        traced_total, _ = _avg_leg(pipeline, latent, num_frames, traced_reps, max(warmups, 2), "TRACED")
        delta = traced_total - eager_total
        print(
            f"\nWARM_HARNESS EAGER_vs_TRACED eager_total={eager_total:.1f}ms "
            f"traced_total={traced_total:.1f}ms delta={delta:+.1f}ms "
            f"({'traced FASTER' if delta < 0 else 'traced SLOWER'})",
            flush=True,
        )

    pipeline.release_traces()
