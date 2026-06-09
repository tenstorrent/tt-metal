# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.utils.ltx import (
    DEFAULT_LTX_PROMPT,
    default_ltx_checkpoint,
    default_ltx_gemma,
    print_ltx_timing_table,
)
from models.tt_dit.utils.patchifiers import AudioLatentShape, VideoPixelShape
from models.tt_dit.utils.test import line_params, ring_params

# Trace region for LTX_TRACED=1. Holds both stage traces' command streams (s1 + larger-seq
# s2); measured need is ~236 MB at 1080p (get_trace_buffers_size), so 300 MB gives headroom.
ring_trace_params = {**ring_params, "trace_region_size": 300_000_000}


@pytest.mark.parametrize(
    "no_prompt",
    [{"1": True, "0": False}.get(os.environ.get("NO_PROMPT"), True)],
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 2), (2, 2), 0, 1, 2, False, line_params, ttnn.Topology.Linear, True],
        [(2, 4), (2, 4), 0, 1, 1, True, line_params, ttnn.Topology.Linear, True],
        # BH on 2x4
        [(2, 4), (2, 4), 1, 0, 2, True, line_params, ttnn.Topology.Linear, False],
        # WH (ring) on 4x8. Requires increased worker_l1_size to avoid code-size error in RingAttention.
        [(4, 8), (4, 8), 1, 0, 4, True, {"worker_l1_size": 1344544, **ring_params}, ttnn.Topology.Ring, True],
        # BH (linear) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, line_params, ttnn.Topology.Linear, False],
        # BH (ring) on 4x8
        [(4, 8), (4, 8), 1, 0, 2, False, ring_trace_params, ttnn.Topology.Ring, False],
        [(4, 32), (4, 32), 1, 0, 2, False, ring_params, ttnn.Topology.Ring, False],
    ],
    ids=[
        "2x2sp0tp1",
        "2x4sp0tp1",
        "bh_2x4sp1tp0",
        "wh_4x8sp1tp0",
        "bh_4x8sp1tp0_linear",
        "bh_4x8sp1tp0_ring",
        "bh_4x32sp1tp0",
    ],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_distilled(
    mesh_device,
    mesh_shape,
    sp_axis,
    tp_axis,
    num_links,
    dynamic_load,
    topology,
    is_fsdp,
    no_prompt,
):
    """LTX-2.3 distilled 2-stage AV pipeline."""
    ckpt = default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")
    gemma = default_ltx_gemma()

    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    # Stage-2 output target: 1080p @ 24fps, ~6s. Constraints from the pipeline:
    #   - height/width must be divisible by 64 (asserted at the top of generate(),
    #     so half-res stage-1 is divisible by 32 for the VAE).
    #     1920 is already % 64; 1080 is not → round up to 1088 (next multiple of 64).
    #     If you need exactly 1080 lines, post-crop the 8 extra rows.
    #   - num_frames must satisfy (num_frames - 1) % 8 == 0 so the VAE decoder maps
    #     latent_frames → num_frames frames exactly. 144 (== 6s × 24fps) doesn't
    #     satisfy this; 145 does (19 latent frames → 145 decoded frames ≈ 6.04s).
    num_frames = int(os.environ.get("NUM_FRAMES", "145"))
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))

    run_warmup = os.environ.get("RUN_WARMUP", "0") in ("1", "true", "True")
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=ckpt,
        gemma_path=gemma,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=run_warmup,
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    prompt = os.environ.get("PROMPT", DEFAULT_LTX_PROMPT)

    def run(*, prompt, number, seed):
        output_filename = os.environ.get("OUTPUT_PATH", f"ltx_av_fast_{width}x{height}_{number}.mp4")
        logger.info(f"Running LTX AV Fast: '{prompt[:80]}...'")
        logger.info(f"Config: {height}x{width}, {num_frames} frames")

        if int(ttnn.distributed_context_get_rank()) != 0:
            logger.info(f"Skipping generation on rank {ttnn.distributed_context_get_rank()}")
            return

        pipeline.generate(
            prompt,
            output_path=output_filename,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
        )
        logger.info(f"Saved video to: {output_filename}")
        print_ltx_timing_table(
            pipeline,
            label="LTX DISTILLED",
            num_frames=num_frames,
            height=height,
            width=width,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            topology=topology,
            output_path=output_filename,
            prompt=prompt,
        )

    if no_prompt:
        seed = int(os.environ.get("SEED", "10"))
        run(prompt=prompt, number=0, seed=seed)
        # Traced: gen #0 captures (lazily, on first step of each stage); gen #1 is pure
        # replay — its Stage 1/2 denoise times are the steady-state measurement.
        if traced:
            logger.info("=== traced steady-state pass (gen #1, pure replay) ===")
            run(prompt=prompt, number=1, seed=seed)
    else:
        for i in itertools.count():
            new_prompt = input("Enter the input prompt, or q to exit: ")
            if new_prompt:
                prompt = new_prompt
            if prompt[0] == "q":
                break
            run(prompt=prompt, number=i, seed=i)

    if traced:
        pipeline.release_traces()


@pytest.mark.skipif(
    not os.path.exists(default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors")),
    reason="needs the LTX checkpoint (set LTX_CHECKPOINT to a local .safetensors)",
)
@pytest.mark.parametrize(
    "mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, device_params, topology, is_fsdp",
    [
        [(2, 4), (2, 4), 1, 0, 2, True, ring_trace_params, ttnn.Topology.Linear, False],
    ],
    ids=["bh_2x4sp1tp0"],
    indirect=["mesh_device", "device_params"],
)
def test_audio_decode_girl(mesh_device, mesh_shape, sp_axis, tp_axis, num_links, dynamic_load, topology, is_fsdp):
    """Just the audio section of the AV pipeline: real checkpoint weights, real girl-clip
    latent shape (num_frames -> als.frames), no transformer/gemma (decode_audio never
    encodes prompts). Profiles cold (weight load + compile) vs warm (steady-state per-gen)
    decode wall. LTX_TRACED=1 traces the main vocoder."""
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    num_frames = int(os.environ.get("NUM_FRAMES", "145"))  # ~6.04s @ 24fps
    height = int(os.environ.get("HEIGHT", "1088"))
    width = int(os.environ.get("WIDTH", "1920"))
    traced = os.environ.get("LTX_TRACED", "0") in ("1", "true", "True")

    # AUDIO_DEPTHWISE=mac reproduces the pre-conv1d (main) audio path for A/B profiling.
    import models.tt_dit.layers.audio_ops as _ao

    _ao._USE_CONV1D_DEPTHWISE = os.environ.get("AUDIO_DEPTHWISE", "conv1d") != "mac"

    pipeline = LTXDistilledPipeline.create_pipeline(
        mesh_device=mesh_device,
        checkpoint_name=default_ltx_checkpoint("ltx-2.3-22b-distilled-1.1.safetensors"),
        gemma_path=default_ltx_gemma(),  # built as a lazy shell only; never loaded here
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        num_links=num_links,
        dynamic_load=dynamic_load,
        topology=topology,
        is_fsdp=is_fsdp,
        run_warmup=False,
        traced=traced,
        num_frames=num_frames,
        height=height,
        width=width,
    )

    vps = VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=24)
    als = AudioLatentShape.from_video_pixel_shape(vps)
    # AUDIO_LATENT=<path> replays a real generated latent (dumped via LTX_DUMP_AUDIO_LATENT) so
    # the decoded audio is the actual clip, not vocoder noise on a random latent.
    _lat = os.environ.get("AUDIO_LATENT")
    if _lat:
        latent = torch.load(_lat).float()
    else:
        torch.manual_seed(0)
        latent = torch.randn(1, als.frames, pipeline.in_channels, dtype=torch.float32) * 0.5

    t0 = time.perf_counter()
    audio = pipeline.decode_audio(latent, num_frames, fps=24.0)  # cold: weight load + compile (+ capture)
    cold_ms = (time.perf_counter() - t0) * 1000

    N = 5
    t0 = time.perf_counter()
    for _ in range(N):
        audio = pipeline.decode_audio(latent, num_frames, fps=24.0)
    warm_ms = (time.perf_counter() - t0) * 1000 / N
    wav = audio.waveform
    sr = audio.sampling_rate
    out_dir = os.environ.get("AUDIO_OUT")

    def _save(w, name):
        if not out_dir:
            return
        import soundfile as sf

        os.makedirs(out_dir, exist_ok=True)
        a = w[0] if w.dim() == 3 else w  # (2, T) -> (T, 2)
        path = os.path.join(out_dir, name)
        sf.write(path, a.transpose(0, 1).cpu().numpy(), int(sr))
        logger.info(f"wrote {path}")

    _save(wav, f"girl_audio_{'conv1d' if _ao._USE_CONV1D_DEPTHWISE else 'mac'}{'_traced' if traced else ''}.wav")

    # Per-1s-interval conv1d-vs-MAC on the real weights: a localized burst (the static) shows
    # up as one interval spiking well above the uniform reduction-order difference, where an
    # aggregate metric would average it out.
    if os.environ.get("AUDIO_COMPARE_DEPTHWISE", "0") == "1":
        _ao._USE_CONV1D_DEPTHWISE = True
        w_conv = pipeline.decode_audio(latent, num_frames, fps=24.0).waveform
        _ao._USE_CONV1D_DEPTHWISE = False
        w_mac = pipeline.decode_audio(latent, num_frames, fps=24.0).waveform
        _save(w_conv, "girl_audio_conv1d.wav")
        _save(w_mac, "girl_audio_mac.wav")
        rms_rows = []
        for start in range(0, w_conv.shape[-1], sr):  # 1s intervals
            c, m = w_conv[..., start : start + sr], w_mac[..., start : start + sr]
            sig = m.pow(2).mean().sqrt().item() + 1e-9
            mx = (c - m).abs().max().item()
            rr = ((c - m) ** 2).mean().sqrt().item() / sig
            rms_rows.append(rr)
            logger.info(f"[interval] t={start/sr:4.1f}s  conv-vs-mac max|Δ|={mx:.3e} rmse/σ={rr:.3e}")
        worst = max(rms_rows)
        med = sorted(rms_rows)[len(rms_rows) // 2]
        print(
            f"\nAUDIO_GIRL conv-vs-mac worst_interval_rmse/σ={worst:.3e} median={med:.3e} ratio={worst/(med+1e-9):.2f}",
            flush=True,
        )
        # No interval may spike far above the uniform difference — that is a localized burst.
        assert worst < 5 * med + 1e-6, f"interval rmse/σ {worst:.3e} >> median {med:.3e}: localized divergence (static)"

    pipeline.release_traces()
    dur = wav.shape[-1] / sr
    print(
        f"\nAUDIO_GIRL depthwise={'conv1d' if _ao._USE_CONV1D_DEPTHWISE else 'mac'} traced={traced} "
        f"latent_frames={als.frames} out={tuple(wav.shape)} {dur:.2f}s@{sr}Hz cold={cold_ms:.1f}ms warm={warm_ms:.1f}ms",
        flush=True,
    )
    assert torch.isfinite(wav).all(), "decoded waveform has non-finite samples"
    assert abs(dur - num_frames / 24.0) < 0.2, f"duration {dur:.2f}s != ~{num_frames/24.0:.2f}s"
