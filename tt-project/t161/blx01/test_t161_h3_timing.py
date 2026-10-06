# t161: warm MiniMax-H3 fl2va e2e timing on the t48 tree (ltx-rt base H3 code). 768p native, dense, 50 steps,
# first+last keyframes, host bicubic upscale to 1920x1080 (the 2026-09-30 g15blx02 protocol). Env: T161_MODE,
# T161_SECONDS (comma list for capture), T161_OUT, T161_FIRST, T161_LAST. Broker jobs are capped at 600 s, so:
# capture = 2-step pass per length with dispatch off (kernel manifest); fill = build the pipeline only (writes the
# weight cache); time = 2-step warmup at the real shape, then the timed 50-step call (warm headline).
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler
from models.tt_dit.pipelines.events import profiler_event_callback
from models.tt_dit.pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from models.tt_dit.tests.models.minimax_h3.common import GALAXY_MESHES
from models.tt_dit.tests.models.minimax_h3.common_av import to_uint8_frames, weights_dir, write_artifacts

PROMPT = (
    "A slow cinematic camera pan across the scene, natural motion, soft ambient sound, "
    "consistent lighting from the first frame to the last."
)
STAGES = (
    ("encoder", "encoder"),
    ("vae_encode", "vae_encode"),
    ("denoise", "denoising"),
    ("vae", "vae"),
    ("audio", "audio"),
)


def upscale(video, width=1920, height=1080, chunk=16):
    frames = video[0].permute(1, 0, 2, 3).float()
    out = np.empty((frames.shape[0], height, width, 3), dtype=np.uint8)
    for s in range(0, frames.shape[0], chunk):
        up = F.interpolate(frames[s : s + chunk], size=(height, width), mode="bicubic", align_corners=False)
        out[s : s + chunk] = up.clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return out


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_t161_h3_fl2va(mesh_device, reset_seeds):
    mode = os.environ.get("T161_MODE", "time")
    lengths = [float(v) for v in os.environ["T161_SECONDS"].split(",")]
    out = Path(os.environ["T161_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    first = Image.open(os.environ["T161_FIRST"]).convert("RGB")
    last = Image.open(os.environ["T161_LAST"]).convert("RGB")

    def kwargs(seconds, steps):
        num_frames = align_num_frames(round(seconds * MINIMAX_H3_FPS))
        return dict(
            image=first, last_image=last, num_frames=num_frames, height=768, width=1344, num_inference_steps=steps
        )

    t0 = time.time()
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device, weights_dir=weights_dir(), vae_output_type="float", warmup=False
    )
    build_s = time.time() - t0
    logger.info(f"T161 pipeline built in {build_s:.1f}s (mode={mode})")
    if mode == "fill":
        return
    if mode == "capture":
        for seconds in lengths:
            pipeline(PROMPT, seed=0, **kwargs(seconds, 2))
            logger.info(f"T161 capture {seconds:g}s done")
        return

    seconds = lengths[0]
    t0 = time.time()
    pipeline(PROMPT, seed=0, **kwargs(seconds, 2))
    ttnn.synchronize_device(mesh_device)
    warm_s = time.time() - t0
    logger.info(f"T161 warmup (2 steps) {warm_s:.1f}s")

    prof = BenchmarkProfiler()
    ttnn.synchronize_device(mesh_device)
    t0 = time.time()
    with prof("run", iteration=0):
        output = pipeline(PROMPT, seed=0, on_event=profiler_event_callback(prof, 0), **kwargs(seconds, 50))
        ttnn.synchronize_device(mesh_device)
    gen_wall = time.time() - t0
    t0 = time.time()
    up = upscale(output.video)
    up_s = time.time() - t0
    stages = {k: prof.get_duration(s, 0) for k, s in STAGES if prof.contains_step(s, 0)}
    sl = pipeline.last_seq_len
    rec = dict(
        seconds_req=seconds,
        num_frames=output.num_frames,
        fps=output.fps,
        video_seconds=output.num_frames / output.fps,
        height=768,
        width=1344,
        steps=50,
        mesh=list(mesh_device.shape),
        tp=pipeline.tp_factor,
        sp=pipeline.sp_factor,
        seq_len=getattr(sl, "logical", None),
        padded_len=getattr(sl, "padded", None),
        stages_s=stages,
        generate_wall_s=gen_wall,
        upscale_s=up_s,
        e2e_1080p_s=gen_wall + up_s,
        pipeline_build_s=build_s,
        warmup_2step_s=warm_s,
        prompt=PROMPT,
    )
    logger.info(f"T161_RESULT {json.dumps(rec)}")
    (out / "timings.json").write_text(json.dumps(rec, indent=2))
    audio = output.audio.float().cpu().numpy()
    write_artifacts(up, audio, output.sampling_rate, out, stem="seed0_1920x1080")
    for label, i in (("first", 0), ("mid", len(up) // 2), ("last", len(up) - 1)):
        Image.fromarray(up[i]).save(out / f"seed0_1920x1080_{label}.png")
    np.save(out / "seed0_frames_u8_every16.npy", to_uint8_frames(output)[::16])
