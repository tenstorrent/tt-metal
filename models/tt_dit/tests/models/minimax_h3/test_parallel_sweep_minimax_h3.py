# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TP/SP parallel-configuration sweep for `t2va` at one working point (default 16:9 / 15 s / 768P).

`test_pipeline_minimax_h3.py` measures the shipped preset across aspect ratios and durations. This
file holds the working point fixed and varies the parallel configuration instead: which mesh shape is
opened and which axis carries TP versus SP. Every row asks for all 32 devices.

On a Wormhole Galaxy the system mesh is 8x4 and only 8x4, 4x8, 1x32 and 32x1 open with ring fabric
(2x16 and 16x2 are rejected by `system_mesh.cpp`). 4x8 is exactly the transpose of 8x4 -- row r of
the 4x8 device-id grid is column r of the 8x4 grid -- so `4x8 tp0/sp1` and `8x4 tp1/sp0` are the same
physical rings and only one is listed. That leaves three distinct configurations:

    4x8  tp0 sp1   TP=4 / SP=8   the shipped `_PRESETS_WH` entry (the baseline)
    4x8  tp1 sp0   TP=8 / SP=4   halves the SP ring (KV traffic) and doubles the TP ring (AGMM traffic)
    1x32 tp0 sp1   TP=1 / SP=32  no TP collectives at all; every head on every device, KV ring of 32

TP=16 (2x16) is out on two counts: the mesh shape does not open, and 56 heads do not divide by 16.

Knobs, all environment variables so the driver can loop without editing the file:

    H3_SWEEP_STEPS        scheduler steps (default 10 -> 9 forwards). Per-forward time is flat across
                          steps, so 10 gives the same ms/forward as 50 at a fifth of the wall clock;
                          confirm a winner at 50 before quoting it against the perf doc's tables.
    H3_SWEEP_ASPECT       "16,9" (default)
    H3_SWEEP_DURATION_S   15 (default)
    H3_SWEEP_OUT          results dir (default ~/h3_parallel_sweep): one JSON line per run in
                          results.jsonl, plus a strided uint8 frame dump and the audio per run so the
                          configurations can be PCC'd against each other (a fast-but-wrong parallel
                          config must not read as a win).

`MINIMAX_H3_DIT_FSDP=1` is required at 15 s on the 12 GB part; the test asserts it is on rather than
OOM 15 minutes in. `TT_DIT_CACHE_DIR` should be set: the cache is keyed on the parallel config and
mesh shape, so each new configuration builds its own sharded-weight cache on first run (~15 min from
the 62 GB safetensors) and reuses it afterwards.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .common import _WH_ONLY, _ring_4k
from .common_av import CALIBRATED_FOX_PROMPT, log_timing_table, run_warm_generation, weights_dir

STEPS = int(os.environ.get("H3_SWEEP_STEPS", "10"))
ASPECT = tuple(int(x) for x in os.environ.get("H3_SWEEP_ASPECT", "16,9").split(","))
DURATION_S = float(os.environ.get("H3_SWEEP_DURATION_S", "15"))
OUT_DIR = Path(os.environ.get("H3_SWEEP_OUT", str(Path.home() / "h3_parallel_sweep"))).expanduser()
# Every 6th frame of the (T, H, W, 3) uint8 video: ~60 frames at 15 s, enough for a PCC between
# configurations without a 1.1 GB dump per run.
FRAME_STRIDE = 6
SEED = 0
NUM_LINKS = 4  # `get_num_links` reports (4, 4) for a TG on both axes

# (mesh_shape, tp_axis, sp_axis). See the module docstring for why these three and no others.
CONFIGS = [
    pytest.param((4, 8), 0, 1, id="4x8_tp0_sp1"),
    pytest.param((4, 8), 1, 0, id="4x8_tp1_sp0"),
    pytest.param((1, 32), 0, 1, id="1x32_tp0_sp1"),
]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001 - a missing git must not fail a 20-minute run at the end
        return "unknown"


@_WH_ONLY
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "tp_axis", "sp_axis"), CONFIGS, indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [_ring_4k], indirect=True, ids=["ring4k"])
def test_t2va_parallel_sweep(mesh_device, device_params, tp_axis, sp_axis, reset_seeds):
    shape = tuple(mesh_device.shape)
    config_id = f"{shape[0]}x{shape[1]}_tp{tp_axis}_sp{sp_axis}"
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    height, width = resolve_canvas_size(*ASPECT)
    num_frames = align_num_frames(round(DURATION_S * MINIMAX_H3_FPS))
    video_seconds = num_frames / MINIMAX_H3_FPS
    logger.info(
        f"parallel sweep {config_id}: {ASPECT[0]}:{ASPECT[1]} -> {width}x{height}, {num_frames} frames, "
        f"{STEPS} steps, TP={shape[tp_axis]} axis {tp_axis} / SP={shape[sp_axis]} axis {sp_axis}"
    )
    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning("TT_DIT_CACHE_DIR is unset: every run re-reads the 62 GB safetensors")

    # Every parallel setting is passed, so `resolve_mesh_preset` is not consulted and an unlisted shape
    # runs. That also means the preset's Wormhole residency choice has to be repeated here.
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        tp_axis=tp_axis,
        sp_axis=sp_axis,
        num_links=NUM_LINKS,
        topology=ttnn.Topology.Ring,
        coresident=False,
    )
    assert pipeline.dit_fsdp or os.environ.get("H3_SWEEP_ALLOW_NO_FSDP"), (
        "DiT FSDP is off; 15 s does not fit a 12 GB Wormhole chip without it. Set MINIMAX_H3_DIT_FSDP=1 "
        "(or H3_SWEEP_ALLOW_NO_FSDP=1 to measure the OOM deliberately)."
    )

    t0 = time.time()
    output = run_warm_generation(
        pipeline,
        CALIBRATED_FOX_PROMPT,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=STEPS,
        seed=SEED,
    )
    wall = time.time() - t0

    num_forwards = STEPS - 1
    total = log_timing_table(
        pipeline,
        f"t2va parallel sweep {config_id}",
        num_forwards=num_forwards,
        video_seconds=video_seconds,
        extra=(
            f" | {ASPECT[0]}:{ASPECT[1]} {width}x{height}, {num_frames} frames, {STEPS} steps, "
            f"padded_len {pipeline.last_padded_len}"
        ),
    )
    rows = dict(pipeline.last_timings)
    denoise = rows.get("Denoise")
    per_forward_ms = denoise / num_forwards * 1000 if denoise else None

    # Correctness evidence, not a gate: the frames are dumped for a PCC against the baseline
    # configuration, and the cheap sanity numbers go in the record so a NaN run is visible in the table.
    video = output.video
    assert torch.isfinite(video).all(), "video contains NaN or Inf"
    frames = video[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    assert frames.shape[0] == num_frames, f"expected {num_frames} frames, got {frames.shape[0]}"
    frames_path = OUT_DIR / f"{config_id}_s{STEPS}.frames.npy"
    audio_path = OUT_DIR / f"{config_id}_s{STEPS}.audio.npy"
    np.save(frames_path, frames[::FRAME_STRIDE])
    np.save(audio_path, output.audio.float().cpu().numpy())

    record = {
        "config": config_id,
        "mesh_shape": list(shape),
        "tp_axis": tp_axis,
        "sp_axis": sp_axis,
        "tp_factor": pipeline.tp_factor,
        "sp_factor": pipeline.sp_factor,
        "num_links": NUM_LINKS,
        "dit_fsdp": pipeline.dit_fsdp,
        "aspect": list(ASPECT),
        "canvas": [width, height],
        "num_frames": num_frames,
        "steps": STEPS,
        "num_forwards": num_forwards,
        "padded_len": pipeline.last_padded_len,
        "rows_per_device": pipeline.last_padded_len // pipeline.sp_factor,
        "timings_s": rows,
        "total_s": total,
        "per_forward_ms": per_forward_ms,
        "realtime_factor": total / video_seconds,
        "warm_call_wall_s": wall,
        "frame_mean": float(frames.mean()),
        "frame_std": float(frames.std()),
        "audio_absmax": float(output.audio.abs().max()),
        "frames_path": str(frames_path),
        "audio_path": str(audio_path),
        "commit": _git_head(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(OUT_DIR / "results.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"SWEEP RESULT {json.dumps(record)}")

    pipeline.release_traces()
