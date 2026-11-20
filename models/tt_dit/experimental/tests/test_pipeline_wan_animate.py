# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Wan2.2-Animate-14B: reference image + driving video -> mp4.

Driven by environment variables so the CLI (``experimental/wan_animate/run_animate.py``) can call it:

    WAN_ANIMATE_SRC_ROOT   directory from the official preprocess_data.py (required)
    WAN_ANIMATE_MODE       animate | replace                       (default animate)
    WAN_ANIMATE_OUT        output mp4 path                          (default wan_animate_<mode>.mp4)
    WAN_ANIMATE_SECONDS    clip length cap in seconds               (default 3)
    WAN_ANIMATE_WIDTH/HEIGHT  generation size, multiples of 16      (default: the pose video's size)
    WAN_ANIMATE_STEPS      denoising steps                          (default 20, upstream default)
    WAN_ANIMATE_SEED       (default 0)
    WAN_ANIMATE_RELIGHT_LORA  relighting LoRA (adapter_model.safetensors or relighting_lora.ckpt)
    WAN_ANIMATE_CHECKPOINT Diffusers checkpoint (default Wan-AI/Wan2.2-Animate-14B-Diffusers)
    WAN_ANIMATE_PROMPT     text prompt (default: the upstream default prompt)

    pytest models/tt_dit/experimental/tests/test_pipeline_wan_animate.py -k bh_4x8_linear -s
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
from loguru import logger

import ttnn

from models.tt_dit.utils.test import line_params_req_exact_devices, ring_params_req_exact_devices, skip_if_unsupported_num_links
from models.tt_dit.experimental.pipelines.pipeline_wan_animate import DEFAULT_CHECKPOINT, DEFAULT_PROMPT, WanAnimatePipeline
from models.tt_dit.experimental.wan_animate.video_io import load_preprocessed, write_mp4


def _env(name, default=None, cast=str):
    v = os.environ.get(name)
    return default if v in (None, "") else cast(v)


@pytest.mark.parametrize(
    ("mesh_device", "num_links", "device_params", "topology", "is_fsdp"),
    [
        # Blackhole Galaxy. The Wan 2.2 I2V doc runs BH Galaxy with the linear fabric; try it first.
        pytest.param((4, 8), 2, line_params_req_exact_devices, ttnn.Topology.Linear, False, id="bh_4x8_linear"),
        pytest.param((4, 8), 2, ring_params_req_exact_devices, ttnn.Topology.Ring, False, id="bh_4x8_ring"),
        # Wormhole Galaxy (12 GB/chip: weights FSDP-sharded on the SP axis).
        pytest.param((4, 8), 4, ring_params_req_exact_devices, ttnn.Topology.Ring, True, id="wh_4x8_ring"),
        # Blackhole LoudBox (bring-up only; 480p).
        pytest.param((2, 4), 2, line_params_req_exact_devices, ttnn.Topology.Linear, False, id="bh_2x4_linear"),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_wan_animate(mesh_device, num_links, device_params, topology, is_fsdp):
    src_root = _env("WAN_ANIMATE_SRC_ROOT")
    if not src_root:
        pytest.skip("set WAN_ANIMATE_SRC_ROOT to a preprocess_data.py output directory")
    skip_if_unsupported_num_links(mesh_device, num_links)

    mode = _env("WAN_ANIMATE_MODE", "animate")
    seconds = _env("WAN_ANIMATE_SECONDS", 3.0, float)
    out_path = _env("WAN_ANIMATE_OUT", f"wan_animate_{mode}.mp4")

    probe = load_preprocessed(src_root, mode=mode, max_frames=1)
    max_frames = max(1, int(round(seconds * probe.fps)))
    inputs = load_preprocessed(src_root, mode=mode, max_frames=max_frames)
    src_w, src_h = inputs.size
    width = _env("WAN_ANIMATE_WIDTH", src_w // 16 * 16, int)
    height = _env("WAN_ANIMATE_HEIGHT", src_h // 16 * 16, int)
    logger.info(
        f"{len(inputs.pose)} frames @ {inputs.fps:.1f} fps, source {src_w}x{src_h}, generating {width}x{height}"
    )

    t0 = time.perf_counter()
    pipe = WanAnimatePipeline(
        mesh_device=mesh_device,
        width=width,
        height=height,
        checkpoint_name=_env("WAN_ANIMATE_CHECKPOINT", DEFAULT_CHECKPOINT),
        relighting_lora_path=_env("WAN_ANIMATE_RELIGHT_LORA") if mode == "replace" else None,
        config_overrides={"topology": topology, "num_links": num_links, "is_fsdp": is_fsdp, "dynamic_load": False},
    )
    init_s = time.perf_counter() - t0
    init_timings = dict(pipe.timings.sections)

    t1 = time.perf_counter()
    frames = pipe(
        image=inputs.image,
        pose_video=inputs.pose,
        face_video=inputs.face,
        background_video=inputs.background,
        mask_video=inputs.mask,
        mode=mode,
        prompt=_env("WAN_ANIMATE_PROMPT", DEFAULT_PROMPT),
        num_inference_steps=_env("WAN_ANIMATE_STEPS", 20, int),
        seed=_env("WAN_ANIMATE_SEED", 0, int),
    )
    gen_s = time.perf_counter() - t1

    write_mp4(frames, out_path, inputs.fps)
    stats = {
        "mode": mode,
        "mesh": list(mesh_device.shape),
        "size": [width, height],
        "frames": int(frames.shape[0]),
        "fps": inputs.fps,
        "init_s": round(init_s, 1),
        "generate_s": round(gen_s, 1),
        "init_sections_s": {k: round(v, 1) for k, v in init_timings.items()},
        "generate_sections_s": {k: round(v, 1) for k, v in pipe.timings.sections.items()},
        "pixel_mean": float(frames.mean()),
        "pixel_std": float(frames.std()),
        "temporal_mad": float(np.abs(np.diff(frames.astype(np.float32), axis=0)).mean()) if len(frames) > 1 else 0.0,
        "out": os.path.abspath(out_path),
    }
    with open(os.path.splitext(out_path)[0] + "_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("RESULTS " + json.dumps(stats))

    # Sanity only (quality is judged by eye; see the runbook's "what good output looks like").
    assert frames.shape[0] == len(inputs.pose)
    assert frames.shape[1:] == (height, width, 3)
    assert 5.0 < stats["pixel_std"], "output is (nearly) flat: likely NaN/zero latents or a dead VAE"
    assert 0.2 < stats["temporal_mad"], "output does not move: motion conditioning is not reaching the model"
