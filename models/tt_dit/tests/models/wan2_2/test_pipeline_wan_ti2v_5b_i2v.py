# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for Wan2.2 TI2V-5B image-to-video on a single BH Galaxy (4x8).

All runs are at 1280x704. 480p is out-of-distribution for this checkpoint -- its VAE
compresses 16x16 spatially and the patchify layer takes the total to 4x32x32, so 832x480
yields only 15x26 tokens and produces visibly soft output that a broken conditioning
implementation could hide behind.
"""

import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.pipelines.wan.pipeline_wan_ti2v_5b_i2v import WanTI2V5BI2VPipeline
from models.tt_dit.utils.test import ring_params_req_exact_devices, skip_if_unsupported_num_links

from .common import check_first_frame_matches_seed, check_output_sanity
from .test_pipeline_wan_i2v import create_fractal_image

MESH_PARAMS = [
    [(4, 8), (4, 8), {**ring_params_req_exact_devices, "trace_region_size": 150000000}, ttnn.Topology.Ring],
]


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    MESH_PARAMS,
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_ti2v_5b_i2v_smoke(mesh_device, mesh_shape, topology):
    """Construction smoke test: the warmup call runs a full 2-step generate, so this alone
    exercises the per-token timestep upload, the shape-gated AdaLN, the device blend and the
    post-loop re-pin."""
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    pipeline = WanTI2V5BI2VPipeline.create_pipeline(
        mesh_device=mesh_device, height=704, width=1280, num_frames=81, run_warmup=True
    )

    assert pipeline.transformer_2 is None, "5B is dense (single expert)"
    assert pipeline._expand_timesteps is True, "I2V requires per-token timesteps"
    assert pipeline._first_frame_mask is not None, "warmup must have built the mask"
    assert pipeline.condition_buffer is not None, "warmup must have allocated the condition buffer"
    logger.info(f"I2V smoke OK; host image encode took {pipeline.last_image_encode_seconds:.3f}s")


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    MESH_PARAMS,
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_ti2v_5b_i2v_generate(mesh_device, mesh_shape, topology):
    """Real I2V generation, gated on frame 0 matching the seed image.

    TI2V-5B pins latent frame 0 to the conditioning latent at every step and once more after
    the loop, and the Wan2.2 VAE decoder is causal in T, so decoded pixel frame 0 is
    essentially a VAE round-trip of the seed. That makes the correlation floor much tighter
    than the 0.3 the 14B path uses -- its own docstring calls 0.3 provisional and invites
    tightening once real values are observed.
    """
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    height, width = 704, 1280
    num_frames = int(os.environ.get("WAN5B_I2V_FRAMES", 81))
    steps = int(os.environ.get("WAN5B_I2V_STEPS", 40))
    # Observed 0.9984 with the fractal seed at 1280x704/81f/40 steps. 0.95 is well clear of
    # that while leaving room for natural photos, which may round-trip slightly worse than a
    # synthetic image. The 14B path's 0.3 is left alone -- it conditions differently.
    pcc_floor = float(os.environ.get("WAN5B_I2V_PCC_FLOOR", 0.95))
    seed_image = create_fractal_image(width, height)

    pipeline = WanTI2V5BI2VPipeline.create_pipeline(
        mesh_device=mesh_device, height=height, width=width, num_frames=num_frames, run_warmup=True
    )

    with torch.no_grad():
        frames = pipeline(
            prompts=["A slow cinematic push in, the scene coming to life with gentle motion."],
            image_prompt=seed_image,
            num_inference_steps=steps,
            seed=42,
            guidance_scale=5.0,
            guidance_scale_2=None,
            output_type="uint8",
        )

    frames_u8 = np.asarray(frames).astype(np.uint8)
    while frames_u8.ndim > 4 and frames_u8.shape[0] == 1:
        frames_u8 = frames_u8[0]

    check_output_sanity(frames_u8, num_frames=num_frames, height=height, width=width)
    check_first_frame_matches_seed(frames_u8, seed_image=seed_image, width=width, height=height, pcc_floor=pcc_floor)


@pytest.mark.parametrize(
    "mesh_device, mesh_shape, device_params, topology",
    MESH_PARAMS,
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
def test_pipeline_ti2v_5b_i2v_demo(mesh_device, mesh_shape, topology):
    """Animate an arbitrary image and write an mp4 plus first/mid/last PNG previews.

    All knobs are env-overridable::

        I2V_IMAGE=/path/to/photo.jpg \
        I2V_PROMPT="..." I2V_STEPS=40 I2V_FRAMES=81 I2V_SEED=42 \
        I2V_OUT=/home/user/tt-data/teja/i2v_demo.mp4 \
          pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan_ti2v_5b_i2v.py -k demo -sv

    `PYTHONPATH` must include the user site or the mp4 export silently degrades to PNGs --
    `imageio_ffmpeg` lives there, not in `python_env`.
    """
    if not ttnn.device.is_blackhole():
        pytest.skip("TI2V-5B targets BH Galaxy")

    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(*mesh_shape))
    skip_if_unsupported_num_links(mesh_device, 2)

    import PIL.Image

    height, width = 704, 1280
    num_frames = int(os.environ.get("I2V_FRAMES", 81))
    steps = int(os.environ.get("I2V_STEPS", 40))
    seed = int(os.environ.get("I2V_SEED", 42))
    fps = int(os.environ.get("I2V_FPS", 24))
    guidance = float(os.environ.get("I2V_GUIDANCE", 5.0))
    flow_shift_env = os.environ.get("I2V_FLOW_SHIFT")
    flow_shift = float(flow_shift_env) if flow_shift_env else None
    out_path = os.environ.get("I2V_OUT", "/home/user/tt-data/teja/i2v_demo.mp4")
    prompt = os.environ.get(
        "I2V_PROMPT", "The scene comes to life with natural, gentle motion. Cinematic, detailed, high quality."
    )

    image_path = os.environ.get("I2V_IMAGE")
    if image_path:
        seed_image = PIL.Image.open(image_path).convert("RGB")
        logger.info(f"I2V demo seed image: {image_path} ({seed_image.size[0]}x{seed_image.size[1]})")
    else:
        seed_image = create_fractal_image(width, height)
        logger.info("I2V demo seed image: synthetic fractal (set I2V_IMAGE to use your own)")

    pipeline = WanTI2V5BI2VPipeline.create_pipeline(
        mesh_device=mesh_device, height=height, width=width, num_frames=num_frames, run_warmup=True
    )

    import time

    t0 = time.perf_counter()
    with torch.no_grad():
        frames = pipeline(
            prompts=[prompt],
            image_prompt=seed_image,
            num_inference_steps=steps,
            seed=seed,
            guidance_scale=guidance,
            guidance_scale_2=None,
            flow_shift=flow_shift,
            output_type="uint8",
        )
    logger.info(f"I2V_DEMO_TIME: {time.perf_counter() - t0:.2f}s ({num_frames}f / {steps} steps, {width}x{height})")

    frames_u8 = np.asarray(frames).astype(np.uint8)
    while frames_u8.ndim > 4 and frames_u8.shape[0] == 1:
        frames_u8 = frames_u8[0]

    base = os.path.splitext(out_path)[0]
    seed_image.resize((width, height)).save(f"{base}_seed.png")
    for tag, idx in (("first", 0), ("mid", frames_u8.shape[0] // 2), ("last", frames_u8.shape[0] - 1)):
        PIL.Image.fromarray(frames_u8[idx]).save(f"{base}_{tag}.png")
    logger.info(f"I2V_DEMO_PNGS: {base}_{{seed,first,mid,last}}.png")
    try:
        from models.tt_dit.utils.video import export_to_video

        export_to_video(frames_u8, out_path, fps=fps)
        logger.info(f"I2V_DEMO_VIDEO: {out_path}")
    except Exception as e:  # noqa: BLE001
        logger.info(f"I2V_DEMO_VIDEO failed ({e!r}); PNGs still written")

    check_output_sanity(frames_u8, num_frames=num_frames, height=height, width=width)
    check_first_frame_matches_seed(frames_u8, seed_image=seed_image, width=width, height=height, pcc_floor=0.95)
