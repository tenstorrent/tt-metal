# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for the Qwen-Image-Edit pipeline on WH Galaxy (TP=8 x SP=4).

Constructs the formal ``QwenImageEditPipeline`` (device denoise + host VL/VAE), runs a real edit on
the sample cat image, saves the result, and reports per-forward / denoise / wall latency.

Run:
  TT_DIT_CACHE_DIR=/home/tt-admin/teja/.ttdit_cache \
    python_env/bin/python -m pytest \
    models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage_edit.py -s
"""
from __future__ import annotations

import pytest
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.qwenimage_edit.pipeline_qwenimage_edit import QwenImageEditPipeline
from ....utils.test import line_params_req_exact_devices

SAMPLE_IMAGE = "models/sample_data/huggingface_cat_image.jpg"
PROMPT = "Give the cat a blue wizard hat."
OUT_PATH = "models/tt_dit/pipelines/qwenimage_edit/edit_pipeline_output.png"


@pytest.mark.parametrize(
    "device_params",
    [{**line_params_req_exact_devices, "trace_region_size": 130000000}],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize(
    ("mesh_device", "cfg_parallel", "device_vae", "device_vae_encode", "num_inference_steps"),
    [
        pytest.param((4, 8), False, True, False, 50, id="4x8_devdecode_hostencode_50steps"),
        pytest.param((4, 8), False, True, True, 50, id="4x8_devvae_full_50steps"),
        pytest.param((4, 8), True, True, True, 50, id="4x8_cfgpar_devvae_full_50steps"),
    ],
    indirect=["mesh_device"],
)
def test_qwenimage_edit_pipeline(
    *,
    mesh_device: ttnn.MeshDevice,
    cfg_parallel: bool,
    device_vae: bool,
    device_vae_encode: bool,
    num_inference_steps: int,
) -> None:
    pipeline = QwenImageEditPipeline.create_pipeline(
        mesh_device=mesh_device,
        cfg_parallel=cfg_parallel,
        device_vae=device_vae,
        device_vae_encode=device_vae_encode,
    )

    image = Image.open(SAMPLE_IMAGE).convert("RGB")
    suffix = ("cfgpar_" if cfg_parallel else "") + ("devvae_full" if device_vae_encode else "devdecode")
    out_path = OUT_PATH.replace(".png", f"_{suffix}.png")
    logger.info(
        f"running edit: '{PROMPT}' | cfg_parallel={cfg_parallel} device_vae={device_vae} encode={device_vae_encode}, {num_inference_steps} steps"
    )

    images = pipeline(
        image=image,  # letterboxed internally (aspect-preserving, SP-aligned)
        prompt=PROMPT,
        negative_prompt=" ",
        num_inference_steps=num_inference_steps,
        true_cfg_scale=4.0,
        seed=0,
    )

    images[0].save(out_path)
    logger.info(f"[qwen-image-edit] saved: {out_path} @ {images[0].size}")

    assert len(images) == 1
