# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Quasar SDXL UNet tests.

* ``load_torch_unet`` loads the HF SDXL base UNet from the local cache and, when the 10 GB of
  weights are not available locally (the usual case on a simulator box), builds the same
  architecture with random weights from the HF config instead. PCC against the torch reference
  is just as meaningful with random weights, and the test stays runnable offline.
* ``load_scheduler`` does the same for the Euler discrete scheduler config.
* ``to_device_tile`` / ``to_device_nhwc`` build device inputs the Quasar way: host tilize,
  then ``quasar.to_device``. The NCHW -> [B, 1, H*W, C] permute+reshape that the original
  tests did on device is done in torch on host, so a test only exercises the ops of the
  module under test.
"""

import os

import torch
from loguru import logger

import ttnn
from models.demos.stable_diffusion_xl_base.quasar.tt import qsr

SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
RANDOM_WEIGHTS_ENV = "SDXL_QUASAR_RANDOM_WEIGHTS"


def use_random_weights():
    return os.environ.get(RANDOM_WEIGHTS_ENV, "0") not in ("", "0", "false", "False")


def load_torch_unet(model_location=SDXL_BASE_REPO, is_ci_env=False, is_ci_v2_env=False, seed=0):
    """SDXL base UNet (diffusers ``UNet2DConditionModel``) in eval mode, fp32."""
    from diffusers import UNet2DConditionModel

    subfolder = None if is_ci_v2_env else "unet"
    if not use_random_weights():
        try:
            unet = UNet2DConditionModel.from_pretrained(
                model_location,
                torch_dtype=torch.float32,
                use_safetensors=True,
                local_files_only=True,
                subfolder=subfolder,
            )
            unet.eval()
            return unet
        except Exception as e:  # weights not cached locally
            logger.warning(
                f"SDXL UNet weights not available locally ({type(e).__name__}); "
                f"building the UNet from its config with random weights (set {RANDOM_WEIGHTS_ENV}=1 to skip the lookup)"
            )
    config = UNet2DConditionModel.load_config(
        model_location, subfolder=subfolder, local_files_only=is_ci_v2_env or is_ci_env
    )
    torch.manual_seed(seed)
    unet = UNet2DConditionModel.from_config(config)
    unet.eval()
    return unet


def load_scheduler(pipeline_location=SDXL_BASE_REPO, is_ci_env=False, is_ci_v2_env=False):
    """diffusers ``EulerDiscreteScheduler`` with the SDXL base config (fetched, else built inline)."""
    from diffusers import EulerDiscreteScheduler

    try:
        return EulerDiscreteScheduler.from_pretrained(
            pipeline_location, subfolder="scheduler", local_files_only=is_ci_env or is_ci_v2_env
        )
    except Exception as e:
        logger.warning(f"SDXL scheduler config not available ({type(e).__name__}); using the inline SDXL base config")
        return EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            trained_betas=None,
            prediction_type="epsilon",
            interpolation_type="linear",
            use_karras_sigmas=False,
            timestep_spacing="leading",
            steps_offset=1,
        )


def to_device_tile(torch_tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None):
    return qsr.from_torch(
        torch_tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG
    )


def to_device_nhwc(torch_nchw, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None):
    """NCHW torch tensor -> device tensor of shape [B, 1, H*W, C] (channels last, flattened).

    Returns the device tensor and ``[B, C, H, W]`` as the model modules expect.
    """
    B, C, H, W = torch_nchw.shape
    flat = torch.permute(torch_nchw, (0, 2, 3, 1)).reshape(B, 1, H * W, C)
    return to_device_tile(flat, device, dtype=dtype, layout=layout, memory_config=memory_config), [B, C, H, W]


def from_device_nhwc(ttnn_tensor, output_shape, batch=1):
    """[B, 1, H*W, C] device tensor -> NCHW torch tensor given ``[C, H, W]``."""
    C, H, W = output_shape
    out = ttnn.to_torch(ttnn_tensor)
    out = out.reshape(batch, H, W, C)
    return torch.permute(out, (0, 3, 1, 2))
