# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 lightx2v distilled I2V pipeline.

A thin subclass of :class:`WanPipelineI2V` that:

1. Uses ``Wan-AI/Wan2.2-I2V-A14B-Diffusers`` for tokenizer, text encoder, VAE,
   and scheduler — the lightx2v repo ships only the DiT.
2. Replaces both expert transformer state dicts with lightx2v's flat
   ``.safetensors`` files (4-step distill, high-noise + low-noise pair).
3. Defaults ``boundary_ratio=0.5`` (2 high-noise + 2 low-noise steps).

Test/caller passes ``num_inference_steps=4`` and ``guidance_scale=1.0`` for both
stages since CFG is baked into the distill.
"""
from __future__ import annotations

import os

from ...utils import cache
from ...utils.lightx2v_loader import load_lightx2v_state_dict
from .pipeline_wan import WanPipeline
from .pipeline_wan_i2v import WanPipelineI2V


class WanDistillPipelineI2V(WanPipelineI2V):
    LIGHTX2V_REPO = "lightx2v/Wan2.2-Distill-Models"
    HIGH_NOISE_FILE = "wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors"
    LOW_NOISE_FILE = "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"
    BASE_DIFFUSERS_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    CACHE_NAMESPACE = "Wan2.2-Distill-lightx2v-4step"
    DISTILL_BOUNDARY_RATIO = 0.5

    def __init__(
        self,
        *args,
        lightx2v_local_dir: str | None = None,
        allow_download: bool | None = None,
        **kwargs,
    ):
        kwargs["checkpoint_name"] = kwargs.get("checkpoint_name") or self.BASE_DIFFUSERS_REPO
        kwargs["boundary_ratio"] = self.DISTILL_BOUNDARY_RATIO

        if allow_download is None:
            allow_download = os.environ.get("TT_DIT_ALLOW_HF_DOWNLOAD") == "1"
        if lightx2v_local_dir is None:
            lightx2v_local_dir = os.environ.get("LIGHTX2V_LOCAL_DIR")

        self._lightx2v_local_dir = lightx2v_local_dir
        self._allow_download = allow_download

        super().__init__(*args, **kwargs)

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]
        filename = self.HIGH_NOISE_FILE if idx == 0 else self.LOW_NOISE_FILE
        cache.load_model(
            state.model,
            model_name=self.CACHE_NAMESPACE,
            subfolder=state.subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=lambda f=filename: load_lightx2v_state_dict(
                self.LIGHTX2V_REPO,
                f,
                allow_download=self._allow_download,
                local_dir=self._lightx2v_local_dir,
            ),
        )

    @staticmethod
    def create_pipeline(*args, **kwargs):
        kwargs["checkpoint_name"] = kwargs.get("checkpoint_name") or WanDistillPipelineI2V.BASE_DIFFUSERS_REPO
        return WanPipeline.create_pipeline(*args, pipeline_class=WanDistillPipelineI2V, **kwargs)
