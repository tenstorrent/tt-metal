# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""IndexTeam/Index-anisora V3.2 I2V pipeline.

A thin subclass of :class:`WanPipelineI2V` that:

1. Uses ``Wan-AI/Wan2.2-I2V-A14B-Diffusers`` for tokenizer, text encoder, VAE,
   and scheduler — the AniSora HF repo only ships the DiT experts.
2. Replaces both expert transformer state dicts with AniSora's ``V3.2``
   ``diffusion_pytorch_model.safetensors`` files (high-noise + low-noise pair).
3. Defaults to 40 sampling steps with ``boundary_ratio=0.9`` and
   ``guidance_scale=3.5`` for both stages, matching the upstream config in
   ``anisoraV3.2/wan/configs/wan_i2v_A14B.py``.

AniSora's safetensors use the same original-Wan key naming as the lightx2v
distill checkpoints (verified: 1095 keys, identical layout). The
``wan_lightx2v_to_diffusers_key`` rename function in
:mod:`models.tt_dit.utils.lightx2v_loader` is reused as-is.

A ``random_weights`` mode is provided for smoke-testing without HuggingFace
downloads. It (a) skips the two ~28 GB transformer subfolder downloads by
constructing config-only random ``TorchWanTransformer3DModel`` instances and
(b) feeds those random ``state_dict()``s into the TT model in place of the
AniSora safetensors. Tokenizer, text encoder, VAE, and scheduler still come
from the base diffusers repo (~12 GB total).
"""
from __future__ import annotations

import os
from contextlib import nullcontext

from ...utils import cache
from ...utils.lightx2v_loader import load_lightx2v_state_dict
from .pipeline_wan import WanPipeline
from .pipeline_wan_distill import _patch_torch_transformer_random
from .pipeline_wan_i2v import WanPipelineI2V


class AniSoraPipeline(WanPipelineI2V):
    HF_REPO = "IndexTeam/Index-anisora"
    HIGH_NOISE_FILE = "V3.2/high_noise_model/diffusion_pytorch_model.safetensors"
    LOW_NOISE_FILE = "V3.2/low_noise_model/diffusion_pytorch_model.safetensors"
    BASE_DIFFUSERS_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    CACHE_NAMESPACE = "Index-anisora-V3.2"
    RANDOM_CACHE_NAMESPACE = "Index-anisora-random"
    ANISORA_BOUNDARY_RATIO = 0.9

    def __init__(
        self,
        *args,
        anisora_local_dir: str | None = None,
        allow_download: bool | None = None,
        random_weights: bool | None = None,
        **kwargs,
    ):
        kwargs["checkpoint_name"] = kwargs.get("checkpoint_name") or self.BASE_DIFFUSERS_REPO
        kwargs["boundary_ratio"] = self.ANISORA_BOUNDARY_RATIO

        if allow_download is None:
            allow_download = os.environ.get("TT_DIT_ALLOW_HF_DOWNLOAD") == "1"
        if anisora_local_dir is None:
            anisora_local_dir = os.environ.get("ANISORA_LOCAL_DIR")
        if random_weights is None:
            random_weights = os.environ.get("TT_DIT_RANDOM_WEIGHTS") == "1"

        self._anisora_local_dir = anisora_local_dir
        self._allow_download = allow_download
        self._random_weights = random_weights

        ctx = _patch_torch_transformer_random() if random_weights else nullcontext()
        with ctx:
            super().__init__(*args, **kwargs)

    def _prepare_transformer(self, idx: int):
        if self._random_weights:
            state = self.transformer_states[idx]
            cache.load_model(
                state.model,
                model_name=self.RANDOM_CACHE_NAMESPACE,
                subfolder=state.subfolder,
                parallel_config=self.parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                is_fsdp=self.is_fsdp,
                get_torch_state_dict=lambda s=state: s.torch_model.state_dict(),
            )
            return

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
                self.HF_REPO,
                f,
                allow_download=self._allow_download,
                local_dir=self._anisora_local_dir,
            ),
        )

    @staticmethod
    def create_pipeline(*args, random_weights: bool | None = None, **kwargs):
        kwargs["checkpoint_name"] = kwargs.get("checkpoint_name") or AniSoraPipeline.BASE_DIFFUSERS_REPO
        if "allow_download" in kwargs:
            allow = kwargs.pop("allow_download")
            if allow:
                os.environ["TT_DIT_ALLOW_HF_DOWNLOAD"] = "1"
        if "anisora_local_dir" in kwargs:
            val = kwargs.pop("anisora_local_dir")
            if val is not None:
                os.environ["ANISORA_LOCAL_DIR"] = val
        if random_weights:
            os.environ["TT_DIT_RANDOM_WEIGHTS"] = "1"
        try:
            return WanPipeline.create_pipeline(*args, pipeline_class=AniSoraPipeline, **kwargs)
        finally:
            os.environ.pop("TT_DIT_RANDOM_WEIGHTS", None)
