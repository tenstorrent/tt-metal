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

import contextlib
import os

import ttnn
from models.tt_dit.experimental.pipelines.pipeline_wan_distill import _patch_torch_transformer_random
from models.tt_dit.experimental.utils.lightx2v_loader import load_lightx2v_state_dict
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.tt_dit.utils import cache


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
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        anisora_local_dir: str | None = None,
        allow_download: bool | None = None,
        random_weights: bool | None = None,
    ) -> None:
        if allow_download is None:
            allow_download = os.environ.get("TT_DIT_ALLOW_HF_DOWNLOAD") == "1"
        if anisora_local_dir is None:
            anisora_local_dir = os.environ.get("ANISORA_LOCAL_DIR")
        if random_weights is None:
            random_weights = os.environ.get("TT_DIT_RANDOM_WEIGHTS") == "1"

        self._anisora_local_dir = anisora_local_dir
        self._allow_download = allow_download
        self._random_weights = random_weights

        ctx = _patch_torch_transformer_random() if random_weights else contextlib.nullcontext()
        with ctx:
            super().__init__(device=device, config=config)

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]
        if self._random_weights:
            cache.load_model(
                state.model,
                model_name=self.RANDOM_CACHE_NAMESPACE,
                subfolder=state.checkpoint.subfolder,
                parallel_config=self.parallel_config,
                mesh_shape=tuple(self.mesh_device.shape),
                mesh_device=self.mesh_device,
                is_fsdp=self.is_fsdp,
                get_torch_state_dict=lambda s=state: s.checkpoint.state_dict(),
            )
            return

        filename = self.HIGH_NOISE_FILE if idx == 0 else self.LOW_NOISE_FILE
        cache.load_model(
            state.model,
            model_name=self.CACHE_NAMESPACE,
            subfolder=state.checkpoint.subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            mesh_device=self.mesh_device,
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=lambda f=filename: load_lightx2v_state_dict(
                self.HF_REPO,
                f,
                allow_download=self._allow_download,
                local_dir=self._anisora_local_dir,
            ),
        )

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_links: int | None = None,
        dynamic_load: bool | None = None,
        topology: ttnn.Topology | None = None,
        is_fsdp: bool | None = None,
    ) -> AniSoraPipeline:
        config = WanPipelineConfig.default(
            mesh_shape=mesh_device.shape,
            checkpoint_name=cls.BASE_DIFFUSERS_REPO,
            height=height,
            width=width,
            num_frames=num_frames,
            num_links=num_links,
            topology=topology,
            dynamic_load=dynamic_load,
            is_fsdp=is_fsdp,
            boundary_ratio=cls.ANISORA_BOUNDARY_RATIO,
            model_type="i2v",
        )
        return cls(device=mesh_device, config=config)
