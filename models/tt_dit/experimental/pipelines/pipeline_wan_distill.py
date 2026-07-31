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

A ``random_weights`` mode is provided for smoke-testing without HuggingFace
downloads. It (a) skips the two ~28 GB transformer subfolder downloads by
constructing config-only random ``TorchWanTransformer3DModel`` instances and
(b) feeds those random ``state_dict()``s into the TT model in place of the
lightx2v safetensors. Tokenizer, text encoder, VAE, and scheduler still come
from the base diffusers repo (~12 GB total).
"""
from __future__ import annotations

import contextlib
import os

import torch

import ttnn
from models.tt_dit.experimental.utils.lightx2v_loader import load_lightx2v_state_dict
from models.tt_dit.models.transformers.wan2_2.transformer_wan import TorchWanTransformer3DModel
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipelineConfig
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.tt_dit.utils import cache

# Hard-coded config for Wan2.2-I2V-A14B-Diffusers transformer subfolders. Used
# only in random_weights mode so we don't have to fetch transformer/config.json
# from HF. Both transformer and transformer_2 share this architecture; only
# weights differ between high-noise and low-noise experts.
_RANDOM_I2V_TRANSFORMER_CONFIG = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=40,
    attention_head_dim=128,
    in_channels=36,  # I2V: 4 mask + 16 image latent + 16 noise latent
    out_channels=16,
    text_dim=4096,
    freq_dim=256,
    ffn_dim=13824,
    num_layers=40,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=1024,
    image_dim=None,
    added_kv_proj_dim=None,
    pos_embed_seq_len=None,
)


@contextlib.contextmanager
def _patch_torch_transformer_random(seed: int = 0):
    """Replace ``TorchWanTransformer3DModel.from_pretrained`` with a stub that
    instantiates from a hard-coded I2V config and never touches the network or
    disk. Restored on exit."""
    cls = TorchWanTransformer3DModel
    sentinel = object()
    saved = cls.__dict__.get("from_pretrained", sentinel)

    def _stub(_cls, *args, **kwargs):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            return _cls(**_RANDOM_I2V_TRANSFORMER_CONFIG)

    cls.from_pretrained = classmethod(_stub)
    try:
        yield
    finally:
        if saved is sentinel:
            # `from_pretrained` was inherited; delete our override to restore inheritance.
            try:
                delattr(cls, "from_pretrained")
            except AttributeError:
                pass
        else:
            cls.from_pretrained = saved


class WanDistillPipelineI2V(WanPipelineI2V):
    LIGHTX2V_REPO = "lightx2v/Wan2.2-Distill-Models"
    HIGH_NOISE_FILE = "wan2.2_i2v_A14b_high_noise_lightx2v_4step.safetensors"
    LOW_NOISE_FILE = "wan2.2_i2v_A14b_low_noise_lightx2v_4step.safetensors"
    BASE_DIFFUSERS_REPO = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    CACHE_NAMESPACE = "Wan2.2-Distill-lightx2v-4step"
    RANDOM_CACHE_NAMESPACE = "Wan2.2-Distill-random"
    DISTILL_BOUNDARY_RATIO = 0.5

    def __init__(
        self,
        *,
        device: ttnn.MeshDevice,
        config: WanPipelineConfig,
        lightx2v_local_dir: str | None = None,
        allow_download: bool | None = None,
        random_weights: bool | None = None,
    ) -> None:
        if allow_download is None:
            allow_download = os.environ.get("TT_DIT_ALLOW_HF_DOWNLOAD") == "1"
        if lightx2v_local_dir is None:
            lightx2v_local_dir = os.environ.get("LIGHTX2V_LOCAL_DIR")
        if random_weights is None:
            random_weights = os.environ.get("TT_DIT_RANDOM_WEIGHTS") == "1"

        self._lightx2v_local_dir = lightx2v_local_dir
        self._allow_download = allow_download
        self._random_weights = random_weights

        ctx = _patch_torch_transformer_random() if random_weights else contextlib.nullcontext()
        with ctx:
            super().__init__(device=device, config=config)

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        # When CFG is baked in (guidance_scale=1.0), encode_prompt returns
        # negative_prompt_embeds=None. The base loop still calls this for the
        # negative buffer; forwarding None into the text embedder hits a
        # NoneType.padded_shape in Linear. combined_step already short-circuits
        # on do_classifier_free_guidance=False, so leaving the buffer as-is is
        # safe.
        if prompt_embeds is None:
            return buffer
        return super().prepare_text_conditioning(tt_model, prompt_embeds, buffer, traced)

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]
        if self._random_weights:
            # Use the (random) state_dict the WanCheckpoint loaded under the
            # _patch_torch_transformer_random monkey-patch. Cache under a
            # separate namespace so a real-weights run doesn't reuse it.
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
                self.LIGHTX2V_REPO,
                f,
                allow_download=self._allow_download,
                local_dir=self._lightx2v_local_dir,
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
    ) -> WanDistillPipelineI2V:
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
            boundary_ratio=cls.DISTILL_BOUNDARY_RATIO,
            model_type="i2v",
        )
        return cls(device=mesh_device, config=config)
