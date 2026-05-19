# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 I2V pipeline with LoRA adapters fused into the base PyTorch weights.

Each expert transformer (high-noise + low-noise) gets its own ordered LoRA
stack. Stacks are fused on CPU before TT conversion, so inference uses
vanilla I2V machinery with no LoRA-specific runtime cost.

Single-LoRA usage::

    pipe = WanPipelineI2VLora.create_pipeline(
        mesh_device=...,
        lora_high="/path/high.safetensors",
        lora_low="/path/low.safetensors",
    )

Multi-LoRA stacking (e.g. LightX2V LoRA + style LoRA, applied in order)::

    pipe = WanPipelineI2VLora.create_pipeline(
        mesh_device=...,
        lora_high=[
            LoRASpec("/path/lightx2v_high.safetensors", scale=1.0),
            LoRASpec("/path/style_high.safetensors", scale=0.5),
        ],
        lora_low=[LoRASpec("/path/lightx2v_low.safetensors", scale=1.0)],
    )

The TT cache is keyed by a SHA1 hash of the ordered ``(path, scale)`` tuples
per expert, so distinct stacks never alias.
"""
from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file

from models.tt_dit.experimental.utils.lora import (
    LoRAArg,
    LoRASpec,
    fuse_lora_state_dict,
    lora_stack_cache_namespace,
    normalize_lora_arg,
    verify_fusion_changed_weights,
)
from models.tt_dit.pipelines.wan.pipeline_wan import WanPipeline
from models.tt_dit.pipelines.wan.pipeline_wan_i2v import WanPipelineI2V
from models.tt_dit.utils import cache


def _has_lora_keys(state_dict: dict) -> bool:
    return any(
        ("lora_A" in k)
        or ("lora_B" in k)
        or ("lora_down" in k)
        or ("lora_up" in k)
        or k.endswith(".diff")
        or k.endswith(".diff_b")
        or k.startswith("lora_unet_")
        for k in state_dict
    )


class WanPipelineI2VLora(WanPipelineI2V):
    """Wan2.2 I2V with LoRA stacks fused into the base PyTorch weights."""

    def __init__(
        self,
        *args,
        lora_high: LoRAArg = None,
        lora_low: LoRAArg = None,
        **kwargs,
    ):
        high_specs = normalize_lora_arg(lora_high)
        low_specs = normalize_lora_arg(lora_low)

        if not high_specs and not low_specs:
            raise ValueError(
                "WanPipelineI2VLora requires at least one LoRA. "
                "Pass lora_high and/or lora_low as a path, LoRASpec, or list."
            )

        for label, specs in [("lora_high", high_specs), ("lora_low", low_specs)]:
            for spec in specs:
                if not Path(spec.path).is_file():
                    raise FileNotFoundError(f"{label}: file does not exist: {spec.path}")

        self._lora_specs: dict[int, list[LoRASpec]] = {0: high_specs, 1: low_specs}
        self._cache_namespace = lora_stack_cache_namespace(self._lora_specs)
        # Lazily-built fused state dicts keyed by transformer index. Cleared
        # after handoff to TT cache to free CPU memory.
        self._fused_state_dicts: dict[int, dict[str, torch.Tensor] | None] = {0: None, 1: None}

        super().__init__(*args, **kwargs)

    def prepare_text_conditioning(self, tt_model, prompt_embeds, buffer, traced=False):
        # When guidance_scale=1.0 the encoder returns negative_prompt_embeds=None.
        # The base loop still calls this for the negative buffer; forwarding None
        # would hit NoneType in Linear. combined_step short-circuits when
        # do_classifier_free_guidance is False, so leaving the buffer untouched
        # is safe.
        if prompt_embeds is None:
            return buffer
        return super().prepare_text_conditioning(tt_model, prompt_embeds, buffer, traced)

    def _build_fused_state_dict(self, idx: int) -> dict[str, torch.Tensor] | None:
        specs = self._lora_specs[idx]
        state = self.transformer_states[idx]
        if not specs:
            logger.info(f"No LoRA for expert idx={idx} ('{state.subfolder}') -- using base weights")
            return None

        base_sd = state.torch_model.state_dict()
        fused_sd = base_sd
        for spec in specs:
            logger.info(f"Loading LoRA for '{state.subfolder}' from {spec.path} (scale={spec.scale})")
            lora_sd = load_file(str(spec.path))
            if not _has_lora_keys(lora_sd):
                raise RuntimeError(
                    f"No LoRA-style keys (lora_A/lora_B, lora_down/lora_up, diff/diff_b) found in {spec.path}"
                )
            fused_sd = fuse_lora_state_dict(fused_sd, lora_sd, scale=spec.scale)

        verify_fusion_changed_weights(
            base_sd,
            fused_sd,
            label=f"{state.subfolder} (stack of {len(specs)})",
        )
        return fused_sd

    def _prepare_transformer(self, idx: int):
        state = self.transformer_states[idx]

        if not self._lora_specs[idx]:
            super()._prepare_transformer(idx)
            return

        def _get_state_dict(idx_=idx):
            cached = self._fused_state_dicts.get(idx_)
            if cached is not None:
                return cached
            sd = self._build_fused_state_dict(idx_)
            self._fused_state_dicts[idx_] = sd
            return sd

        cache.load_model(
            state.model,
            model_name=self._cache_namespace,
            subfolder=state.subfolder,
            parallel_config=self.parallel_config,
            mesh_shape=tuple(self.mesh_device.shape),
            is_fsdp=self.is_fsdp,
            get_torch_state_dict=_get_state_dict,
        )
        self._fused_state_dicts[idx] = None

    @staticmethod
    def create_pipeline(*args, **kwargs):
        kwargs["checkpoint_name"] = kwargs.get("checkpoint_name") or "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
        return WanPipeline.create_pipeline(*args, pipeline_class=WanPipelineI2VLora, **kwargs)
