# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mistral decoder block base classes.

This file intentionally carries two layers:
- ``DecoderBlock2DBase``: DeepSeek-style TT orchestration class (convert/config/state/forward)
- ``Mistral4DecoderBlock2DBase``: eager HF-compatible ``nn.Module`` block used by parity tests
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config
from transformers.models.mistral4.modeling_mistral4 import Mistral4Attention

import ttnn
from models.demos.mistral_small_4_119B.tt.decoder_block.decoder_block_base import DecoderBlockBase
from models.demos.mistral_small_4_119B.tt.mistral4_rmsnorm import TtMistral4RMSNorm
from models.demos.mistral_small_4_119B.tt.mla.mla2d import MistralSmall4MLA2D
from models.demos.mistral_small_4_119B.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.config_dataclass import KvCacheConfig
from models.demos.mistral_small_4_119B.tt_utils.config_helpers import sub_state_dict
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.tt_transformers.tt.common import PagedAttentionConfig


def _has_distinct_buffer(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    try:
        return a.buffer_address() != b.buffer_address()
    except Exception:
        return a is not b


class DecoderBlock2DBase(DecoderBlockBase):
    """DeepSeek-style TT decoder block wiring for Mistral components."""

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        (state_dict,) = state_dicts
        assert state_dict is not None, "Expected a state dict for DecoderBlock."
        return {
            "mla_norm": DistributedRMSNorm.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "input_layernorm."),) * mesh_device.shape[0],
                output_path / "mla_norm",
                mesh_device,
            ),
            "mla": MistralSmall4MLA2D.convert_weights(
                hf_config, (sub_state_dict(state_dict, "self_attn."),), output_path / "mla", mesh_device
            ),
            "mlp_norm": DistributedRMSNorm.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "post_attention_layernorm."),) * mesh_device.shape[0],
                output_path / "mlp_norm",
                mesh_device,
            ),
            "mlp": cls.convert_mlp_weights(
                hf_config,
                sub_state_dict(state_dict, "mlp."),
                output_path / "mlp",
                mesh_device,
            ),
        }

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL | None,
        mla_cache: torch.Tensor | None = None,
        kv_cache_override: KvCacheConfig | None = None,
    ) -> ModelState:
        # ``DistributedRMSNorm``, MLA decode, and dense MLP all call ``ccl.populate_*`` even on a 1×1 mesh
        # (e.g. ``all_gather_async`` with one rank); callers that pass ``None`` for "no multi-device" still
        # need a real ``CCL`` for semaphores and link metadata.
        if ccl is None:
            ccl = CCL(mesh_device)
        return {
            "mla_norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "mla": cls.create_mla_state(hf_config, paged_config, mesh_device, ccl, mla_cache, kv_cache_override),
            "mlp_norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "mlp": cls.create_mlp_state(hf_config, mesh_device, ccl),
        }

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        user_id: int,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        mla_norm_out = DistributedRMSNorm.forward_prefill(x, cfg["mla_norm"])
        mla_out = MistralSmall4MLA2D.forward_prefill(mla_norm_out, user_id, cfg["mla"], rope_tensors, page_table)
        ttnn.deallocate(mla_norm_out)

        x += mla_out
        ttnn.deallocate(mla_out)

        mlp_norm_out = DistributedRMSNorm.forward_prefill(x, cfg["mlp_norm"])
        mlp_out = cls.forward_mlp_prefill(mlp_norm_out, cfg["mlp"])
        ttnn.deallocate(mlp_norm_out)

        x += mlp_out
        ttnn.deallocate(mlp_out)
        return x

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        mla_norm_in = ttnn.to_memory_config(x, **cfg["mla_norm_reshard"])
        mla_norm_out = DistributedRMSNorm.forward_decode(mla_norm_in, cfg["mla_norm"])
        if _has_distinct_buffer(mla_norm_in, x):
            ttnn.deallocate(mla_norm_in)

        mla_norm_out = ttnn.to_memory_config(mla_norm_out, **cfg["mla_reshard"])
        mla_out = MistralSmall4MLA2D.forward_decode(mla_norm_out, position_idxs, cfg["mla"], rope_tensors, page_table)
        ttnn.deallocate(mla_norm_out)

        x += mla_out
        ttnn.deallocate(mla_out)

        mlp_norm_in = ttnn.to_memory_config(x, **cfg["mlp_norm_reshard"])
        mlp_norm_out = DistributedRMSNorm.forward_decode(mlp_norm_in, cfg["mlp_norm"])
        if _has_distinct_buffer(mlp_norm_in, x):
            ttnn.deallocate(mlp_norm_in)

        mlp_norm_out = ttnn.to_memory_config(mlp_norm_out, **cfg["mlp_reshard"])
        mlp_out = cls.forward_mlp_decode(mlp_norm_out, cfg["mlp"])
        ttnn.deallocate(mlp_norm_out)

        x += mlp_out
        ttnn.deallocate(mlp_out)
        return x

    @classmethod
    @abstractmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
    ) -> ModelState:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        raise NotImplementedError


class Mistral4DecoderBlock2DBase(nn.Module):
    """Pre-norm decoder block with injected FFN submodule (dense or MoE)."""

    def __init__(
        self,
        config: Mistral4Config,
        layer_idx: int,
        mlp: nn.Module,
        *,
        attn: Mistral4Attention | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = attn if attn is not None else Mistral4Attention(config=config, layer_idx=layer_idx)
        self.mlp = mlp

        self.input_layernorm = TtMistral4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TtMistral4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


__all__ = ["DecoderBlock2DBase", "Mistral4DecoderBlock2DBase"]
