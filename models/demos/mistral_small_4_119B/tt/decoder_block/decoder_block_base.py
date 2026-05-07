# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Decoder block contracts for Mistral-Small-4.

This module now exposes both:
1) a DeepSeek-style TT orchestration base class (`DecoderBlockBase`) used by TT decoder-block modules
2) the eager factory helper (`build_mistral4_decoder_block`) used by `TtMistral4DecoderLayer`.
"""

from __future__ import annotations

from abc import abstractmethod
from time import perf_counter
from typing import Any

from loguru import logger
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

import ttnn
from models.demos.mistral_small_4_119B.tt.mla.mla2d import MistralSmall4MLA2D
from models.demos.mistral_small_4_119B.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.mistral_small_4_119B.tt_utils.abstract_module import AbstractModule
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.config_dataclass import KvCacheConfig, ReshardConfig
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
)
from models.tt_transformers.tt.common import PagedAttentionConfig


class DecoderBlockBase(AbstractModule):
    """DeepSeek-style TT decoder-block orchestration base."""

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
        batch_size_per_row: int,
    ) -> ModelPrefillConfig:
        mla_norm_config = DistributedRMSNorm.prefill_model_config(hf_config, mesh_device)
        mlp_norm_config = DistributedRMSNorm.prefill_model_config(hf_config, mesh_device)
        mla_config = cls.prefill_mla_config(hf_config, mesh_device, batch_size_per_row=batch_size_per_row)

        return {
            "mla_norm_reshard": ReshardConfig(memory_config=mla_norm_config["input_memory_config"]),
            "mla_norm": mla_norm_config,
            "mla_reshard": ReshardConfig(memory_config=mla_config["input_memory_config"]),
            "mla": mla_config,
            "mlp_norm_reshard": ReshardConfig(memory_config=mlp_norm_config["input_memory_config"]),
            "mlp_norm": mlp_norm_config,
            "mlp_reshard": ReshardConfig(memory_config=ttnn.DRAM_MEMORY_CONFIG),
            "mlp": cls.prefill_mlp_config(hf_config, mesh_device, fabric_config),
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
        batch_size_per_row: int,
    ) -> ModelDecodeConfig:
        mla_norm_config = DistributedRMSNorm.decode_model_config(
            hf_config, mesh_device, batch_size_per_row=batch_size_per_row
        )
        mlp_norm_config = DistributedRMSNorm.decode_model_config(
            hf_config, mesh_device, batch_size_per_row=batch_size_per_row
        )
        mla_config = cls.decode_mla_config(hf_config, mesh_device, batch_size_per_row=batch_size_per_row)
        mlp_config = cls.decode_mlp_config(hf_config, mesh_device, fabric_config, batch_size_per_row=batch_size_per_row)

        if "shared_expert" in mlp_config:
            mlp_input_memory_config = mlp_config["shared_expert"]["input_memory_config"]
        else:
            mlp_input_memory_config = mlp_config["input_memory_config"]

        return {
            "mla_norm_reshard": ReshardConfig(memory_config=mla_norm_config["input_memory_config"]),
            "mla_norm": mla_norm_config,
            "mla_reshard": ReshardConfig(memory_config=mla_config["input_memory_config"]),
            "mla": mla_config,
            "mlp_norm_reshard": ReshardConfig(memory_config=mlp_norm_config["input_memory_config"]),
            "mlp_norm": mlp_norm_config,
            "mlp_reshard": ReshardConfig(memory_config=mlp_input_memory_config),
            "mlp": mlp_config,
        }

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        logger.info(f"Creating {cls.__name__} shared state...")
        mlp_start = perf_counter()
        mlp_shared_state = cls.create_mlp_shared_state(hf_config, mesh_device)
        logger.info(f"Created {cls.__name__} MLP shared state in {perf_counter() - mlp_start:.2f}s")
        return {MESH_DEVICE_STATE_DICT_KEY: mesh_device, "mlp": mlp_shared_state}

    @classmethod
    @abstractmethod
    def prefill_mla_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, batch_size_per_row: int
    ) -> ModelPrefillConfig:
        return MistralSmall4MLA2D.prefill_model_config(hf_config, mesh_device, batch_size_per_row=batch_size_per_row)

    @classmethod
    @abstractmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
    ) -> ModelPrefillConfig:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def decode_mla_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, batch_size_per_row: int
    ) -> ModelDecodeConfig:
        return MistralSmall4MLA2D.decode_model_config(hf_config, mesh_device, batch_size_per_row=batch_size_per_row)

    @classmethod
    @abstractmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        fabric_config: ttnn.FabricConfig,
        batch_size_per_row: int,
    ) -> ModelDecodeConfig:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create_mla_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
        mla_cache: Any | None = None,
        kv_cache_override: KvCacheConfig | None = None,
    ) -> ModelState:
        return MistralSmall4MLA2D.create_state(hf_config, paged_config, mesh_device, ccl, mla_cache, kv_cache_override)

    @classmethod
    @abstractmethod
    def create_mlp_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        raise NotImplementedError


def build_mistral4_decoder_block(
    config: Mistral4Config, layer_idx: int
) -> Mistral4DenseDecoderBlock2D | Mistral4MoEDecoderBlock2D:
    """Instantiate the correct block type for ``layer_idx`` (dense vs MoE), matching HF routing."""
    from models.demos.mistral_small_4_119B.tt.decoder_block.decoder_block_2d import Mistral4DenseDecoderBlock2D
    from models.demos.mistral_small_4_119B.tt.decoder_block.moe_decoder_block_2d import Mistral4MoEDecoderBlock2D

    if layer_idx >= config.first_k_dense_replace:
        return Mistral4MoEDecoderBlock2D(config, layer_idx)
    return Mistral4DenseDecoderBlock2D(config, layer_idx)


__all__ = ["DecoderBlockBase", "build_mistral4_decoder_block"]
