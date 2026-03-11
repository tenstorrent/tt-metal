# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from time import perf_counter
from typing import Sequence

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import ReshardConfig
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
)
from models.demos.deepseek_v3.utils.shared_state_addon import SharedStateAddOn
from models.tt_transformers.tt.common import PagedAttentionConfig


class DecoderBlockBase(SharedStateAddOn, AbstractModule):
    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        mla_norm_config = DistributedRMSNorm.prefill_model_config(hf_config, mesh_device)
        mlp_norm_config = DistributedRMSNorm.prefill_model_config(hf_config, mesh_device)

        mla_config = cls.prefill_mla_config(hf_config, mesh_device)

        return {
            "mla_norm_reshard": ReshardConfig(memory_config=mla_norm_config["input_memory_config"]),
            "mla_norm": mla_norm_config,
            "mla_reshard": ReshardConfig(memory_config=mla_config["input_memory_config"]),
            "mla": mla_config,
            "mlp_norm_reshard": ReshardConfig(memory_config=mlp_norm_config["input_memory_config"]),
            "mlp_norm": mlp_norm_config,
            "mlp_reshard": ReshardConfig(memory_config=ttnn.DRAM_MEMORY_CONFIG),
            "mlp": cls.prefill_mlp_config(hf_config, mesh_device),
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelDecodeConfig:
        mla_norm_config = DistributedRMSNorm.decode_model_config(hf_config, mesh_device)
        mlp_norm_config = DistributedRMSNorm.decode_model_config(hf_config, mesh_device)

        mla_config = cls.decode_mla_config(hf_config, mesh_device)
        mlp_config = cls.decode_mlp_config(hf_config, mesh_device)

        # Get the input_memory_config for the mlp_reshard
        # For MoE blocks, input_memory_config is nested inside shared_expert
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
        mlp_shared_state = cls.create_mlp_shared_state(
            hf_config,
            mesh_device,
        )
        logger.info(f"Created {cls.__name__} MLP shared state in {perf_counter() - mlp_start:.2f}s")
        state = {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "mlp": mlp_shared_state,
        }
        logger.info(f"Created {cls.__name__} shared state")
        return state

    @classmethod
    @abstractmethod
    def prefill_mla_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        """
        Prefill configuration for the MLA component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLA configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        """
        Prefill configuration for the MLP component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def decode_mla_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelDecodeConfig:
        """
        Decode configuration for the MLA component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLA configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelDecodeConfig:
        """
        Decode configuration for the MLP component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def create_mla_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
        mla_cache: Sequence[torch.Tensor] | None = None,
    ) -> ModelState:
        """
        Create the state for the MLA component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLA configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def create_mlp_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        """
        Create the shared state for the MLP component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
