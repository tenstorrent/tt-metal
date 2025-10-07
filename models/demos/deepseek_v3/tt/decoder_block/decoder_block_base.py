# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from pathlib import Path
from typing import Sequence

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.mla_1d import MLA1D
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import ReshardConfig
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dicts
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.demos.deepseek_v3.utils.shared_state_addon import SharedStateAddOn
from models.tt_transformers.tt.common import PagedAttentionConfig


class DecoderBlockBase(SharedStateAddOn, AbstractModule):
    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        return {
            "mla_norm": DistributedRMSNorm.convert_weights(
                hf_config, sub_state_dicts(state_dicts, "input_layernorm."), output_path / "mla_norm", mesh_device
            ),
            "mla": MLA1D.convert_weights(
                hf_config, sub_state_dicts(state_dicts, "self_attn."), output_path / "mla", mesh_device
            ),
            "mlp_norm": DistributedRMSNorm.convert_weights(
                hf_config,
                sub_state_dicts(state_dicts, "post_attention_layernorm."),
                output_path / "mlp_norm",
                mesh_device,
            ),
            "mlp": cls.convert_mlp_weights(
                hf_config,
                sub_state_dicts(state_dicts, "mlp."),
                output_path / "mlp",
                mesh_device,
            ),
        }

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelPrefillConfig:
        return {
            "mla_norm": DistributedRMSNorm.prefill_model_config(hf_config, mesh_device),
            "mla": MLA1D.prefill_model_config(hf_config, mesh_device),
            "mlp_norm": DistributedRMSNorm.prefill_model_config(hf_config, mesh_device),
            "mlp": cls.prefill_mlp_config(hf_config, mesh_device, is_padding_layer),
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelDecodeConfig:
        mla_norm_config = DistributedRMSNorm.decode_model_config(hf_config, mesh_device)
        mlp_norm_config = DistributedRMSNorm.decode_model_config(hf_config, mesh_device)

        mla_config = MLA1D.decode_model_config(hf_config, mesh_device)

        return {
            "mla_norm_reshard": ReshardConfig(memory_config=mla_norm_config["input_memory_config"]),
            "mla_norm": mla_norm_config,
            "mla_reshard": ReshardConfig(memory_config=mla_config["input_memory_config"]),
            "mla": mla_config,
            "mlp_norm_reshard": ReshardConfig(memory_config=mlp_norm_config["input_memory_config"]),
            "mlp_norm": mlp_norm_config,
            "mlp_reshard": ReshardConfig(memory_config=ttnn.DRAM_MEMORY_CONFIG),
            "mlp": cls.decode_mlp_config(hf_config, mesh_device, is_padding_layer),
        }

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL1D,
        is_padding_layer: tuple[bool, ...] | None = None,
        mla_cache: Sequence[torch.Tensor] | None = None,
    ) -> ModelState:
        return {
            "mla_norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "mla": MLA1D.create_state(hf_config, paged_config, mesh_device, ccl, mla_cache),
            "mlp_norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "mlp": cls.create_mlp_state(
                hf_config,
                mesh_device,
                ccl,
                is_padding_layer,
            ),
        }

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelState:
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "mlp": cls.create_mlp_shared_state(
                hf_config,
                mesh_device,
                is_padding_layer,
            ),
        }

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        user_id: int,
        row_idx: int,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        # MLA norm
        mla_norm_out = DistributedRMSNorm.forward_prefill(x, cfg["mla_norm"])

        # MLA
        mla_out = MLA1D.forward_prefill(mla_norm_out, user_id, row_idx, cfg["mla"], rope_tensors, page_table)
        ttnn.deallocate(mla_norm_out)

        # MLA Residual
        x += mla_out
        ttnn.deallocate(mla_out)

        # MLP norm
        mlp_norm_out = DistributedRMSNorm.forward_prefill(x, cfg["mlp_norm"])

        # MLP
        mlp_out = cls.forward_mlp_prefill(mlp_norm_out, row_idx, cfg["mlp"])
        ttnn.deallocate(mlp_norm_out)

        # MLP Residual
        x += mlp_out
        ttnn.deallocate(mlp_out)

        return x

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        row_idx: int,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        # MLA norm
        mla_norm_in = ttnn.to_memory_config(x, **cfg["mla_norm_reshard"])
        mla_norm_out = DistributedRMSNorm.forward_decode(mla_norm_in, cfg["mla_norm"])
        ttnn.deallocate(mla_norm_in)

        # MLA
        mla_norm_out = ttnn.to_memory_config(mla_norm_out, **cfg["mla_reshard"])
        mla_out = MLA1D.forward_decode(mla_norm_out, position_idxs, row_idx, cfg["mla"], rope_tensors, page_table)
        ttnn.deallocate(mla_norm_out)

        # MLA Residual
        x += mla_out
        ttnn.deallocate(mla_out)

        # MLP norm
        mlp_norm_in = ttnn.to_memory_config(x, **cfg["mlp_norm_reshard"])
        mlp_norm_out = DistributedRMSNorm.forward_decode(mlp_norm_in, cfg["mlp_norm"])
        ttnn.deallocate(mlp_norm_in)

        # MLP
        mlp_norm_out = ttnn.to_memory_config(mlp_norm_out, **cfg["mlp_reshard"])
        mlp_out = cls.forward_mlp_decode(mlp_norm_out, row_idx, cfg["mlp"])
        ttnn.deallocate(mlp_norm_out)

        # MLP Residual
        x += mlp_out
        ttnn.deallocate(mlp_out)

        return x

    @classmethod
    @abstractmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        """
        Convert weights for the MLP component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelPrefillConfig:
        """
        Prefill configuration for the MLP component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelDecodeConfig:
        """
        Decode configuration for the MLP component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL1D,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelState:
        """
        Create the state for the MLP component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def create_mlp_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_padding_layer: tuple[bool, ...] | None = None,
    ) -> ModelState:
        """
        Create the shared state for the MLP component of the decoder layer.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, row_idx: int, cfg: RunPrefillConfig) -> ttnn.Tensor:
        """
        Forward pass for the MLP component during prefill.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, row_idx: int, cfg: RunDecodeConfig) -> ttnn.Tensor:
        """
        Forward pass for the MLP component during decode.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
