# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_base import DecoderBlockBase
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.tt_transformers.tt.common import PagedAttentionConfig


class DecoderBlock2DBase(DecoderBlockBase):
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
            "mla": MLA2D.convert_weights(
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
        ccl: CCL,
        mla_cache: torch.Tensor | None = None,
    ) -> ModelState:
        return {
            "mla_norm": DistributedRMSNorm.create_state(hf_config, mesh_device, ccl),
            "mla": cls.create_mla_state(hf_config, paged_config, mesh_device, ccl, mla_cache),
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
        # MLA norm
        mla_norm_out = DistributedRMSNorm.forward_prefill(x, cfg["mla_norm"])

        # MLA
        mla_out = MLA2D.forward_prefill(mla_norm_out, user_id, cfg["mla"], rope_tensors, page_table)
        ttnn.deallocate(mla_norm_out)

        # MLA Residual
        x += mla_out
        ttnn.deallocate(mla_out)

        # MLP norm
        mlp_norm_out = DistributedRMSNorm.forward_prefill(x, cfg["mlp_norm"])

        # MLP
        mlp_out = cls.forward_mlp_prefill(mlp_norm_out, cfg["mlp"])
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
        mla_out = MLA2D.forward_decode(mla_norm_out, position_idxs, cfg["mla"], rope_tensors, page_table)
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
        mlp_out = cls.forward_mlp_decode(mlp_norm_out, cfg["mlp"])
        ttnn.deallocate(mlp_norm_out)

        # MLP Residual
        x += mlp_out
        ttnn.deallocate(mlp_out)

        return x

    @classmethod
    @abstractmethod
    def prefill_mla_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        return MLA2D.prefill_model_config(hf_config, mesh_device)

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
        return MLA2D.decode_model_config(hf_config, mesh_device)

    @classmethod
    @abstractmethod
    def create_mla_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
        mla_cache: torch.Tensor | None = None,
    ) -> ModelState:
        return MLA2D.create_state(hf_config, paged_config, mesh_device, ccl, mla_cache)

    @classmethod
    @abstractmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
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
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
    ) -> ModelState:
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        """
        Forward pass for the MLP component during prefill.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

    @classmethod
    @abstractmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        """
        Forward pass for the MLP component during decode.
        This method should be implemented by subclasses to handle specific MLP configurations.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")
