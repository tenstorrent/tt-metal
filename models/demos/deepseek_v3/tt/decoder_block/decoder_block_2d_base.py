# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_base import DecoderBlockBase
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)

# Import tensor logging utility
from models.demos.deepseek_v3.utils.tensor_logger import log_tensor
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
        kv_cache_override: KvCacheConfig | None = None,
    ) -> ModelState:
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
        # Log initial input
        log_tensor(x, "decoder_block_input", "x")
        log_tensor(position_idxs, "decoder_block_input", "position_idxs")

        def x_to_torch(x: ttnn.Tensor) -> torch.Tensor:
            return ttnn.to_torch(
                x, mesh_composer=ttnn.ConcatMesh2dToTensor(x.device(), dims=(0, -1), mesh_shape=x.device().shape)
            )

        def comp_pcc_and_assert(x_torch: torch.Tensor, x1_torch: torch.Tensor, name: str = ""):
            passing, pcc_message = comp_pcc(x_torch, x1_torch)
            logger.info(f"FROM {name} -> PCC: {pcc_message}")
            assert passing, f"FROM {name} -> PCC value is lower than 0.99 for some of the outputs. Check Warnings!"

        x_initial_torch = x_to_torch(x)

        # MLA norm resharding
        mla_norm_in = ttnn.to_memory_config(x, **cfg["mla_norm_reshard"])
        ttnn.synchronize_device(x.device())
        log_tensor(mla_norm_in, "mla_norm_reshard", "mla_norm_in", {"config": cfg["mla_norm_reshard"]})
        comp_pcc_and_assert(x_initial_torch, x_to_torch(mla_norm_in), "x_initial_torch vs mla_norm_in")
        comp_pcc_and_assert(x_initial_torch, x_to_torch(x), "x_initial_torch vs x")

        # MLA norm
        mla_norm_out = DistributedRMSNorm.forward_decode(mla_norm_in, cfg["mla_norm"])
        ttnn.synchronize_device(x.device())
        log_tensor(mla_norm_out, "mla_norm", "mla_norm_out")
        ttnn.deallocate(mla_norm_in)
        comp_pcc_and_assert(x_initial_torch, x_to_torch(x), "x_initial_torch vs x")

        # MLA resharding
        mla_norm_out = ttnn.to_memory_config(mla_norm_out, **cfg["mla_reshard"])
        ttnn.synchronize_device(x.device())
        log_tensor(mla_norm_out, "mla_reshard", "mla_norm_out_resharded", {"config": cfg["mla_reshard"]})
        comp_pcc_and_assert(x_initial_torch, x_to_torch(x), "x_initial_torch vs x")

        # MLA forward
        comp_pcc_and_assert(x_initial_torch, x_to_torch(x), "before x_initial_torch vs x")
        mla_out = MLA2D.forward_decode(mla_norm_out, position_idxs, cfg["mla"], rope_tensors, page_table)
        ttnn.synchronize_device(x.device())
        log_tensor(mla_out, "mla", "mla_out")
        ttnn.deallocate(mla_norm_out)
        comp_pcc_and_assert(x_initial_torch, x_to_torch(x), "x_initial_torch vs x")

        # MLA Residual: y = mla(x) + x
        log_tensor(x, "mla_residual", "x_before_mla_residual", {"operation": "y = mla_out + x"})
        y = mla_out + x
        ttnn.synchronize_device(x.device())
        log_tensor(y, "mla_residual", "y_after_mla_residual", {"operation": "y = mla_out + x"})
        ttnn.deallocate(mla_out)

        # MLP norm resharding
        mlp_norm_in = ttnn.to_memory_config(y, **cfg["mlp_norm_reshard"])
        ttnn.synchronize_device(x.device())
        log_tensor(mlp_norm_in, "mlp_norm_reshard", "mlp_norm_in", {"config": cfg["mlp_norm_reshard"]})

        # MLP norm
        mlp_norm_out = DistributedRMSNorm.forward_decode(mlp_norm_in, cfg["mlp_norm"])
        ttnn.synchronize_device(x.device())
        log_tensor(mlp_norm_out, "mlp_norm", "mlp_norm_out")
        ttnn.deallocate(mlp_norm_in)

        # MLP resharding
        mlp_norm_out = ttnn.to_memory_config(mlp_norm_out, **cfg["mlp_reshard"])
        ttnn.synchronize_device(x.device())
        log_tensor(mlp_norm_out, "mlp_reshard", "mlp_norm_out_resharded", {"config": cfg["mlp_reshard"]})

        # MLP forward
        mlp_out = cls.forward_mlp_decode(mlp_norm_out, cfg["mlp"])
        ttnn.synchronize_device(x.device())
        log_tensor(mlp_out, "mlp", "mlp_out")
        ttnn.deallocate(mlp_norm_out)

        # MLP Residual: z = mlp(y) + y
        log_tensor(y, "mlp_residual", "y_before_mlp_residual", {"operation": "z = mlp_out + y"})
        z = mlp_out + y
        ttnn.synchronize_device(x.device())
        log_tensor(z, "mlp_residual", "z_after_mlp_residual", {"operation": "z = mlp_out + y"})
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(y)

        # Log final output
        log_tensor(z, "decoder_block_output", "final_output")

        return z

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
        kv_cache_override: KvCacheConfig | None = None,
    ) -> ModelState:
        return MLA2D.create_state(hf_config, paged_config, mesh_device, ccl, mla_cache, kv_cache_override)

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
