# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from abc import abstractmethod
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
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
from models.tt_transformers.tt.common import PagedAttentionConfig


# DEBUG: Helper for hang debugging
def _debug_print(msg: str, flush: bool = True):
    """Print debug message with timestamp and flush to ensure immediate output."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[DEBUG {timestamp}] {msg}", file=sys.stderr, flush=flush)


def _has_distinct_buffer(a: ttnn.Tensor, b: ttnn.Tensor) -> bool:
    try:
        return a.buffer_address() != b.buffer_address()
    except Exception:
        return a is not b


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
        _debug_print(f"DecoderBlock2DBase.forward_prefill: START (user_id={user_id})")
        # MLA norm
        _debug_print("DecoderBlock2DBase.forward_prefill: DistributedRMSNorm (mla_norm) START")
        mla_norm_out = DistributedRMSNorm.forward_prefill(x, cfg["mla_norm"])
        _debug_print("DecoderBlock2DBase.forward_prefill: DistributedRMSNorm (mla_norm) DONE")

        # MLA
        _debug_print("DecoderBlock2DBase.forward_prefill: MLA2D.forward_prefill START")
        mla_out = MLA2D.forward_prefill(mla_norm_out, user_id, cfg["mla"], rope_tensors, page_table)
        _debug_print("DecoderBlock2DBase.forward_prefill: MLA2D.forward_prefill DONE")
        _debug_print("DecoderBlock2DBase.forward_prefill: deallocate mla_norm_out START")
        ttnn.deallocate(mla_norm_out)
        _debug_print("DecoderBlock2DBase.forward_prefill: deallocate mla_norm_out DONE")

        # MLA Residual
        _debug_print("DecoderBlock2DBase.forward_prefill: MLA residual add START")
        x += mla_out
        _debug_print("DecoderBlock2DBase.forward_prefill: MLA residual add DONE")
        _debug_print("DecoderBlock2DBase.forward_prefill: deallocate mla_out START")
        ttnn.deallocate(mla_out)
        _debug_print("DecoderBlock2DBase.forward_prefill: deallocate mla_out DONE")

        # MLP norm
        _debug_print("DecoderBlock2DBase.forward_prefill: DistributedRMSNorm (mlp_norm) START")
        mlp_norm_out = DistributedRMSNorm.forward_prefill(x, cfg["mlp_norm"])
        _debug_print("DecoderBlock2DBase.forward_prefill: DistributedRMSNorm (mlp_norm) DONE")

        # MLP
        _debug_print("DecoderBlock2DBase.forward_prefill: forward_mlp_prefill START")
        mlp_out = cls.forward_mlp_prefill(mlp_norm_out, cfg["mlp"])
        _debug_print("DecoderBlock2DBase.forward_prefill: forward_mlp_prefill DONE")
        _debug_print("DecoderBlock2DBase.forward_prefill: deallocate mlp_norm_out START")
        ttnn.deallocate(mlp_norm_out)
        _debug_print("DecoderBlock2DBase.forward_prefill: deallocate mlp_norm_out DONE")

        # MLP Residual
        _debug_print("DecoderBlock2DBase.forward_prefill: MLP residual add START")
        x += mlp_out
        _debug_print("DecoderBlock2DBase.forward_prefill: MLP residual add DONE")
        _debug_print("DecoderBlock2DBase.forward_prefill: deallocate mlp_out START")
        ttnn.deallocate(mlp_out)
        _debug_print("DecoderBlock2DBase.forward_prefill: deallocate mlp_out DONE")

        _debug_print("DecoderBlock2DBase.forward_prefill: END")
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
        _debug_print("DecoderBlock2DBase.forward_decode: START")
        # MLA norm
        _debug_print("DecoderBlock2DBase.forward_decode: to_memory_config (mla_norm_reshard) START")
        mla_norm_in = ttnn.to_memory_config(x, **cfg["mla_norm_reshard"])
        _debug_print("DecoderBlock2DBase.forward_decode: to_memory_config (mla_norm_reshard) DONE")
        _debug_print("DecoderBlock2DBase.forward_decode: DistributedRMSNorm (mla_norm) START")
        mla_norm_out = DistributedRMSNorm.forward_decode(mla_norm_in, cfg["mla_norm"])
        _debug_print("DecoderBlock2DBase.forward_decode: DistributedRMSNorm (mla_norm) DONE")
        if _has_distinct_buffer(mla_norm_in, x):
            _debug_print("DecoderBlock2DBase.forward_decode: deallocate mla_norm_in START")
            ttnn.deallocate(mla_norm_in)
            _debug_print("DecoderBlock2DBase.forward_decode: deallocate mla_norm_in DONE")

        # MLA
        _debug_print("DecoderBlock2DBase.forward_decode: create_sharded_memory_config (mla_reshard) START")
        mla_reshard_memory_config = ttnn.create_sharded_memory_config(
            mla_norm_out.shape,
            **cfg["mla_reshard"],
        )
        _debug_print("DecoderBlock2DBase.forward_decode: create_sharded_memory_config (mla_reshard) DONE")
        _debug_print("DecoderBlock2DBase.forward_decode: to_memory_config (mla_reshard) START")
        mla_norm_out = ttnn.to_memory_config(mla_norm_out, memory_config=mla_reshard_memory_config)
        _debug_print("DecoderBlock2DBase.forward_decode: to_memory_config (mla_reshard) DONE")
        _debug_print("DecoderBlock2DBase.forward_decode: MLA2D.forward_decode START")
        mla_out = MLA2D.forward_decode(mla_norm_out, position_idxs, cfg["mla"], rope_tensors, page_table)
        _debug_print("DecoderBlock2DBase.forward_decode: MLA2D.forward_decode DONE")
        _debug_print("DecoderBlock2DBase.forward_decode: deallocate mla_norm_out START")
        ttnn.deallocate(mla_norm_out)
        _debug_print("DecoderBlock2DBase.forward_decode: deallocate mla_norm_out DONE")

        # MLA Residual
        _debug_print("DecoderBlock2DBase.forward_decode: MLA residual add START")
        x += mla_out
        _debug_print("DecoderBlock2DBase.forward_decode: MLA residual add DONE")
        _debug_print("DecoderBlock2DBase.forward_decode: deallocate mla_out START")
        ttnn.deallocate(mla_out)
        _debug_print("DecoderBlock2DBase.forward_decode: deallocate mla_out DONE")

        # MLP norm
        _debug_print("DecoderBlock2DBase.forward_decode: to_memory_config (mlp_norm_reshard) START")
        mlp_norm_in = ttnn.to_memory_config(x, **cfg["mlp_norm_reshard"])
        _debug_print("DecoderBlock2DBase.forward_decode: to_memory_config (mlp_norm_reshard) DONE")
        _debug_print("DecoderBlock2DBase.forward_decode: DistributedRMSNorm (mlp_norm) START")
        mlp_norm_out = DistributedRMSNorm.forward_decode(mlp_norm_in, cfg["mlp_norm"])
        _debug_print("DecoderBlock2DBase.forward_decode: DistributedRMSNorm (mlp_norm) DONE")
        if _has_distinct_buffer(mlp_norm_in, x):
            _debug_print("DecoderBlock2DBase.forward_decode: deallocate mlp_norm_in START")
            ttnn.deallocate(mlp_norm_in)
            _debug_print("DecoderBlock2DBase.forward_decode: deallocate mlp_norm_in DONE")

        # MLP
        _debug_print("DecoderBlock2DBase.forward_decode: to_memory_config (mlp_reshard) START")
        mlp_norm_out = ttnn.to_memory_config(mlp_norm_out, **cfg["mlp_reshard"])
        _debug_print("DecoderBlock2DBase.forward_decode: to_memory_config (mlp_reshard) DONE")
        _debug_print("DecoderBlock2DBase.forward_decode: forward_mlp_decode START")
        mlp_out = cls.forward_mlp_decode(mlp_norm_out, cfg["mlp"])
        _debug_print("DecoderBlock2DBase.forward_decode: forward_mlp_decode DONE")
        _debug_print("DecoderBlock2DBase.forward_decode: deallocate mlp_norm_out START")
        ttnn.deallocate(mlp_norm_out)
        _debug_print("DecoderBlock2DBase.forward_decode: deallocate mlp_norm_out DONE")

        # MLP Residual
        _debug_print("DecoderBlock2DBase.forward_decode: MLP residual add START")
        x += mlp_out
        _debug_print("DecoderBlock2DBase.forward_decode: MLP residual add DONE")
        _debug_print("DecoderBlock2DBase.forward_decode: deallocate mlp_out START")
        ttnn.deallocate(mlp_out)
        _debug_print("DecoderBlock2DBase.forward_decode: deallocate mlp_out DONE")

        _debug_print("DecoderBlock2DBase.forward_decode: END")
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
