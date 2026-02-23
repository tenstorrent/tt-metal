# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d_base import DecoderBlock2DBase
from models.demos.deepseek_v3.tt.mlp.non_expert import NonExpert
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class DecoderBlock2D(DecoderBlock2DBase):
    @classmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        return NonExpert.convert_weights(hf_config, (state_dict,) * mesh_device.shape[0], output_path, mesh_device)

    @classmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        return NonExpert.prefill_model_config(hf_config, mesh_device)

    @classmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelDecodeConfig:
        return NonExpert.decode_model_config(hf_config, mesh_device)

    @classmethod
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
    ) -> ModelState:
        return NonExpert.create_state(hf_config, mesh_device, ccl)

    @classmethod
    def create_mlp_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        return {}

    @classmethod
    def _all_gather_helper(
        cls,
        x: ttnn.Tensor,
        cfg: RunPrefillConfig | RunDecodeConfig,
        all_gather_config_key: str = "all_gather",
    ) -> ttnn.Tensor | None:
        """
        Helper function to perform all_gather operation if needed.

        Returns:
            Gathered tensor if all_gather was performed, None otherwise.
        """
        ccl = cfg.get("ccl")
        if ccl is None:
            return None

        return ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg[all_gather_config_key]))

    @classmethod
    def _reduce_scatter_helper(
        cls,
        output: ttnn.Tensor,
        cfg: RunPrefillConfig | RunDecodeConfig,
        reduce_scatter_config_key: str,
    ) -> ttnn.Tensor:
        """
        Helper function to perform reduce_scatter operation.

        Returns:
            Scattered tensor after reduce_scatter.
        """
        ccl = cfg.get("ccl")
        return ttnn.experimental.reduce_scatter_minimal_async(
            output, **ccl.populate_reduce_scatter_runtime_args(cfg[reduce_scatter_config_key])
        )

    @classmethod
    def _forward_mlp_common(
        cls,
        x: ttnn.Tensor,
        cfg: RunPrefillConfig | RunDecodeConfig,
        forward_fn,
        hidden_size: int,
        tp_size: int,
        reduce_scatter_key: str,
        mode_name: str,
    ) -> ttnn.Tensor:
        """
        Common implementation for forward_mlp_prefill and forward_mlp_decode.

        Args:
            x: Input tensor
            cfg: Configuration
            forward_fn: The forward function to call (NonExpert.forward_prefill or NonExpert.forward_decode)
            hidden_size: The full hidden size
            tp_size: Tensor parallel size
            reduce_scatter_key: Config key for reduce_scatter ("reduce_scatter_async" or "reduce_scatter")
            mode_name: Name of the mode for logging ("prefill" or "decode")
        """
        x_dim = x.shape[-1]

        # Check if input is TP-sharded
        if x_dim == hidden_size // tp_size:
            # Input is TP-sharded, need to gather
            x_gathered = cls._all_gather_helper(x, cfg)
            if x_gathered is None:
                # If no CCL object, can't do collective ops
                return forward_fn(x, cfg)

            # Run NonExpert with gathered input
            output = forward_fn(x_gathered, cfg)

            # Perform reduce_scatter
            output_scattered = cls._reduce_scatter_helper(output, cfg, reduce_scatter_key)

            # Cleanup
            ttnn.deallocate(output)
            if x_gathered is not x:
                ttnn.deallocate(x_gathered)

            return output_scattered
        elif x_dim == hidden_size:
            # Already full hidden size, no collective ops needed
            return forward_fn(x, cfg)
        else:
            # Unexpected dimension
            from loguru import logger

            logger.warning(
                f"DecoderBlock2D forward_mlp_{mode_name}: Unexpected input dimension {x_dim}. "
                f"Expected either {hidden_size} (full) or {hidden_size // tp_size} (TP-sharded). "
                f"Proceeding without collective operations."
            )
            return forward_fn(x, cfg)

    @classmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        # Get dimensions for tensor parallel check
        mesh_device = cfg.get(MESH_DEVICE_STATE_DICT_KEY)
        if mesh_device is None:
            # If no mesh device, no tensor parallelism
            return NonExpert.forward_prefill(x, cfg)

        tp_size = mesh_device.shape[1]  # Width of mesh is TP size
        if tp_size == 1:
            # No tensor parallelism
            return NonExpert.forward_prefill(x, cfg)

        # Calculate expected dimensions
        # NonExpert/MLP expects full hidden_size, not the sharded size
        # We can infer from the linear_pc_gen configuration
        hidden_size = cfg["linear_pc_gen"].dim

        return cls._forward_mlp_common(
            x,
            cfg,
            NonExpert.forward_prefill,
            hidden_size,
            tp_size,
            "reduce_scatter_async",
            "prefill",
        )

    @classmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        # Get dimensions for tensor parallel check
        mesh_device = cfg.get(MESH_DEVICE_STATE_DICT_KEY)
        if mesh_device is None:
            # If no mesh device, no tensor parallelism
            return NonExpert.forward_decode(x, cfg)

        tp_size = mesh_device.shape[1]  # Width of mesh is TP size
        if tp_size == 1:
            # No tensor parallelism
            return NonExpert.forward_decode(x, cfg)

        # For decode, we need to check the actual hidden size
        # The w1 weight shape tells us the input dimension
        hidden_size = cfg["w1"]["weight"].shape[-1] * tp_size  # Input features to w1

        return cls._forward_mlp_common(
            x,
            cfg,
            NonExpert.forward_decode,
            hidden_size,
            tp_size,
            "reduce_scatter",
            "decode",
        )
