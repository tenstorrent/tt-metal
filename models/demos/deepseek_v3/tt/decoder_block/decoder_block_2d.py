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
        # We need to infer hidden_size from the configuration
        # The w1/w3 weights go from dim -> hidden_dim, w2 goes from hidden_dim -> dim
        # After TP sharding, w1/w3 output hidden_dim/tp_size per device
        # We can infer from the linear_pc_gen configuration
        hidden_size = cfg["linear_pc_gen"].dim
        x_dim = x.shape[-1]

        # Check if input is TP-sharded
        if x_dim == hidden_size // tp_size:
            # Input is TP-sharded, need to gather
            ccl = cfg.get("ccl")
            if ccl is None:
                # If no CCL object, can't do collective ops
                return NonExpert.forward_prefill(x, cfg)

            # Perform all_gather
            x_gathered = ttnn.experimental.all_gather_async(
                x, **ccl.populate_all_gather_runtime_args(cfg["all_gather"])
            )

            # Run NonExpert with gathered input
            output = NonExpert.forward_prefill(x_gathered, cfg)

            # Perform reduce_scatter
            output_scattered = ttnn.experimental.reduce_scatter_minimal_async(
                output, **ccl.populate_reduce_scatter_runtime_args(cfg["reduce_scatter_async"])
            )

            # Cleanup
            ttnn.deallocate(output)
            if x_gathered is not x:
                ttnn.deallocate(x_gathered)

            return output_scattered
        else:
            # Already full hidden size or different configuration
            return NonExpert.forward_prefill(x, cfg)

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
        # In decode mode, the configuration might be different
        # We can use the weight dimensions to infer this
        # The w1 weight shape tells us the input dimension
        hidden_size = cfg["w1"]["weight"].shape[-1] * tp_size  # Input features to w1
        x_dim = x.shape[-1]

        # Check if input is TP-sharded
        if x_dim == hidden_size // tp_size:
            # Input is TP-sharded, need to gather
            ccl = cfg.get("ccl")
            if ccl is None:
                # If no CCL object, can't do collective ops
                return NonExpert.forward_decode(x, cfg)

            # Perform all_gather
            x_gathered = ttnn.experimental.all_gather_async(
                x, **ccl.populate_all_gather_runtime_args(cfg["all_gather"])
            )

            # Run NonExpert with gathered input
            output = NonExpert.forward_decode(x_gathered, cfg)

            # Perform reduce_scatter
            output_scattered = ttnn.experimental.reduce_scatter_minimal_async(
                output, **ccl.populate_reduce_scatter_runtime_args(cfg["reduce_scatter"])
            )

            # Cleanup
            ttnn.deallocate(output)
            if x_gathered is not x:
                ttnn.deallocate(x_gathered)

            return output_scattered
        else:
            # Already full hidden size or different configuration
            return NonExpert.forward_decode(x, cfg)
