# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from abc import abstractmethod
from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d_base import DecoderBlock2DBase
from models.demos.deepseek_v3.tt.mlp.shared_expert import SharedExpert
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)

fabric_config = (
    ttnn.FabricConfig.FABRIC_1D_RING if (os.getenv("USE_TORUS_MODE") is not None) else ttnn.FabricConfig.FABRIC_1D
)


class MoEDecoderBlock2D(DecoderBlock2DBase):
    @classmethod
    @abstractmethod
    def convert_mlp_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.MeshDevice,
    ) -> WeightConfig:
        return {
            "shared_expert": SharedExpert.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "shared_experts."),) * mesh_device.shape[0],
                output_path / "shared_experts",
                mesh_device,
            ),
            "moe": MoE.convert_weights(hf_config, (state_dict,), output_path / "moe", mesh_device),
        }

    @classmethod
    @abstractmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelPrefillConfig:
        return {
            "shared_expert": SharedExpert.prefill_model_config(hf_config, mesh_device),
            "moe": MoE.prefill_model_config(hf_config, mesh_device),
        }

    @classmethod
    @abstractmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelDecodeConfig:
        return {
            "shared_expert": SharedExpert.decode_model_config(hf_config, mesh_device),
            "moe": MoE.decode_model_config(hf_config, mesh_device),
        }

    @classmethod
    @abstractmethod
    def create_mlp_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
    ) -> ModelState:
        return {
            "shared_expert": SharedExpert.create_state(hf_config, mesh_device, ccl),
            "moe": MoE.create_state(hf_config, mesh_device, ccl),
        }

    @classmethod
    @abstractmethod
    def create_mlp_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
    ) -> ModelState:
        return {
            "shared_expert": {},
            "moe": MoE.create_shared_state(hf_config, mesh_device),
        }

    @classmethod
    @abstractmethod
    def forward_mlp_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        # Handle all_gather if input is TP-sharded
        hidden_size = cfg["moe"]["hidden_size"]
        tp_size = cfg["moe"]["mesh_device"].shape[1]
        x_dim = x.shape[-1]

        if x_dim == hidden_size // tp_size:
            # Input is TP-sharded, need to gather
            # Use MoE's all_gather config which outputs the correct memory layout for MoEGate
            ccl_moe = cfg["moe"]["ccl"]
            x_gathered = ttnn.experimental.all_gather_async(
                x, **ccl_moe.populate_all_gather_runtime_args(cfg["moe"]["revert_tp"])
            )
        else:
            # Should always be TP-sharded at this point
            assert False, f"Expected TP-sharded input with dim {hidden_size // tp_size}, got dim {x_dim}"

        # Run both MoE and SharedExpert with the same gathered input
        mlp_out = MoE.forward_prefill(x_gathered, cfg["moe"])
        # SharedExpert now always expects collective ops to be handled by caller
        shared_expert_out = SharedExpert.forward_prefill(x_gathered, cfg["shared_expert"])

        mlp_out = ttnn.sum(combined_out, dim=0, keepdim=True)
        combined_out = ttnn.add(mlp_out, shared_expert_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(shared_expert_out)

        # Handle reduce_scatter if input was TP-sharded
        if x_dim == hidden_size // tp_size:
            # Single reduce_scatter on combined output using MoE's config for consistency
            ccl_moe = cfg["moe"]["ccl"]
            output = ttnn.experimental.reduce_scatter_minimal_async(
                combined_out,
                **ccl_moe.populate_reduce_scatter_runtime_args(cfg["moe"]["final_output_reduce_scatter"]),
            )
            ttnn.deallocate(combined_out)
            # Cleanup gathered tensor
            if x_gathered is not x:
                ttnn.deallocate(x_gathered)
        else:
            # Should always be TP-sharded at this point
            assert False, f"Expected TP-sharded input with dim {hidden_size // tp_size}, got dim {x_dim}"

        return output

    @classmethod
    @abstractmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        # Handle all_gather if input is TP-sharded
        hidden_size = cfg["moe"]["hidden_size"]
        tp_size = cfg["moe"]["mesh_device"].shape[1]
        x_dim = x.shape[-1]

        if x_dim == hidden_size // tp_size:
            # Input is TP-sharded, need to gather
            # Use MoE's all_gather config which outputs the correct memory layout for MoEGate
            ccl_moe = cfg["moe"]["ccl"]
            x_gathered = ttnn.experimental.all_gather_async(
                x, **ccl_moe.populate_all_gather_runtime_args(cfg["moe"]["revert_tp"])
            )
        else:
            # Should always be TP-sharded at this point
            assert False, f"Expected TP-sharded input with dim {hidden_size // tp_size}, got dim {x_dim}"

        # Run both MoE and SharedExpert with the same gathered input
        mlp_out = MoE.forward_decode(x_gathered, cfg["moe"])
        # SharedExpert now always expects collective ops to be handled by caller
        shared_expert_out = SharedExpert.forward_decode(x_gathered, cfg["shared_expert"])

        # We sum the experts from MoE along with SharedExpert inside a single reduce by concatting first, instead
        # of a reduce on the MoE experts followed by an add with the SharedExpert. This enables us to use
        # the optimized reduce_scatter.
        shared_expert_out = ttnn.to_memory_config(shared_expert_out, ttnn.L1_MEMORY_CONFIG)
        combined_out = ttnn.concat([mlp_out, shared_expert_out], dim=0)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(shared_expert_out)

        # Handle summing experts
        # Handle reduce_scatter if input was TP-sharded
        if x_dim == hidden_size // tp_size:
            # Single reduce_scatter on combined output using MoE's config for consistency
            ccl_moe = cfg["moe"]["ccl"]

            if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING:
                summed_experts = ttnn.experimental.deepseek_moe_fast_reduce_nc(
                    combined_out,
                    dim=0,
                    split_size=int(combined_out.shape[-1] // tp_size),
                    output_memory_config=cfg["moe"]["ring_sum_experts_output_memory_config"],
                )
                ttnn.deallocate(combined_out)
                output = ttnn.experimental.deepseek_moe_reduce_scatter(
                    summed_experts, **cfg["moe"]["ring_final_output_reduce_scatter"]
                )
            else:
                summed_experts = ttnn.sum(
                    combined_out, dim=0, keepdim=True, memory_config=cfg["moe"]["sum_experts_output_memory_config"]
                )
                ttnn.deallocate(combined_out)
                output = ttnn.experimental.reduce_scatter_minimal_async(
                    summed_experts,
                    **ccl_moe.populate_reduce_scatter_runtime_args(cfg["moe"]["final_output_reduce_scatter"]),
                )

            # Cleanup gathered tensor
            if x_gathered is not x:
                ttnn.deallocate(x_gathered)
        else:
            # Should always be TP-sharded at this point
            assert False, f"Expected TP-sharded input with dim {hidden_size // tp_size}, got dim {x_dim}"

        return output
