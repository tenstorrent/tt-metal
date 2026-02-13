# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

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
            # Single all_gather using SharedExpert's all_gather config for both modules
            ccl_shared = cfg["shared_expert"]["ccl"]
            x_gathered = ttnn.experimental.all_gather_async(
                x, **ccl_shared.populate_all_gather_runtime_args(cfg["shared_expert"]["all_gather"])
            )
        else:
            # Already full hidden size
            x_gathered = x

        # Run both MoE and SharedExpert with the same gathered input
        mlp_out = MoE.forward_prefill(x_gathered, cfg["moe"])
        mlp_out = ttnn.sum(mlp_out, dim=0, keepdim=True, memory_config=cfg["moe"]["sum_experts_output_memory_config"])

        # SharedExpert now always expects collective ops to be handled by caller
        shared_expert_out = SharedExpert.forward_prefill(x_gathered, cfg["shared_expert"])

        # Add outputs first, then reduce_scatter the combined result
        combined_out = ttnn.add(mlp_out, shared_expert_out)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(shared_expert_out)

        # Handle reduce_scatter if input was TP-sharded
        if x_dim == hidden_size // tp_size:
            # Single reduce_scatter on combined output using shared_expert's config
            ccl_shared = cfg["shared_expert"]["ccl"]
            output = ttnn.experimental.reduce_scatter_minimal_async(
                combined_out,
                **ccl_shared.populate_reduce_scatter_runtime_args(cfg["shared_expert"]["reduce_scatter_async"]),
            )
            ttnn.deallocate(combined_out)
            # Cleanup gathered tensor
            if x_gathered is not x:
                ttnn.deallocate(x_gathered)
        else:
            # If not TP-sharded, combined output is the final output
            output = combined_out

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
            # Single all_gather using SharedExpert's all_gather config for both modules
            ccl_shared = cfg["shared_expert"]["ccl"]
            x_gathered = ttnn.experimental.all_gather_async(
                x, **ccl_shared.populate_all_gather_runtime_args(cfg["shared_expert"]["all_gather"])
            )
        else:
            # Already full hidden size
            x_gathered = x

        # Run both MoE and SharedExpert with the same gathered input
        mlp_out = MoE.forward_decode(x_gathered, cfg["moe"])

        # SharedExpert now always expects collective ops to be handled by caller
        shared_expert_out = SharedExpert.forward_decode(x_gathered, cfg["shared_expert"])

        # We sum the experts from MoE along with SharedExpert inside a single reduce by concatting first, instead
        # of a reduce on the MoE experts, and then a following add with the SharedExpert. This enables us to use
        # the optimized reduce_scatter.
        combined_out = ttnn.concat([mlp_out, shared_expert_out], dim=0)
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(shared_expert_out)

        # Handle reduce_scatter if input was TP-sharded
        if x_dim == hidden_size // tp_size:
            # Add outputs first, then reduce_scatter the combined result
            # Use the optimized version if ring fabric is available

            if cfg["moe"]["optimized_final_output_reduce_scatter"].topology == ttnn.Topology.Ring:
                combined_out = ttnn.experimental.deepseek_moe_fast_reduce_nc(
                    combined_out,
                    dim=0,
                    split_size=combined_out[-1] // tp_size,
                    output_memory_config=cfg["moe"]["optimized_sum_experts_output_memory_config"],
                )

                # Single reduce_scatter on combined output
                output = ttnn.experimental.deepseek_moe_reduce_scatter(
                    combined_out, **cfg["moe"]["optimized_final_output_reduce_scatter"]
                )
            else:
                combined_out = ttnn.sum(
                    combined_out, dim=0, keepdim=True, memory_config=cfg["sum_experts_output_memory_config"]
                )

                # Single reduce_scatter on combined output
                output = ttnn.experimental.reduce_scatter_minimal_async(
                    combined_out,
                    **ccl_shared.populate_reduce_scatter_runtime_args(cfg["moe"]["final_output_reduce_scatter"]),
                )

            ttnn.deallocate(combined_out)
            # Cleanup gathered tensor
            if x_gathered is not x:
                ttnn.deallocate(x_gathered)
        else:
            # If not TP-sharded, combined output is the final output
            output = ttnn.sum(combined_out, dim=0, keepdim=True, memory_config=cfg["sum_experts_output_memory_config"])

        return output
