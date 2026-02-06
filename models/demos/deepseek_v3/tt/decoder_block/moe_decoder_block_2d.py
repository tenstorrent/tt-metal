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
        is_mlp_tensor_parallel: bool = True,
    ) -> WeightConfig:
        return {
            "shared_expert": SharedExpert.convert_weights(
                hf_config,
                (sub_state_dict(state_dict, "shared_experts."),) * mesh_device.shape[0],
                output_path / "shared_experts",
                mesh_device,
                is_mlp_tensor_parallel,
            ),
            "moe": MoE.convert_weights(hf_config, (state_dict,), output_path / "moe", mesh_device),
        }

    @classmethod
    @abstractmethod
    def prefill_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_mlp_tensor_parallel: bool = True,
    ) -> ModelPrefillConfig:
        return {
            "shared_expert": SharedExpert.prefill_model_config(
                hf_config, mesh_device, is_mlp_tensor_parallel=is_mlp_tensor_parallel
            ),
            "moe": MoE.prefill_model_config(hf_config, mesh_device, is_mlp_tensor_parallel=is_mlp_tensor_parallel),
        }

    @classmethod
    @abstractmethod
    def decode_mlp_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        is_mlp_tensor_parallel: bool = True,
    ) -> ModelDecodeConfig:
        return {
            "shared_expert": SharedExpert.decode_model_config(
                hf_config, mesh_device, is_mlp_tensor_parallel=is_mlp_tensor_parallel
            ),
            "moe": MoE.decode_model_config(hf_config, mesh_device, is_mlp_tensor_parallel=is_mlp_tensor_parallel),
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
        # Check if tensor parallelism should be handled at module level (default) or decoder block level
        is_mlp_tensor_parallel = cfg["shared_expert"].get("is_mlp_tensor_parallel", True)

        if not is_mlp_tensor_parallel:
            # When is_mlp_tensor_parallel=False, handle AG/RS at decoder block level
            # Perform All-Gather at decoder block level
            ccl = cfg["shared_expert"]["ccl"]
            x = ttnn.experimental.all_gather_async(
                x, **ccl.populate_all_gather_runtime_args(cfg["shared_expert"]["all_gather"])
            )

            # After All-Gather, ensure tensor has the memory configuration expected by MoE gate
            if "moe_gate" in cfg["moe"] and "input_memory_config" in cfg["moe"]["moe_gate"]:
                x = ttnn.to_memory_config(x, cfg["moe"]["moe_gate"]["input_memory_config"])

            # Forward through MoE and SharedExpert - pass is_mlp_tensor_parallel=False directly
            mlp_out = MoE.forward_prefill(x, cfg["moe"], is_mlp_tensor_parallel=False)
            mlp_out += SharedExpert.forward_prefill(x, cfg["shared_expert"], is_mlp_tensor_parallel=False)

            # Perform Reduce-Scatter at decoder block level
            mlp_out = ttnn.experimental.reduce_scatter_minimal_async(
                mlp_out, **ccl.populate_reduce_scatter_runtime_args(cfg["shared_expert"]["reduce_scatter_async"])
            )
        else:
            # Default behavior: each module handles its own AG/RS - pass is_mlp_tensor_parallel=True directly
            mlp_out = MoE.forward_prefill(x, cfg["moe"], is_mlp_tensor_parallel=True)
            mlp_out += SharedExpert.forward_prefill(x, cfg["shared_expert"], is_mlp_tensor_parallel=True)

        return mlp_out

    @classmethod
    @abstractmethod
    def forward_mlp_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        # Check if tensor parallelism should be handled at module level (default) or decoder block level
        is_mlp_tensor_parallel = cfg["shared_expert"].get("is_mlp_tensor_parallel", True)

        if not is_mlp_tensor_parallel:
            # When is_mlp_tensor_parallel=False, handle AG/RS at decoder block level
            # Perform All-Gather at decoder block level
            ccl = cfg["shared_expert"]["ccl"]
            x = ttnn.experimental.all_gather_async(
                x, **ccl.populate_all_gather_runtime_args(cfg["shared_expert"]["all_gather"])
            )

            # After All-Gather, ensure tensor has the memory configuration expected by MoE gate
            if "moe_gate" in cfg["moe"] and "input_memory_config" in cfg["moe"]["moe_gate"]:
                x = ttnn.to_memory_config(x, cfg["moe"]["moe_gate"]["input_memory_config"])

            # Forward through MoE and SharedExpert - pass is_mlp_tensor_parallel=False directly
            mlp_out = MoE.forward_decode(x, cfg["moe"], is_mlp_tensor_parallel=False)
            mlp_out += SharedExpert.forward_decode(x, cfg["shared_expert"], is_mlp_tensor_parallel=False)

            # Perform Reduce-Scatter at decoder block level
            mlp_out = ttnn.experimental.reduce_scatter_minimal_async(
                mlp_out, **ccl.populate_reduce_scatter_runtime_args(cfg["shared_expert"]["reduce_scatter_async"])
            )
        else:
            # Default behavior: each module handles its own AG/RS - pass is_mlp_tensor_parallel=True directly
            mlp_out = MoE.forward_decode(x, cfg["moe"], is_mlp_tensor_parallel=True)
            mlp_out += SharedExpert.forward_decode(x, cfg["shared_expert"], is_mlp_tensor_parallel=True)

        return mlp_out
