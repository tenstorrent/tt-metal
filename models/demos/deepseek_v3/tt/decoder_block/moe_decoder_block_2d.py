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
    def forward_mlp_prefill(
        cls, x: ttnn.Tensor, cfg: RunPrefillConfig, is_mlp_tensor_parallel: bool = True
    ) -> ttnn.Tensor:
        if not is_mlp_tensor_parallel:
            # When not tensor parallel at MLP level, handle all_gather and reduce_scatter here
            # All gather at the beginning
            x = ttnn.experimental.all_gather_async(
                x, **cfg["moe"]["ccl"].populate_all_gather_runtime_args(cfg["moe"]["revert_tp"])
            )

            # Call MoE and SharedExpert without tensor parallelism
            mlp_out = MoE.forward_prefill(x, cfg["moe"], is_tensor_parallel=False)
            mlp_out += SharedExpert.forward_prefill(x, cfg["shared_expert"], is_tensor_parallel=False)

            # Reduce scatter at the end
            ccl = cfg["moe"]["ccl"]
            mlp_out = ttnn.experimental.reduce_scatter_minimal_async(
                mlp_out, **ccl.populate_reduce_scatter_runtime_args(cfg["moe"]["final_output_reduce_scatter"])
            )
        else:
            # Default behavior with tensor parallelism handled inside each component
            mlp_out = MoE.forward_prefill(x, cfg["moe"])
            mlp_out += SharedExpert.forward_prefill(x, cfg["shared_expert"])
        return mlp_out

    @classmethod
    @abstractmethod
    def forward_mlp_decode(
        cls, x: ttnn.Tensor, cfg: RunDecodeConfig, is_mlp_tensor_parallel: bool = True
    ) -> ttnn.Tensor:
        if not is_mlp_tensor_parallel:
            # When not tensor parallel at MLP level, handle all_gather and reduce_scatter here
            # All gather at the beginning
            x = ttnn.experimental.all_gather_async(
                x, **cfg["moe"]["ccl"].populate_all_gather_runtime_args(cfg["moe"]["revert_tp"])
            )

            # Call MoE and SharedExpert without tensor parallelism
            mlp_out = MoE.forward_decode(x, cfg["moe"], is_tensor_parallel=False)
            mlp_out += SharedExpert.forward_decode(x, cfg["shared_expert"], is_tensor_parallel=False)

            # Reduce scatter at the end
            ccl = cfg["moe"]["ccl"]
            mlp_out = ttnn.experimental.reduce_scatter_minimal_async(
                mlp_out, **ccl.populate_reduce_scatter_runtime_args(cfg["moe"]["final_output_reduce_scatter"])
            )
        else:
            # Default behavior with tensor parallelism handled inside each component
            mlp_out = MoE.forward_decode(x, cfg["moe"])
            mlp_out += SharedExpert.forward_decode(x, cfg["shared_expert"])
        return mlp_out
