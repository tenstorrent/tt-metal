# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.experts import Experts as MoEExperts
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllToAllCombineConfig,
    AllToAllDispatchConfig,
    MulConfig,
    ReduceScatterAsyncConfig,
    RepeatConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import even_int_div
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.demos.deepseek_v3.utils.shared_state_addon import SharedStateAddOn


class MoE(SharedStateAddOn, AbstractModule):
    """MoE module from DeepSeek-R1.
    See the `AbstractModule` docstring for usage info.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        weight_config = {}

        # Create subdirectories for each submodule
        moe_gate_path = output_path / "moe_gate"
        moe_experts_path = output_path / "moe_experts"

        weight_config["moe_gate"] = MoEGate.convert_weights(hf_config, state_dict, moe_gate_path, mesh_device, "gate.")
        weight_config["moe_experts"] = MoEExperts.convert_weights(hf_config, state_dict, moe_experts_path, mesh_device)

        return weight_config

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        ccl: CCL1D,
    ) -> ModelState:
        """Create model state containing CCL-related communication configurations.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            ccl: CCL1D instance for communication configuration
        Returns:
            ModelState containing CCL configurations
        """

        num_devices = mesh_device.get_num_devices()
        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)

        expert_mapping_tensors = ttnn.from_torch(
            torch.eye(num_devices, dtype=torch.int32)
            .repeat_interleave(num_experts_per_device, dim=0)
            .unsqueeze(0)
            .unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        return {
            "expert_mapping_tensors": expert_mapping_tensors,
            # CCL-specific parameters (semaphores and num_links)
            "all_to_all_dispatch": {
                "global_semaphore": ccl.get_semaphore(0),
                "num_links": 1,
            },
            "all_to_all_combine": {
                "global_semaphore": ccl.get_semaphore(0),
                "num_links": 1,
            },
            "final_output_reduce_scatter": {
                "from_remote_multi_device_global_semaphore": ccl.get_semaphore(1),
                "to_remote_multi_device_global_semaphore": ccl.get_semaphore(1),
                "num_links": ccl.get_max_links(1),
            },
        }

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        mode: str,
    ) -> ModelDecodeConfig | ModelPrefillConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """

        if mode == "decode":
            memory_config = ttnn.L1_MEMORY_CONFIG
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG

        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)

        # Construct the config
        return {
            "device": mesh_device,
            "num_devices": mesh_device.get_num_devices(),
            "num_experts_per_device": num_experts_per_device,
            "hidden_size": hf_config.hidden_size,
            "num_experts_per_tok": hf_config.num_experts_per_tok,
            "num_dispatch_devices": tuple(mesh_device.shape)[0],
            "moe_gate": MoEGate.model_config(hf_config, mesh_device, mode),
            "all_to_all_dispatch_output_memory_config": memory_config,
            "all_to_all_dispatch_metadata_memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "activations_repeat": RepeatConfig(repeat_dims=ttnn.Shape((1, num_experts_per_device, 1, 1))),
            "moe_experts": MoEExperts._create_model_config(hf_config, mesh_device, mode),
            "all_to_all_combine_output_memory_config": memory_config,
            "topk_weights_repeat": RepeatConfig(repeat_dims=ttnn.Shape((hf_config.hidden_size, 1, 1, 1))),
            "mul_experts_output_with_weights": MulConfig(memory_config=memory_config),
            "input_memory_config": memory_config,
            "output_memory_config": memory_config,
            "all_to_all_dispatch": AllToAllDispatchConfig(cluster_axis=0, memory_config=memory_config),
            "all_to_all_combine": AllToAllCombineConfig(axis=0, memory_config=memory_config),
            "final_output_reduce_scatter": ReduceScatterAsyncConfig(
                mesh_device=mesh_device,
                cluster_axis=1,
                dim=3,
                math_op=ttnn.ReduceType.Sum,
                memory_config=memory_config,
                topology=ttnn.Topology.Linear,
            ),
        }

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, "prefill")

    @classmethod
    def create_runtime_output_buffers(
        cls, mesh_device: ttnn.Device, batch_size: int, seq_len: int, cfg: RunDecodeConfig | RunPrefillConfig
    ) -> dict:
        batch_size_per_device = even_int_div(batch_size, cfg["num_dispatch_devices"])

        all_to_all_dispatch_output_tensors = ttnn.from_torch(
            torch.zeros([1, batch_size, seq_len, cfg["hidden_size"]]),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=cfg["all_to_all_dispatch_output_memory_config"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        all_to_all_dispatch_metadata_tensors = ttnn.from_torch(
            torch.zeros([1, batch_size, seq_len, cfg["num_experts_per_tok"]], dtype=torch.int32),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=cfg["all_to_all_dispatch_metadata_memory_config"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        all_to_all_combine_output_tensors = ttnn.from_torch(
            torch.zeros([cfg["num_experts_per_tok"], batch_size_per_device, seq_len, cfg["hidden_size"]]),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=cfg["all_to_all_combine_output_memory_config"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        return {
            "all_to_all_dispatch_output_tensors": all_to_all_dispatch_output_tensors,
            "all_to_all_dispatch_metadata_tensors": all_to_all_dispatch_metadata_tensors,
            "all_to_all_combine_output_tensors": all_to_all_combine_output_tensors,
        }

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        assert x.memory_config() == cfg["input_memory_config"]
        seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
        batch_size_per_device = x.shape[
            -2
        ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device
        batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size

        output_buffers = cls.create_runtime_output_buffers(cfg["device"], batch_size, seq_len, cfg)

        # 1. MoE gate
        topk_experts_weights, topk_experts_indices = MoEGate.forward(x, cfg["moe_gate"])
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, shape=(batch_size_per_device, 1, seq_len, cfg["hidden_size"]))

        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_indices_rm = ttnn.reshape(
            topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"])
        )
        ttnn.all_to_all_dispatch(
            x_rm,
            topk_experts_indices_rm,
            cfg["expert_mapping_tensors"],
            output_tensors=[
                output_buffers["all_to_all_dispatch_output_tensors"],
                output_buffers["all_to_all_dispatch_metadata_tensors"],
            ],
            **cfg["all_to_all_dispatch"],
        )
        post_all_to_all_dispatch_output = ttnn.reshape(
            output_buffers["all_to_all_dispatch_output_tensors"], shape=(1, 1, batch_size * seq_len, cfg["hidden_size"])
        )
        post_all_to_all_dispatch_output = ttnn.repeat(post_all_to_all_dispatch_output, **cfg["activations_repeat"])
        post_all_to_all_dispatch_output = ttnn.to_layout(post_all_to_all_dispatch_output, ttnn.TILE_LAYOUT)
        experts_output = MoEExperts._forward(post_all_to_all_dispatch_output, cfg["moe_experts"])
        ttnn.deallocate(post_all_to_all_dispatch_output)
        experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
        experts_output = ttnn.reshape(
            experts_output, shape=(cfg["num_experts_per_device"], batch_size, seq_len, cfg["hidden_size"])
        )
        ttnn.all_to_all_combine(
            experts_output,
            cfg["expert_mapping_tensors"],
            output_buffers["all_to_all_dispatch_metadata_tensors"],
            optional_output_tensor=output_buffers["all_to_all_combine_output_tensors"],
            **cfg["all_to_all_combine"],
        )
        post_combine_output_tensor = ttnn.reshape(
            output_buffers["all_to_all_combine_output_tensors"],
            shape=(cfg["num_experts_per_tok"], 1, batch_size_per_device * seq_len, cfg["hidden_size"]),
        )
        post_combine_output_tensor = ttnn.to_layout(post_combine_output_tensor, ttnn.TILE_LAYOUT)
        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, **cfg["topk_weights_repeat"])
        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
        topk_experts_weights = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_experts_weights_rm)
        post_combine_output_tensor = ttnn.mul(
            post_combine_output_tensor, topk_experts_weights, **cfg["mul_experts_output_with_weights"]
        )
        post_combine_output_tensor = ttnn.sum(post_combine_output_tensor, dim=0, keepdim=True)
        post_combine_output_tensor = ttnn.experimental.reduce_scatter_async(
            post_combine_output_tensor, **cfg["final_output_reduce_scatter"]
        )

        return post_combine_output_tensor

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)
