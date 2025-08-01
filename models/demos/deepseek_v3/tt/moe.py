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
    MeshDeviceStub,
    MulConfig,
    ReduceScatterAsyncConfig,
    RepeatConfig,
    ReshapeConfig,
)
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class MoE(AbstractModule):
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

        weight_config["moe_gate"] = MoEGate.convert_weights(hf_config, state_dict, output_path, mesh_device, "gate.")
        weight_config["moe_experts"] = MoEExperts.convert_weights(hf_config, state_dict, output_path, mesh_device)

        return weight_config

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        ccl: CCL1D,
        mode: str,
        batch_size: int,
        seq_len: int,
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

        mesh_shape = list(mesh_device.shape)

        num_devices = mesh_device.get_num_devices()
        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)
        num_dispatch_devices = tuple(mesh_device.shape)[0]

        batch_size_per_device = batch_size // num_dispatch_devices

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
        all_to_all_dispatch_output_tensors = ttnn.from_torch(
            torch.zeros([1, batch_size, seq_len, hf_config.hidden_size]),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=memory_config,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        all_to_all_dispatch_metadata_tensors = ttnn.from_torch(
            torch.zeros([1, batch_size, seq_len, hf_config.num_experts_per_tok], dtype=torch.int32),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        all_to_all_combine_output_tensors = ttnn.from_torch(
            torch.zeros([hf_config.num_experts_per_tok, batch_size_per_device, seq_len, hf_config.hidden_size]),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=memory_config,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # Construct the config
        return {
            "moe_gate": MoEGate.model_config(hf_config, mesh_device, mode),
            "input_reshape": ReshapeConfig(shape=(batch_size_per_device, 1, seq_len, hf_config.hidden_size)),
            "topk_indices_reshape": ReshapeConfig(
                shape=(batch_size_per_device, 1, seq_len, hf_config.num_experts_per_tok)
            ),
            "all_to_all_dispatch": AllToAllDispatchConfig(
                cluster_axis=0, num_links=1, memory_config=memory_config, global_semaphore=ccl.get_semaphore(0)
            ),
            "expert_mapping_tensors": expert_mapping_tensors,
            "all_to_all_dispatch_output_tensors": all_to_all_dispatch_output_tensors,
            "all_to_all_dispatch_metadata_tensors": all_to_all_dispatch_metadata_tensors,
            "all_to_all_output_reshape": ReshapeConfig(shape=(1, 1, batch_size * seq_len, hf_config.hidden_size)),
            "activations_repeat": RepeatConfig(repeat_dims=ttnn.Shape((1, num_experts_per_device, 1, 1))),
            "moe_experts": MoEExperts._create_model_config(hf_config, mesh_device, mode),
            "experts_output_reshape": ReshapeConfig(
                shape=(num_experts_per_device, batch_size, seq_len, hf_config.hidden_size)
            ),
            "all_to_all_combine": AllToAllCombineConfig(
                axis=0, num_links=1, memory_config=memory_config, global_semaphore=ccl.get_semaphore(0)
            ),
            "all_to_all_combine_output_tensors": all_to_all_combine_output_tensors,
            "combine_output_reshape": ReshapeConfig(
                shape=(hf_config.num_experts_per_tok, 1, batch_size_per_device * seq_len, hf_config.hidden_size)
            ),
            "topk_weights_repeat": RepeatConfig(repeat_dims=ttnn.Shape((hf_config.hidden_size, 1, 1, 1))),
            "mul_experts_output_with_weights": MulConfig(memory_config=memory_config),
            "final_output_reduce_scatter": ReduceScatterAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_shape),
                cluster_axis=1,
                dim=3,
                from_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
                to_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
                math_op=ttnn.ReduceType.Sum,
                num_links=ccl.get_max_links(1),
                memory_config=memory_config,
                topology=ttnn.Topology.Linear,
            ),
            "input_memory_config": memory_config,
            "output_memory_config": memory_config,
        }

    @classmethod
    def decode_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL1D, batch_size: int
    ) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, ccl, "decode", batch_size, seq_len=1)

    @classmethod
    def prefill_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL1D, batch_size: int, seq_len: int
    ) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, ccl, "prefill", batch_size, seq_len)

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        assert x.memory_config() == cfg["input_memory_config"]

        topk_experts_weights, topk_experts_indices = MoEGate.forward(x, cfg["moe_gate"])
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, **cfg["input_reshape"])

        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_indices_rm = ttnn.reshape(topk_experts_indices_rm, **cfg["topk_indices_reshape"])
        ttnn.all_to_all_dispatch(
            x_rm,
            topk_experts_indices_rm,
            cfg["expert_mapping_tensors"],
            output_tensors=[cfg["all_to_all_dispatch_output_tensors"], cfg["all_to_all_dispatch_metadata_tensors"]],
            **cfg["all_to_all_dispatch"],
        )
        post_all_to_all_dispatch_output = ttnn.reshape(
            cfg["all_to_all_dispatch_output_tensors"], **cfg["all_to_all_output_reshape"]
        )
        post_all_to_all_dispatch_output = ttnn.repeat(post_all_to_all_dispatch_output, **cfg["activations_repeat"])
        post_all_to_all_dispatch_output = ttnn.to_layout(post_all_to_all_dispatch_output, ttnn.TILE_LAYOUT)
        experts_output = MoEExperts._forward(post_all_to_all_dispatch_output, cfg["moe_experts"])
        ttnn.deallocate(post_all_to_all_dispatch_output)
        experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
        experts_output = ttnn.reshape(experts_output, **cfg["experts_output_reshape"])
        ttnn.all_to_all_combine(
            experts_output,
            cfg["expert_mapping_tensors"],
            cfg["all_to_all_dispatch_metadata_tensors"],
            optional_output_tensor=cfg["all_to_all_combine_output_tensors"],
            **cfg["all_to_all_combine"],
        )
        post_combine_output_tensor = ttnn.reshape(
            cfg["all_to_all_combine_output_tensors"], **cfg["combine_output_reshape"]
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
