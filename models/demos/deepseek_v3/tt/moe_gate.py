# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    BinaryOpConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    MulConfig,
    ReshapeConfig,
    ScatterConfig,
    TopKConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    TOPK_MIN_WIDTH,
    even_int_div,
    save_and_get_path,
)
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)


class MoEGate(AbstractModule):
    """MoE gate module from DeepSeek-R1.
    See the `AbstractModule` docstring for usage info.
    """

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
        prefix: str = "",
    ) -> WeightConfig:
        tt_gate_proj_weight = ttnn.from_torch(
            state_dict[f"{prefix}weight"].T.unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_e_score_correction_bias = ttnn.from_torch(
            state_dict[f"{prefix}e_score_correction_bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        eps = 1e-20  # no hf config for this
        tt_norm_eps = ttnn.from_torch(
            torch.tensor([eps]).repeat(1, hf_config.num_experts_per_tok).unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tt_expert_scale = ttnn.from_torch(
            torch.tensor([hf_config.routed_scaling_factor])
            .repeat(1, hf_config.num_experts_per_tok)
            .unsqueeze(0)
            .unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        torch_expert_group_mask = torch.full((1, 1, 1, hf_config.n_group), -float("inf"))
        torch_ones_src_tensor = torch.ones((1, 1, 1, hf_config.topk_group))
        tt_expert_group_mask = ttnn.from_torch(
            torch_expert_group_mask,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        tt_ones_src_tensor = ttnn.from_torch(
            torch_ones_src_tensor,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        return {
            "gate_proj": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"gate_proj.input_tensor_b",
                    tt_gate_proj_weight,
                )
            },
            "add_score_correction_bias": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"e_score_correction_bias.input_tensor_b",
                    tt_e_score_correction_bias,
                )
            },
            "add_norm_eps": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"add_norm_eps.input_tensor_b",
                    tt_norm_eps,
                )
            },
            "multiply_expert_scale": {
                "input_tensor_b": save_and_get_path(
                    output_path / f"multiply_expert_scale.input_tensor_b",
                    tt_expert_scale,
                )
            },
            "scatter_top_expert_groups": {
                "input": save_and_get_path(
                    output_path / f"scatter_top_expert_groups.input",
                    tt_expert_group_mask,
                ),
                "src": save_and_get_path(
                    output_path / f"scatter_top_expert_groups.src",
                    tt_ones_src_tensor,
                ),
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

        # Construct the config
        return {
            "gate_proj": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            ),
            "add_score_correction_bias": BinaryOpConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=memory_config,
                dtype=ttnn.bfloat16,
            ),
            "add_norm_eps": BinaryOpConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=memory_config,
                dtype=ttnn.bfloat16,
            ),
            "multiply_expert_scale": BinaryOpConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=memory_config,
                dtype=ttnn.bfloat16,
            ),
            "reshape_scores": ReshapeConfig(
                shape=(1, -1, hf_config.n_group, even_int_div(hf_config.n_routed_experts, hf_config.n_group)),
            ),
            "topk_within_expert_groups": TopKConfig(
                k=2,  # no hf config for this
                dim=3,
            ),
            "topk_expert_groups": TopKConfig(
                k=hf_config.topk_group,
                dim=3,
            ),
            "scatter_top_expert_groups": ScatterConfig(
                input=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                src=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                dim=3,
            ),
            "reshape_group_mask": ReshapeConfig(
                shape=(1, -1, hf_config.n_group, 1),
            ),
            "reshape_active_experts": ReshapeConfig(
                shape=(1, 1, -1, hf_config.n_routed_experts),
            ),
            "mul_scores_with_mask": MulConfig(
                memory_config=memory_config,
            ),
            "topk_experts": TopKConfig(
                k=hf_config.num_experts_per_tok,
                dim=3,
            ),
            "input_memory_config": memory_config,
            "output_memory_config": memory_config,
        }

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, "decode")

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, "prefill")

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        assert x.memory_config() == cfg["input_memory_config"]

        # Gate projections
        logits = ttnn.linear(x, **cfg["gate_proj"])
        # Sigmoid activation
        scores = ttnn.sigmoid(logits)
        ttnn.deallocate(logits)
        # Add score correction bias
        # Expand bias to match scores shape(dynamic shape)
        scores_correction_bias = cfg["add_score_correction_bias"]["input_tensor_b"]
        scores_correction_bias = ttnn.repeat(scores_correction_bias, ttnn.Shape((1, 1, scores.shape[2], 1)))
        scores_correction_bias = ttnn.to_layout(scores_correction_bias, ttnn.TILE_LAYOUT)
        scores_with_bias = ttnn.add(
            scores,
            scores_correction_bias,
            memory_config=cfg["add_score_correction_bias"]["memory_config"],
            dtype=cfg["add_score_correction_bias"]["dtype"],
        )
        # Reshape scores to expert groups
        expert_scores_grouped = ttnn.reshape(scores_with_bias, **cfg["reshape_scores"])
        num_experts_per_group = expert_scores_grouped.shape[3]

        if expert_scores_grouped.shape[3] < TOPK_MIN_WIDTH:
            # Pad to 64 for topk op
            expert_scores_grouped = ttnn.pad(
                expert_scores_grouped,
                [(0, 0), (0, 0), (0, 0), (0, TOPK_MIN_WIDTH - expert_scores_grouped.shape[3])],
                value=-float("inf"),
            )
        # calculate top-2 scores with expert groups
        topk_scores_within_expert_groups, topk_indices_within_expert_groups = ttnn.topk(
            expert_scores_grouped, **cfg["topk_within_expert_groups"]
        )
        ttnn.deallocate(expert_scores_grouped)
        ttnn.deallocate(topk_indices_within_expert_groups)
        # sum top-2 scores within expert groups
        expert_group_scores = ttnn.sum(topk_scores_within_expert_groups, dim=3)
        ttnn.deallocate(topk_scores_within_expert_groups)
        # reshape to 4D tensor
        expert_group_scores = ttnn.unsqueeze(expert_group_scores, dim=0)

        if expert_group_scores.shape[3] < TOPK_MIN_WIDTH:
            expert_group_scores = ttnn.pad(
                expert_group_scores,
                [(0, 0), (0, 0), (0, 0), (0, TOPK_MIN_WIDTH - expert_group_scores.shape[3])],
                value=-float("inf"),
            )
        # calculate top-k expert groups
        topk_expert_groups_scores, topk_expert_groups_indices = ttnn.topk(
            expert_group_scores, **cfg["topk_expert_groups"]
        )
        ttnn.deallocate(expert_group_scores)
        ttnn.deallocate(topk_expert_groups_scores)

        # create full expert_groups_mask(dynamic shape)
        input_mask = cfg["scatter_top_expert_groups"]["input"]
        input_mask = ttnn.repeat(input_mask, ttnn.Shape((1, 1, scores.shape[2], 1)))

        # create full src tensor of ones
        src_tensor = cfg["scatter_top_expert_groups"]["src"]
        src_tensor = ttnn.repeat(src_tensor, ttnn.Shape((1, 1, scores.shape[2], 1)))

        # scatter top-k expert groups indices to full expert_groups_mask
        active_groups_mask = ttnn.scatter(
            input=input_mask,
            index=topk_expert_groups_indices,
            src=src_tensor,
            dim=cfg["scatter_top_expert_groups"]["dim"],
        )
        ttnn.deallocate(topk_expert_groups_indices)
        active_groups_mask = ttnn.reshape(active_groups_mask, **cfg["reshape_group_mask"])

        # expand active_groups_mask to all the experts
        active_experts_mask = ttnn.repeat(active_groups_mask, ttnn.Shape((1, 1, 1, num_experts_per_group)))
        ttnn.deallocate(active_groups_mask)
        active_experts_mask = ttnn.reshape(active_experts_mask, **cfg["reshape_active_experts"])
        active_experts_scores = ttnn.mul(scores_with_bias, active_experts_mask, **cfg["mul_scores_with_mask"])
        ttnn.deallocate(scores_with_bias)
        ttnn.deallocate(active_experts_mask)

        # calculate top-k experts
        topk_experts_scores_with_bias, topk_experts_indices = ttnn.topk(active_experts_scores, **cfg["topk_experts"])
        ttnn.deallocate(active_experts_scores)
        ttnn.deallocate(topk_experts_scores_with_bias)

        # gather original scores without bias
        topk_experts_scores = ttnn.gather(scores, dim=3, index=topk_experts_indices)
        ttnn.deallocate(scores)

        # normalize scores
        topk_expert_scores_sum = ttnn.sum(topk_experts_scores, dim=3, keepdim=True)
        topk_experts_scores_normalized = ttnn.div(topk_experts_scores, topk_expert_scores_sum)
        ttnn.deallocate(topk_expert_scores_sum)
        ttnn.deallocate(topk_experts_scores)

        # add norm eps
        norm_eps = cfg["add_norm_eps"]["input_tensor_b"]
        # expand norm_eps to match topk_experts_scores_normalized shape(dynamic shape)
        norm_eps = ttnn.repeat(norm_eps, ttnn.Shape((1, 1, topk_experts_scores_normalized.shape[2], 1)))
        norm_eps = ttnn.to_layout(norm_eps, ttnn.TILE_LAYOUT)
        topk_experts_scores_normalized = ttnn.add(
            topk_experts_scores_normalized,
            norm_eps,
            memory_config=cfg["add_norm_eps"]["memory_config"],
            dtype=cfg["add_norm_eps"]["dtype"],
        )
        ttnn.deallocate(norm_eps)

        # multiply by expert scale
        expert_scale = cfg["multiply_expert_scale"]["input_tensor_b"]
        # expand expert_scale to match topk_experts_scores_normalized shape(dynamic shape)
        expert_scale = ttnn.repeat(expert_scale, ttnn.Shape((1, 1, topk_experts_scores_normalized.shape[2], 1)))
        expert_scale = ttnn.to_layout(expert_scale, ttnn.TILE_LAYOUT)
        topk_experts_scores_normalized = ttnn.mul(
            topk_experts_scores_normalized,
            expert_scale,
            memory_config=cfg["multiply_expert_scale"]["memory_config"],
            dtype=cfg["multiply_expert_scale"]["dtype"],
        )
        ttnn.deallocate(expert_scale)

        return topk_experts_scores_normalized, topk_experts_indices

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)
