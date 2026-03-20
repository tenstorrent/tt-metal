# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn


# DEBUG: Helper for hang debugging
def _debug_print(msg: str, flush: bool = True):
    """Print debug message with timestamp and flush to ensure immediate output."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[DEBUG {timestamp}] {msg}", file=sys.stderr, flush=flush)


from models.demos.deepseek_v3.reference.reference_utils import topk_bitonic
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    BinaryOpConfig,
    FromWeightConfig,
    LinearConfig,
    LinearFallbackConfig,
    MeshDeviceStub,
    MulConfig,
    ReshapeConfig,
    ScatterConfig,
    TopKConfig,
    TopKFallbackConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    TOPK_MIN_WIDTH,
    even_int_div,
    get_dequantized_tensor,
    shard_and_save,
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
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
        prefix: str = "",
    ) -> WeightConfig:
        (state_dict,) = state_dicts
        assert state_dict is not None
        gate_weight = get_dequantized_tensor(state_dict, f"{prefix}weight")
        score_correction_bias = get_dequantized_tensor(
            state_dict, f"{prefix}e_score_correction_bias", dtype=torch.float32
        )
        return {
            "gate_proj": {
                "input_tensor_b": shard_and_save(
                    output_path / f"gate_proj.input_tensor_b",
                    gate_weight.T.unsqueeze(0).unsqueeze(0),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                )
            },
            "add_score_correction_bias": {
                "input_tensor_b": shard_and_save(
                    output_path / f"e_score_correction_bias.input_tensor_b",
                    score_correction_bias.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.float32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            },
            "multiply_expert_scale": {
                "input_tensor_b": shard_and_save(
                    output_path / f"multiply_expert_scale.input_tensor_b",
                    torch.tensor([hf_config.routed_scaling_factor])
                    .repeat(1, hf_config.num_experts_per_tok)
                    .unsqueeze(0)
                    .unsqueeze(0),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            },
            "scatter_top_expert_groups": {
                "input": shard_and_save(
                    output_path / f"scatter_top_expert_groups.input",
                    torch.full((1, 1, 1, hf_config.n_group), -float("inf")),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                "src": shard_and_save(
                    output_path / f"scatter_top_expert_groups.src",
                    torch.ones((1, 1, 1, hf_config.topk_group)),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
            },
        }

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        mode: str,
        topk_fallback: bool = False,
        use_bitonic_sort: bool = True,
    ) -> ModelDecodeConfig | ModelPrefillConfig:
        """Generate decode configuration for this module.
        Note: topk_fallback and use_bitonic_sort are defaulted to True and not required in future when we have equivalent topk op.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            mode: "decode" or "prefill"
            topk_fallback: whether to use topk fallback
            use_bitonic_sort: whether to use bitonic sort
        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """

        if mode == "decode":
            memory_config = ttnn.L1_MEMORY_CONFIG

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
                    dim=-1,
                ),
                "topk_expert_groups": TopKConfig(
                    k=hf_config.topk_group,
                    dim=-1,
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
                    dim=-1,
                ),
                "topk_fallback": topk_fallback,
                "topk_fallback_config": TopKFallbackConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dtype=ttnn.bfloat16,
                    memory_config=memory_config,
                    use_bitonic_sort=use_bitonic_sort,
                ),
                "linear_fallback": False,
                "linear_fallback_config": LinearFallbackConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dtype=ttnn.bfloat16,
                ),
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                # "input_memory_config": ttnn.create_sharded_memory_config(  # Bad PCC
                #         shape=(USERS_PER_ROW, HIDDEN_SIZE),
                #         core_grid=ttnn.CoreGrid(y=7, x=8),
                #         strategy=ttnn.ShardStrategy.WIDTH,
                #     ),
                "input_memory_config": memory_config,
                "output_memory_config": memory_config,
            }
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG

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
                    dim=-1,
                ),
                "topk_expert_groups": TopKConfig(
                    k=hf_config.topk_group,
                    dim=-1,
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
                    dim=-1,
                ),
                "topk_fallback": topk_fallback,
                "topk_fallback_config": TopKFallbackConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dtype=ttnn.bfloat16,
                    memory_config=memory_config,
                    use_bitonic_sort=use_bitonic_sort,
                ),
                "linear_fallback": False,
                "linear_fallback_config": LinearFallbackConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dtype=ttnn.bfloat16,
                ),
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "input_memory_config": memory_config,
                "output_memory_config": memory_config,
            }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        topk_fallback: bool = False,
        use_bitonic_sort: bool = True,
    ) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, "decode", topk_fallback, use_bitonic_sort)

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        topk_fallback: bool = False,
        use_bitonic_sort: bool = True,
    ) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, "prefill", topk_fallback, use_bitonic_sort)

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        _debug_print(f"MoEGate.forward: START (input shape={x.shape})")
        assert x.memory_config() == cfg["input_memory_config"]

        # Gate projections
        _debug_print("MoEGate.forward: gate projection (linear) START")
        if cfg["linear_fallback"]:
            logits = cls.linear_fallback_op(x, **cfg["linear_fallback_config"], **cfg["gate_proj"])
        else:
            logits = ttnn.linear(x, **cfg["gate_proj"])
        _debug_print("MoEGate.forward: gate projection (linear) DONE")
        # Sigmoid activation
        _debug_print("MoEGate.forward: sigmoid START")
        scores = ttnn.sigmoid(logits)
        _debug_print("MoEGate.forward: sigmoid DONE")
        _debug_print("MoEGate.forward: deallocate logits START")
        ttnn.deallocate(logits)
        _debug_print("MoEGate.forward: deallocate logits DONE")
        # Add score correction bias
        # Expand bias to match scores shape(dynamic shape)
        _debug_print("MoEGate.forward: add score correction bias START")
        scores_correction_bias = cfg["add_score_correction_bias"]["input_tensor_b"]
        scores_correction_bias = ttnn.repeat(scores_correction_bias, ttnn.Shape((1, 1, scores.shape[2], 1)))
        scores_correction_bias = ttnn.to_layout(scores_correction_bias, ttnn.TILE_LAYOUT)
        scores_with_bias = ttnn.add(
            scores,
            scores_correction_bias,
            memory_config=cfg["add_score_correction_bias"]["memory_config"],
            dtype=cfg["add_score_correction_bias"]["dtype"],
        )
        _debug_print("MoEGate.forward: add score correction bias DONE")
        # Reshape scores to expert groups
        _debug_print("MoEGate.forward: reshape scores to expert groups START")
        expert_scores_grouped = ttnn.reshape(scores_with_bias, **cfg["reshape_scores"])
        _debug_print("MoEGate.forward: reshape scores to expert groups DONE")
        num_experts_per_group = expert_scores_grouped.shape[3]

        # calculate top-2 scores with expert groups
        _debug_print("MoEGate.forward: topk within expert groups START")
        if cfg["topk_fallback"]:
            topk_scores_within_expert_groups, topk_indices_within_expert_groups = cls.topk_fallback_op(
                expert_scores_grouped, **cfg["topk_fallback_config"], **cfg["topk_within_expert_groups"]
            )
        else:
            if expert_scores_grouped.shape[3] < TOPK_MIN_WIDTH:
                # Pad to 64 for topk op
                expert_scores_grouped = ttnn.pad(
                    expert_scores_grouped,
                    [(0, 0), (0, 0), (0, 0), (0, TOPK_MIN_WIDTH - expert_scores_grouped.shape[3])],
                    value=-float("inf"),
                )
            topk_scores_within_expert_groups, topk_indices_within_expert_groups = ttnn.topk(
                expert_scores_grouped, **cfg["topk_within_expert_groups"]
            )
        _debug_print("MoEGate.forward: topk within expert groups DONE")
        _debug_print("MoEGate.forward: deallocate expert_scores_grouped, topk_indices_within_expert_groups START")
        ttnn.deallocate(expert_scores_grouped)
        ttnn.deallocate(topk_indices_within_expert_groups)
        _debug_print("MoEGate.forward: deallocate expert_scores_grouped, topk_indices_within_expert_groups DONE")
        # sum top-2 scores within expert groups
        _debug_print("MoEGate.forward: sum top-2 scores within expert groups START")
        expert_group_scores = ttnn.sum(topk_scores_within_expert_groups, dim=3)
        _debug_print("MoEGate.forward: sum top-2 scores within expert groups DONE")
        _debug_print("MoEGate.forward: deallocate topk_scores_within_expert_groups START")
        ttnn.deallocate(topk_scores_within_expert_groups)
        _debug_print("MoEGate.forward: deallocate topk_scores_within_expert_groups DONE")
        # reshape to 4D tensor
        _debug_print("MoEGate.forward: unsqueeze expert_group_scores START")
        expert_group_scores = ttnn.unsqueeze(expert_group_scores, dim=0)
        _debug_print("MoEGate.forward: unsqueeze expert_group_scores DONE")

        # calculate top-k expert groups
        _debug_print("MoEGate.forward: topk expert groups START")
        if cfg["topk_fallback"]:
            topk_expert_groups_scores, topk_expert_groups_indices = cls.topk_fallback_op(
                expert_group_scores, **cfg["topk_fallback_config"], **cfg["topk_expert_groups"]
            )
        else:
            if expert_group_scores.shape[3] < TOPK_MIN_WIDTH:
                expert_group_scores = ttnn.pad(
                    expert_group_scores,
                    [(0, 0), (0, 0), (0, 0), (0, TOPK_MIN_WIDTH - expert_group_scores.shape[3])],
                    value=-float("inf"),
                )
            topk_expert_groups_scores, topk_expert_groups_indices = ttnn.topk(
                expert_group_scores, **cfg["topk_expert_groups"]
            )
        _debug_print("MoEGate.forward: topk expert groups DONE")
        _debug_print("MoEGate.forward: deallocate expert_group_scores, topk_expert_groups_scores START")
        ttnn.deallocate(expert_group_scores)
        ttnn.deallocate(topk_expert_groups_scores)
        _debug_print("MoEGate.forward: deallocate expert_group_scores, topk_expert_groups_scores DONE")

        # create full expert_groups_mask(dynamic shape)
        _debug_print("MoEGate.forward: create expert_groups_mask START")
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
        _debug_print("MoEGate.forward: create expert_groups_mask DONE")
        _debug_print("MoEGate.forward: deallocate topk_expert_groups_indices START")
        ttnn.deallocate(topk_expert_groups_indices)
        _debug_print("MoEGate.forward: deallocate topk_expert_groups_indices DONE")
        _debug_print("MoEGate.forward: to_layout and reshape active_groups_mask START")
        active_groups_mask = ttnn.to_layout(active_groups_mask, ttnn.TILE_LAYOUT)
        active_groups_mask = ttnn.reshape(active_groups_mask, **cfg["reshape_group_mask"])
        _debug_print("MoEGate.forward: to_layout and reshape active_groups_mask DONE")

        # expand active_groups_mask to all the experts
        _debug_print("MoEGate.forward: expand active_groups_mask to all experts START")
        active_experts_mask = ttnn.repeat(active_groups_mask, ttnn.Shape((1, 1, 1, num_experts_per_group)))
        _debug_print("MoEGate.forward: deallocate active_groups_mask START")
        ttnn.deallocate(active_groups_mask)
        _debug_print("MoEGate.forward: deallocate active_groups_mask DONE")
        _debug_print("MoEGate.forward: reshape active_experts_mask START")
        active_experts_mask = ttnn.reshape(active_experts_mask, **cfg["reshape_active_experts"])
        _debug_print("MoEGate.forward: reshape active_experts_mask DONE")
        _debug_print("MoEGate.forward: mul active_experts_scores START")
        active_experts_scores = ttnn.mul(scores_with_bias, active_experts_mask, **cfg["mul_scores_with_mask"])
        _debug_print("MoEGate.forward: mul active_experts_scores DONE")
        _debug_print("MoEGate.forward: deallocate scores_with_bias, active_experts_mask START")
        ttnn.deallocate(scores_with_bias)
        ttnn.deallocate(active_experts_mask)
        _debug_print("MoEGate.forward: deallocate scores_with_bias, active_experts_mask DONE")

        # calculate top-k experts
        _debug_print("MoEGate.forward: topk experts START")
        if cfg["topk_fallback"]:
            topk_experts_scores_with_bias, topk_experts_indices = cls.topk_fallback_op(
                active_experts_scores, **cfg["topk_fallback_config"], **cfg["topk_experts"]
            )
        else:
            topk_experts_scores_with_bias, topk_experts_indices = ttnn.topk(
                active_experts_scores, **cfg["topk_experts"]
            )
        _debug_print("MoEGate.forward: topk experts DONE")
        _debug_print("MoEGate.forward: deallocate active_experts_scores, topk_experts_scores_with_bias START")
        ttnn.deallocate(active_experts_scores)
        ttnn.deallocate(topk_experts_scores_with_bias)
        _debug_print("MoEGate.forward: deallocate active_experts_scores, topk_experts_scores_with_bias DONE")

        # gather original scores without bias
        _debug_print("MoEGate.forward: gather original scores START")
        topk_experts_scores = ttnn.gather(scores, dim=3, index=topk_experts_indices)
        _debug_print("MoEGate.forward: gather original scores DONE")
        _debug_print("MoEGate.forward: deallocate scores START")
        ttnn.deallocate(scores)
        _debug_print("MoEGate.forward: deallocate scores DONE")

        # normalize scores
        _debug_print("MoEGate.forward: normalize scores START")
        topk_expert_scores_sum = ttnn.sum(topk_experts_scores, dim=3, keepdim=True) + 1e-20  # add norm eps
        topk_experts_scores_normalized = ttnn.div(topk_experts_scores, topk_expert_scores_sum)
        _debug_print("MoEGate.forward: normalize scores DONE")
        _debug_print("MoEGate.forward: deallocate topk_expert_scores_sum, topk_experts_scores START")
        ttnn.deallocate(topk_expert_scores_sum)
        ttnn.deallocate(topk_experts_scores)
        _debug_print("MoEGate.forward: deallocate topk_expert_scores_sum, topk_experts_scores DONE")

        # multiply by expert scale
        _debug_print("MoEGate.forward: multiply by expert scale START")
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
        _debug_print("MoEGate.forward: multiply by expert scale DONE")
        _debug_print("MoEGate.forward: deallocate expert_scale START")
        ttnn.deallocate(expert_scale)
        _debug_print("MoEGate.forward: deallocate expert_scale DONE")

        _debug_print("MoEGate.forward: END")
        return topk_experts_scores_normalized, topk_experts_indices

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        _debug_print("MoEGate.forward_prefill: START")
        result = cls.forward(x, cfg)
        _debug_print("MoEGate.forward_prefill: END")
        return result

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        _debug_print("MoEGate.forward_decode: START")
        result = cls.forward(x, cfg)
        _debug_print("MoEGate.forward_decode: END")
        return result

    @classmethod
    def topk_fallback_op(
        cls,
        input: ttnn.Tensor,
        mesh_device: ttnn.Device,
        dtype: ttnn.DataType,
        memory_config: ttnn.MemoryConfig,
        k: int,
        dim: int,
        largest: bool,
        sorted: bool,
        use_bitonic_sort: bool,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # convert ttnn mesh tensor to torch tensor
        logger.info(f"topk_fallback_op: input shape: {input.shape}")
        torch_input = ttnn.to_torch(
            input,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0].unsqueeze(0)

        if use_bitonic_sort:
            topk_fn = topk_bitonic
        else:
            topk_fn = torch.topk

        torch_topk_scores, torch_topk_indices = topk_fn(torch_input, k=k, dim=dim, largest=largest, sorted=sorted)

        ttnn_topk_scores = ttnn.from_torch(
            torch_topk_scores,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
            dtype=dtype,
            memory_config=memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        ttnn_topk_indices = ttnn.from_torch(
            torch_topk_indices,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
            dtype=ttnn.uint16,
            memory_config=memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        return ttnn_topk_scores, ttnn_topk_indices

    @classmethod
    def linear_fallback_op(
        cls,
        input_tensor: ttnn.Tensor,
        input_tensor_b: ttnn.Tensor,
        mesh_device: ttnn.Device,
        dtype: ttnn.DataType,
        memory_config: ttnn.MemoryConfig,
        compute_kernel_config=None,
    ) -> ttnn.Tensor:
        """Linear fallback operation using torch.nn.functional.linear"""
        # convert ttnn mesh tensors to torch tensors
        logger.info(f"linear_fallback_op: input shape: {input_tensor.shape}, weight shape: {input_tensor_b.shape}")

        torch_input = ttnn.to_torch(
            input_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0].unsqueeze(0)

        torch_weight = ttnn.to_torch(
            input_tensor_b,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0][0]

        torch_input_2d = torch_input.squeeze(0).squeeze(0)  # [seq_len, hidden_dim]
        torch_weight_2d = torch_weight.T  # [output_dim, hidden_dim]

        # use torch linear: input @ weight.T
        torch_output = torch.nn.functional.linear(torch_input_2d, torch_weight_2d)

        # Restore dimensions and convert back to ttnn
        torch_output = torch_output.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, output_dim]

        ttnn_output = ttnn.from_torch(
            torch_output,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, None), mesh_shape=tuple(mesh_device.shape)),
            dtype=dtype,
            memory_config=memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        return ttnn_output
