# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
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
    COMPUTE_KERNEL_CONFIG_HIFI4,
    TOPK_MIN_WIDTH,
    even_int_div,
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
        return {
            "gate_proj": {
                "input_tensor_b": shard_and_save(
                    output_path / f"gate_proj.input_tensor_b",
                    state_dict[f"{prefix}weight"].T.unsqueeze(0).unsqueeze(0),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    # Dequantized checkpoints already store gate weights in bf16.
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                )
            },
            "add_score_correction_bias": {
                "input_tensor_b": shard_and_save(
                    output_path / f"e_score_correction_bias.input_tensor_b",
                    state_dict[f"{prefix}e_score_correction_bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    # Reference test path casts module parameters to bf16 before forward.
                    # Store bias in bf16 to match that oracle, then compute the add in fp32.
                    dtype=ttnn.bfloat16,
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
            gate_kernel = COMPUTE_KERNEL_CONFIG_HIFI4 if topk_fallback else COMPUTE_KERNEL_CONFIG_HIFI2
            routing_dtype = ttnn.float32 if topk_fallback else ttnn.bfloat16

            return {
                "gate_proj": LinearConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    compute_kernel_config=gate_kernel,
                ),
                "add_score_correction_bias": BinaryOpConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    dtype=routing_dtype,
                ),
                "multiply_expert_scale": BinaryOpConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    dtype=routing_dtype,
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
                    dtype=routing_dtype,
                    memory_config=memory_config,
                    use_bitonic_sort=use_bitonic_sort,
                ),
                # Keep gate projection on TT by default; topk fallback is only for selection.
                "linear_fallback": False,
                "linear_fallback_config": LinearFallbackConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dtype=routing_dtype,
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
            gate_kernel = COMPUTE_KERNEL_CONFIG_HIFI4 if topk_fallback else COMPUTE_KERNEL_CONFIG_HIFI2
            routing_dtype = ttnn.float32 if topk_fallback else ttnn.bfloat16

            return {
                "gate_proj": LinearConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    compute_kernel_config=gate_kernel,
                ),
                "add_score_correction_bias": BinaryOpConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    dtype=routing_dtype,
                ),
                "multiply_expert_scale": BinaryOpConfig(
                    input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                    memory_config=memory_config,
                    dtype=routing_dtype,
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
                    dtype=routing_dtype,
                    memory_config=memory_config,
                    use_bitonic_sort=use_bitonic_sort,
                ),
                # Keep gate projection on TT by default; topk fallback is only for selection.
                "linear_fallback": False,
                "linear_fallback_config": LinearFallbackConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dtype=routing_dtype,
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
        assert x.memory_config() == cfg["input_memory_config"]

        def _cast_for_device_topk(tensor: ttnn.Tensor) -> ttnn.Tensor:
            # Current TT topk kernel only supports BF16/BF8_B inputs.
            if tensor.dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
                return tensor
            casted = ttnn.typecast(tensor, dtype=ttnn.bfloat16)
            ttnn.deallocate(tensor)
            return casted

        # Gate projections
        if cfg["linear_fallback"]:
            logits = cls.linear_fallback_op(x, **cfg["linear_fallback_config"], **cfg["gate_proj"])
        else:
            if cfg["topk_fallback"]:
                linear_input = x
                if linear_input.dtype != ttnn.float32:
                    linear_input = ttnn.typecast(linear_input, dtype=ttnn.float32)
                # Keep logits in fp32 only for routing-fallback mode.
                logits = ttnn.linear(linear_input, dtype=ttnn.float32, **cfg["gate_proj"])
                if linear_input is not x:
                    ttnn.deallocate(linear_input)
            else:
                logits = ttnn.linear(x, **cfg["gate_proj"])

        if cfg["topk_fallback"]:
            topk_weights, topk_indices = cls.routing_fallback_from_logits(
                logits=logits,
                score_correction_bias=cfg["add_score_correction_bias"]["input_tensor_b"],
                expert_scale=cfg["multiply_expert_scale"]["input_tensor_b"],
                mesh_device=cfg["topk_fallback_config"]["mesh_device"],
                dtype=cfg["topk_fallback_config"]["dtype"],
                memory_config=cfg["output_memory_config"],
                use_bitonic_sort=cfg["topk_fallback_config"]["use_bitonic_sort"],
                n_group=cfg["reshape_scores"]["shape"][2],
                topk_group=cfg["topk_expert_groups"]["k"],
                num_experts_per_tok=cfg["topk_experts"]["k"],
            )
            ttnn.deallocate(logits)
            return topk_weights, topk_indices
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
        ttnn.deallocate(scores_correction_bias)

        # Reshape scores to expert groups
        expert_scores_grouped = ttnn.reshape(scores_with_bias, **cfg["reshape_scores"])
        num_experts_per_group = expert_scores_grouped.shape[3]

        # calculate top-2 scores with expert groups
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
            expert_scores_grouped = _cast_for_device_topk(expert_scores_grouped)
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

        # calculate top-k expert groups
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
            expert_group_scores = _cast_for_device_topk(expert_group_scores)
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
        ttnn.deallocate(input_mask)
        ttnn.deallocate(src_tensor)
        ttnn.deallocate(topk_expert_groups_indices)
        active_groups_mask = ttnn.to_layout(active_groups_mask, ttnn.TILE_LAYOUT)
        active_groups_mask = ttnn.reshape(active_groups_mask, **cfg["reshape_group_mask"])

        # expand active_groups_mask to all the experts
        active_experts_mask = ttnn.repeat(active_groups_mask, ttnn.Shape((1, 1, 1, num_experts_per_group)))
        ttnn.deallocate(active_groups_mask)
        active_experts_mask = ttnn.reshape(active_experts_mask, **cfg["reshape_active_experts"])
        active_experts_scores = ttnn.mul(scores_with_bias, active_experts_mask, **cfg["mul_scores_with_mask"])
        ttnn.deallocate(scores_with_bias)
        ttnn.deallocate(active_experts_mask)

        # calculate top-k experts
        if cfg["topk_fallback"]:
            topk_experts_scores_with_bias, topk_experts_indices = cls.topk_fallback_op(
                active_experts_scores, **cfg["topk_fallback_config"], **cfg["topk_experts"]
            )
        else:
            active_experts_scores = _cast_for_device_topk(active_experts_scores)
            topk_experts_scores_with_bias, topk_experts_indices = ttnn.topk(
                active_experts_scores, **cfg["topk_experts"]
            )
        ttnn.deallocate(active_experts_scores)
        ttnn.deallocate(topk_experts_scores_with_bias)

        # gather original scores without bias
        topk_experts_scores = ttnn.gather(scores, dim=3, index=topk_experts_indices)
        ttnn.deallocate(scores)

        # normalize scores
        topk_expert_scores_sum = ttnn.sum(topk_experts_scores, dim=3, keepdim=True) + 1e-20  # add norm eps
        topk_experts_scores_normalized = ttnn.div(topk_experts_scores, topk_expert_scores_sum)
        ttnn.deallocate(topk_expert_scores_sum)
        ttnn.deallocate(topk_experts_scores)

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
    def routing_fallback_from_logits(
        cls,
        logits: ttnn.Tensor,
        score_correction_bias: ttnn.Tensor,
        expert_scale: ttnn.Tensor,
        mesh_device: ttnn.Device,
        dtype: ttnn.DataType,
        memory_config: ttnn.MemoryConfig,
        use_bitonic_sort: bool,
        n_group: int,
        topk_group: int,
        num_experts_per_tok: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # Keep gate projection on TT; run routing selection in torch for parity.
        torch_logits = ttnn.to_torch(
            logits,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0].unsqueeze(0)

        # Bias is replicated across mesh; use one replica.
        torch_bias = ttnn.to_torch(ttnn.get_device_tensors(score_correction_bias)[0]).reshape(-1)
        torch_bias = torch_bias.to(torch.bfloat16).to(torch.float32)

        torch_scale = ttnn.to_torch(ttnn.get_device_tensors(expert_scale)[0]).reshape(-1)
        routed_scaling_factor = torch_scale.to(torch.float32)[0]

        torch_logits_2d = torch_logits.squeeze(0).squeeze(0).to(torch.float32).reshape(-1, torch_logits.shape[-1])
        torch_scores_2d = torch.sigmoid(torch_logits_2d)
        torch_scores_for_choice = torch_scores_2d + torch_bias.unsqueeze(0)

        topk_fn = topk_bitonic if use_bitonic_sort else torch.topk
        num_tokens = torch_scores_for_choice.shape[0]

        grouped_scores = torch_scores_for_choice.view(num_tokens, n_group, -1)
        group_scores = topk_fn(grouped_scores, k=2, dim=-1, largest=True, sorted=True)[0].sum(dim=-1)

        group_idx = topk_fn(group_scores, k=topk_group, dim=-1, largest=True, sorted=True)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(num_tokens, n_group, torch_scores_for_choice.shape[-1] // n_group)
        score_mask = score_mask.reshape(num_tokens, -1)

        tmp_scores = torch_scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_idx = topk_fn(tmp_scores, k=num_experts_per_tok, dim=-1, largest=True, sorted=True)[1]
        topk_weight = torch_scores_2d.gather(1, topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * routed_scaling_factor

        torch_topk_scores = topk_weight.unsqueeze(0).unsqueeze(0)
        torch_topk_indices = topk_idx.unsqueeze(0).unsqueeze(0)

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
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

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

        torch_input_2d = torch_input.squeeze(0).squeeze(0).to(torch.float32)  # [seq_len, hidden_dim]
        torch_weight_2d = torch_weight.T.to(torch.float32)  # [output_dim, hidden_dim]

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
