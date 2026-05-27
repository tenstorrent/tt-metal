# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.minimax_m27.reference.reference_utils import topk_bitonic
from models.demos.minimax_m27.utils.abstract_module import AbstractModule
from models.demos.minimax_m27.utils.config_dataclass import (
    BinaryOpConfig,
    FromWeightConfig,
    LinearConfig,
    LinearFallbackConfig,
    MeshDeviceStub,
    TopKConfig,
    TopKFallbackConfig,
)
from models.demos.minimax_m27.utils.config_helpers import COMPUTE_KERNEL_CONFIG_HIFI2, shard_and_save
from models.demos.minimax_m27.utils.run_config import ModelPrefillConfig, RunPrefillConfig, WeightConfig


class MoEGate(AbstractModule):
    """MoE gate module for MiniMax M2.7.

    Implements the flat sigmoid + correction-bias top-k routing used by
    ``MiniMaxM2MoEGate`` / ``MiniMaxM2SparseMoeBlock`` (no grouped routing,
    no routed scaling factor).

    See the ``AbstractModule`` docstring for usage info.
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
        """Convert MoE-gate weights.

        Expected state-dict keys (under ``prefix``, which should be the
        block-level prefix, e.g. ``""`` for the standalone gate or for a
        ``MiniMaxM2SparseMoeBlock``):
            - ``gate.weight``                  (router projection)
            - ``e_score_correction_bias``      (additive routing bias)
        """
        (state_dict,) = state_dicts
        assert state_dict is not None
        return {
            "gate_proj": {
                "input_tensor_b": shard_and_save(
                    output_path / f"gate_proj.input_tensor_b",
                    state_dict[f"{prefix}gate.weight"].T.unsqueeze(0).unsqueeze(0),
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
                    state_dict[f"{prefix}e_score_correction_bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0),
                    shard_dims=(None, None),
                    mesh_device=mesh_device,
                    dtype=ttnn.float32,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            },
        }

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        topk_fallback: bool = False,
        use_bitonic_sort: bool = True,
    ) -> ModelPrefillConfig:
        """Generate prefill configuration for this module.

        Note: ``topk_fallback`` and ``use_bitonic_sort`` exist as a workaround
        until we have an equivalent on-device top-k op.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            topk_fallback: whether to use topk fallback
            use_bitonic_sort: whether to use bitonic sort
        Returns:
            ModelPrefillConfig containing operator configurations for prefill mode
        """
        memory_config = ttnn.DRAM_MEMORY_CONFIG

        return {
            "gate_proj": LinearConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=memory_config,
                compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            ),
            # Bias is stored as fp32 (see convert_weights). Keep the add in fp32 to match the
            # reference, which runs sigmoid + bias + normalize in fp32 and only casts back to
            # the input dtype at the very end.
            "add_score_correction_bias": BinaryOpConfig(
                input_tensor_b=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
                memory_config=memory_config,
                dtype=ttnn.float32,
            ),
            "output_dtype": ttnn.bfloat16,
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
    ):
        raise NotImplementedError("Decode mode has been removed from minimax_m27.")

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        topk_fallback: bool = False,
        use_bitonic_sort: bool = True,
    ) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, topk_fallback, use_bitonic_sort)

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """MiniMax M2.7 routing.

        Steps (matches reference ``MiniMaxM2MoEGate.forward``):
            1. logits = x @ gate_proj.T
            2. scores = sigmoid(logits.float())        # run sigmoid + downstream math in fp32
            3. scores_for_choice = scores + e_score_correction_bias
            4. top_k_index = topk(scores_for_choice).indices
            5. top_k_weights = gather(scores, top_k_index)  # original (unbiased) scores
            6. top_k_weights /= top_k_weights.sum() + 1e-20
            7. cast back to bfloat16
        """
        assert x.memory_config() == cfg["input_memory_config"]

        # 1) Gate projection
        if cfg["linear_fallback"]:
            logits = cls.linear_fallback_op(x, **cfg["linear_fallback_config"], **cfg["gate_proj"])
        else:
            logits = ttnn.linear(x, **cfg["gate_proj"])

        # 2) Sigmoid scoring in fp32 (matches reference's `sigmoid(router_logits.float())`).
        # `ttnn.sigmoid` does accept fp32 input despite its docstring only listing bfloat16 --
        # the device op validator (`unary_device_operation.cpp::validate_supported_arch_dtype`)
        # has no dtype restriction for `SIGMOID`, and on fp32 it matches `torch.sigmoid`
        # bit-exactly. Casting to fp32 here keeps the whole routing path in fp32.
        logits_fp32 = ttnn.typecast(logits, dtype=ttnn.float32)
        ttnn.deallocate(logits)
        scores = ttnn.sigmoid(logits_fp32)
        ttnn.deallocate(logits_fp32)

        # 3) Add score correction bias (broadcast bias across token dim) -- fp32.
        scores_correction_bias = cfg["add_score_correction_bias"]["input_tensor_b"]
        scores_correction_bias = ttnn.repeat(scores_correction_bias, ttnn.Shape((1, 1, scores.shape[2], 1)))
        scores_correction_bias = ttnn.to_layout(scores_correction_bias, ttnn.TILE_LAYOUT)
        scores_for_choice = ttnn.add(
            scores,
            scores_correction_bias,
            memory_config=cfg["add_score_correction_bias"]["memory_config"],
            dtype=cfg["add_score_correction_bias"]["dtype"],
        )

        # 4) Top-k expert selection (we only need indices from the biased scores)
        if cfg["topk_fallback"]:
            topk_biased_scores, topk_experts_indices = cls.topk_fallback_op(
                scores_for_choice, **cfg["topk_fallback_config"], **cfg["topk_experts"]
            )
        else:
            topk_biased_scores, topk_experts_indices = ttnn.topk(scores_for_choice, **cfg["topk_experts"])
        ttnn.deallocate(scores_for_choice)
        ttnn.deallocate(topk_biased_scores)

        # 5) Gather *unbiased* sigmoid scores at the selected expert indices (still fp32).
        topk_experts_scores = ttnn.gather(scores, dim=3, index=topk_experts_indices)
        ttnn.deallocate(scores)

        # 6) Normalize in fp32.
        topk_expert_scores_sum = ttnn.sum(topk_experts_scores, dim=3, keepdim=True) + 1e-20
        topk_experts_scores_normalized = ttnn.div(topk_experts_scores, topk_expert_scores_sum)
        ttnn.deallocate(topk_expert_scores_sum)
        ttnn.deallocate(topk_experts_scores)

        # 7) Cast back to bf16 to match the reference's final `.to(input_dtype)`.
        topk_experts_scores_bf16 = ttnn.typecast(topk_experts_scores_normalized, dtype=cfg["output_dtype"])
        ttnn.deallocate(topk_experts_scores_normalized)

        return topk_experts_scores_bf16, topk_experts_indices

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        raise NotImplementedError("Decode mode has been removed from minimax_m27.")

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
