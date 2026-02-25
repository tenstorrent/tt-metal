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
        assert x.memory_config() == cfg["input_memory_config"]

        # INSTRUMENTATION: Save router input and config in SAME format as MoEGateAdapter
        import os

        if os.environ.get("SAVE_ROUTER_OUTPUTS") == "1":
            try:
                import json
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_outputs_binary")
                save_dir.mkdir(parents=True, exist_ok=True)

                # Get mesh device from config
                mesh_device = cfg["mesh_device"]

                # Save input as binary (same as MoEGateAdapter)
                x_torch = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

                # Convert bfloat16 to float32 for numpy compatibility
                if x_torch.dtype == torch.bfloat16:
                    x_torch_save = x_torch.float()
                else:
                    x_torch_save = x_torch

                with open(save_dir / "full_reference_input.bin", "wb") as f:
                    f.write(x_torch_save.cpu().numpy().tobytes())

                # Save input metadata
                torch.save(
                    {
                        "input_shape": x_torch.shape,
                        "input_dtype": str(x_torch.dtype),
                        "input_min": float(x_torch_save.min()),
                        "input_max": float(x_torch_save.max()),
                        "input_mean": float(x_torch_save.mean()),
                    },
                    save_dir / "full_reference_input_metadata.pt",
                )

                # Save configuration (extract relevant parts from cfg)
                config_to_save = {
                    "implementation": "Full Reference MoEGate",
                    "mesh_device_shape": tuple(mesh_device.shape) if mesh_device else None,
                    "topk_fallback": cfg.get("topk_fallback", False),
                    "use_bitonic_sort": cfg.get("use_bitonic_sort", False),
                    # Add more config items as needed
                }
                with open(save_dir / "full_reference_config.json", "w") as f:
                    json.dump(config_to_save, f, indent=2)

                logger.info(f"[Reference MoEGate] Saved input and config to {save_dir}")
                logger.info(f"  Input shape: {x_torch.shape}, dtype: {x_torch.dtype}")
                logger.info(
                    f"  Input stats - min: {x_torch_save.min():.6f}, max: {x_torch_save.max():.6f}, mean: {x_torch_save.mean():.6f}"
                )

            except Exception as e:
                logger.error(f"[Reference MoEGate] Failed to save input/config: {e}")

        # Keep old instrumentation for backward compatibility
        import os

        if os.environ.get("SAVE_ROUTER_INPUTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                # Convert to torch for saving (handle distributed tensors)
                mesh_device = cfg["mesh_device"]
                try:
                    input_torch = ttnn.to_torch(x)
                except:
                    # Tensor is distributed, use mesh composer
                    input_torch = ttnn.to_torch(
                        x,
                        mesh_composer=ttnn.ConcatMesh2dToTensor(
                            mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                        ),
                    )

                # Save both tensor data and binary representation for exact comparison
                torch.save(
                    {
                        "tensor": input_torch,
                        "bytes": input_torch.cpu().float().numpy().tobytes(),  # Convert to float32 for numpy
                        "shape": input_torch.shape,
                        "dtype": input_torch.dtype,
                        "implementation": "MoEGate",
                    },
                    save_dir / "router_input.pt",
                )

                print(f"[REF] ✅ Saved router input: shape {input_torch.shape}, dtype {input_torch.dtype}")
            except Exception as e:
                print(f"[REF] ❌ Failed to save router input: {e}")

        # Gate projections
        if cfg["linear_fallback"]:
            logits = cls.linear_fallback_op(x, **cfg["linear_fallback_config"], **cfg["gate_proj"])
        else:
            logits = ttnn.linear(x, **cfg["gate_proj"])

        # Checkpoint 1: After gate projection
        if os.environ.get("SAVE_ROUTER_CHECKPOINTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                mesh_device = cfg["mesh_device"]
                logits_torch = ttnn.to_torch(
                    logits,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "tensor": logits_torch,
                        "bytes": logits_torch.cpu().float().numpy().tobytes(),  # Convert to float32 for numpy
                        "shape": logits_torch.shape,
                        "dtype": logits_torch.dtype,
                    },
                    save_dir / "checkpoint_logits.pt",
                )
                print(f"[REF] Checkpoint 1: Saved logits {logits_torch.shape}")
            except Exception as e:
                print(f"[REF] Failed to save checkpoint_logits: {e}")
        # Sigmoid activation
        scores = ttnn.sigmoid(logits)
        ttnn.deallocate(logits)

        # Checkpoint 2: After sigmoid
        if os.environ.get("SAVE_ROUTER_CHECKPOINTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                mesh_device = cfg["mesh_device"]
                scores_torch = ttnn.to_torch(
                    scores,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "tensor": scores_torch,
                        "bytes": scores_torch.cpu().float().numpy().tobytes(),  # Convert to float32 for numpy
                        "shape": scores_torch.shape,
                        "dtype": scores_torch.dtype,
                    },
                    save_dir / "checkpoint_scores_sigmoid.pt",
                )

                # Also save original scores for comparison before bias
                torch.save(
                    {
                        "tensor": scores_torch,
                        "bytes": scores_torch.cpu().numpy().tobytes(),
                        "shape": scores_torch.shape,
                        "dtype": scores_torch.dtype,
                    },
                    save_dir / "checkpoint_scores_original.pt",
                )
                print(f"[REF] Checkpoint 2: Saved scores after sigmoid {scores_torch.shape}")
            except Exception as e:
                print(f"[REF] Failed to save checkpoint_scores_sigmoid: {e}")

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

        # Checkpoint 3: After bias correction
        if os.environ.get("SAVE_ROUTER_CHECKPOINTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                mesh_device = cfg["mesh_device"]
                scores_biased_torch = ttnn.to_torch(
                    scores_with_bias,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "tensor": scores_biased_torch,
                        "bytes": scores_biased_torch.cpu().float().numpy().tobytes(),  # Convert to float32 for numpy
                        "shape": scores_biased_torch.shape,
                        "dtype": scores_biased_torch.dtype,
                    },
                    save_dir / "checkpoint_scores_biased.pt",
                )
                print(f"[REF] Checkpoint 3: Saved scores after bias {scores_biased_torch.shape}")
            except Exception as e:
                print(f"[REF] Failed to save checkpoint_scores_biased: {e}")

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
        active_groups_mask = ttnn.to_layout(active_groups_mask, ttnn.TILE_LAYOUT)
        active_groups_mask = ttnn.reshape(active_groups_mask, **cfg["reshape_group_mask"])

        # expand active_groups_mask to all the experts
        active_experts_mask = ttnn.repeat(active_groups_mask, ttnn.Shape((1, 1, 1, num_experts_per_group)))
        ttnn.deallocate(active_groups_mask)
        active_experts_mask = ttnn.reshape(active_experts_mask, **cfg["reshape_active_experts"])
        active_experts_scores = ttnn.mul(scores_with_bias, active_experts_mask, **cfg["mul_scores_with_mask"])
        ttnn.deallocate(scores_with_bias)
        ttnn.deallocate(active_experts_mask)

        # Checkpoint 4: After masking
        if os.environ.get("SAVE_ROUTER_CHECKPOINTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                mesh_device = cfg["mesh_device"]
                masked_torch = ttnn.to_torch(
                    active_experts_scores,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "tensor": masked_torch,
                        "bytes": masked_torch.cpu().float().numpy().tobytes(),
                        "shape": masked_torch.shape,
                        "dtype": masked_torch.dtype,
                    },
                    save_dir / "checkpoint_masked.pt",
                )
                print(f"[REF] Checkpoint 4: Saved masked scores {masked_torch.shape}")
            except Exception as e:
                print(f"[REF] Failed to save checkpoint_masked: {e}")

        # Checkpoint 4: After group masking
        if os.environ.get("SAVE_ROUTER_CHECKPOINTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                mesh_device = cfg["mesh_device"]
                masked_torch = ttnn.to_torch(
                    active_experts_scores,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "tensor": masked_torch,
                        "bytes": masked_torch.cpu().float().numpy().tobytes(),  # Convert to float32 for numpy
                        "shape": masked_torch.shape,
                        "dtype": masked_torch.dtype,
                    },
                    save_dir / "checkpoint_masked.pt",
                )
                print(f"[REF] Checkpoint 4: Saved masked scores {masked_torch.shape}")
            except Exception as e:
                print(f"[REF] Failed to save checkpoint_masked: {e}")

        # calculate top-k experts
        if cfg["topk_fallback"]:
            topk_experts_scores_with_bias, topk_experts_indices = cls.topk_fallback_op(
                active_experts_scores, **cfg["topk_fallback_config"], **cfg["topk_experts"]
            )
        else:
            topk_experts_scores_with_bias, topk_experts_indices = ttnn.topk(
                active_experts_scores, **cfg["topk_experts"]
            )
        ttnn.deallocate(active_experts_scores)
        ttnn.deallocate(topk_experts_scores_with_bias)

        # gather original scores without bias
        topk_experts_scores = ttnn.gather(scores, dim=3, index=topk_experts_indices)
        ttnn.deallocate(scores)

        # Checkpoint 5: After topk
        if os.environ.get("SAVE_ROUTER_CHECKPOINTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                mesh_device = cfg["mesh_device"]
                topk_indices_torch = ttnn.to_torch(
                    topk_experts_indices,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )
                topk_weights_torch = ttnn.to_torch(
                    topk_experts_scores,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "tensor": topk_indices_torch,
                        "bytes": topk_indices_torch.cpu().numpy().tobytes(),
                        "shape": topk_indices_torch.shape,
                        "dtype": topk_indices_torch.dtype,
                    },
                    save_dir / "checkpoint_topk_indices.pt",
                )

                torch.save(
                    {
                        "tensor": topk_weights_torch,
                        "bytes": topk_weights_torch.cpu().float().numpy().tobytes(),  # Convert to float32 for numpy
                        "shape": topk_weights_torch.shape,
                        "dtype": topk_weights_torch.dtype,
                    },
                    save_dir / "checkpoint_topk_weights.pt",
                )
                print(
                    f"[REF] Checkpoint 5: Saved topk indices {topk_indices_torch.shape}, weights {topk_weights_torch.shape}"
                )
            except Exception as e:
                print(f"[REF] Failed to save checkpoint_topk: {e}")

        # normalize scores
        topk_expert_scores_sum = ttnn.sum(topk_experts_scores, dim=3, keepdim=True) + 1e-20  # add norm eps
        topk_experts_scores_normalized = ttnn.div(topk_experts_scores, topk_expert_scores_sum)
        ttnn.deallocate(topk_expert_scores_sum)
        ttnn.deallocate(topk_experts_scores)

        # Checkpoint 6: After normalization
        if os.environ.get("SAVE_ROUTER_CHECKPOINTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                mesh_device = cfg["mesh_device"]
                normalized_torch = ttnn.to_torch(
                    topk_experts_scores_normalized,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "tensor": normalized_torch,
                        "bytes": normalized_torch.cpu().float().numpy().tobytes(),  # Convert to float32 for numpy
                        "shape": normalized_torch.shape,
                        "dtype": normalized_torch.dtype,
                    },
                    save_dir / "checkpoint_normalized.pt",
                )
                print(f"[REF] Checkpoint 6: Saved normalized weights {normalized_torch.shape}")
            except Exception as e:
                print(f"[REF] Failed to save checkpoint_normalized: {e}")

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

        # Checkpoint 7: After scaling (final)
        if os.environ.get("SAVE_ROUTER_CHECKPOINTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                mesh_device = cfg["mesh_device"]
                final_torch = ttnn.to_torch(
                    topk_experts_scores_normalized,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "tensor": final_torch,
                        "bytes": final_torch.cpu().float().numpy().tobytes(),  # Convert to float32 for numpy
                        "shape": final_torch.shape,
                        "dtype": final_torch.dtype,
                    },
                    save_dir / "checkpoint_final.pt",
                )
                print(f"[REF] Checkpoint 7: Saved final scaled weights {final_torch.shape}")
            except Exception as e:
                print(f"[REF] Failed to save checkpoint_final: {e}")

        # Save router outputs for comparison
        import os

        if os.environ.get("SAVE_ROUTER_CHECKPOINTS", "0") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_debug/reference")
                save_dir.mkdir(parents=True, exist_ok=True)

                # Convert to torch for saving
                mesh_device = cfg["mesh_device"]
                weights_torch = ttnn.to_torch(
                    topk_experts_scores_normalized,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )
                indices_torch = ttnn.to_torch(
                    topk_experts_indices,
                    mesh_composer=ttnn.ConcatMesh2dToTensor(
                        mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)
                    ),
                )

                torch.save(
                    {
                        "weights": weights_torch,
                        "indices": indices_torch,
                        "shape_weights": weights_torch.shape,
                        "shape_indices": indices_torch.shape,
                    },
                    save_dir / "router_output.pt",
                )  # Same filename as our impl
                print(f"[REF] Saved router output: weights {weights_torch.shape}, indices {indices_torch.shape}")

                # Also print debug info
                print(f"[REF] First token experts: {indices_torch[0, 0, 0, :].tolist()}")
                if indices_torch.shape[2] > 1:
                    print(f"[REF] Last token experts: {indices_torch[0, 0, -1, :].tolist()}")
                    # Check diversity
                    unique_combos = len(
                        set(tuple(indices_torch[0, 0, i, :].tolist()) for i in range(min(10, indices_torch.shape[2])))
                    )
                    print(f"[REF] Expert diversity: {unique_combos}/10 unique combinations")
            except Exception as e:
                print(f"[REF] Failed to save router outputs: {e}")

        # INSTRUMENTATION: Save binary router outputs (same format as MoEGateAdapter)
        if os.environ.get("SAVE_ROUTER_OUTPUTS") == "1":
            try:
                from pathlib import Path

                import torch

                save_dir = Path("/tmp/router_outputs_binary")
                save_dir.mkdir(parents=True, exist_ok=True)

                # Convert to torch and save as binary
                weights_torch = ttnn.to_torch(
                    topk_experts_scores_normalized, mesh_composer=ttnn.ConcatMeshToTensor(cfg["mesh_device"], dim=0)
                )
                indices_torch = ttnn.to_torch(
                    topk_experts_indices, mesh_composer=ttnn.ConcatMeshToTensor(cfg["mesh_device"], dim=0)
                )

                # Convert bfloat16 to float32 for numpy compatibility
                if weights_torch.dtype == torch.bfloat16:
                    weights_torch_save = weights_torch.float()
                else:
                    weights_torch_save = weights_torch

                # Save as binary for exact comparison
                with open(save_dir / "full_reference_weights.bin", "wb") as f:
                    f.write(weights_torch_save.cpu().numpy().tobytes())
                with open(save_dir / "full_reference_indices.bin", "wb") as f:
                    f.write(indices_torch.cpu().numpy().tobytes())

                # Also save shapes for reconstruction
                torch.save(
                    {
                        "weights_shape": weights_torch.shape,
                        "indices_shape": indices_torch.shape,
                        "weights_dtype": str(weights_torch.dtype),
                        "indices_dtype": str(indices_torch.dtype),
                    },
                    save_dir / "full_reference_metadata.pt",
                )

                logger.info(f"[Reference MoEGate] Saved router outputs to {save_dir}")
                logger.info(f"  Weights shape: {weights_torch.shape}, sum: {weights_torch.sum()}")
                logger.info(f"  Indices shape: {indices_torch.shape}")
                logger.info(
                    f"  First token experts: {indices_torch[0, 0, 0, :8].tolist() if indices_torch.ndim >= 4 else indices_torch[0, :8].tolist()}"
                )

            except Exception as e:
                logger.error(f"[Reference MoEGate] Failed to save router outputs: {e}")

        return topk_experts_scores_normalized, topk_experts_indices

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
