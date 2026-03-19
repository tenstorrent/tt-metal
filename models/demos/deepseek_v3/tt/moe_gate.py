# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
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
from models.demos.deepseek_v3.utils.config_helpers import COMPUTE_KERNEL_CONFIG_HIFI2, even_int_div, shard_and_save
from models.demos.deepseek_v3.utils.run_config import (
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.demos.deepseek_v3_b1.micro_ops.deepseek_moe_gate.op import DeepseekMoeGateSingleCore


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
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelState:
        """Create input_indices, output_indices and output_tensor for each MoE layer

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            ccl: CCL instance for communication configuration
        Returns:
            ModelState containing input_indices, output_indices and output_tensor for each MoE layer
        """
        ttnn_output_tensor = ttnn.zeros(
            shape=(1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_output_indices = ttnn.zeros(
            shape=(1, 32, 32),
            dtype=ttnn.uint16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn_input_indices = ttnn.arange(
            start=0,
            end=16 * 16,
            step=1,
            dtype=ttnn.int32,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn_input_indices = ttnn.unsqueeze(ttnn_input_indices, dim=0)
        ttnn_input_indices = ttnn.reshape(ttnn_input_indices, (1, 16, 16))
        ttnn_input_indices = ttnn.transpose(ttnn_input_indices, dim1=-2, dim2=-1)
        ttnn_input_indices = ttnn.typecast(ttnn_input_indices, dtype=ttnn.uint16)
        return {
            "gate_routing": {
                "ttnn_output_tensor": ttnn_output_tensor,
                "ttnn_input_indices": ttnn_input_indices,
                "ttnn_output_indices": ttnn_output_indices,
            },
            "mesh_device": mesh_device,
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
                "input_memory_config": memory_config,
                "output_memory_config": memory_config,
                "routed_scaling_factor": hf_config.routed_scaling_factor,
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
                "routed_scaling_factor": hf_config.routed_scaling_factor,
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

        # Gate projections
        if cfg["linear_fallback"]:
            logits = cls.linear_fallback_op(x, **cfg["linear_fallback_config"], **cfg["gate_proj"])
        else:
            logits = ttnn.linear(x, **cfg["gate_proj"])

        mesh_device = cfg["mesh_device"]
        num_experts = cfg["add_score_correction_bias"].input_tensor_b.shape[3]
        assert num_experts == 256, "num_experts should be 256"
        total_batch_size_per_device = logits.shape[2]

        # create the shard spec and memory config for the input, logits and output
        grid = mesh_device.compute_with_storage_grid_size()
        num_device_cores = grid.x * grid.y
        start_index = 0
        # we can only have one token per core at a time
        end_index = min(num_device_cores, total_batch_size_per_device)
        # this while loop is designed to handle the huge batch size (4096)
        topk_experts_scores_list = []
        topk_experts_indices_list = []
        while True:
            batch_size_per_device = end_index - start_index
            # get the ceil of batch size per core
            core_grid = ttnn.num_cores_to_corerangeset(
                batch_size_per_device,
                ttnn.CoreCoord(grid.x, grid.y),
                row_wise=True,
            )

            input_output_shard_shape = (32, 32)
            reshaped_input_shape = (batch_size_per_device, 16, 16)

            # currently we cannot convert the tile size of logits and input indices to 16*16,
            # but the memory layout is the same since the length is 256
            input_output_shard_spec = ttnn.ShardSpec(
                core_grid,
                input_output_shard_shape,
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            input_output_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_output_shard_spec
            )

            cur_logits = logits[:, :, start_index:end_index, :]
            cur_logits = ttnn.reshape(cur_logits, reshaped_input_shape)

            # change the memory config of the logits
            cur_logits = ttnn.to_memory_config(cur_logits, memory_config=input_output_mem_config)

            # create the bias tensor (don't put it before line 398)
            scores_correction_bias = cfg["add_score_correction_bias"]["input_tensor_b"]
            scores_correction_bias = ttnn.repeat(scores_correction_bias, ttnn.Shape((batch_size_per_device, 1)))
            scores_correction_bias = ttnn.reshape(scores_correction_bias, reshaped_input_shape)
            scores_correction_bias = ttnn.transpose(scores_correction_bias, dim1=-2, dim2=-1)
            scores_correction_bias = ttnn.to_layout(scores_correction_bias, ttnn.TILE_LAYOUT)
            scores_correction_bias = ttnn.to_memory_config(
                scores_correction_bias, memory_config=input_output_mem_config
            )

            # create the output tensor, input indices and output indices
            ttnn_output_tensor = cfg["gate_routing"]["ttnn_output_tensor"]
            ttnn_output_tensor = ttnn.repeat(ttnn_output_tensor, (batch_size_per_device, 1, 1))
            ttnn_output_tensor = ttnn.to_memory_config(ttnn_output_tensor, memory_config=input_output_mem_config)

            ttnn_input_indices = cfg["gate_routing"]["ttnn_input_indices"]
            ttnn_input_indices = ttnn.repeat(ttnn_input_indices, (batch_size_per_device, 1, 1))
            ttnn_input_indices = ttnn.to_memory_config(ttnn_input_indices, memory_config=input_output_mem_config)

            ttnn_output_indices = cfg["gate_routing"]["ttnn_output_indices"]
            ttnn_output_indices = ttnn.repeat(ttnn_output_indices, (batch_size_per_device, 1, 1))
            ttnn_output_indices = ttnn.to_memory_config(ttnn_output_indices, memory_config=input_output_mem_config)

            eps = 1e-20
            scaling_factor = cfg["routed_scaling_factor"]
            enable_sigmoid = True
            topk_experts_scores, topk_experts_indices = DeepseekMoeGateSingleCore.op(
                cur_logits,
                scores_correction_bias,
                ttnn_output_tensor,
                ttnn_input_indices,
                ttnn_output_indices,
                eps,
                scaling_factor,
                enable_sigmoid,
            )

            topk_experts_scores = ttnn.reshape(
                topk_experts_scores, (-1, topk_experts_scores.shape[-2], topk_experts_scores.shape[-1])
            )
            topk_experts_scores = ttnn.to_memory_config(topk_experts_scores, memory_config=ttnn.L1_MEMORY_CONFIG)
            topk_experts_indices = ttnn.reshape(
                topk_experts_indices, (-1, topk_experts_indices.shape[-2], topk_experts_indices.shape[-1])
            )
            topk_experts_indices = ttnn.to_memory_config(topk_experts_indices, memory_config=ttnn.L1_MEMORY_CONFIG)
            topk_experts_indices = ttnn.typecast(topk_experts_indices, dtype=ttnn.int32)
            topk_experts_scores_list.append(topk_experts_scores[:batch_size_per_device, :, :])
            topk_experts_indices_list.append(topk_experts_indices[:batch_size_per_device, :, :])
            ttnn.deallocate(cur_logits)
            ttnn.deallocate(scores_correction_bias)
            ttnn.deallocate(ttnn_output_tensor)
            ttnn.deallocate(ttnn_input_indices)
            ttnn.deallocate(ttnn_output_indices)

            if end_index >= total_batch_size_per_device:
                break
            start_index = end_index
            end_index = min(start_index + num_device_cores, total_batch_size_per_device)

        topk_experts_weights = ttnn.concat(topk_experts_scores_list, dim=0)
        topk_experts_indices = ttnn.concat(topk_experts_indices_list, dim=0)
        # here we only take the 1x8  out of 32x32
        topk_experts_weights = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_indices = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_weights = topk_experts_weights[:, 0, :8]
        topk_experts_indices = topk_experts_indices[:, 0, :]
        topk_experts_weights = ttnn.unsqueeze(topk_experts_weights, dim=0)
        topk_experts_weights = ttnn.unsqueeze(topk_experts_weights, dim=0)
        topk_experts_indices = ttnn.unsqueeze(topk_experts_indices, dim=0)
        topk_experts_indices = ttnn.unsqueeze(topk_experts_indices, dim=0)
        topk_experts_weights = ttnn.to_memory_config(topk_experts_weights, memory_config=cfg["output_memory_config"])
        topk_experts_indices = ttnn.to_memory_config(topk_experts_indices, memory_config=cfg["output_memory_config"])
        # if we do typecast on a row_major tensor, then we may see a hang
        topk_experts_indices = ttnn.to_layout(topk_experts_indices, ttnn.TILE_LAYOUT)
        topk_experts_indices = ttnn.typecast(topk_experts_indices, dtype=ttnn.uint16)
        topk_experts_indices = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_indices = ttnn.slice(topk_experts_indices, [0, 0, 0, 0], [1, 1, topk_experts_indices.shape[2], 8])
        for tensor in topk_experts_scores_list:
            ttnn.deallocate(tensor)
        for tensor in topk_experts_indices_list:
            ttnn.deallocate(tensor)

        return topk_experts_weights, topk_experts_indices

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return cls.forward(x, cfg)

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
