# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
        assert x.memory_config() == cfg["input_memory_config"]
        SEND_CORES = (0, 3, 6, 9)
        RECV_CORES = (1, 2, 4, 5, 7, 8, 10, 11)
        L = 1
        C = 1
        M = x.shape[2]
        K = x.shape[3]
        N = cfg["gate_proj"]["input_tensor_b"].shape[3]

        in0_core_coords = cfg["mesh_device"].get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        core2dram = {}
        for dram_bank_id, core_coords in enumerate(in0_core_coords):
            core2dram[core_coords] = dram_bank_id

        in0_num_cores = len(in0_core_coords)

        # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
        in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

        ring2cores = {}
        for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
            # key: ring_pos, value: (core_coord, dram_bank_id, send_flag)
            ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in SEND_CORES else 0)

        in0_core_range = [ttnn.CoreRange(in0_core_coord, in0_core_coord) for in0_core_coord in in0_core_coords_sorted]
        in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

        # --------------------------------------------------------------------------
        # Constants
        # --------------------------------------------------------------------------
        in0_dtype = ttnn.bfloat16
        w_dtype = ttnn.bfloat16
        num_dram_banks = len(in0_core_coords)

        dram_core_coords = [ttnn.CoreCoord(core2dram[in0_core_coord], 0) for in0_core_coord in in0_core_coords_sorted]
        dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
        dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

        # --------------------------------------------------------------------------
        # Tensor shapes and memory configurations
        # --------------------------------------------------------------------------
        # Define tensor shapes - same for both accuracy and performance testing
        input_shape = (in0_num_cores, M, K)

        in0_shard_spec = ttnn.ShardSpec(
            grid=in0_core_range_set,
            shard_shape=(M, K),
            shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

        # Each core gets a copy of the original (M, K) input
        input_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in0_shard_spec
        )

        # ------------------------------------------------------------------------
        # Create DRAM shard spec for w
        # Tensor shape: (L, K, N) -> Sharded across N cores
        # ------------------------------------------------------------------------
        w_shard_height = L * (76 + 1) * ttnn.TILE_SIZE
        w_shard_width = 2 * ttnn.TILE_SIZE

        w_shard_spec = ttnn.ShardSpec(
            dram_core_range_set, (w_shard_height, w_shard_width), ttnn.ShardOrientation.ROW_MAJOR
        )

        w_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w_shard_spec)

        # ------------------------------------------------------------------------
        # Create DRAM shard spec for output
        # Tensor shape: (M, N) -> Sharded across 8 cores
        # ------------------------------------------------------------------------
        output_shard_height = M
        output_shard_width = ttnn.TILE_SIZE
        output_shard_spec = ttnn.ShardSpec(
            in0_core_range_set, (output_shard_height, output_shard_width), ttnn.ShardOrientation.ROW_MAJOR
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )
        tt_output = ttnn.empty(
            (M, in0_num_cores * ttnn.TILE_SIZE),
            dtype=in0_dtype,
            device=cfg["mesh_device"],
            layout=ttnn.TILE_LAYOUT,
            memory_config=output_mem_config,
        )

        # ------------------------------------------------------------------------
        # Prepare the tensors
        # --------------------------------------------------------------------------
        torch_w = ttnn.to_torch(
            cfg["gate_proj"]["input_tensor_b"],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                cfg["mesh_device"], dims=(0, 1), mesh_shape=tuple(cfg["mesh_device"].shape)
            ),
        )[0, 0]
        torch_bias = ttnn.to_torch(
            cfg["add_score_correction_bias"]["input_tensor_b"],
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                cfg["mesh_device"], dims=(0, 1), mesh_shape=tuple(cfg["mesh_device"].shape)
            ),
        )[0, 0].to(torch.bfloat16)

        # ------------------------------------------------------------------------
        # Prepare w tensor (padded, and reordered)
        from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_mm import prepare_w_tensor

        torch_w_reordered = prepare_w_tensor(torch_w, torch_bias, L, K, N, ring2cores)
        # Create tt_w tensor with DRAM sharding
        tt_w = ttnn.from_torch(
            torch_w_reordered,
            dtype=w_dtype,
            device=cfg["mesh_device"],
            layout=ttnn.TILE_LAYOUT,
            memory_config=w_mem_config,
        )

        # --------------------------------------------------------------------------
        # Run the operation
        # --------------------------------------------------------------------------
        # Collect accuracy metrics for all layers and experts
        all_outputs = []
        all_accuracy_metrics = {}

        import itertools

        breakpoint()
        for layer_id, column_id in itertools.product(range(L), range(C)):
            ttnn.experimental.deepseek.moe.moe_gate_mm(
                x,
                w_tensor=tt_w,
                output_tensor=tt_output,
                layer_id=layer_id,
                column_id=column_id,
            )

            tt_to_torch_output = ttnn.to_torch(tt_output)
            all_outputs.append(tt_to_torch_output)

        tt_to_torch_outputs = torch.stack(all_outputs)

        # prepare the output tensor

        # gather original scores without bias
        topk_experts_scores = ttnn.gather(scores_flat, dim=3, index=topk_experts_indices)
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
