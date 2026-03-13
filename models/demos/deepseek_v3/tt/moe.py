# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from time import perf_counter

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.experts import Experts as MoEExperts
from models.demos.deepseek_v3.tt.moe_gate import MoEGate
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    AllToAllCombineConfig,
    AllToAllDispatchConfig,
    AllToAllDispatchMetadataConfig,
    DeepseekMoEReduceScatterConfig,
    MeshDeviceStub,
    MoEComputeConfig,
    MorehFullConfig,
    MulConfig,
    ReduceScatterAsyncMinimalConfig,
    RepeatConfig,
    SelectiveReduceCombineConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
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
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        assert (
            len(state_dicts) == 1 and state_dicts[0] is not None
        ), f"MoE expects exactly one non-padding state dict, got {len(state_dicts)}"
        (state_dict,) = state_dicts
        assert state_dict is not None

        return {
            "moe_gate": MoEGate.convert_weights(
                hf_config, (state_dict,), output_path / "moe_gate", mesh_device, "gate."
            ),
            "moe_experts": MoEExperts.convert_weights(
                hf_config, (state_dict,), output_path / "moe_experts", mesh_device
            ),
        }

    @classmethod
    def create_shared_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelState:
        """Create shared model state containing tensors that are constant across all instances.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
        Returns:
            ModelState containing shared tensors
        """
        num_dispatch_device_rows = mesh_device.shape[0]

        logger.info(
            "Creating MoE shared state: expert mapping tensor "
            f"(num_devices={num_devices}, experts_per_device={num_experts_per_device})..."
        )
        expert_mapping_start = perf_counter()
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
        logger.info(f"Created MoE expert mapping tensor in {perf_counter() - expert_mapping_start:.2f}s")

        logger.info(
            "Creating MoE shared state: remap topk mask "
            f"(dispatch_rows={num_dispatch_device_rows}, experts={hf_config.n_routed_experts})..."
        )
        remap_mask_start = perf_counter()
        remap_topk_mask = ttnn.from_torch(
            torch.ones((1, num_dispatch_device_rows, 1, hf_config.n_routed_experts), dtype=torch.bfloat16),
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        logger.info(f"Created MoE remap topk mask in {perf_counter() - remap_mask_start:.2f}s")

        return {
            "remap_topk_mask": remap_topk_mask,
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
        }

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        ccl: CCL,
    ) -> ModelState:
        """Create model state containing CCL-related communication configurations.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
            ccl: CCL instance for communication configuration
        Returns:
            ModelState containing CCL configurations
        """
        # Store CCL object for runtime semaphore initialization
        return {
            # CCL-specific parameters (semaphores and num_links)
            "all_to_all_dispatch": {
                "num_links": 4,
            },
            "all_to_all_combine": {
                "num_links": 4,
            },
            "ccl": ccl,
        }

    @classmethod
    def model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        mode: str,
        topk_fallback: bool = False,
    ) -> ModelDecodeConfig | ModelPrefillConfig:
        """Generate decode configuration for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device the model will be placed later on
        Returns:
            ModelDecodeConfig containing operator configurations for decode mode
        """

        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)
        expert_mapping_tensor = cls._create_expert_mapping_tensor(
            mesh_device, fabric_config, mode, num_experts_per_device
        )

        if mode == "decode":
            memory_config = ttnn.L1_MEMORY_CONFIG

            HIDDEN_SIZE = hf_config.hidden_size
            TP_SIZE = mesh_device.shape[1]

            shard_core_grid = ttnn.CoreGrid(y=7, x=4)
            per_core_width = (HIDDEN_SIZE // TP_SIZE) // shard_core_grid.num_cores
            input_output_memory_config = ttnn.create_sharded_memory_config(
                shape=(
                    ttnn.core.roundup(USERS_PER_ROW, ttnn.TILE_SIZE),
                    ttnn.core.roundup(per_core_width, ttnn.TILE_SIZE),
                ),
                core_grid=shard_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Construct the config
            decode_config = {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
                "fabric_config": fabric_config,
                "num_experts_per_device": num_experts_per_device,
                "hidden_size": hf_config.hidden_size,
                "num_experts_per_tok": hf_config.num_experts_per_tok,
                "num_dispatch_devices": mesh_device.shape[0],
                "expert_mapping_tensor": expert_mapping_tensor,
                "moe_gate": MoEGate.model_config(hf_config, mesh_device, mode, topk_fallback=topk_fallback),
                "all_to_all_dispatch_output_memory_config": memory_config,
                "all_to_all_dispatch_metadata_memory_config": ttnn.DRAM_MEMORY_CONFIG,
                "activations_repeat": RepeatConfig(repeat_dims=ttnn.Shape((1, num_experts_per_device, 1, 1))),
                "moe_experts": MoEExperts._create_model_config(hf_config, mesh_device, mode),
                "all_to_all_combine_output_memory_config": memory_config,
                "topk_weights_repeat": RepeatConfig(repeat_dims=ttnn.Shape((hf_config.hidden_size, 1, 1, 1))),
                "mul_experts_output_with_weights": MulConfig(memory_config=memory_config),
                "input_memory_config": input_output_memory_config,
                "output_memory_config": input_output_memory_config,
                "all_to_all_dispatch": AllToAllDispatchConfig(cluster_axis=0, memory_config=memory_config),
                "all_to_all_combine": AllToAllCombineConfig(cluster_axis=0, memory_config=memory_config),
                "sum_experts_output_memory_config": memory_config,
                "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                    cluster_axis=1,
                    dim=3,
                    memory_config=input_output_memory_config,
                ),
                "ring_sum_experts_output_memory_config": DeepseekMoEReduceScatterConfig.create_default_input_memory_config(
                    USERS_PER_ROW, HIDDEN_SIZE, TP_SIZE
                ),
                "ring_final_output_reduce_scatter": DeepseekMoEReduceScatterConfig(
                    cluster_axis=1,
                    dim=3,
                    output_memory_config=input_output_memory_config,
                ),
                "revert_tp": AllGatherAsyncConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dim=-1,  # Last dimension
                    # memory_config=ttnn.create_sharded_memory_config(  # Bad PCC
                    #     shape=(USERS_PER_ROW, HIDDEN_SIZE),
                    #     core_grid=ttnn.CoreGrid(y=7, x=8),
                    #     strategy=ttnn.ShardStrategy.WIDTH,
                    # ),
                    memory_config=memory_config,
                    cluster_axis=1,
                ),
            }

            # optimized MoE ops only functional for quad torus
            if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING and mesh_device.shape[0] == 16:
                batch = USERS_PER_ROW * mesh_device.shape[0]
                seq_len = 1

                decode_config["quad_ring_all_to_all_dispatch_metadata"] = AllToAllDispatchMetadataConfig(
                    cluster_axis=0,
                    worker_mode=ttnn.WorkerMode.DIRECT,
                    dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
                    output_tensors=AllToAllDispatchMetadataConfig.create_preallocated_dispatch_output_tensors(
                        mesh_device,
                        batch,
                        HIDDEN_SIZE,
                        hf_config.num_experts_per_tok,
                    ),
                )
                decode_config["quad_ring_moreh_full"] = MorehFullConfig(
                    shape=ttnn.Shape([hf_config.num_experts_per_tok, USERS_PER_ROW, hf_config.hidden_size]),
                    fill_value=0,
                    device=mesh_device,
                    dtype=torch.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                decode_config["quad_ring_moe_compute"] = MoEComputeConfig(
                    output_height_shard_dim=4,
                    output_width_shard_dim=4,
                    cluster_axis=0,
                )
                decode_config["quad_ring_selective_reduce_combine"] = SelectiveReduceCombineConfig(
                    hidden_size=hf_config.hidden_size,
                    batch=batch,
                    seq=seq_len,
                    select_experts_k=hf_config.num_experts_per_tok,
                    experts=hf_config.n_routed_experts,
                    axis=0,
                    token_parallel_core_dim=4,
                    data_parallel_core_dim=4,
                    worker_cores=ttnn.experimental.get_moe_combine_cores(mesh_device),
                    mux_core_range_set=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(4, 7))}),
                )

            return decode_config
        else:
            memory_config = ttnn.DRAM_MEMORY_CONFIG
            # Construct the config
            return {
                "mesh_device": MeshDeviceStub(mesh_device.shape),
                "num_devices": mesh_device.get_num_devices(),
                "fabric_config": fabric_config,
                "num_experts_per_device": num_experts_per_device,
                "hidden_size": hf_config.hidden_size,
                "num_experts_per_tok": hf_config.num_experts_per_tok,
                "num_dispatch_devices": mesh_device.shape[0],
                "moe_gate": MoEGate.model_config(hf_config, mesh_device, mode, topk_fallback=topk_fallback),
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
                "all_to_all_combine": AllToAllCombineConfig(cluster_axis=0, memory_config=memory_config),
                "sum_experts_output_memory_config": memory_config,
                "final_output_reduce_scatter": ReduceScatterAsyncMinimalConfig(
                    cluster_axis=1,
                    dim=3,
                    memory_config=memory_config,
                ),
                "revert_tp": AllGatherAsyncConfig(
                    mesh_device=MeshDeviceStub(mesh_device.shape),
                    dim=-1,  # Last dimension
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cluster_axis=1,
                ),
            }

    @classmethod
    def _create_expert_mapping_tensor(
        cls,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        mode: str,
        num_experts_per_device: int,
    ):
        num_devices = mesh_device.get_num_devices()
        if mode == "decode":
            # optimized MoE ops only functional for quad torus
            num_dispatch_devices = mesh_device.shape[0]
            if fabric_config == ttnn.FabricConfig.FABRIC_1D_RING and num_dispatch_devices == 16:
                num_experts = num_devices * num_experts_per_device
                tp_size = mesh_device.shape[1]
                num_experts_per_cluster = num_experts // tp_size

                e = torch.arange(num_experts, dtype=torch.int32)
                torch_expert_mapping_tensor = (
                    (
                        ((e % num_experts_per_cluster) // num_experts_per_device) * tp_size
                        + (e // num_experts_per_cluster)
                    )
                    .unsqueeze(0)
                    .repeat(num_devices, 1)
                )
            else:
                torch_expert_mapping_tensor = (
                    torch.eye(num_devices, dtype=torch.int32)
                    .repeat_interleave(num_experts_per_device, dim=0)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
        else:
            torch_expert_mapping_tensor = (
                torch.eye(num_devices, dtype=torch.int32)
                .repeat_interleave(num_experts_per_device, dim=0)
                .unsqueeze(0)
                .unsqueeze(0)
            )

        expert_mapping_tensor = ttnn.from_torch(
            torch_expert_mapping_tensor,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        return expert_mapping_tensor

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        topk_fallback: bool = False,
    ) -> ModelDecodeConfig:
        return cls.model_config(hf_config, mesh_device, fabric_config, "decode", topk_fallback=topk_fallback)

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        topk_fallback: bool = False,
    ) -> ModelPrefillConfig:
        return cls.model_config(hf_config, mesh_device, fabric_config, "prefill", topk_fallback=topk_fallback)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
        batch_size_per_device = x.shape[
            -2
        ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device
        batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size

        # Note: all_gather is handled by the caller (decoder block or test)

        # MoE Gate
        topk_experts_weights, topk_experts_indices = cls._fwd_moe_gate(x, cfg)

        # MOE
        post_combine_output_tensor = cls._fwd_decode_moe(
            x,
            topk_experts_indices,
            topk_experts_weights,
            cfg,
            batch_size_per_device,
            batch_size,
            seq_len,
        )

        # Note: sum_experts and reduce_scatter is handled by the caller (decoder block or test)
        return post_combine_output_tensor

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        # Chunk the full MoE prefill path at 16K tokens to avoid OOM.
        # Use global token count (local seq_len * num_dispatch_devices) to decide.
        chunk_tokens = int(cfg.get("prefill_chunk_size", 16384))
        num_dispatch_devices = int(cfg.get("num_dispatch_devices", 1))
        global_tokens = x.shape[2] * num_dispatch_devices
        if global_tokens > chunk_tokens:
            chunk_size = max(1, chunk_tokens // max(1, num_dispatch_devices))
            return cls._forward_chunked_prefill(x, cfg, chunk_size)
        return cls._forward_prefill_impl(x, cfg)

    @classmethod
    def _forward_chunked_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig, chunk_size: int) -> ttnn.Tensor:
        chunk_size = max(1, chunk_size)
        _, _, seq_len, _ = x.shape
        output_chunks: list[ttnn.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            x_chunk = ttnn.slice(x, [0, 0, start, 0], [x.shape[0], x.shape[1], end, x.shape[3]])
            output_chunks.append(cls._forward_impl(x_chunk, cfg))
            ttnn.deallocate(x_chunk)

        if len(output_chunks) == 1:
            return output_chunks[0]
        output = ttnn.concat(output_chunks, dim=2)
        for chunk in output_chunks:
            ttnn.deallocate(chunk)
        return output

    @classmethod
    def _forward_prefill_impl(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        # Validate input dimensions
        hidden_size = cfg["hidden_size"]
        mesh_device = cfg.get("mesh_device")
        tp_size = mesh_device.shape[1] if mesh_device else 1

        x_dim = x.shape[-1]
        expected_dims = [hidden_size, hidden_size // tp_size] if tp_size > 1 else [hidden_size]

        if x_dim not in expected_dims:
            raise ValueError(
                f"MoE: Unexpected input dimension {x_dim}. Expected one of {expected_dims}. "
                f"(hidden_size={hidden_size}, tp_size={tp_size})"
            )

        seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
        batch_size_per_device = x.shape[
            -2
        ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device
        batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size

        # Note: all_gather is handled by the caller (decoder block or test)

        # MoE Gate
        topk_experts_weights, topk_experts_indices = cls._fwd_moe_gate(x, cfg)

        # MOE
        post_combine_output_tensor = cls._fwd_prefill_moe(
            x,
            topk_experts_indices,
            topk_experts_weights,
            cfg,
            batch_size_per_device,
            batch_size,
            seq_len,
        )

        # Note: sum_experts and reduce_scatter is handled by the caller (decoder block or test)
        return post_combine_output_tensor

    @classmethod
    def _fwd_moe_gate(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return MoEGate.forward(x, cfg["moe_gate"])

    @classmethod
    def _fwd_repeat_permute_expert_weights(
        cls, topk_experts_weights: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig
    ) -> ttnn.Tensor:
        topk_experts_weights_rm = ttnn.to_layout(topk_experts_weights, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_weights_rm = ttnn.repeat(topk_experts_weights_rm, **cfg["topk_weights_repeat"])
        topk_experts_weights_rm = ttnn.permute(topk_experts_weights_rm, (3, 1, 2, 0))
        topk_experts_weights = ttnn.to_layout(topk_experts_weights_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_experts_weights_rm)
        return topk_experts_weights

    @classmethod
    def _fwd_decode_moe(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: RunDecodeConfig | RunPrefillConfig,
        batch_size_per_device: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(
            x_rm,
            shape=(batch_size_per_device, 1, seq_len, cfg["hidden_size"]),
        )

        if cfg["fabric_config"] == ttnn.FabricConfig.FABRIC_1D_RING and cfg["num_dispatch_devices"] == 16:
            ccl = cfg["ccl"]

            # L1 sharded topk_experts_weights need to be deallocated before moe_compute
            # configure weights for post combine scaling prior to that deallocation, and store in DRAM
            permuted_topk_experts_weights = ttnn.permute(
                topk_experts_weights, (3, 1, 0, 2), memory_config=ttnn.L1_MEMORY_CONFIG
            )
            permuted_topk_experts_weights = ttnn.to_layout(
                permuted_topk_experts_weights, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # NOTE: needs to run prior to all_to_all_dispatch_metadata
            preallocated_combine_output = ttnn.moreh_full(**cfg["quad_ring_moreh_full"])

            (
                dispatch_output_sparse_buffer,
                dispatch_output_expert_indices,
                dispatch_output_expert_scores,
            ) = ttnn.experimental.all_to_all_dispatch_metadata(
                x_rm,
                topk_experts_indices,  # TODO: (GR) need to change shard spec of aho gate module output (once it's merged)
                topk_experts_weights,  # TODO: (GR) need to change shard spec of aho gate module output (once it's merged)
                cfg["expert_mapping_tensor"],
                **ccl.populate_all_to_all_dispatch_metadata_args(cfg["quad_ring_all_to_all_dispatch_metadata"]),
            )

            # deallocation required in order to free up L1 space for moe_compute
            ttnn.deallocate(x_rm)
            ttnn.deallocate(topk_experts_indices)
            ttnn.deallocate(topk_experts_weights)

            # NOTE: we are actively working on fusing moe_compute and selective_reduce_combine

            w0_w1 = ()  # TODO: (GR)
            w2 = ()  # TODO: (GR)
            layer_id = 0  # TODO: (GR)
            (
                compute_output_token_counts,
                compute_output_dense_expert_activation,
                compute_ouput_dense_e_t,
                _,  # tile layout output of selective tilize (same buffer as output)
                compute_output,
            ) = ttnn.experimental.moe_compute(
                dispatch_output_sparse_buffer,
                dispatch_output_expert_indices,
                dispatch_output_expert_scores,
                cfg["expert_mapping_tensor"],
                w0_w1,
                w2,
                layer_id=layer_id,
                **cfg["quad_ring_moe_compute"],
            )

            # NOTE: can't deallocate dispatch output tensors as they are preallocated and reused across layers

            combine_output = ttnn.experimental.selective_reduce_combine(
                compute_output,
                compute_output_dense_expert_activation,
                compute_ouput_dense_e_t,
                compute_output_token_counts,
                output_tensor=preallocated_combine_output,
                **ccl.populate_selective_reduce_combine_args(cfg["quad_ring_selective_reduce_combine"]),
            )

            ttnn.deallocate(compute_output)
            ttnn.deallocate(compute_output_dense_expert_activation)
            ttnn.deallocate(compute_ouput_dense_e_t)
            ttnn.deallocate(compute_output_token_counts)

            combine_output = ttnn.to_layout(
                combine_output,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            combine_output = ttnn.unsqueeze(combine_output, dim=1)

            post_combine_output_tensor = ttnn.mul(
                combine_output, permuted_topk_experts_weights, **cfg["mul_experts_output_with_weights"]
            )
        else:
            # Repeat + Permute Expert weights
            topk_experts_weights = cls._fwd_repeat_permute_expert_weights(topk_experts_weights, cfg)

            topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
            topk_experts_indices_rm = ttnn.reshape(
                topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"])
            )

            all_to_all_dispatch_output_tensors, all_to_all_dispatch_metadata_tensors = ttnn.all_to_all_dispatch(
                x_rm,
                topk_experts_indices_rm,
                cfg["expert_mapping_tensor"],
                **cfg["all_to_all_dispatch"],
            )

            post_all_to_all_dispatch_output = ttnn.reshape(
                all_to_all_dispatch_output_tensors, shape=(1, 1, batch_size * seq_len, cfg["hidden_size"])
            )
            post_all_to_all_dispatch_output = ttnn.to_layout(post_all_to_all_dispatch_output, ttnn.TILE_LAYOUT)
            # repeat remap_topk_mask for the num_tokens known at runtime

            remap_topk_mask = ttnn.repeat(
                cfg["remap_topk_mask"], ttnn.Shape((1, batch_size_per_device, 1, 1))
            )  # TODO: move to static path

            _, sparsity_t = ttnn.moe_expert_token_remap(
                remap_topk_mask,
                cfg["expert_mapping_tensor"],
                all_to_all_dispatch_metadata_tensors,
                reduction_size=cfg["sparsity_block_size"],
            )

            experts_output = MoEExperts._forward(post_all_to_all_dispatch_output, sparsity_t, cfg["moe_experts"])
            ttnn.deallocate(post_all_to_all_dispatch_output)

            experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
            experts_output = ttnn.reshape(
                experts_output, shape=(cfg["num_experts_per_device"], batch_size, seq_len, cfg["hidden_size"])
            )

            all_to_all_combine_output_tensors = ttnn.all_to_all_combine(
                experts_output,
                all_to_all_dispatch_metadata_tensors,
                cfg["expert_mapping_tensor"],
                **cfg["all_to_all_combine"],
            )

            post_combine_output_tensor = ttnn.reshape(
                all_to_all_combine_output_tensors,
                shape=(cfg["num_experts_per_tok"], 1, batch_size_per_device * seq_len, cfg["hidden_size"]),
            )
            post_combine_output_tensor = ttnn.to_layout(post_combine_output_tensor, ttnn.TILE_LAYOUT)

            post_combine_output_tensor = ttnn.mul(
                post_combine_output_tensor, topk_experts_weights, **cfg["mul_experts_output_with_weights"]
            )

        return post_combine_output_tensor

    @classmethod
    def _fwd_prefill_moe(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: RunDecodeConfig | RunPrefillConfig,
        batch_size_per_device: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
        # Repeat + Permute Expert weights
        topk_experts_weights = cls._fwd_repeat_permute_expert_weights(topk_experts_weights, cfg)

        tokens = batch_size * seq_len
        x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(
            x_rm,
            shape=(batch_size_per_device, 1, seq_len, cfg["hidden_size"]),
        )

        topk_experts_indices_rm = ttnn.to_layout(topk_experts_indices, ttnn.ROW_MAJOR_LAYOUT)
        topk_experts_indices_rm = ttnn.reshape(
            topk_experts_indices_rm, shape=(batch_size_per_device, 1, seq_len, cfg["num_experts_per_tok"])
        )

        # Chunk along local batch dimension to keep all_to_all_dispatch output small in prefill.
        chunk_size = min(batch_size_per_device, max(1, cfg.get("moe_chunk_size", batch_size_per_device)))
        output_chunks: list[ttnn.Tensor] = []

        def _slice_topk_weights(batch_start: int, batch_end: int) -> ttnn.Tensor:
            token_start = batch_start * seq_len
            token_end = batch_end * seq_len
            return ttnn.slice(
                topk_experts_weights,
                [0, 0, token_start, 0],
                [cfg["num_experts_per_tok"], 1, token_end, cfg["hidden_size"]],
            )

        for batch_start in range(0, batch_size_per_device, chunk_size):
            batch_end = min(batch_start + chunk_size, batch_size_per_device)
            batch_chunk = batch_end - batch_start
            batch_size_chunk = batch_chunk * cfg["num_dispatch_devices"]

            x_chunk = ttnn.slice(
                x_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["hidden_size"]],
            )
            topk_indices_chunk = ttnn.slice(
                topk_experts_indices_rm,
                [batch_start, 0, 0, 0],
                [batch_end, 1, seq_len, cfg["num_experts_per_tok"]],
            )

            all_to_all_dispatch_output_tensors, all_to_all_dispatch_metadata_tensors = ttnn.all_to_all_dispatch(
                x_chunk,
                topk_indices_chunk,
                cfg["expert_mapping_tensors"],
                **cfg["all_to_all_dispatch"],
            )
            ttnn.deallocate(x_chunk)
            ttnn.deallocate(topk_indices_chunk)

            dispatch_chunk = ttnn.reshape(
                all_to_all_dispatch_output_tensors,
                shape=(1, 1, batch_size_chunk * seq_len, cfg["hidden_size"]),
            )
            dispatch_chunk = ttnn.repeat(dispatch_chunk, **cfg["activations_repeat"])
            dispatch_chunk = ttnn.to_layout(dispatch_chunk, ttnn.TILE_LAYOUT)
            ttnn.deallocate(all_to_all_dispatch_output_tensors)

            experts_output = MoEExperts._forward(dispatch_chunk, cfg["moe_experts"])
            ttnn.deallocate(dispatch_chunk)

            experts_output = ttnn.to_layout(experts_output, ttnn.ROW_MAJOR_LAYOUT)
            experts_output = ttnn.reshape(
                experts_output, shape=(cfg["num_experts_per_device"], batch_size_chunk, seq_len, cfg["hidden_size"])
            )

            all_to_all_dispatch_metadata_tensors = ttnn.reshape(
                all_to_all_dispatch_metadata_tensors,
                shape=(1, batch_size_chunk, seq_len, cfg["num_experts_per_tok"]),
            )

            all_to_all_combine_output_tensors = ttnn.all_to_all_combine(
                experts_output,
                all_to_all_dispatch_metadata_tensors,
                cfg["expert_mapping_tensors"],
                **cfg["all_to_all_combine"],
            )
            ttnn.deallocate(experts_output)
            ttnn.deallocate(all_to_all_dispatch_metadata_tensors)

            post_combine_output_tensor = ttnn.reshape(
                all_to_all_combine_output_tensors,
                shape=(cfg["num_experts_per_tok"], 1, batch_chunk * seq_len, cfg["hidden_size"]),
            )
            post_combine_output_tensor = ttnn.to_layout(post_combine_output_tensor, ttnn.TILE_LAYOUT)

            topk_weights_chunk = _slice_topk_weights(batch_start, batch_end)
            post_combine_output_tensor = ttnn.mul(
                post_combine_output_tensor, topk_weights_chunk, **cfg["mul_experts_output_with_weights"]
            )
            ttnn.deallocate(topk_weights_chunk)

            output_chunks.append(post_combine_output_tensor)

        if len(output_chunks) == 1:
            post_combine_output_tensor = output_chunks[0]
        else:
            post_combine_output_tensor = ttnn.concat(output_chunks, dim=2)
            for chunk in output_chunks:
                ttnn.deallocate(chunk)

        ttnn.deallocate(x_rm)
        ttnn.deallocate(topk_experts_indices_rm)
        return post_combine_output_tensor

    @classmethod
    def _fwd_all_gather(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        return ttnn.experimental.all_gather_async(x, **cfg["ccl"].populate_all_gather_runtime_args(cfg["revert_tp"]))

    @classmethod
    def _fwd_reduce_scatter(
        cls, post_combine_output_tensor: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig, ccl: CCL
    ) -> ttnn.Tensor:
        # Use standard reduce_scatter (composite fallback) to avoid shard shape constraints
        # encountered by the minimal async path in some decode configurations.
        rs_cfg = cfg["final_output_reduce_scatter"]
        rs_kwargs = {
            "dim": rs_cfg["dim"],
            "cluster_axis": rs_cfg.get("cluster_axis"),
            "subdevice_id": rs_cfg.get("subdevice_id"),
            "memory_config": rs_cfg.get("memory_config"),
            "intermediate_memory_config": rs_cfg.get("intermediate_memory_config"),
            "num_links": rs_cfg.get("num_links"),
            "topology": rs_cfg.get("topology"),
            "chunks_per_sync": rs_cfg.get("chunks_per_sync"),
            "num_workers_per_link": rs_cfg.get("num_workers_per_link"),
            "num_buffers_per_channel": rs_cfg.get("num_buffers_per_channel"),
        }
        rs_kwargs = {k: v for k, v in rs_kwargs.items() if v is not None}
        return ttnn.reduce_scatter(post_combine_output_tensor, **rs_kwargs)

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig, handle_tensor_parallel: bool = False) -> ttnn.Tensor:
        # Handle all_gather if tensor parallel is enabled
        if handle_tensor_parallel:
            x = cls._fwd_all_gather(x, cfg)

        # Run the forward pass
        output = cls.forward_decode(x, cfg)

        # Handle sum_experts and reduce_scatter if tensor parallel is enabled
        if handle_tensor_parallel:
            ccl = cfg["ccl"]
            tp_size = cfg["mesh_device"].shape[1]

            if cfg["fabric_config"] == ttnn.FabricConfig.FABRIC_1D_RING and tp_size == 8:
                output = ttnn.experimental.deepseek_moe_fast_reduce_nc(
                    output,
                    dim=0,
                    split_size=output.shape[-1] // tp_size,
                    output_memory_config=cfg["ring_sum_experts_output_memory_config"],
                )
                output = ttnn.experimental.deepseek_moe_reduce_scatter(
                    output, **cfg["ring_final_output_reduce_scatter"]
                )
            else:
                output = ttnn.sum(output, dim=0, keepdim=True, memory_config=cfg["sum_experts_output_memory_config"])
                output = cls._fwd_reduce_scatter(output, cfg, ccl)

        return output

    @classmethod
    def forward_prefill(
        cls, x: ttnn.Tensor, cfg: RunPrefillConfig, handle_tensor_parallel: bool = False
    ) -> ttnn.Tensor:
        # Handle all_gather if tensor parallel is enabled
        if handle_tensor_parallel:
            x = cls._fwd_all_gather(x, cfg)

        # Run the forward pass
        output = cls.forward_prefill(x, cfg)

        # Handle sum_experts and reduce_scatter if tensor parallel is enabled
        if handle_tensor_parallel:
            ccl = cfg["ccl"]
            output = ttnn.sum(output, dim=0, keepdim=True, memory_config=cfg["sum_experts_output_memory_config"])
            output = cls._fwd_reduce_scatter(output, cfg, ccl)

        return output
