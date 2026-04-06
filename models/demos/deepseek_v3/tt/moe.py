# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from time import perf_counter
from typing import Iterable

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
    DeepseekMoEReduceScatterConfig,
    MeshDeviceStub,
    MulConfig,
    ReduceScatterAsyncMinimalConfig,
    RepeatConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, get_shared_experts_per_device
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


def cluster_distance(d0: int, d1: int, mesh_shape: tuple[int, int], cluster_axis: int) -> int | None:
    """Calculate Manhattan distance between two devices along the cluster axis.

    Returns None if devices are not on the same cluster line, otherwise returns
    the distance along the cluster axis.
    """
    c0 = (d0 // mesh_shape[1], d0 % mesh_shape[1])
    c1 = (d1 // mesh_shape[1], d1 % mesh_shape[1])

    return None if c0[1 - cluster_axis] != c1[1 - cluster_axis] else abs(c0[cluster_axis] - c1[cluster_axis])


def map_shared_experts(
    expert_mapping_tensor: torch.Tensor,
    shared_expert_ids_to_devices: dict[int, list[int]],
    mesh_shape: Iterable[int],
    cluster_axis: int,
) -> torch.Tensor:
    """
    Map shared experts to their nearest on-axis device for dispatch operations.

    This function extends the expert mapping tensor to include shared experts by determining
    the optimal device assignment for each shared expert based on cluster topology. For each
    dispatching device, it selects the nearest receiving device on the same cluster axis
    that has the shared expert.

    Args:
        expert_mapping_tensor: 2D tensor of shape [devices, routed_experts] containing
            linearized mesh coordinates of the device owning each expert.
        shared_expert_ids_to_devices: Dictionary mapping shared expert IDs to lists of
            device IDs where they are replicated. Expert IDs must be contiguous
            continuations of routed expert IDs.
        mesh_shape: Tuple/list representing the dimensions of the device mesh (e.g., (4, 4)).
        cluster_axis: Axis along which devices are clustered (0 or 1). Determines the
            direction of nearest-neighbor search for shared experts.

    Returns:
        torch.Tensor: Extended mapping tensor of shape [devices, routed_experts + shared_experts]
            where each entry [d, e] contains the device ID that device d should dispatch
            expert e to. For shared experts, this is the nearest device on the same
            cluster axis that has the expert.

    Raises:
        RuntimeError: If shared experts are not distributed evenly across devices.
        RuntimeError: If more than 3 experts per device would result (current limitation).
        RuntimeError: If shared expert IDs are not contiguous with routed expert IDs.

    Notes:
        - The function uses Manhattan distance along the cluster axis to find nearest devices.
        - If no device with the shared expert is on the same cluster axis, a default
          device is selected (the first in the list).
        - This mapping is critical for efficient MoE dispatch operations in distributed systems.
    """

    # assuming [devices, experts] -> linearized mesh coordinate of owning device
    assert len(expert_mapping_tensor.shape) == 2

    devices = expert_mapping_tensor.shape[0]
    routed_experts = expert_mapping_tensor.shape[1]

    routed_experts_per_device = routed_experts // devices

    shared_experts = len(shared_expert_ids_to_devices)

    shared_experts_per_device = get_shared_experts_per_device(shared_expert_ids_to_devices, devices)

    if not len(set(shared_experts_per_device)) == 1:
        raise RuntimeError("Shared Experts must be distributed such that all devices have an equal number of experts")

    # this is a fairly soft limitation at the moment, small changes to moe_compute are required to lift it
    if shared_experts_per_device[0] + routed_experts_per_device > 3:
        raise RuntimeError("At the moment MoE supports up to 3 experts per device")

    if list(range(routed_experts)) + sorted([se for se in shared_expert_ids_to_devices]) != list(
        range(routed_experts + shared_experts)
    ):
        raise RuntimeError("Shared expert IDs should be a contigious continuation of routed expert IDs ")

    routed_and_shared_expert_mapping = torch.cat(
        [expert_mapping_tensor, torch.zeros((devices, shared_experts), dtype=expert_mapping_tensor.dtype)], dim=1
    )
    for disp_d in range(devices):
        for se, rec_ds in shared_expert_ids_to_devices.items():
            min_distance = mesh_shape[cluster_axis] + 1

            # just pick one as the default case. If none of the device assignments are on the same cluster axis as
            # disp_d then this expert will also get skipped by dispatch on disp_d
            routed_and_shared_expert_mapping[disp_d, se] = rec_ds[0]
            for rec_d in rec_ds:
                distance = cluster_distance(disp_d, rec_d, mesh_shape, cluster_axis)
                if distance is not None and distance < min_distance:
                    routed_and_shared_expert_mapping[disp_d, se] = rec_d
                    min_distance = distance

    return routed_and_shared_expert_mapping


class MoE(SharedStateAddOn, AbstractModule):
    """MoE module from DeepSeek-R1.
    See the `AbstractModule` docstring for usage info.
    """

    PREFILL_TOKEN_CHUNK_SIZE = 16384
    PREFILL_BATCH_CHUNK_SIZE = 256

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
        num_devices = mesh_device.get_num_devices()
        num_experts_per_device = MoEExperts._get_num_experts_per_device(hf_config, mesh_device)
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
            "expert_mapping_tensors": expert_mapping_tensors,
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
        num_links = ccl.get_max_links(axis=0)
        return {
            "all_to_all_dispatch": {
                "num_links": num_links,
            },
            "all_to_all_combine": {
                "num_links": num_links,
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
        batch_size_per_row: int,
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

        if mode == "decode":
            memory_config = ttnn.L1_MEMORY_CONFIG

            HIDDEN_SIZE = hf_config.hidden_size
            TP_SIZE = mesh_device.shape[1]

            shard_core_grid = ttnn.CoreGrid(y=7, x=4)
            per_core_width = (HIDDEN_SIZE // TP_SIZE) // shard_core_grid.num_cores
            input_output_memory_config = ttnn.create_sharded_memory_config(
                shape=(
                    ttnn.core.roundup(batch_size_per_row, ttnn.TILE_SIZE),
                    ttnn.core.roundup(per_core_width, ttnn.TILE_SIZE),
                ),
                core_grid=shard_core_grid,
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

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
                    batch_size_per_row,
                    HIDDEN_SIZE,
                    TP_SIZE,
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
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        batch_size_per_row: int,
        topk_fallback: bool = False,
    ) -> ModelDecodeConfig:
        return cls.model_config(
            hf_config,
            mesh_device,
            fabric_config,
            "decode",
            batch_size_per_row=batch_size_per_row,
            topk_fallback=topk_fallback,
        )

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        fabric_config: ttnn.FabricConfig,
        topk_fallback: bool = False,
    ) -> ModelPrefillConfig:
        return cls.model_config(
            hf_config,
            mesh_device,
            fabric_config,
            "prefill",
            batch_size_per_row=USERS_PER_ROW,
            topk_fallback=topk_fallback,
        )

    @classmethod
    def forward(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        # Chunk the full MoE prefill path at 16K tokens to avoid OOM.
        # Use global token count (local seq_len * num_dispatch_devices) to decide.
        chunk_tokens = cls.PREFILL_TOKEN_CHUNK_SIZE
        num_dispatch_devices = cfg["num_dispatch_devices"]
        global_tokens = x.shape[2] * num_dispatch_devices
        if global_tokens > chunk_tokens:
            chunk_size = max(1, chunk_tokens // max(1, num_dispatch_devices))
            return cls._forward_chunked_prefill(x, cfg, chunk_size)
        return cls._forward_impl(x, cfg)

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
    def _forward_impl(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> ttnn.Tensor:
        # Validate input dimensions
        hidden_size = cfg["hidden_size"]
        mesh_device = cfg["mesh_device"]
        tp_size = mesh_device.shape[1]

        x_dim = x.shape[-1]
        expected_dims = [hidden_size, hidden_size // tp_size] if tp_size > 1 else [hidden_size]

        if x_dim not in expected_dims:
            raise ValueError(
                f"MoE: Unexpected input dimension {x_dim}. Expected one of {expected_dims}. "
                f"(hidden_size={hidden_size}, tp_size={tp_size})"
            )

        # breakpoint()
        ccl = cfg["ccl"]  # CCL runtime initialization in execution order
        seq_len = 1  # a2a dispatch and combine require DP=num_dispatch_devices, hence in prefill for bs=1, we interchange the seq_len with batch_size dimensions
        batch_size_per_device = x.shape[
            -2
        ]  # Input is expected to be DP. In prefill, this is equivalent to seq_len_per_device
        batch_size = batch_size_per_device * cfg["num_dispatch_devices"]  # Global batch size

        # Note: all_gather is handled by the caller (decoder block or test)

        # MoE Gate
        topk_experts_weights, topk_experts_indices = cls._fwd_moe_gate(x, cfg)

        # MOE
        post_combine_output_tensor = cls._fwd_moe(
            x,
            topk_experts_indices,
            topk_experts_weights,
            cfg,
            batch_size_per_device,
            batch_size,
            seq_len,
        )
        ttnn.deallocate(topk_experts_weights)
        ttnn.deallocate(topk_experts_indices)

        # Note: sum_experts and reduce_scatter is handled by the caller (decoder block or test)

        return post_combine_output_tensor

    @classmethod
    def _fwd_moe_gate(cls, x: ttnn.Tensor, cfg: RunDecodeConfig | RunPrefillConfig) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return MoEGate.forward(x, cfg["moe_gate"])

    @classmethod
    def _fwd_moe(
        cls,
        x: ttnn.Tensor,
        topk_experts_indices: ttnn.Tensor,
        topk_experts_weights: ttnn.Tensor,
        cfg: RunDecodeConfig | RunPrefillConfig,
        batch_size_per_device: int,
        batch_size: int,
        seq_len: int,
    ) -> ttnn.Tensor:
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

        # Chunk along local batch dimension to keep prefill intermediates (especially topk tilize) small.
        chunk_size = min(batch_size_per_device, cls.PREFILL_BATCH_CHUNK_SIZE)
        output_chunks: list[ttnn.Tensor] = []

        def _slice_topk_weights(batch_start: int, batch_end: int) -> ttnn.Tensor:
            token_start = batch_start * seq_len
            token_end = batch_end * seq_len
            topk_weights_chunk = ttnn.slice(
                topk_experts_weights,
                [0, 0, token_start, 0],
                [1, 1, token_end, cfg["num_experts_per_tok"]],
            )
            topk_weights_chunk_rm = ttnn.to_layout(topk_weights_chunk, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(topk_weights_chunk)
            topk_weights_chunk_rm = ttnn.repeat(topk_weights_chunk_rm, **cfg["topk_weights_repeat"])
            topk_weights_chunk_rm = ttnn.permute(topk_weights_chunk_rm, (3, 1, 2, 0))
            topk_weights_chunk = ttnn.to_layout(topk_weights_chunk_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(topk_weights_chunk_rm)
            return topk_weights_chunk

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
            dispatch_chunk_rm = dispatch_chunk
            dispatch_chunk = ttnn.to_layout(dispatch_chunk_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(dispatch_chunk_rm)
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
            post_combine_output_tensor_rm = post_combine_output_tensor
            post_combine_output_tensor = ttnn.to_layout(post_combine_output_tensor_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(all_to_all_combine_output_tensors)

            topk_weights_chunk = _slice_topk_weights(batch_start, batch_end)
            post_combine_weighted_output_tensor = ttnn.mul(
                post_combine_output_tensor, topk_weights_chunk, **cfg["mul_experts_output_with_weights"]
            )
            ttnn.deallocate(post_combine_output_tensor)
            ttnn.deallocate(topk_weights_chunk)

            post_combine_output_tensor = ttnn.sum(post_combine_weighted_output_tensor, dim=0, keepdim=True)
            ttnn.deallocate(post_combine_weighted_output_tensor)
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
    def forward_prefill(
        cls, x: ttnn.Tensor, cfg: RunPrefillConfig, handle_tensor_parallel: bool = False
    ) -> ttnn.Tensor:
        # Handle all_gather if tensor parallel is enabled
        if handle_tensor_parallel:
            x = cls._fwd_all_gather(x, cfg)

        # Run the forward pass
        output = cls.forward(x, cfg)

        # Handle sum_experts and reduce_scatter if tensor parallel is enabled
        if handle_tensor_parallel:
            ccl = cfg["ccl"]
            output = ttnn.sum(output, dim=0, keepdim=True, memory_config=cfg["sum_experts_output_memory_config"])
            output = cls._fwd_reduce_scatter(output, cfg, ccl)

        return output

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig, handle_tensor_parallel: bool = False) -> ttnn.Tensor:
        # Handle all_gather if tensor parallel is enabled
        if handle_tensor_parallel:
            x = cls._fwd_all_gather(x, cfg)

        # Run the forward pass
        output = cls.forward(x, cfg)

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
