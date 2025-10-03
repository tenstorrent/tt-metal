# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import math
from pathlib import Path
from typing import Sequence

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.utility_functions import nearest_y
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.rms_norm.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    ReduceScatterAsyncMinimalConfig,
    ReshardConfig,
    SavedWeight,
)
from models.demos.deepseek_v3.utils.config_helpers import (
    MAX_BATCH_SIZE,
    dequantize,
    even_int_div,
    get_mesh_coords,
    get_state_dicts,
    shard_and_save,
    sub_state_dicts,
)
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.tt_transformers.tt.common import PagedAttentionConfig


class MLA(AbstractModule):
    """
    Multi-Latent Attention Module for 1D tensor parallelism.
    """

    HF_TTNN_MAPPING = {
        "q_a_proj": "wq_a",
        "q_b_proj": "wq_b",
        "kv_a_proj_with_mqa": "wkv_a",
        "kv_b_proj": "wkv_b",
        "o_proj": "wo",
        "q_a_layernorm": "q_norm",
        "kv_a_layernorm": "kv_norm",
    }

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        """Convert PyTorch weights to TTNN format for 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            state_dicts: Tuple of state dictionaries containing model weights
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device
        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """
        assert cls.is_device_supported(mesh_device)

        num_shards = mesh_device.shape[0]
        weight_block_height, weight_block_width = hf_config.quantization_config["weight_block_size"]

        dim = hf_config.hidden_size
        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim
        q_lora_rank = hf_config.q_lora_rank
        q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        # Norm weights
        norm_weight_configs = {
            ttnn_name: RMSNorm.convert_weights(
                hf_config,
                sub_state_dicts(state_dicts, f"{hf_name}."),
                output_path / ttnn_name,
                mesh_device,
            )
            for hf_name, ttnn_name in [("q_a_layernorm", "q_norm"), ("kv_a_layernorm", "kv_norm")]
        }

        # Regular non-split weights
        linear_weight_configs = {  # TODO: add dequant
            ttnn_name: {
                "input_tensor_b": cls._convert_metaweight(
                    output_path / f"{ttnn_name}.input_tensor_b",
                    dequantize(
                        get_state_dicts(state_dicts, f"{hf_name}.weight", shape, dtype=torch.float8_e4m3fn),
                        get_state_dicts(state_dicts, f"{hf_name}.weight_scale_inv", dtype=torch.float32),
                        (1, weight_block_height, weight_block_width),
                    ),
                    mesh_dims,
                    mesh_device,
                ),
            }
            for hf_name, ttnn_name, shape, mesh_dims in [
                ("q_a_proj", "wq_a", (q_lora_rank, dim), (0, -2)),
                ("q_b_proj", "wq_b", (num_heads * q_head_dim, q_lora_rank), (0, -1)),
                ("kv_a_proj_with_mqa", "wkv_a", (kv_lora_rank + qk_rope_head_dim, dim), (0, -2)),
                ("o_proj", "wo", (dim, num_heads * v_head_dim), (0, -1)),
            ]
        }

        # wkv_b (Needs Special handling!!)
        torch_weights = dequantize(
            get_state_dicts(
                state_dicts,
                f"kv_b_proj.weight",
                shape=(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank),
                dtype=torch.float8_e4m3fn,
            ),
            get_state_dicts(state_dicts, f"kv_b_proj.weight_scale_inv", dtype=torch.float32),
            (1, weight_block_height, weight_block_width),
        ).reshape(num_shards, num_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)

        torch_weights_k = torch_weights[..., :qk_nope_head_dim, :].transpose(
            -2, -1
        )  # [num_heads, kv_lora_rank, qk_nope_head_dim]
        torch_weights_v = torch_weights[..., qk_nope_head_dim:, :]  # [num_heads, v_head_dim, kv_lora_rank]

        return {
            **norm_weight_configs,
            **linear_weight_configs,
            "wkv_b1": {
                "input_tensor_b": cls._convert_metaweight(
                    output_path / "wkv_b1.input_tensor_b", torch_weights_k, (0, -3), mesh_device
                ),
            },
            "wkv_b2": {
                "input_tensor_b": cls._convert_metaweight(
                    output_path / "wkv_b2.input_tensor_b", torch_weights_v, (0, -3), mesh_device
                ),
            },
        }

    @classmethod
    def _convert_metaweight(
        cls,
        path: Path,
        torch_metaweight: torch.Tensor,
        dims: tuple[int | None, int | None],
        mesh_device: ttnn.MeshDevice,
    ) -> SavedWeight:
        return shard_and_save(
            path,
            torch_metaweight.transpose(-2, -1),
            shard_dims=dims,
            mesh_device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @classmethod
    def is_device_supported(cls, mesh_device: ttnn.Device) -> bool:
        """
        We only support 1D tensor parallelism, with TP=8

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        return tuple(mesh_device.shape)[1] == 8

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelPrefillConfig:
        """Prefill model config for an MLP with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for prefill mode
        """

        grid_size = mesh_device.compute_with_storage_grid_size()

        # Extract dimensions from HF config
        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim
        mscale = hf_config.rope_scaling["mscale"]
        rope_factor = hf_config.rope_scaling["factor"]

        input_memory_config = ttnn.DRAM_MEMORY_CONFIG

        wq_a_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wq_b_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wkv_a_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wkv_b1_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wkv_b2_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wo_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # FlashMLA
        q_chunk_size = 128  # TODO: Make dynamic?
        k_chunk_size = 128  # TODO: Make dynamic?

        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        flash_mla_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        scale = qk_head_dim**-0.5
        if rope_factor > 1.0:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            scale = scale * mscale * mscale

        flash_mla_config = {
            "head_dim_v": kv_lora_rank,
            "scale": scale,
            "program_config": sdpa_program_config,
            "compute_kernel_config": flash_mla_compute_kernel_config,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "attn_mask": None,
            "is_causal": True,
        }

        # Norms
        q_norm_config = RMSNorm.prefill_model_config(
            hf_config,
            mesh_device,
        )
        kv_norm_config = RMSNorm.prefill_model_config(
            hf_config,
            mesh_device,
        )

        # Set up CCLs

        # Q
        wq_a_rs_config = ReduceScatterAsyncMinimalConfig(
            cluster_axis=1,
            dim=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        wq_a_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_device.shape),
            cluster_axis=1,
            dim=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # KV
        wkv_a_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_device.shape),
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        wkv_a_r_config = {
            "dims": [1],
            "output": None,
            "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        }

        # WO
        wo_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_device.shape),
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        return {
            "num_heads": num_heads,
            "kv_lora_rank": kv_lora_rank,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "qk_head_dim": qk_head_dim,
            "v_head_dim": v_head_dim,
            "input_memory_config": input_memory_config,
            "wq_a": wq_a_config,
            "wq_b": wq_b_config,
            "wkv_a": wkv_a_config,
            "wkv_b1": wkv_b1_config,
            "wkv_b2": wkv_b2_config,
            "wo": wo_config,
            "flash_mla": flash_mla_config,
            "q_norm": q_norm_config,
            "kv_norm": kv_norm_config,
            "wq_a_rs_prefill": wq_a_rs_config,
            "wq_a_ag_prefill": wq_a_ag_config,
            "wkv_a_ag_prefill": wkv_a_ag_config,
            "wkv_a_r_prefill": wkv_a_r_config,
            "wo_ag_prefill": wo_ag_config,
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
    ) -> ModelDecodeConfig:
        """Generate decode operator configuration for this MLP layer.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """

        grid_size = mesh_device.compute_with_storage_grid_size()
        num_cores = grid_size.x * grid_size.y

        # Extract dimensions from HF config
        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim
        mscale = hf_config.rope_scaling["mscale"]
        rope_factor = hf_config.rope_scaling["factor"]

        mesh_shape = list(mesh_device.shape)
        num_heads_local = even_int_div(num_heads, mesh_shape[1])

        input_memory_config = ttnn.DRAM_MEMORY_CONFIG

        wq_a_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wq_b_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wkv_a_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wkv_b1_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wkv_b2_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        wo_config = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # Resharding for q_rope
        # TODO: Should be dynamic based on batch size?
        q_rope_shape = (1, MAX_BATCH_SIZE, num_heads_local, qk_rope_head_dim)
        q_rope_shard_height = nearest_y(q_rope_shape[2], ttnn.TILE_SIZE)
        q_rope_shard_width = q_rope_shape[3]
        q_rope_num_cores = q_rope_shape[1]
        q_rope_core_grid = ttnn.num_cores_to_corerangeset(q_rope_num_cores, grid_size, row_wise=True)
        q_rope_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(q_rope_shard_height, q_rope_shard_width),
            core_grid=q_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        q_rope_reshard_config = ReshardConfig(
            memory_config=q_rope_mem_cfg,
        )
        q_rope_out_reshard_config = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Resharding for kv_rope
        # TODO: Should be dynamic based on batch size?
        # TODO: Split batch when adding DP
        kv_rope_shape = (1, MAX_BATCH_SIZE, 1, qk_rope_head_dim)
        kv_rope_shard_height = nearest_y(kv_rope_shape[2], ttnn.TILE_SIZE)
        kv_rope_shard_width = kv_rope_shape[3]
        kv_rope_num_cores = kv_rope_shape[1]
        kv_rope_core_grid = ttnn.num_cores_to_corerangeset(kv_rope_num_cores, grid_size, row_wise=True)
        kv_rope_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(kv_rope_shard_height, kv_rope_shard_width),
            core_grid=kv_rope_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        kv_rope_reshard_config = ReshardConfig(
            memory_config=kv_rope_mem_cfg,
        )
        kv_rope_out_reshard_config = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Resharding for kvpe
        kvpe_shape = (1, even_int_div(MAX_BATCH_SIZE, mesh_shape[1]), 1, kv_lora_rank + qk_rope_head_dim)
        kvpe_shard_height = nearest_y(kvpe_shape[2], ttnn.TILE_SIZE)
        kvpe_shard_width = kvpe_shape[3]
        kvpe_num_cores = kvpe_shape[1]
        kvpe_core_grid = ttnn.num_cores_to_corerangeset(kvpe_num_cores, grid_size, row_wise=True)
        kvpe_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(kvpe_shard_height, kvpe_shard_width),
            core_grid=kvpe_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        kvpe_reshard_config = ReshardConfig(
            memory_config=kvpe_mem_cfg,
        )

        # FlashMLA
        q_chunk_size = 0  # Unused in decode mode
        k_chunk_size = 128  # TODO: Make dynamic?

        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        flash_mla_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        q_num_cores = num_cores
        q_num_cores = min(even_int_div(MAX_BATCH_SIZE, mesh_shape[1]) * num_heads, q_num_cores)
        block_height = nearest_y(
            (even_int_div(MAX_BATCH_SIZE, mesh_shape[1]) * num_heads) // q_num_cores, ttnn.TILE_SIZE
        )
        block_width = kv_lora_rank + qk_rope_head_dim

        q_core_grid = ttnn.num_cores_to_corerangeset(q_num_cores, grid_size, row_wise=True)
        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, block_width),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        flash_mla_out_mem_config = ttnn.create_sharded_memory_config(
            shape=(block_height, kv_lora_rank),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        scale = qk_head_dim**-0.5
        if rope_factor > 1.0:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            scale = scale * mscale * mscale

        flash_mla_reshard_config = ReshardConfig(
            memory_config=q_mem_config,
        )
        flash_mla_config = {
            "head_dim_v": kv_lora_rank,
            "scale": scale,
            "program_config": sdpa_program_config,
            "compute_kernel_config": flash_mla_compute_kernel_config,
            "memory_config": flash_mla_out_mem_config,
        }
        flash_mla_out_reshard_config = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Norms
        q_norm_config = RMSNorm.decode_model_config(
            hf_config,
            mesh_device,
        )
        kv_norm_config = RMSNorm.decode_model_config(
            hf_config,
            mesh_device,
        )

        # Set up CCLs

        # Q
        wq_a_rs_config = ReduceScatterAsyncMinimalConfig(
            cluster_axis=1,
            dim=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        wq_a_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # Q all-to-all
        wq_a2a_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        wq_a2a_rs_config = ReduceScatterAsyncMinimalConfig(
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # KV
        wkv_a_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        wkv_a_r_config = {
            "dims": [1],
            "output": None,
            "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        }
        wkv_a_rs_config = ReduceScatterAsyncMinimalConfig(
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # FlashMLA all-to-all
        flash_mla_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        flash_mla_rs_config = ReduceScatterAsyncMinimalConfig(
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # WO
        wo_ag_config = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        return {
            "num_heads": num_heads,
            "kv_lora_rank": kv_lora_rank,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "qk_head_dim": qk_head_dim,
            "v_head_dim": v_head_dim,
            "input_memory_config": input_memory_config,
            "wq_a": wq_a_config,
            "wq_b": wq_b_config,
            "wkv_a": wkv_a_config,
            "wkv_b1": wkv_b1_config,
            "wkv_b2": wkv_b2_config,
            "wo": wo_config,
            "q_rope_reshard": q_rope_reshard_config,
            "q_rope_out_reshard": q_rope_out_reshard_config,
            "kv_rope_reshard": kv_rope_reshard_config,
            "kv_rope_out_reshard": kv_rope_out_reshard_config,
            "kvpe_reshard": kvpe_reshard_config,
            "flash_mla_reshard": flash_mla_reshard_config,
            "flash_mla": flash_mla_config,
            "flash_mla_out_reshard": flash_mla_out_reshard_config,
            "q_norm": q_norm_config,
            "kv_norm": kv_norm_config,
            "wq_a_rs_decode": wq_a_rs_config,
            "wq_a_ag_decode": wq_a_ag_config,
            "wq_a2a_ag_decode": wq_a2a_ag_config,
            "wq_a2a_rs_decode": wq_a2a_rs_config,
            "wkv_a_ag_decode": wkv_a_ag_config,
            "wkv_a_r_decode": wkv_a_r_config,
            "wkv_a_rs_decode": wkv_a_rs_config,
            "flash_mla_ag_decode": flash_mla_ag_config,
            "flash_mla_rs_decode": flash_mla_rs_config,
            "wo_ag_decode": wo_ag_config,
        }

    @classmethod
    def get_valid_paged_config(
        cls, max_seq_len: int, batch_size: int, dp_factor: int, block_size: int = ttnn.TILE_SIZE
    ) -> PagedAttentionConfig:
        """Get a valid paged attention configuration for MLA.

        This function also calculates max_num_blocks such that each user will have max_seq_len available.
        For DP, the max_num_blocks is divided by the batch size of the DP shard, not the total batch size.

        Args:
            max_seq_len: Maximum sequence length
            batch_size: Batch size for the model
            block_size: Block size for paged attention (default is TILE_SIZE)

        Returns:
            A PagedAttentionConfig object with valid parameters
        """
        assert max_seq_len % block_size == 0, f"max_seq_len {max_seq_len} must be divisible by block_size {block_size}."
        assert (
            block_size % ttnn.TILE_SIZE == 0
        ), f"block_size {block_size} must be a multiple of TILE_SIZE {ttnn.TILE_SIZE}."

        batch_per_shard = even_int_div(batch_size, dp_factor)
        max_num_blocks = even_int_div(
            max_seq_len * batch_per_shard, block_size
        )  # Such that each user will have max_seq_len available

        return PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    @classmethod
    def create_page_table(
        cls,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        page_table: torch.Tensor | None = None,
        batch_size: int = MAX_BATCH_SIZE,
    ) -> ttnn.Tensor:
        """Helper function to allocate the page table for MLA on device.

        When doing DP, this function replicates the page table across DP shards.
        Assumptions:
            - If user X on DP shard 1 is on position N, with page id P,
                and if user X on DP shard 2 is also on position N, it will also be on page id P.
                As such, the max_num_blocks is only cut by the batch size of the DP shard, not the total batch size.

        Args:
            page_table: A torch tensor version of the page table
            paged_config: PagedAttentionConfig containing page table configuration
            mesh_device: TTNN mesh device

        Returns:
            Device-allocated version of the page table representing the page table
        """  # TODO: update docs
        assert cls.is_device_supported(mesh_device), f"Mesh device shape {mesh_device.shape} must be supported by MLA."

        if page_table is None:
            max_num_blocks = paged_config.max_num_blocks
            _, dp_factor = mesh_device.shape
            batch_per_shard = even_int_div(batch_size, dp_factor)

            page_table = torch.randperm(max_num_blocks, dtype=torch.int32)  # Randperm not necessary, but more rigorous
            page_table = page_table.reshape(batch_per_shard, even_int_div(max_num_blocks, batch_per_shard))

        assert page_table.numel() == paged_config.max_num_blocks

        return ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
        caches: Sequence[torch.Tensor] | None = None,
    ) -> ModelState:
        kvpe_dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
        cache_shape = (paged_config.max_num_blocks * mesh_device.shape[1], 1, paged_config.block_size, kvpe_dim)

        assert (
            caches is None
            or len(caches) == mesh_device.shape[0]
            and all(cache.shape == cache_shape for cache in caches)
        )
        if caches is None:
            caches = (torch.zeros(cache_shape),) * mesh_device.shape[0]

        tt_cache = ttnn.as_tensor(
            torch.concatenate(tuple(caches)),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, 0),
        )

        # CCL states setup (Must be in order of execution)
        get_rs_params = lambda axis: {
            "multi_device_global_semaphore": ccl.get_reduce_scatter_sem(axis=axis),
            "barrier_semaphore": ccl.get_barrier_sem(axis=axis),
            "num_links": ccl.get_max_links(axis=axis),
        }
        get_ag_params = lambda axis: {
            "multi_device_global_semaphore": ccl.get_gather_sem(axis=axis),
            "barrier_semaphore": ccl.get_barrier_sem(axis=axis),
            "num_links": ccl.get_max_links(axis=axis),
        }
        ccl_states_prefill = {
            "wq_a_rs_prefill": get_rs_params(1),
            "wq_a_ag_prefill": get_ag_params(1),
            "wkv_a_ag_prefill": get_ag_params(1),
            "wo_ag_prefill": get_ag_params(1),
        }
        ccl_states_decode = {
            "wq_a_rs_decode": get_rs_params(1),
            "wq_a_ag_decode": get_ag_params(1),
            "wq_a2a_ag_decode": get_ag_params(1),
            "wq_a2a_rs_decode": get_rs_params(1),
            "wkv_a_ag_decode": get_ag_params(1),
            "wkv_a_rs_decode": get_rs_params(1),
            "flash_mla_ag_decode": get_ag_params(1),
            "flash_mla_rs_decode": get_rs_params(1),
            "wo_ag_decode": get_ag_params(1),
        }

        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "mesh_shape": mesh_device.shape,
            "kvpe_cache": tt_cache,
            **ccl_states_prefill,
            **ccl_states_decode,
        }

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        row_idx: int,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Forward pass of MLA in decode mode.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            cfg: RunConfig containing weights and op configurations
            position_idxs: List of position indices for the current batch
            rope_tensors: Dictionary containing RoPE tensors
            page_table: Page table tensor for paged attention
        Returns:
            Output tensor after MLA computation

        """
        _, mla_tp_factor = mesh_shape = cfg["mesh_shape"]

        num_heads = cfg["num_heads"]
        num_heads_local = even_int_div(num_heads, mla_tp_factor)
        kv_lora_rank = cfg["kv_lora_rank"]
        qk_nope_head_dim = cfg["qk_nope_head_dim"]
        qk_rope_head_dim = cfg["qk_rope_head_dim"]
        qk_head_dim = cfg["qk_head_dim"]
        v_head_dim = cfg["v_head_dim"]

        kvpe_cache = cfg["kvpe_cache"]

        bsz = x.shape[2]
        scale = 1.0 / mla_tp_factor

        # wq_a and wq_b
        tt_q = ttnn.linear(x, **cfg["wq_a"])
        tt_q = ttnn.experimental.reduce_scatter_minimal_async(tt_q, **cfg["wq_a_rs_decode"])
        tt_q = ttnn.experimental.all_gather_async(tt_q, **cfg["wq_a_ag_decode"])

        tt_q = RMSNorm.forward_decode(tt_q, cfg["q_norm"])
        tt_q = ttnn.linear(tt_q, **cfg["wq_b"])

        tt_q = ttnn.reshape(tt_q, (bsz, 1, num_heads_local, qk_head_dim))
        tt_q_nope = ttnn.slice(tt_q, [0, 0, 0, 0], [bsz, 1, num_heads_local, qk_nope_head_dim])
        tt_q_rope = ttnn.slice(tt_q, [0, 0, 0, qk_nope_head_dim], [bsz, 1, num_heads_local, qk_head_dim])

        # wkv_b1
        tt_q_nope = ttnn.permute(tt_q_nope, (1, 2, 0, 3))  # [1, num_heads_local, bsz, qk_nope_head_dim]
        tt_q_nope = ttnn.linear(tt_q_nope, **cfg["wkv_b1"])  # [1, num_heads_local, bsz, kv_lora_rank]
        tt_q_nope = ttnn.permute(tt_q_nope, (0, 2, 1, 3))  # [1, bsz, num_heads_local, qk_nope_head_dim]

        # Q RoPE
        tt_q_rope = ttnn.permute(
            tt_q_rope, (1, 0, 2, 3)
        )  # [1, bsz, num_heads_local, qk_rope_head_dim], should be no-op
        tt_q_rope = ttnn.to_memory_config(tt_q_rope, **cfg["q_rope_reshard"])
        tt_q_rope = ttnn.experimental.rotary_embedding_llama(
            tt_q_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
        tt_q_rope = ttnn.to_memory_config(tt_q_rope, **cfg["q_rope_out_reshard"])

        # Q ready for FlashMLA
        tt_q = ttnn.concat([tt_q_nope, tt_q_rope], dim=-1)

        # FIXME: All-to-All here!! (tt_q)
        # The following code does the following:
        # [1, bsz, num_heads_local, kv_lora_rank + qk_rope_head_dim] -> [1, bsz_local, num_heads, kv_lora_rank + qk_rope_head_dim]
        # Using the following algorithm: 1. AG on in_dim, 2. Scale by number of devices, 3. RS on out_dim
        tt_q = ttnn.permute(tt_q, (0, 2, 1, 3))  # [1, num_heads_local, bsz_local, kv_lora_rank + qk_rope_head_dim]
        tt_q = ttnn.experimental.all_gather_async(
            tt_q, **cfg["wq_a2a_ag_decode"]
        )  # [1, num_heads, bsz_local, kv_lora_rank + qk_rope_head_dim]
        tt_q = ttnn.permute(tt_q, (0, 2, 1, 3))  # [1, bsz_local, num_heads, kv_lora_rank + qk_rope_head_dim]
        tt_q = ttnn.experimental.reduce_scatter_minimal_async(tt_q, **cfg["wq_a2a_rs_decode"])
        tt_q = tt_q * scale  # Scale the input tensor

        # KVPE Stuff
        tt_kv = ttnn.linear(x, **cfg["wkv_a"])

        # AG + Reduce b/c sub-tile RS not supported
        tt_kv = ttnn.experimental.all_gather_async(
            tt_kv, **cfg["wkv_a_ag_decode"]
        )  # [1, num_devices, bsz, kv_lora_rank + qk_rope_head_dim]
        tt_kv = ttnn.experimental.fast_reduce_nc(
            tt_kv, **cfg["wkv_a_r_decode"]
        )  # [1, 1, bsz, kv_lora_rank + qk_rope_head_dim]

        tt_kv_nope = ttnn.slice(tt_kv, [0, 0, 0, 0], [1, 1, bsz, kv_lora_rank])
        tt_kv_rope = ttnn.slice(tt_kv, [0, 0, 0, kv_lora_rank], [1, 1, bsz, kv_lora_rank + qk_rope_head_dim])
        ttnn.deallocate(tt_kv)

        # KV Norm
        tt_kv_nope = RMSNorm.forward_decode(tt_kv_nope, cfg["kv_norm"])

        # KV RoPE
        tt_kv_rope = ttnn.permute(tt_kv_rope, (0, 2, 1, 3))  # [1, bsz, 1, qk_rope_head_dim]
        tt_kv_rope = ttnn.to_memory_config(tt_kv_rope, **cfg["kv_rope_reshard"])
        # TODO: Use DP tensors
        # Currently, not using DP tensors because sub-tile RS is not supported
        tt_kv_rope = ttnn.experimental.rotary_embedding_llama(
            tt_kv_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=True,
        )
        tt_kv_rope = ttnn.to_memory_config(tt_kv_rope, **cfg["kv_rope_out_reshard"])
        tt_kv_rope = ttnn.permute(tt_kv_rope, (0, 2, 1, 3))  # [1, 1, bsz, qk_rope_head_dim]

        tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)

        # FIXME: Reduce-Scatter here!! (tt_kvpe)
        tt_kvpe = ttnn.pad(tt_kvpe, [(0, 0), (0, ttnn.TILE_SIZE - 1), (0, 0), (0, 0)], 0)
        tt_kvpe = ttnn.permute(tt_kvpe, (0, 2, 1, 3))  # [1, bsz, ttnn.TILE_SIZE, kv_lora_rank + qk_rope_head_dim]
        tt_kvpe = ttnn.experimental.reduce_scatter_minimal_async(tt_kvpe, **cfg["wkv_a_rs_decode"])
        tt_kvpe = tt_kvpe[:, :, :1, :]  # [1, bsz_local, 1, kv_lora_rank + qk_rope_head_dim]
        tt_kvpe = tt_kvpe * scale  # Scale the input tensor

        tt_kvpe = ttnn.to_memory_config(tt_kvpe, **cfg["kvpe_reshard"])
        ttnn.deallocate(tt_kv_nope)
        ttnn.deallocate(tt_kv_rope)

        # Update KVPE Cache
        ttnn.experimental.paged_update_cache(
            kvpe_cache,
            tt_kvpe,
            update_idxs_tensor=position_idxs,
            page_table=page_table,
            mesh_coords=set(get_mesh_coords(mesh_shape, row_idx)),
        )

        # FlashMLA
        tt_q = ttnn.to_memory_config(tt_q, **cfg["flash_mla_reshard"])
        attn_out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            tt_q,
            kvpe_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=position_idxs,
            **cfg["flash_mla"],
        )  #  [1, bsz_local, num_heads, kv_lora_rank]
        ttnn.deallocate(tt_q)
        attn_out = ttnn.to_memory_config(attn_out, **cfg["flash_mla_out_reshard"])

        # FIXME: All-to-All here!! (attn_out)
        # TODO: add deallocation of intermediate tensors
        attn_out = ttnn.experimental.all_gather_async(
            attn_out, **cfg["flash_mla_ag_decode"]
        )  # [1, bsz, num_heads, kv_lora_rank]
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))  # [1, num_heads, bsz, kv_lora_rank]
        attn_out = ttnn.experimental.reduce_scatter_minimal_async(
            attn_out, **cfg["flash_mla_rs_decode"]
        )  # [1, num_heads_local, bsz, kv_lora_rank]
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))  # [1, bsz, num_heads_local, kv_lora_rank]
        attn_out = attn_out * scale  # Scale the output tensor

        # wkv_b2
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))  # [1, num_heads_local, bsz, kv_lora_rank]
        v_out = ttnn.linear(attn_out, **cfg["wkv_b2"])  # [1, num_heads_local, bsz, v_head_dim]

        # wo
        v_out = ttnn.experimental.all_gather_async(v_out, **cfg["wo_ag_decode"])  # [1, num_heads, bsz, v_head_dim]
        v_out = ttnn.permute(v_out, (0, 2, 1, 3))  # [1, bsz, num_heads, v_head_dim]

        v_out = ttnn.reshape(v_out, (1, 1, bsz, num_heads * v_head_dim))
        out = ttnn.linear(v_out, **cfg["wo"])  # [1, 1, bsz, dim]

        return out

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        batch_idx: int,
        row_idx: int,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Forward pass of MLA in prefill mode.

        Args:
            x: Input tensor
            cfg: RunConfig containing weights and op configurations
            batch_idx: Batch index for cache updates (wrt to global batch size)
            rope_tensors: Dictionary containing RoPE tensors
            page_table: Page table tensor for paged attention
            row_idx: Row index in the mesh

        Returns:
            Output tensor after MLP computation
        """

        sdpa_dp_factor, mla_tp_factor = mesh_shape = cfg["mesh_shape"]

        num_heads = cfg["num_heads"]
        num_heads_local = even_int_div(num_heads, mla_tp_factor)
        kv_lora_rank = cfg["kv_lora_rank"]
        qk_nope_head_dim = cfg["qk_nope_head_dim"]
        qk_rope_head_dim = cfg["qk_rope_head_dim"]
        qk_head_dim = cfg["qk_head_dim"]
        v_head_dim = cfg["v_head_dim"]

        kvpe_cache = cfg["kvpe_cache"]

        seq_len = x.shape[2]

        # wq_a and wq_b
        tt_q = ttnn.linear(x, **cfg["wq_a"])

        tt_q = ttnn.experimental.reduce_scatter_minimal_async(tt_q, **cfg["wq_a_rs_prefill"])
        tt_q = ttnn.experimental.all_gather_async(tt_q, **cfg["wq_a_ag_prefill"])

        tt_q = RMSNorm.forward_prefill(tt_q, cfg["q_norm"])
        tt_q = ttnn.linear(tt_q, **cfg["wq_b"])

        tt_q = ttnn.reshape(tt_q, (1, seq_len, num_heads_local, qk_head_dim))
        tt_q = ttnn.permute(tt_q, (0, 2, 1, 3))  # [1, num_heads_local, seq_len, qk_head_dim]

        tt_q_nope = ttnn.slice(tt_q, [0, 0, 0, 0], [1, num_heads_local, seq_len, qk_nope_head_dim])
        tt_q_rope = ttnn.slice(tt_q, [0, 0, 0, qk_nope_head_dim], [1, num_heads_local, seq_len, qk_head_dim])

        # wkv_b1
        tt_q_nope = ttnn.linear(tt_q_nope, **cfg["wkv_b1"])  # [1, num_heads_local, seq_len, kv_lora_rank]

        # Q RoPE
        tt_q_rope = ttnn.experimental.rotary_embedding_llama(
            tt_q_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=False,
        )

        # Q ready for FlashMLA
        tt_q = ttnn.concat([tt_q_nope, tt_q_rope], dim=-1)

        # KVPE Stuff
        tt_kv = ttnn.linear(x, **cfg["wkv_a"])

        tt_kv = ttnn.experimental.all_gather_async(
            tt_kv, **cfg["wkv_a_ag_prefill"]
        )  # [1, 1, seq_len / num_devices, kv_lora_rank + qk_rope_head_dim]
        tt_kv = ttnn.experimental.fast_reduce_nc(
            tt_kv, **cfg["wkv_a_r_prefill"]
        )  # [1, 1, seq_len, kv_lora_rank + qk_rope_head_dim]

        tt_kv_nope = ttnn.slice(tt_kv, [0, 0, 0, 0], [1, 1, seq_len, kv_lora_rank])
        tt_kv_rope = ttnn.slice(tt_kv, [0, 0, 0, kv_lora_rank], [1, 1, seq_len, kv_lora_rank + qk_rope_head_dim])
        ttnn.deallocate(tt_kv)

        # KV Norm
        tt_kv_nope = RMSNorm.forward_prefill(tt_kv_nope, cfg["kv_norm"])

        # KV RoPE
        tt_kv_rope = ttnn.experimental.rotary_embedding_llama(
            tt_kv_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=False,
        )

        tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)
        # TODO: Add Norm here for KVPE
        ttnn.deallocate(tt_kv_nope)
        ttnn.deallocate(tt_kv_rope)

        tt_kvpe = ttnn.typecast(tt_kvpe, dtype=kvpe_cache.dtype)

        # Update KVPE Cache
        local_batch_idx = batch_idx % sdpa_dp_factor  # Local batch index within the DP shard
        col_idx = batch_idx // sdpa_dp_factor  # Which DP shard the batch belongs to
        ttnn.experimental.paged_fill_cache(
            kvpe_cache,
            tt_kvpe,
            page_table=page_table,
            batch_idx=local_batch_idx,
            mesh_coords=set(get_mesh_coords(mesh_shape, row_idx, col_idx)),
        )

        # FlashMLA
        attn_out = ttnn.transformer.flash_mla_prefill(
            tt_q,
            tt_kvpe,
            **cfg["flash_mla"],
        )  # [1, num_heads_local, seq_len, kv_lora_rank]
        ttnn.deallocate(tt_q)

        # wkv_b2
        v_out = ttnn.linear(attn_out, **cfg["wkv_b2"])  # [1, num_heads_local, seq_len, v_head_dim]
        v_out = ttnn.experimental.all_gather_async(v_out, **cfg["wo_ag_prefill"])  # [1, num_heads, seq_len, v_head_dim]

        # wo
        v_out = ttnn.permute(v_out, (0, 2, 1, 3))  # [1, seq_len, num_heads, v_head_dim]
        v_out = ttnn.reshape(v_out, (1, 1, seq_len, num_heads * v_head_dim))
        out = ttnn.linear(v_out, **cfg["wo"])  # [1, 1, seq_len, dim]

        return out
