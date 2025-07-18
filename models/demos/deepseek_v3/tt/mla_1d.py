# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import math
from pathlib import Path
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl_1d import CCL1D
from models.demos.deepseek_v3.tt.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    AllGatherAsyncConfig,
    FromWeightConfig,
    LinearConfig,
    MeshDeviceStub,
    ReduceScatterAsyncConfig,
    ReshardConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import save_and_get_path
from models.demos.deepseek_v3.utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.utility_functions import nearest_y


class MLA1D(AbstractModule):
    """
    Multi-Latent Attention Module for 1D tensor parallelism.
    """

    MAX_BATCH_SIZE = ttnn.TILE_SIZE
    TG_GRID = (8, 4)

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        """Convert PyTorch weights to TTNN format for 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """

        weight_config = {}

        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        v_head_dim = hf_config.v_head_dim

        def add_weight_config(
            torch_weight,
            our_name,
            kwarg_name,
            dtype,
            mem_config,
            layout,
            mesh_mapper,
        ):
            """Helper function to convert and save weights, updating weight_config."""
            ttnn_weight = ttnn.as_tensor(
                torch_weight,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=mesh_mapper,
                layout=layout,
                memory_config=mem_config,
            )
            ttnn_weight = ttnn.unsqueeze_to_4D(ttnn_weight)

            # Add to weight config
            weight_file_path = output_path / f"{our_name}.{kwarg_name}.weight"
            weight_config[our_name] = {kwarg_name: save_and_get_path(weight_file_path, ttnn_weight)}

        hf_ttnn_name_mapping = {
            "q_a_proj": "wq_a",
            "q_b_proj": "wq_b",
            "kv_a_proj_with_mqa": "wkv_a",
            "kv_b_proj": "wkv_b",
            "o_proj": "wo",
        }

        # wq_a
        hf_name = "q_a_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[None, -2],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wq_b
        hf_name = "q_b_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[None, -1],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wkv_a
        hf_name = "kv_a_proj_with_mqa"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[None, -2],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wkv_b1
        hf_name = "kv_b_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]

        # This weight needs to be split
        torch_weight = torch_weight.view(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))
        torch_weight = torch_weight.reshape(num_heads, -1, kv_lora_rank)

        torch_weight_k = torch_weight[:, :qk_nope_head_dim, :]  # [num_heads, qk_nope_head_dim, kv_lora_rank]
        torch_weight_v = torch_weight[:, qk_nope_head_dim:, :].transpose(
            -2, -1
        )  # [num_heads, kv_lora_rank, v_head_dim]

        add_weight_config(
            torch_weight_k,
            our_name + "1",
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[None, -3],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        add_weight_config(
            torch_weight_v,
            our_name + "2",
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[None, -3],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # wo
        hf_name = "o_proj"
        our_name = hf_ttnn_name_mapping[hf_name]
        torch_weight = state_dict[f"{our_name}.weight"]
        torch_weight = torch.transpose(torch_weight, -2, -1)

        add_weight_config(
            torch_weight,
            our_name,
            "input_tensor_b",
            dtype=ttnn.bfloat8_b,
            mem_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=[None, -1],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # Norm weights
        our_name = "q_norm"
        q_norm_state_dict = {"weight": state_dict[f"{our_name}.weight"]}
        weight_config["q_norm"] = RMSNorm.convert_weights(
            hf_config, q_norm_state_dict, output_path, mesh_device, norm_category="q_norm"
        )

        our_name = "kv_norm"
        kv_norm_state_dict = {"weight": state_dict[f"{our_name}.weight"]}
        weight_config["kv_norm"] = RMSNorm.convert_weights(
            hf_config, kv_norm_state_dict, output_path, mesh_device, norm_category="k_norm"
        )

        return weight_config

    @classmethod
    def prefill_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL1D
    ) -> ModelPrefillConfig:
        """Prefill model config for an MLP with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device
            ccl: CCL1D object for communication configuration

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
        original_seq_len = hf_config.rope_scaling["original_max_position_embeddings"]
        max_seq_len = hf_config.max_seq_len

        mesh_shape = list(mesh_device.shape)

        config: ModelPrefillConfig = {}

        config["hf_config"] = hf_config
        config["mesh_shape"] = mesh_shape

        config["wq_a"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wq_b"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_a"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_b1"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_b2"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wo"] = LinearConfig(
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
        if max_seq_len > original_seq_len:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            scale = scale * mscale * mscale

        config["flash_mla"] = {
            "head_dim_v": kv_lora_rank,
            "scale": scale,
            "program_config": sdpa_program_config,
            "compute_kernel_config": flash_mla_compute_kernel_config,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "attn_mask": None,
            "is_causal": True,
        }

        # Norms
        config["q_norm"] = RMSNorm.prefill_model_config(hf_config, mesh_device, norm_category="q_norm")
        config["kv_norm"] = RMSNorm.prefill_model_config(hf_config, mesh_device, norm_category="k_norm")

        # Set up CCLs
        # **Must be in order of execution**

        # Q
        config["wq_a_rs"] = ReduceScatterAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=3,
            from_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            to_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            math_op=ttnn.ReduceType.Sum,
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        config["wq_a_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=3,
            multi_device_global_semaphore=ccl.get_semaphore(1),
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # KV
        config["wkv_a_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            multi_device_global_semaphore=ccl.get_semaphore(1),
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        config["wkv_a_r"] = {
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
        config["wo_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            multi_device_global_semaphore=ccl.get_semaphore(1),
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        return config

    @classmethod
    def decode_model_config(
        cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL1D
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
        original_seq_len = hf_config.rope_scaling["original_max_position_embeddings"]
        max_seq_len = hf_config.max_seq_len

        mesh_shape = list(mesh_device.shape)
        num_heads_local = num_heads // mesh_shape[1]

        config: ModelDecodeConfig = {}
        config["hf_config"] = hf_config
        config["mesh_shape"] = mesh_shape

        config["wq_a"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wq_b"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_a"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_b1"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_b2"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wo"] = LinearConfig(
            input_tensor_b=FromWeightConfig(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # Resharding for q_rope
        # TODO: Should be dynamic based on batch size?
        q_rope_shape = (1, MLA1D.MAX_BATCH_SIZE, num_heads_local, qk_rope_head_dim)
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
        config["q_rope_reshard"] = ReshardConfig(
            memory_config=q_rope_mem_cfg,
        )
        config["q_rope_out_reshard"] = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Resharding for kv_rope
        # TODO: Should be dynamic based on batch size?
        # TODO: Split batch when adding DP
        kv_rope_shape = (1, MLA1D.MAX_BATCH_SIZE, 1, qk_rope_head_dim)
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
        config["kv_rope_reshard"] = ReshardConfig(
            memory_config=kv_rope_mem_cfg,
        )
        config["kv_rope_out_reshard"] = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Resharding for kvpe
        kvpe_shape = (1, MLA1D.MAX_BATCH_SIZE // mesh_shape[1], 1, kv_lora_rank + qk_rope_head_dim)
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
        config["kvpe_reshard"] = ReshardConfig(
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
        q_num_cores = min(MLA1D.MAX_BATCH_SIZE // mesh_shape[1] * num_heads, q_num_cores)
        block_height = nearest_y((MLA1D.MAX_BATCH_SIZE // mesh_shape[1] * num_heads) // q_num_cores, ttnn.TILE_SIZE)
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
        if max_seq_len > original_seq_len:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            scale = scale * mscale * mscale

        config["flash_mla_reshard"] = ReshardConfig(
            memory_config=q_mem_config,
        )
        config["flash_mla"] = {
            "head_dim_v": kv_lora_rank,
            "scale": scale,
            "program_config": sdpa_program_config,
            "compute_kernel_config": flash_mla_compute_kernel_config,
            "memory_config": flash_mla_out_mem_config,
        }
        config["flash_mla_out_reshard"] = ReshardConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Norms
        config["q_norm"] = RMSNorm.decode_model_config(hf_config, mesh_device, norm_category="q_norm")
        config["kv_norm"] = RMSNorm.decode_model_config(hf_config, mesh_device, norm_category="k_norm")

        # Set up CCLs
        # **Must be in order of execution**

        # Q
        config["wq_a_rs"] = ReduceScatterAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=3,
            from_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            to_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            math_op=ttnn.ReduceType.Sum,
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        config["wq_a_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=3,
            multi_device_global_semaphore=ccl.get_semaphore(1),
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # Q all-to-all
        config["wq_a2a_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            multi_device_global_semaphore=ccl.get_semaphore(1),
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        config["wq_a2a_rs"] = ReduceScatterAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            from_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            to_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            math_op=ttnn.ReduceType.Sum,
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # KV
        config["wkv_a_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            multi_device_global_semaphore=ccl.get_semaphore(1),
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        config["wkv_a_r"] = {
            "dims": [1],
            "output": None,
            "compute_kernel_config": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            ),
        }
        config["wkv_a_rs"] = ReduceScatterAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            from_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            to_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            math_op=ttnn.ReduceType.Sum,
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # FlashMLA all-to-all
        config["flash_mla_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            multi_device_global_semaphore=ccl.get_semaphore(1),
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        config["flash_mla_rs"] = ReduceScatterAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            from_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            to_remote_multi_device_global_semaphore=ccl.get_semaphore(1),
            math_op=ttnn.ReduceType.Sum,
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        # WO
        config["wo_ag"] = AllGatherAsyncConfig(
            mesh_device=MeshDeviceStub(mesh_shape),
            cluster_axis=1,
            dim=1,
            multi_device_global_semaphore=ccl.get_semaphore(1),
            num_links=ccl.get_max_links(1),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        return config

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        mode: str,
    ) -> Any:
        kv_lora_rank = hf_config.kv_lora_rank
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        max_seq_len = hf_config.max_seq_len

        kvpe_dim = kv_lora_rank + qk_rope_head_dim
        kvpe_cache_dtype = ttnn.bfloat8_b
        kvpe_cache_layout = ttnn.TILE_LAYOUT
        kvpe_cache_mem_config = ttnn.DRAM_MEMORY_CONFIG

        mesh_shape = list(mesh_device.shape)

        cache = torch.zeros(
            (
                MLA1D.MAX_BATCH_SIZE // (mesh_shape[1] if mode == "decode" else 1),  # Prefill does not support DP yet
                1,  # 1 latent kv heads
                max_seq_len,
                kvpe_dim,
            )
        )

        tt_cache = ttnn.as_tensor(
            cache,
            dtype=kvpe_cache_dtype,
            layout=kvpe_cache_layout,
            device=mesh_device,
            memory_config=kvpe_cache_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            # TODO: Add caching
        )

        return {"kvpe_cache": tt_cache, MESH_DEVICE_STATE_DICT_KEY: mesh_device}

    @classmethod
    def forward_decode(
        self, x: ttnn.Tensor, position_idxs: [int], rope_tensors: dict, cfg: RunPrefillConfig
    ) -> ttnn.Tensor:
        """Forward pass of MLA1D in decode mode.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            position_idxs: List of position indices for the current batch
            rope_tensors: Dictionary containing RoPE tensors
            cfg: RunConfig containing weights and op configurations
        Returns:
            Output tensor after MLA1D computation

        """

        hf_config = cfg["hf_config"]
        num_heads = hf_config.num_attention_heads
        num_heads_local = num_heads // cfg["mesh_shape"][1]
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim

        kvpe_cache = cfg["kvpe_cache"]

        bsz = x.shape[2]
        scale = 1.0 / cfg["mesh_shape"][1]

        # wq_a and wq_b
        tt_q = ttnn.linear(x, **cfg["wq_a"])

        tt_q = ttnn.experimental.reduce_scatter_async(tt_q, **cfg["wq_a_rs"])
        tt_q = ttnn.experimental.all_gather_async(tt_q, **cfg["wq_a_ag"])

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
            tt_q, **cfg["wq_a2a_ag"]
        )  # [1, num_heads, bsz_local, kv_lora_rank + qk_rope_head_dim]
        tt_q = ttnn.permute(tt_q, (0, 2, 1, 3))  # [1, bsz_local, num_heads, kv_lora_rank + qk_rope_head_dim]
        tt_q = ttnn.experimental.reduce_scatter_async(tt_q, **cfg["wq_a2a_rs"])
        tt_q = tt_q * scale  # Scale the input tensor

        # KVPE Stuff
        tt_kv = ttnn.linear(x, **cfg["wkv_a"])

        # AG + Reduce b/c sub-tile RS not supported
        tt_kv = ttnn.experimental.all_gather_async(
            tt_kv, **cfg["wkv_a_ag"]
        )  # [1, num_devices, bsz, kv_lora_rank + qk_rope_head_dim]
        tt_kv = ttnn.experimental.fast_reduce_nc(
            tt_kv, **cfg["wkv_a_r"]
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
        tt_kvpe = ttnn.experimental.reduce_scatter_async(tt_kvpe, **cfg["wkv_a_rs"])
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
        )

        # FlashMLA
        tt_q = ttnn.to_memory_config(tt_q, **cfg["flash_mla_reshard"])
        attn_out = ttnn.transformer.flash_multi_latent_attention_decode(
            tt_q,
            kvpe_cache,
            cur_pos_tensor=position_idxs,
            **cfg["flash_mla"],
        )  #  [1, bsz_local, num_heads, kv_lora_rank]
        ttnn.deallocate(tt_q)
        attn_out = ttnn.to_memory_config(attn_out, **cfg["flash_mla_out_reshard"])

        # FIXME: All-to-All here!! (attn_out)
        attn_out = ttnn.experimental.all_gather_async(
            attn_out, **cfg["flash_mla_ag"]
        )  # [1, bsz, num_heads, kv_lora_rank]
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))  # [1, num_heads, bsz, kv_lora_rank]
        attn_out = ttnn.experimental.reduce_scatter_async(
            attn_out, **cfg["flash_mla_rs"]
        )  # [1, num_heads_local, bsz, kv_lora_rank]
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))  # [1, bsz, num_heads_local, kv_lora_rank]
        attn_out = attn_out * scale  # Scale the output tensor

        # wkv_b2
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))  # [1, num_heads_local, bsz, kv_lora_rank]
        v_out = ttnn.linear(attn_out, **cfg["wkv_b2"])  # [1, num_heads_local, bsz, v_head_dim]

        # wo
        v_out = ttnn.experimental.all_gather_async(v_out, **cfg["wo_ag"])  # [1, num_heads, bsz, v_head_dim]
        v_out = ttnn.permute(v_out, (0, 2, 1, 3))  # [1, bsz, num_heads, v_head_dim]

        v_out = ttnn.reshape(v_out, (1, 1, bsz, num_heads * v_head_dim))
        out = ttnn.linear(v_out, **cfg["wo"])  # [1, 1, bsz, dim]

        return out

    @classmethod
    def forward_prefill(self, x: ttnn.Tensor, user_id: int, rope_tensors: dict, cfg: RunPrefillConfig) -> ttnn.Tensor:
        """Forward pass of the MLP.

        Prefill mode we reshape to respect cfg["max_rows"] and generate program configs from the seq-len lambda.

        Args:
            x: Input tensor
            user_id: Batch index for cache updates
            rope_tensors: Dictionary containing RoPE tensors
            cfg: RunConfig containing weights and op configurations

        Returns:
            Output tensor after MLP computation
        """

        hf_config = cfg["hf_config"]
        num_heads = hf_config.num_attention_heads
        num_heads_local = num_heads // cfg["mesh_shape"][1]
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim

        kvpe_cache = cfg["kvpe_cache"]

        seq_len = x.shape[2]

        # wq_a and wq_b
        tt_q = ttnn.linear(x, **cfg["wq_a"])

        tt_q = ttnn.experimental.reduce_scatter_async(tt_q, **cfg["wq_a_rs"])
        tt_q = ttnn.experimental.all_gather_async(tt_q, **cfg["wq_a_ag"])

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
            tt_kv, **cfg["wkv_a_ag"]
        )  # [1, 1, seq_len / num_devices, kv_lora_rank + qk_rope_head_dim]
        tt_kv = ttnn.experimental.fast_reduce_nc(
            tt_kv, **cfg["wkv_a_r"]
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
        ttnn.fill_cache(
            kvpe_cache,
            tt_kvpe,
            batch_idx=user_id,
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
        v_out = ttnn.experimental.all_gather_async(v_out, **cfg["wo_ag"])  # [1, num_heads, seq_len, v_head_dim]

        # wo
        v_out = ttnn.permute(v_out, (0, 2, 1, 3))  # [1, seq_len, num_heads, v_head_dim]
        v_out = ttnn.reshape(v_out, (1, 1, seq_len, num_heads * v_head_dim))
        out = ttnn.linear(v_out, **cfg["wo"])  # [1, 1, seq_len, dim]

        return out
