# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0


import math
from pathlib import Path
from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.rms_norm import RMSNorm
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import (
    FromWeightConfig,
    LinearConfig,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ReshardConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.demos.deepseek_v3.utils.config_helpers import save_and_get_path
from models.utility_functions import nearest_y


class MLA1D(AbstractModule):
    """ """

    MAX_BATCH_SIZE = ttnn.TILE_SIZE

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

        num_devices = mesh_device.get_num_devices()

        TG_GRID = (8, 4)  # TP, DP

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
                dims=[-2, None],
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
                dims=[-1, None],
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
                dims=[-2, None],
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
                dims=[-3, None],
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
                dims=[-3, None],
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
                dims=[-1, None],
                mesh_shape=list(mesh_device.shape),
            ),
        )

        # Norm weights
        q_norm_state_dict = {"weight": state_dict["q_norm.weight"]}
        weight_config["q_norm"] = RMSNorm.convert_weights(
            hf_config, q_norm_state_dict, output_path, mesh_device, norm_category="q_norm"
        )

        kv_norm_state_dict = {"weight": state_dict["kv_norm.weight"]}
        weight_config["kv_norm"] = RMSNorm.convert_weights(
            hf_config, kv_norm_state_dict, output_path, mesh_device, norm_category="k_norm"
        )

        return weight_config

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelPrefillConfig:
        """Prefill model config for an MLP with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for prefill mode
        """
        # Extract dimensions from HF config
        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        num_devices = mesh_device.get_num_devices()
        grid_size = mesh_device.compute_with_storage_grid_size()

        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim
        mscale = hf_config.rope_scaling["mscale"]
        rope_factor = hf_config.rope_scaling["factor"]

        config: ModelPrefillConfig = {}
        config["hf_config"] = hf_config

        config["wq_a"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wq_b"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_a"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_b1"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_b2"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wo"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
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
        # If max_seq_len > original max_seq_len (4k)
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

        return config

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device) -> ModelDecodeConfig:
        """Generate decode operator configuration for this MLP layer.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """
        # Extract dimensions from HF config
        dim = hf_config.hidden_size
        hidden_dim = hf_config.intermediate_size
        num_devices = mesh_device.get_num_devices()
        grid_size = mesh_device.compute_with_storage_grid_size()

        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim
        mscale = hf_config.rope_scaling["mscale"]
        rope_factor = hf_config.rope_scaling["factor"]

        config: ModelDecodeConfig = {}

        config["hf_config"] = hf_config

        config["wq_a"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wq_b"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_a"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_b1"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wkv_b2"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        config["wo"] = LinearConfig(
            input_tensor_b=FromWeightConfig(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=None,
        )

        # Resharding for q_rope
        # TODO: Should be dynamic based on batch size?
        q_rope_shape = (1, MLA1D.MAX_BATCH_SIZE, num_heads, qk_rope_head_dim)
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
        kvpe_shape = (1, MLA1D.MAX_BATCH_SIZE, 1, kv_lora_rank + qk_rope_head_dim)
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

        q_num_cores = MLA1D.MAX_BATCH_SIZE  # TODO: How to use non-padded batch size here? (might need to be dynamic)
        block_height = nearest_y((MLA1D.MAX_BATCH_SIZE * num_heads) // q_num_cores, ttnn.TILE_SIZE)
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
        # If max_seq_len > original max_seq_len (4k)
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

        return config

    @classmethod
    def _new_state(
        cls,
        stateless_run_prefill_config: AbstractModule.StatelessRunPrefillConfig,
        stateless_run_decode_config: AbstractModule.StatelessRunDecodeConfig,
        mesh_device: ttnn.Device,
    ) -> Any:
        hf_config = stateless_run_prefill_config["hf_config"]

        kv_lora_rank = hf_config.kv_lora_rank
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        max_seq_len = hf_config.max_seq_len

        kvpe_dim = kv_lora_rank + qk_rope_head_dim
        kvpe_cache_dtype = ttnn.bfloat8_b
        kvpe_cache_layout = ttnn.TILE_LAYOUT
        kvpe_cache_mem_config = ttnn.DRAM_MEMORY_CONFIG

        cache = torch.zeros(
            (
                MLA1D.MAX_BATCH_SIZE,
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

        return {"kvpe_cache": tt_cache}

    @classmethod
    def forward_decode(
        self, x: ttnn.Tensor, position_idxs: [int], rope_tensors: dict, cfg: RunPrefillConfig
    ) -> ttnn.Tensor:
        """Straightforward forward pass for decode mode"""

        hf_config = cfg["hf_config"]
        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim

        kvpe_cache = cfg["kvpe_cache"]

        bsz = x.shape[2]

        # wq_a and wq_b
        tt_q = ttnn.linear(x, **cfg["wq_a"])
        tt_q = RMSNorm.forward_decode(tt_q, cfg["q_norm"], None)
        tt_q = ttnn.linear(tt_q, **cfg["wq_b"])

        # TODO: Use local heads here
        tt_q = ttnn.reshape(tt_q, (bsz, 1, num_heads, qk_head_dim))

        tt_q_nope = ttnn.slice(tt_q, [0, 0, 0, 0], [bsz, 1, num_heads, qk_nope_head_dim])
        tt_q_rope = ttnn.slice(tt_q, [0, 0, 0, qk_nope_head_dim], [bsz, 1, num_heads, qk_head_dim])

        # wkv_b1
        tt_q_nope = ttnn.permute(tt_q_nope, (1, 2, 0, 3))  # [1, num_heads, bsz, qk_nope_head_dim]
        tt_q_nope = ttnn.linear(tt_q_nope, **cfg["wkv_b1"])  # [1, num_heads, bsz, kv_lora_rank]
        tt_q_nope = ttnn.permute(tt_q_nope, (0, 2, 1, 3))  # [1, bsz, num_heads, qk_nope_head_dim]

        # Q RoPE
        tt_q_rope = ttnn.permute(tt_q_rope, (1, 0, 2, 3))  # [1, bsz, num_heads, qk_rope_head_dim], should be no-op
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

        # KVPE Stuff
        tt_kv = ttnn.linear(x, **cfg["wkv_a"])
        tt_kv_nope = ttnn.slice(tt_kv, [0, 0, 0, 0], [1, 1, bsz, kv_lora_rank])
        tt_kv_rope = ttnn.slice(tt_kv, [0, 0, 0, kv_lora_rank], [1, 1, bsz, kv_lora_rank + qk_rope_head_dim])
        ttnn.deallocate(tt_kv)

        # KV Norm
        tt_kv_nope = RMSNorm.forward_decode(tt_kv_nope, cfg["kv_norm"], None)

        # KV RoPE
        tt_kv_rope = ttnn.permute(tt_kv_rope, (0, 2, 1, 3))  # [1, bsz, 1, qk_rope_head_dim]
        tt_kv_rope = ttnn.to_memory_config(tt_kv_rope, **cfg["kv_rope_reshard"])
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
        tt_kvpe = ttnn.permute(tt_kvpe, (0, 2, 1, 3))  # [1, bsz, 1, kv_lora_rank + qk_rope_head_dim]
        tt_kvpe = ttnn.to_memory_config(tt_kvpe, **cfg["kvpe_reshard"])
        ttnn.deallocate(tt_kv_nope)
        ttnn.deallocate(tt_kv_rope)

        # Update KVPE Cache
        ttnn.experimental.paged_update_cache(
            kvpe_cache,
            tt_kvpe,
            update_idxs=position_idxs,
        )

        # FlashMLA
        tt_q = ttnn.to_memory_config(tt_q, **cfg["flash_mla_reshard"])
        attn_out = ttnn.transformer.flash_multi_latent_attention_decode(
            tt_q,
            kvpe_cache,
            cur_pos=position_idxs,
            **cfg["flash_mla"],
        )  #  [1, bsz, num_heads, kv_lora_rank]
        ttnn.deallocate(tt_q)
        attn_out = ttnn.to_memory_config(attn_out, **cfg["flash_mla_out_reshard"])

        # wkv_b2
        attn_out = ttnn.permute(attn_out, (0, 2, 1, 3))  # [1, num_heads, bsz, kv_lora_rank]
        v_out = ttnn.linear(attn_out, **cfg["wkv_b2"])  # [1, num_heads, bsz, v_head_dim]
        v_out = ttnn.permute(v_out, (0, 2, 1, 3))  # [1, bsz, num_heads, v_head_dim]

        # wo
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
        assert False, "MLA1D prefill forward is untested!"

        hf_config = cfg["hf_config"]
        num_heads = hf_config.num_attention_heads
        kv_lora_rank = hf_config.kv_lora_rank
        qk_nope_head_dim = hf_config.qk_nope_head_dim
        qk_rope_head_dim = hf_config.qk_rope_head_dim
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        v_head_dim = hf_config.v_head_dim

        kvpe_cache = cfg["kvpe_cache"]

        seq_len = x.shape[2]

        # wq_a and wq_b
        tt_q = ttnn.linear(x, **cfg["wq_a"])
        # TODO: Add norm
        tt_q = ttnn.linear(tt_q, **cfg["wq_b"])

        # TODO: Use local heads here
        tt_q = ttnn.reshape(tt_q, (1, seq_len, num_heads, qk_head_dim))
        tt_q = ttnn.permute(tt_q, (0, 2, 1, 3))  # [1, num_heads, seq_len, qk_head_dim]

        tt_q_nope = ttnn.slice(tt_q, [0, 0, 0, 0], [1, num_heads, seq_len, qk_nope_head_dim])
        tt_q_rope = ttnn.slice(tt_q, [0, 0, 0, qk_nope_head_dim], [1, num_heads, seq_len, qk_head_dim])

        # wkv_b1
        tt_q_nope = ttnn.linear(tt_q_nope, **cfg["wkv_b1"])  # [1, num_heads, seq_len, kv_lora_rank]

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
        tt_kv_nope = ttnn.slice(tt_kv, [0, 0, 0, 0], [1, 1, seq_len, kv_lora_rank])
        tt_kv_rope = ttnn.slice(tt_kv, [0, 0, 0, kv_lora_rank], [1, 1, seq_len, kv_lora_rank + qk_rope_head_dim])
        ttnn.deallocate(tt_kv)

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
        )  # [1, num_heads, seq_len, kv_lora_rank]
        ttnn.deallocate(tt_q)

        # wkv_b2
        v_out = ttnn.linear(attn_out, **cfg["wkv_b2"])  # [1, num_heads, seq_len, v_head_dim]
        v_out = ttnn.permute(v_out, (0, 2, 1, 3))  # [1, seq_len, num_heads, v_head_dim]

        # wo
        v_out = ttnn.reshape(v_out, (1, 1, seq_len, num_heads * v_head_dim))
        out = ttnn.linear(v_out, **cfg["wo"])  # [1, 1, seq_len, dim]

        return out
