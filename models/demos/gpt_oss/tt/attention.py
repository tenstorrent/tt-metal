# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.utility_functions import nearest_y
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name
from models.demos.gpt_oss.utils.substate import substate

from ..tt.sdpa import sdpa as tt_sdpa
from ..utils.general_utils import MAX_SEQ_LEN


class Attention:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        ccl_manager,
        tensor_cache_path=None,
        paged_attention_config=None,
        mesh_config=None,
        create_kv_cache=True,
        transformation_mats=None,
    ):
        self.layer_idx = layer_idx
        self.use_sliding_window = self.layer_idx % 2 == 0
        if self.use_sliding_window:
            self.sliding_window = hf_config.sliding_window
        else:
            self.sliding_window = None
        self.scaling = hf_config.head_dim**-0.5
        self.head_dim = hf_config.head_dim
        self.num_heads = hf_config.num_attention_heads
        self.num_kv_heads = hf_config.num_key_value_heads

        # Use mode-aware MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1]))
        self.num_local_heads = self.mesh_config.shard_size(self.num_heads)
        self.num_local_kv_heads = self.mesh_config.shard_size(self.num_kv_heads)

        self.hidden_size = hf_config.hidden_size
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device
        self.transformation_mats = transformation_mats

        # Extract projection weights from state dict
        q_proj_weight = substate(state_dict, "q_proj")["weight"]  # [num_heads * head_dim, hidden_size]
        k_proj_weight = substate(state_dict, "k_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]
        v_proj_weight = substate(state_dict, "v_proj")["weight"]  # [num_kv_heads * head_dim, hidden_size]

        o_proj = substate(state_dict, "o_proj")["weight"].transpose(-1, -2)
        o_proj_bias = substate(state_dict, "o_proj")["bias"]

        # Create fused QKV weight like tt-transformers
        # Split Q, K, V across devices, then concatenate per device
        qkv_list = []
        for i in range(self.mesh_config.tp):
            # Chunk weights across tensor parallel dimension
            wq_selected = torch.chunk(q_proj_weight, self.mesh_config.tp, dim=0)[i]
            wk_selected = torch.chunk(k_proj_weight, self.mesh_config.tp, dim=0)[i]
            wv_selected = torch.chunk(v_proj_weight, self.mesh_config.tp, dim=0)[i]

            # Transpose for matmul: [hidden_size, local_dim]
            wq = wq_selected.transpose(-2, -1)
            wk = wk_selected.transpose(-2, -1)
            wv = wv_selected.transpose(-2, -1)

            # Concatenate Q, K, V: [hidden_size, local_q_dim + local_k_dim + local_v_dim]
            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        # Concatenate across devices: [hidden_size, total_qkv_dim]
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_size, total_qkv_dim]

        # Clean mesh mapping using MeshConfig
        col_mesh_mapper = self.mesh_config.column_parallel(mesh_device)
        row_mesh_mapper = self.mesh_config.row_parallel(mesh_device)

        # Match tt-transformers: use ShardTensor2dMesh for wqkv (attention.py line 240-242)
        # dims=(2, 3) shards along the last two dimensions of the 4D tensor
        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Handle biases - create fused QKV bias
        q_proj_bias = substate(state_dict, "q_proj")["bias"]
        k_proj_bias = substate(state_dict, "k_proj")["bias"]
        v_proj_bias = substate(state_dict, "v_proj")["bias"]

        qkv_bias_list = []
        for i in range(self.mesh_config.tp):
            q_bias_selected = torch.chunk(q_proj_bias, self.mesh_config.tp, dim=0)[i]
            k_bias_selected = torch.chunk(k_proj_bias, self.mesh_config.tp, dim=0)[i]
            v_bias_selected = torch.chunk(v_proj_bias, self.mesh_config.tp, dim=0)[i]
            qkv_bias = torch.cat([q_bias_selected, k_bias_selected, v_bias_selected], dim=-1)
            qkv_bias_list.append(qkv_bias)

        qkv_bias_cat = torch.cat(qkv_bias_list, dim=-1)  # [total_qkv_dim]
        # Match tt-transformers: use ShardTensorToMesh for bias (attention.py line 173)
        self.wqkv_bias = ttnn.as_tensor(
            qkv_bias_cat,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,  # tt-transformers uses bfloat16 for bias
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Attention sinks (gpt_oss specific feature)
        sinks = state_dict["sinks"].reshape(1, hf_config.num_attention_heads, 1, 1)
        decode_sinks = torch.nn.functional.pad(
            sinks.view(-1, 1), (0, ttnn.TILE_SIZE - sinks.shape[-1]), "constant", value=0.0
        )
        decode_sinks /= self.scaling

        if self.mesh_config.tp > 1:
            o_proj_bias = torch.cat([o_proj_bias] + [torch.zeros_like(o_proj_bias)] * (self.mesh_config.tp - 1), dim=-1)

        self.o_proj = ttnn.as_tensor(
            o_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=row_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.o_proj_bias = ttnn.as_tensor(
            o_proj_bias,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "o_proj_bias"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.decode_sinks = ttnn.as_tensor(
            decode_sinks,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=self.mesh_config.row_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "decode_sinks"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.sinks = ttnn.as_tensor(
            sinks,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=self.mesh_config.sequence_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "sinks"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Set up kv cache
        # Store paged attention config for later use in operations
        self.paged_attention_config = paged_attention_config

        # The cache shape will be different for paged vs non-paged attention
        if create_kv_cache:
            self.init_kv_cache(mesh_device, tensor_cache_path)
        grid_size = mesh_device.compute_with_storage_grid_size()
        # Fix: k/v tensors should be [1, num_local_kv_heads, 1, head_dim] for decode, not [1, 1, num_local_kv_heads, head_dim]
        kv_shape = (1, self.num_local_kv_heads, 1, self.head_dim)
        kv_shard_height = nearest_y(kv_shape[1], ttnn.TILE_SIZE)  # height = num_local_kv_heads
        kv_shard_width = kv_shape[3]  # width = head_dim
        kv_num_cores = kv_shape[2]  # cores = 1 (sequence length for decode)
        kv_core_grid = ttnn.num_cores_to_corerangeset(kv_num_cores, grid_size, row_wise=True)
        self.kv_mem_cfg = ttnn.create_sharded_memory_config(
            shape=(kv_shard_height, kv_shard_width),
            core_grid=kv_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )

        # SDPA setup
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=0,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def init_kv_cache(self, mesh_device, tensor_cache_path):
        """Initialize KV cache for both paged and non-paged attention (like tt_transformers)"""
        if self.paged_attention_config:
            # Paged attention cache shape: [max_num_blocks, num_kv_heads, block_size, head_dim]
            cache_shape = [
                self.paged_attention_config.max_num_blocks,
                self.num_kv_heads,
                self.paged_attention_config.block_size,
                self.head_dim,
            ]
        else:
            # Standard cache shape: [batch_size, num_kv_heads, max_seq_len, head_dim]
            cache_shape = [1, self.num_kv_heads, MAX_SEQ_LEN, self.head_dim]

        k_cache = ttnn.as_tensor(
            torch.zeros(cache_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=self.mesh_config.sequence_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, f"k_cache_{cache_shape}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_cache = ttnn.as_tensor(
            torch.zeros(cache_shape),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=self.mesh_config.sequence_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, f"v_cache_{cache_shape}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.kv_cache = [k_cache, v_cache]

        # Set layer_past to reference the actual KV cache for tt_transformers compatibility
        self.layer_past = self.kv_cache

    def __call__(self, x: ttnn.Tensor, mask, rope_mats, position_idx=None, page_table=None, kv_cache=None):
        batch_size, seq_len, hidden_size = x.shape

        # Determine if we're in decode or prefill mode
        is_decode_mode = batch_size * seq_len == 1
        mode = "decode" if is_decode_mode else "prefill"

        # QKV projection: single matmul + bias
        xqkv_fused = ttnn.matmul(x, self.wqkv, dtype=ttnn.bfloat16)
        xqkv_fused = ttnn.add(xqkv_fused, self.wqkv_bias, output_tensor=xqkv_fused)

        if is_decode_mode:
            # Decode mode: Use nlp_create_qkv_heads_decode for efficient head creation
            # Creates heads with HEIGHT_SHARDED output for efficient RoPE
            tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads_decode(
                xqkv_fused,
                num_heads=self.num_local_heads,
                num_kv_heads=self.num_local_kv_heads,
                memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            )

        else:
            # Prefill mode: Use nlp_create_qkv_heads for efficient head creation
            tt_q, tt_k, tt_v = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.num_local_heads,
                num_kv_heads=self.num_local_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Output shapes: Q [1, num_local_heads, seq_len, head_dim], K/V [1, num_local_kv_heads, seq_len, head_dim]

        ttnn.deallocate(xqkv_fused)

        # Apply RoPE using ttnn.experimental.rotary_embedding_llama (matches tt-transformers)

        tt_q = ttnn.experimental.rotary_embedding_llama(
            tt_q, rope_mats[0], rope_mats[1], self.transformation_mats[mode], is_decode_mode=is_decode_mode
        )
        tt_k = ttnn.experimental.rotary_embedding_llama(
            tt_k, rope_mats[0], rope_mats[1], self.transformation_mats[mode], is_decode_mode=is_decode_mode
        )

        # After RoPE in decode mode, convert back to interleaved for KV cache update
        if is_decode_mode:
            tt_q = ttnn.to_memory_config(tt_q, ttnn.DRAM_MEMORY_CONFIG)
            tt_k = ttnn.to_memory_config(tt_k, ttnn.DRAM_MEMORY_CONFIG)

        if is_decode_mode:
            # Use external kv_cache if provided (like tt-transformers), otherwise use internal cache
            if kv_cache:
                k_cache, v_cache = kv_cache[0], kv_cache[1]
            else:
                k_cache, v_cache = self.kv_cache

            tt_k = ttnn.to_memory_config(tt_k, self.kv_mem_cfg)
            tt_v = ttnn.to_memory_config(tt_v, self.kv_mem_cfg)
            # Paged attention path - like tt-transformers
            ttnn.experimental.paged_update_cache(
                k_cache,
                tt_k,
                update_idxs_tensor=position_idx,
                page_table=page_table,
            )
            ttnn.experimental.paged_update_cache(
                v_cache,
                tt_v,
                update_idxs_tensor=position_idx,
                page_table=page_table,
            )

            tt_k.deallocate(True)
            tt_v.deallocate(True)

            # Use paged SDPA when page_table is available (like tt-transformers)
            if page_table is not None:
                tt_sdpa_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    tt_q,
                    k_cache,
                    v_cache,
                    cur_pos_tensor=position_idx,
                    sliding_window_size=self.sliding_window,
                    attention_sink=self.decode_sinks,
                    page_table_tensor=page_table,
                    scale=self.scaling,
                    program_config=self.sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                tt_sdpa_tensor = ttnn.transformer.scaled_dot_product_attention_decode(
                    tt_q,
                    k_cache,
                    v_cache,
                    cur_pos_tensor=position_idx,
                    sliding_window_size=self.sliding_window,
                    attention_sink=self.decode_sinks,
                    scale=self.scaling,
                    program_config=self.sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            tt_q.deallocate(True)

            tt_sdpa_tensor = ttnn.transpose(tt_sdpa_tensor, 1, 2)
            tt_sdpa_out = ttnn.experimental.nlp_concat_heads(
                tt_sdpa_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )  # [1, 1, num_tokens, dim * nh]
            tt_sdpa_tensor.deallocate(True)
        else:
            # Fill cache (prefill mode)
            assert batch_size == 1, f"Only batch 1 supported, but got {batch_size=}"

            # Use external kv_cache if provided (like tt-transformers), otherwise use internal cache
            if kv_cache:
                k_cache, v_cache = kv_cache[0], kv_cache[1]
            else:
                k_cache, v_cache = self.kv_cache

            # Transpose and cast tensors for cache compatibility
            # After nlp_create_qkv_heads: K/V are [1, num_local_kv_heads, seq_len, head_dim]
            # Cache expects: [batch, num_kv_heads, seq_len, head_dim]
            tt_k = ttnn.typecast(tt_k, k_cache.dtype)

            tt_v = ttnn.typecast(tt_v, v_cache.dtype)

            # Use paged fill cache when page_table is available (like tt-transformers)
            if page_table is not None:
                # In the case that the tokens have been padded along the seq len dimension, we need to fill the cache with the unpadded k/v values.
                # Assume that the page table does not have padding, so we can use it to get the unpadded page len.
                block_size = k_cache.shape[2]

                page_len = page_table.shape[1] * block_size
                tt_k_sliced = tt_k[:, :, :page_len, :] if page_len < tt_k.shape[2] else tt_k
                tt_v_sliced = tt_v[:, :, :page_len, :] if page_len < tt_v.shape[2] else tt_v

                ttnn.experimental.paged_fill_cache(k_cache, tt_k_sliced, page_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v_cache, tt_v_sliced, page_table, batch_idx=0)
            else:
                ttnn.fill_cache(
                    k_cache,
                    tt_k,
                    batch_idx=0,
                )
                ttnn.fill_cache(
                    v_cache,
                    tt_v,
                    batch_idx=0,
                )

            # Transpose tensors back for SDPA computation

            # Reshape for custom SDPA: Q [seq_len, num_kv_heads, Q/K ratio, head_dim], K/V [seq_len, num_kv_heads, head_dim]
            # Q: [1, num_local_heads, seq_len, head_dim] -> transpose -> [1, seq_len, num_local_heads, head_dim]
            tt_q = ttnn.transpose(tt_q, 1, 2)  # [1, seq_len, num_local_heads, head_dim]
            tt_q = ttnn.reshape(
                tt_q, [seq_len, self.num_local_kv_heads, self.num_local_heads // self.num_local_kv_heads, self.head_dim]
            )
            tt_k = ttnn.reshape(tt_k, [seq_len, self.num_local_kv_heads, self.head_dim])
            tt_v = ttnn.reshape(tt_v, [seq_len, self.num_local_kv_heads, self.head_dim])

            tt_sdpa_out, _ = tt_sdpa(
                tt_q,
                tt_k,
                tt_v,
                self.sinks,
                sm_scale=self.scaling,
                tt_mask=mask,
                tt_cache=None,
                position_idx=None,
            )

        tt_out = ttnn.matmul(tt_sdpa_out, self.o_proj, dtype=ttnn.bfloat16)
        tt_sdpa_out.deallocate(True)
        tt_out = ttnn.add(tt_out, self.o_proj_bias, output_tensor=tt_out)

        tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, self.hidden_size))

        # Clean tensor parallel communication (with performance padding)
        if self.mesh_config.tp > 1:
            tt_out = ttnn.unsqueeze(tt_out, 0)
            tt_out = self.mesh_config.allreduce(
                tt_out, self.ccl_manager, pad_size=192 if self.mesh_config.tp == 8 else 0, axis=self.mesh_config.tp_axis
            )
            tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, self.hidden_size))

        return tt_out
