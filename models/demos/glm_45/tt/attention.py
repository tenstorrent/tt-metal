# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import nearest_y
from models.demos.glm_45.config import MeshConfig
from models.demos.glm_45.utils.general_utils import get_cache_file_name
from models.demos.glm_45.utils.substate import substate
from models.tt_transformers.tt.common import get_rot_transformation_mat

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
    ):
        self.layer_idx = layer_idx
        # Disable sliding window attention explicitly
        self.use_sliding_window = False
        self.scaling = hf_config.head_dim**-0.5
        self.head_dim = hf_config.head_dim
        self.num_heads = hf_config.num_attention_heads
        self.num_kv_heads = hf_config.num_key_value_heads
        self.use_qk_norm = getattr(hf_config, "use_qk_norm", False)
        self.rms_norm_eps = getattr(hf_config, "rms_norm_eps", 1e-6)

        # Use MeshConfig for clean parallelization
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.num_local_heads = self.mesh_config.shard_size(self.num_heads)
        self.num_local_kv_heads = self.mesh_config.shard_size(self.num_kv_heads)

        self.hidden_size = hf_config.hidden_size
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device

        # Extract projection weights and biases from state dict
        q_state = substate(state_dict, "q_proj")
        k_state = substate(state_dict, "k_proj")
        v_state = substate(state_dict, "v_proj")
        o_state = substate(state_dict, "o_proj")

        # Strict validation of required attention weights
        missing_attn = []
        if "weight" not in q_state:
            missing_attn.append("self_attn.q_proj.weight")
        if "weight" not in k_state:
            missing_attn.append("self_attn.k_proj.weight")
        if "weight" not in v_state:
            missing_attn.append("self_attn.v_proj.weight")
        if "weight" not in o_state:
            missing_attn.append("self_attn.o_proj.weight")
        if missing_attn:
            raise ValueError(
                f"Missing required attention weights at layer {layer_idx}: {missing_attn}. "
                f"Ensure all projection weights are present."
            )

        q_proj = q_state["weight"].transpose(-1, -2)
        q_proj_bias = q_state.get("bias")

        k_proj = k_state["weight"].transpose(-1, -2)
        k_proj_bias = k_state.get("bias")

        v_proj = v_state["weight"].transpose(-1, -2)
        v_proj_bias = v_state.get("bias")

        o_proj = o_state["weight"].transpose(-1, -2)
        # Some models (e.g., GLM-4.5-Air-1L) do not have o_proj.bias
        o_proj_bias = o_state.get("bias", torch.zeros(hf_config.hidden_size, dtype=torch.bfloat16))

        # Disable attention sinks by setting them to -inf (no contribution after softmax)
        sinks = torch.full((1, hf_config.num_attention_heads, 1, 1), float("-inf"), dtype=torch.bfloat16)
        decode_sinks = torch.full((hf_config.num_attention_heads, ttnn.TILE_SIZE), float("-inf"), dtype=torch.bfloat16)

        # Clean mesh mapping using MeshConfig
        col_mesh_mapper = self.mesh_config.column_parallel(mesh_device)
        row_mesh_mapper = self.mesh_config.row_parallel(mesh_device)

        self.q_proj = ttnn.as_tensor(
            q_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "q_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if q_proj_bias is not None:
            self.q_proj_bias = ttnn.as_tensor(
                q_proj_bias,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                mesh_mapper=col_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "q_proj_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.q_proj_bias = None

        self.k_proj = ttnn.as_tensor(
            k_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "k_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if k_proj_bias is not None:
            self.k_proj_bias = ttnn.as_tensor(
                k_proj_bias,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                mesh_mapper=col_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "k_proj_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.k_proj_bias = None

        self.v_proj = ttnn.as_tensor(
            v_proj,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "v_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Fused WQKV for decode (tensor-parallel column sharded)
        wqkv = torch.cat([q_proj, k_proj, v_proj], dim=-1)
        self.wqkv = ttnn.as_tensor(
            wqkv,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if q_proj_bias is not None or k_proj_bias is not None or v_proj_bias is not None:
            qb = q_proj_bias if q_proj_bias is not None else torch.zeros(q_proj.shape[-1], dtype=torch.bfloat16)
            kb = k_proj_bias if k_proj_bias is not None else torch.zeros(k_proj.shape[-1], dtype=torch.bfloat16)
            vb = v_proj_bias if v_proj_bias is not None else torch.zeros(v_proj.shape[-1], dtype=torch.bfloat16)
            wqkv_b = torch.cat([qb, kb, vb], dim=-1)
            self.wqkv_bias = ttnn.as_tensor(
                wqkv_b,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                mesh_mapper=col_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "wqkv_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.wqkv_bias = None
        if v_proj_bias is not None:
            self.v_proj_bias = ttnn.as_tensor(
                v_proj_bias,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                mesh_mapper=col_mesh_mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, "v_proj_bias"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            self.v_proj_bias = None

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
        # Reshape bias to [1, hidden] so last two dims are [1, hidden]
        # This ensures a valid ROW broadcast in binary_ng add
        self.o_proj_bias = ttnn.as_tensor(
            o_proj_bias.unsqueeze(0),
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

        # Static decode batch for tracing/memconfigs
        self.decode_batch = 2
        grid_size = mesh_device.compute_with_storage_grid_size()
        decode_core_grid = ttnn.num_cores_to_corerangeset(self.decode_batch, grid_size, row_wise=True)
        # Explicit height-sharded memcfgs pinned to static decode batch
        self.decode_heads_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=decode_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.sdpa_decode_out_memcfg = ttnn.create_sharded_memory_config(
            shape=(nearest_y(self.num_local_heads, ttnn.TILE_SIZE), self.head_dim),
            core_grid=decode_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # Precreate rotary transformation mat for decode (batch-sharded)
        trans_t = get_rot_transformation_mat(dhead=ttnn.TILE_SIZE).repeat(1, 1, self.decode_batch, 1)
        trans_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=decode_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.rotary_decode_trans_mat = ttnn.from_torch(
            trans_t,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=trans_memcfg,
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
        apply_rope, tt_cos, tt_sin = rope_mats

        # Decode path with static batch: conform to self.decode_batch by padding or slicing
        if seq_len == 1:
            # Use external kv_cache if provided (like tt-transformers), otherwise use internal cache
            if kv_cache:
                k_cache, v_cache = kv_cache[0], kv_cache[1]
            else:
                k_cache, v_cache = self.kv_cache

            # Use pre-created memcfgs and rotary transform for the static decode batch

            # Conform input batch to static decode_batch (pad if smaller, slice if larger)
            if batch_size < self.decode_batch:
                pad_bs = self.decode_batch - batch_size
                pad_tensor = ttnn.from_torch(
                    torch.zeros((pad_bs, 1, self.hidden_size), dtype=torch.bfloat16),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                )
                x_padded = ttnn.concat([x, pad_tensor], dim=0)
            elif batch_size > self.decode_batch:
                # Keep the first decode_batch entries to match static trace
                x_padded = ttnn.slice(x, (0, 0, 0), (self.decode_batch, 1, self.hidden_size))
            else:
                x_padded = x

            # Fused linear for WQKV on padded hidden states; linear expects [S, 1, B, H]
            x4d = ttnn.reshape(x_padded, (1, 1, self.decode_batch, self.hidden_size))
            xqkv_fused = ttnn.linear(
                x4d,
                self.wqkv,
                bias=self.wqkv_bias,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            (
                q_heads_pre_rot_1BQD,
                k_heads_pre_rot_1BKD,
                v_heads_1BKD,
            ) = ttnn.experimental.nlp_create_qkv_heads_decode(
                xqkv_fused,
                num_heads=self.num_local_heads,
                num_kv_heads=self.num_local_kv_heads,
                memory_config=self.decode_heads_memcfg,
            )
            ttnn.deallocate(xqkv_fused)

            # Optional Q/K RMSNorm before RoPE
            if self.use_qk_norm:
                q_heads_pre_rot_1BQD = ttnn.rms_norm(q_heads_pre_rot_1BQD, epsilon=self.rms_norm_eps)
                k_heads_pre_rot_1BKD = ttnn.rms_norm(k_heads_pre_rot_1BKD, epsilon=self.rms_norm_eps)

            # Apply RoPE using precreated transform (static batch)
            q_heads_1BQD = ttnn.experimental.rotary_embedding_llama(
                q_heads_pre_rot_1BQD, tt_cos, tt_sin, self.rotary_decode_trans_mat, is_decode_mode=True
            )
            k_heads_1BKD = ttnn.experimental.rotary_embedding_llama(
                k_heads_pre_rot_1BKD, tt_cos, tt_sin, self.rotary_decode_trans_mat, is_decode_mode=True
            )
            ttnn.deallocate(q_heads_pre_rot_1BQD)
            ttnn.deallocate(k_heads_pre_rot_1BKD)

            # Pad position idx and page table to static batch if needed
            if position_idx is not None and position_idx.shape[0] != self.decode_batch:
                if position_idx.shape[0] > self.decode_batch:
                    position_idx = ttnn.slice(position_idx, (0,), (self.decode_batch,))
                else:
                    pad_bs = self.decode_batch - position_idx.shape[0]
                    if pad_bs > 0:
                        last_idx = ttnn.slice(position_idx, (position_idx.shape[0] - 1,), (position_idx.shape[0],))
                        pad_pos = ttnn.concat([last_idx] * pad_bs, dim=0)
                        position_idx = ttnn.concat([position_idx, pad_pos], dim=0)

            if page_table is not None and hasattr(page_table, "shape") and page_table.shape[0] != self.decode_batch:
                if page_table.shape[0] > self.decode_batch:
                    _shp = list(page_table.shape)
                    _dims = len(_shp)
                    _start = tuple([0] * _dims)
                    _end = (self.decode_batch,) + tuple(int(x) for x in _shp[1:])
                    page_table = ttnn.slice(page_table, _start, _end)
                else:
                    pad_bs = self.decode_batch - page_table.shape[0]
                    if pad_bs > 0:
                        # Duplicate last row 'pad_bs' times along batch dim (dim 0)
                        _shp = list(page_table.shape)
                        _dims = len(_shp)
                        start = (_shp[0] - 1,) + tuple([0] * (_dims - 1))
                        end = tuple(int(x) for x in _shp)
                        last_row = ttnn.slice(page_table, start, end)
                        pads = [last_row] * pad_bs
                        page_table = ttnn.concat([page_table] + pads, dim=0)

            # KV update: v_heads already in proper sharded layout
            try:
                logger.info(
                    f"Decode KV update (fused): q={q_heads_1BQD.shape} k={k_heads_1BKD.shape} v={v_heads_1BKD.shape} pt_shape={getattr(page_table,'shape',None)} pos_shape={getattr(position_idx,'shape',None)}"
                )
            except Exception:
                pass
            ttnn.experimental.paged_update_cache(
                k_cache,
                k_heads_1BKD,
                update_idxs_tensor=position_idx,
                page_table=page_table,
            )
            ttnn.experimental.paged_update_cache(
                v_cache,
                v_heads_1BKD,
                update_idxs_tensor=position_idx,
                page_table=page_table,
            )
            ttnn.deallocate(v_heads_1BKD)

            # Decode SDPA
            try:
                logger.info(
                    f"SDPA decode call: q={q_heads_1BQD.shape} cacheK_mc={k_cache.memory_config()} cacheV_mc={v_cache.memory_config()} pt_shape={getattr(page_table,'shape',None)} pos={getattr(position_idx,'shape',None)}"
                )
            except Exception:
                pass
            if page_table is not None:
                tt_sdpa_tensor = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_heads_1BQD,
                    k_cache,
                    v_cache,
                    cur_pos_tensor=position_idx,
                    is_causal=True,
                    attn_mask=None,
                    attention_sink=self.decode_sinks,
                    page_table_tensor=page_table,
                    scale=self.scaling,
                    program_config=self.sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                tt_sdpa_tensor = ttnn.transformer.scaled_dot_product_attention_decode(
                    q_heads_1BQD,
                    k_cache,
                    v_cache,
                    cur_pos_tensor=position_idx,
                    is_causal=True,
                    attn_mask=None,
                    attention_sink=self.decode_sinks,
                    scale=self.scaling,
                    program_config=self.sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            ttnn.deallocate(q_heads_1BQD)
            try:
                logger.info(f"SDPA output shape={tt_sdpa_tensor.shape}")
            except Exception:
                pass

            # SDPA output: use precreated memcfg (static batch)
            # Move SDPA output to explicit height-sharded memcfg, then concat heads
            attn_output_11BH = ttnn.to_memory_config(tt_sdpa_tensor, self.sdpa_decode_out_memcfg)
            try:
                logger.info(f"Post-SDPA to_memcfg attn_output_11BH={attn_output_11BH.shape}")
            except Exception:
                pass
            attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
                attn_output_11BH, num_heads=self.num_local_heads
            )
            try:
                logger.info(f"Post-concat heads attn_output_cat={attn_output_cat.shape}")
            except Exception:
                pass
            ttnn.deallocate(attn_output_11BH)
            ttnn.deallocate(tt_sdpa_tensor)
            # Move to DRAM TILE for output matmul
            tt_sdpa_out = ttnn.to_memory_config(attn_output_cat, ttnn.DRAM_MEMORY_CONFIG)
            try:
                logger.info(f"Post-to-DRAM tt_sdpa_out={tt_sdpa_out.shape}")
            except Exception:
                pass
            ttnn.deallocate(attn_output_cat)
            # Set logical batch dimension to actual batch_size (physical remains 32)
            try:
                feat_in = self.num_local_heads * self.head_dim
                tt_sdpa_out = ttnn.reshape(tt_sdpa_out, (1, 1, batch_size, feat_in), (1, 1, ttnn.TILE_SIZE, feat_in))
                logger.info(f"After logical reshape tt_sdpa_out={tt_sdpa_out.shape}")
            except Exception:
                pass
        else:
            # Prefill path: apply optional Q/K RMSNorm and RoPE before SDPA
            tt_q = ttnn.matmul(x, self.q_proj)
            if self.q_proj_bias is not None:
                tt_q = ttnn.add(tt_q, self.q_proj_bias, output_tensor=tt_q)
            tt_q = ttnn.reshape(tt_q, [1, seq_len * batch_size, -1, self.head_dim])

            tt_k = ttnn.matmul(x, self.k_proj)
            if self.k_proj_bias is not None:
                tt_k = ttnn.add(tt_k, self.k_proj_bias, output_tensor=tt_k)
            tt_k = ttnn.reshape(tt_k, [1, seq_len * batch_size, -1, self.head_dim])

            tt_v = ttnn.matmul(x, self.v_proj)
            if self.v_proj_bias is not None:
                tt_v = ttnn.add(tt_v, self.v_proj_bias, output_tensor=tt_v)
            tt_v = ttnn.reshape(tt_v, [1, seq_len * batch_size, -1, self.head_dim])
            if self.use_qk_norm:
                tt_q = ttnn.rms_norm(tt_q, epsilon=self.rms_norm_eps)
                tt_k = ttnn.rms_norm(tt_k, epsilon=self.rms_norm_eps)

            tt_q_rope = apply_rope(tt_q, tt_cos, tt_sin)
            tt_q.deallocate(True)
            tt_q = tt_q_rope
            tt_k_rope = apply_rope(tt_k, tt_cos, tt_sin)
            tt_k.deallocate(True)
            tt_k = tt_k_rope
            # Fill cache (prefill mode)
            assert batch_size == 1, f"Only batch 1 supported, but got {batch_size=}"

            # Use external kv_cache if provided (like tt-transformers), otherwise use internal cache
            if kv_cache:
                k_cache, v_cache = kv_cache[0], kv_cache[1]
            else:
                k_cache, v_cache = self.kv_cache

            # Transpose and cast tensors for cache compatibility
            tt_k = ttnn.transpose(tt_k, 1, 2)
            tt_k = ttnn.typecast(tt_k, k_cache.dtype)

            tt_v = ttnn.transpose(tt_v, 1, 2)
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
            tt_k = ttnn.transpose(tt_k, 1, 2)
            tt_v = ttnn.transpose(tt_v, 1, 2)

            tt_q = ttnn.reshape(tt_q, [batch_size * seq_len, -1, self.num_local_heads, self.head_dim])
            tt_k = ttnn.reshape(tt_k, [batch_size * seq_len, -1, self.head_dim])
            tt_v = ttnn.reshape(tt_v, [batch_size * seq_len, -1, self.head_dim])

            # Disable sliding window mask and attention sinks in prefill
            tt_sdpa_out, _ = tt_sdpa(
                tt_q,
                tt_k,
                tt_v,
                self.sinks,
                sm_scale=self.scaling,
                tt_mask=None,
                tt_cache=None,
                position_idx=None,
            )
            tt_q.deallocate(True)
            tt_k.deallocate(True)
            tt_v.deallocate(True)

        tt_out = ttnn.matmul(tt_sdpa_out, self.o_proj, dtype=ttnn.bfloat16)
        try:
            logger.info(f"Post-o_proj matmul tt_out={tt_out.shape}")
        except Exception:
            pass
        tt_sdpa_out.deallocate(True)
        tt_out = ttnn.add(tt_out, self.o_proj_bias, output_tensor=tt_out)

        # Return shape should match input x: [batch_size, seq_len, hidden]
        if seq_len == 1:
            # Convert [1, 1, batch, hidden] -> [batch, 1, hidden] using transpose + squeeze
            tt_out = ttnn.transpose(tt_out, 0, 2)
            tt_out = ttnn.squeeze(tt_out, dim=2)
            # For static decode path, return early to avoid TP allreduce reshape complexities
            return tt_out
        else:
            tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, self.hidden_size))

        # Clean tensor parallel communication (with performance padding)
        if self.mesh_config.tp > 1:
            tt_out = ttnn.unsqueeze(tt_out, 0)
            tt_out = self.mesh_config.allreduce(tt_out, self.ccl_manager, pad_size=192, axis=self.mesh_config.tp_axis)
            tt_out = ttnn.reshape(tt_out, (batch_size, seq_len, self.hidden_size))

        return tt_out
