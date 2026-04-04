# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 Attention Module

Custom attention implementation for Gemma 4 E4B that handles:
- Dual head_dim (256 for sliding, 512 for global attention layers)
- Partial rotary embeddings (25% of dims for global layers)
- V-norm (RMSNorm without learnable scale)
- Attention scale = head_dim^-0.5 (QK norms replace sqrt(d) scaling)

Note: KV cache sharing is not implemented in this initial version.
All layers compute their own QKV independently.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode


class Gemma4Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.layer_num = layer_num
        self.dtype = dtype
        self.prefetcher = prefetcher

        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_kv_heads = configuration.n_kv_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.paged_attention_config = paged_attention_config
        self.tile_size = configuration.tile_size
        self.rms_norm_add_unit_offset = configuration.rms_norm_add_unit_offset

        # Per-layer head_dim (varies between sliding=256 and global=512)
        self.head_dim = args.get_layer_head_dim(layer_num)
        self.rotary_dim = args.get_layer_rotary_dim(layer_num)  # 256 for sliding, 128 for global
        self.is_sliding = args.is_sliding_layer(layer_num)
        self.sliding_window = configuration.sliding_window if self.is_sliding else None

        # Per-layer QKV size
        self.qkv_size = args.get_layer_qkv_size(layer_num)

        # N150: single device
        self.n_local_heads = self.n_heads  # 8
        self.n_local_kv_heads = self.n_kv_heads  # 2

        # Attention scale
        self.scale = self.head_dim**-0.5

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.model_config = configuration.get_model_config()
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN

        layer_name = configuration.get_state_dict_prefix("Attention", layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        # Weight key names
        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Build fused QKV weight [1, 1, dim, n_heads*hd + 2*n_kv_heads*hd]
        wq = torch.transpose(state_dict[f"{wq_str}.weight"], -2, -1)
        wk = torch.transpose(state_dict[f"{wk_str}.weight"], -2, -1)
        wv = torch.transpose(state_dict[f"{wv_str}.weight"], -2, -1)
        qkv = torch.cat([wq, wk, wv], dim=-1).unsqueeze(0).unsqueeze(0)

        self.wqkv = ttnn.as_tensor(
            qkv,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=cache_name("wqkv"),
        )

        # WO projection
        pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=cache_name("wo"),
        )

        # QK norms
        def norm_reshard(x, norm, mode, norm_config):
            if mode == Mode.DECODE:
                mem_cfg = x.memory_config()
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=x.dtype)
            x = norm(x, mode, norm_config=norm_config)
            if mode == Mode.DECODE:
                x = ttnn.to_memory_config(x, mem_cfg, dtype=x.dtype)
            return x

        if f"{q_norm_str}.weight" in state_dict:
            fn_q_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=q_norm_str,
                add_unit_offset=self.rms_norm_add_unit_offset,
                is_distributed=False,
                tt_ccl=self.tt_ccl,
            )
            self.q_norm = lambda x, mode, norm_config: norm_reshard(x, fn_q_norm, mode, norm_config)
        else:
            self.q_norm = lambda x, mode, norm_config: x

        if f"{k_norm_str}.weight" in state_dict:
            fn_k_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=k_norm_str,
                add_unit_offset=self.rms_norm_add_unit_offset,
                is_distributed=False,
                tt_ccl=self.tt_ccl,
            )
            self.k_norm = lambda x, mode, norm_config: norm_reshard(x, fn_k_norm, mode, norm_config)
        else:
            self.k_norm = lambda x, mode, norm_config: x

        # V-norm: RMSNorm without learnable scale
        # Gemma 4 V-norm has with_scale=False => no weight parameter
        # With add_unit_offset=True, zeros weight => effective weight = 0 + 1 = 1 (identity scale)
        if getattr(args, "use_v_norm", False):
            v_norm_key = f"{layer_name}.v_norm"
            v_norm_state = {f"{v_norm_key}.weight": torch.zeros(self.head_dim)}
            fn_v_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=v_norm_state,
                state_dict_prefix=None,
                weight_cache_path=None,  # Constant weights, no caching needed
                weight_dtype=ttnn.bfloat16,
                weight_key=v_norm_key,
                add_unit_offset=True,
                is_distributed=False,
                tt_ccl=self.tt_ccl,
            )
            self.v_norm = lambda x, mode, norm_config: norm_reshard(x, fn_v_norm, mode, norm_config)
        else:
            self.v_norm = lambda x, mode, norm_config: x

        # Initialize KV cache
        if not use_paged_kv_cache:
            self.init_kv_cache(configuration, weight_cache_path)
        else:
            self.layer_past = None

    def init_kv_cache(self, configuration, weight_cache_path):
        """Initialize empty KV cache with per-layer head_dim."""
        if self.paged_attention_config:
            cache_shape = (
                self.paged_attention_config.max_num_blocks,
                self.n_local_kv_heads,
                self.paged_attention_config.block_size,
                self.head_dim,
            )
        else:
            cache_shape = (
                self.max_batch_size,
                self.n_local_kv_heads,
                self.max_seq_len,
                self.head_dim,
            )

        cache_k = torch.zeros(cache_shape)
        cache_v = torch.zeros(cache_shape)
        kv_cache_dtype = ttnn.bfloat8_b

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=kv_cache_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=(
                    f"{weight_cache_path}/kvcache_l{self.layer_num}_{k_or_v.shape}"
                    if weight_cache_path and not configuration.dummy_weights
                    else None
                ),
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        """Prefill forward pass."""
        batch_size = x_11SH.shape[0]
        if batch_size > 1:
            x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

        seq_len = x_11SH.shape[-2]
        original_seq_len = seq_len

        # Reshape for large sequences
        if seq_len > self.MAX_QKV_MM_SEQ_LEN and seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
            padded_seq_len = (
                (seq_len + self.MAX_QKV_MM_SEQ_LEN - 1) // self.MAX_QKV_MM_SEQ_LEN
            ) * self.MAX_QKV_MM_SEQ_LEN
            pad_len = padded_seq_len - seq_len
            x_11SH = ttnn.pad(x_11SH, padding=[(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)
            seq_len = padded_seq_len

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        # QKV matmul
        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])
        if original_seq_len != seq_len:
            xqkv_fused = xqkv_fused[:, :, :original_seq_len, :]
            seq_len = original_seq_len

        if batch_size > 1:
            xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len // batch_size, -1])

        ttnn.deallocate(x_11SH)

        # Split QKV into heads
        (
            q_heads_pre_rot,
            k_heads_pre_rot,
            v_heads,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # QK norms
        norm_config = self.args.get_norm_config("attn", Mode.PREFILL, None)
        q_heads_pre_rot = self.q_norm(q_heads_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)
        k_heads_pre_rot = self.k_norm(k_heads_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)

        # V-norm
        v_heads = self.v_norm(v_heads, mode=Mode.PREFILL, norm_config=norm_config)

        # Apply partial rotary embeddings
        q_heads, k_heads = self._apply_rotary_prefill(q_heads_pre_rot, k_heads_pre_rot, rot_mats)
        ttnn.deallocate(q_heads_pre_rot)
        ttnn.deallocate(k_heads_pre_rot)

        # Update KV cache
        if kv_cache:
            keys, values = kv_cache[0], kv_cache[1]
        else:
            keys, values = self.layer_past[0], self.layer_past[1]

        ttnn.experimental.paged_fill_cache(keys, k_heads, page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(values, v_heads, page_table, batch_idx=user_id)

        # SDPA
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=True,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # Concat heads and output projection
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        dense_out = ttnn.linear(
            attn_output,
            self.wo,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(attn_output)

        return dense_out

    def forward_decode(self, x, current_pos, rot_mats=None, page_table=None, kv_cache=None):
        """Decode forward pass."""
        xqkv_fused = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(x)

        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, self.max_batch_size, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))

        (
            q_heads_pre_rot,
            k_heads_pre_rot,
            v_heads,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        norm_config = self.args.get_norm_config("attn", Mode.DECODE, None)
        q_heads_pre_rot = self.q_norm(q_heads_pre_rot, mode=Mode.DECODE, norm_config=norm_config)
        k_heads_pre_rot = self.k_norm(k_heads_pre_rot, mode=Mode.DECODE, norm_config=norm_config)
        v_heads = self.v_norm(v_heads, mode=Mode.DECODE, norm_config=norm_config)

        # Apply rotary
        q_heads, k_heads = self._apply_rotary_decode(q_heads_pre_rot, k_heads_pre_rot, rot_mats, current_pos)
        ttnn.deallocate(q_heads_pre_rot)
        ttnn.deallocate(k_heads_pre_rot)

        # KV cache update
        if kv_cache:
            keys, values = kv_cache[0], kv_cache[1]
        else:
            keys, values = self.layer_past[0], self.layer_past[1]

        ttnn.experimental.paged_update_cache(keys, k_heads, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(values, v_heads, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        # SDPA decode
        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            q_heads,
            keys,
            values,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads)

        attn_output = ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)
        attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=self.n_local_heads)

        dense_out = ttnn.linear(
            attn_output,
            self.wo,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(attn_output)

        return dense_out

    def _apply_rotary_prefill(self, q_heads, k_heads, rot_mats):
        """Apply rotary embeddings for prefill. Handles partial rotary for global layers."""
        if self.rotary_dim == self.head_dim:
            # Full rotation (sliding layers): apply RoPE to all dims
            if q_heads.dtype != ttnn.bfloat16:
                q_heads = ttnn.typecast(q_heads, dtype=ttnn.bfloat16)
            if k_heads.dtype != ttnn.bfloat16:
                k_heads = ttnn.typecast(k_heads, dtype=ttnn.bfloat16)

            q_rot = ttnn.experimental.rotary_embedding(q_heads, rot_mats[0], rot_mats[1])
            k_rot = ttnn.experimental.rotary_embedding(k_heads, rot_mats[0], rot_mats[1])
            return q_rot, k_rot

        # Partial rotation (global layers): head_dim=512, rotary_dim=128
        # Split into rotary and pass-through parts
        rotary_dim = self.rotary_dim

        q_rot_part = q_heads[:, :, :, :rotary_dim]
        q_pass_part = q_heads[:, :, :, rotary_dim:]
        k_rot_part = k_heads[:, :, :, :rotary_dim]
        k_pass_part = k_heads[:, :, :, rotary_dim:]

        if q_rot_part.dtype != ttnn.bfloat16:
            q_rot_part = ttnn.typecast(q_rot_part, dtype=ttnn.bfloat16)
        if k_rot_part.dtype != ttnn.bfloat16:
            k_rot_part = ttnn.typecast(k_rot_part, dtype=ttnn.bfloat16)

        q_rotated = ttnn.experimental.rotary_embedding(q_rot_part, rot_mats[0], rot_mats[1])
        k_rotated = ttnn.experimental.rotary_embedding(k_rot_part, rot_mats[0], rot_mats[1])

        q_out = ttnn.concat([q_rotated, q_pass_part], dim=-1)
        k_out = ttnn.concat([k_rotated, k_pass_part], dim=-1)

        return q_out, k_out

    def _apply_rotary_decode(self, q_heads, k_heads, rot_mats, current_pos):
        """Apply rotary embeddings for decode. Handles partial rotary for global layers."""
        int_current_pos = int(ttnn.to_torch(ttnn.get_device_tensors(current_pos)[0])[0])

        if self.rotary_dim == self.head_dim:
            # Full rotation (sliding layers)
            q_rot = ttnn.experimental.rotary_embedding(q_heads, rot_mats[0], rot_mats[1], int_current_pos)
            k_rot = ttnn.experimental.rotary_embedding(k_heads, rot_mats[0], rot_mats[1], int_current_pos)

            q_rot = ttnn.reshape(
                q_rot,
                (1, self.max_batch_size, self.n_local_heads, self.head_dim),
                (1, self.max_batch_size, 32, self.head_dim),
            )
            k_rot = ttnn.reshape(
                k_rot,
                (1, self.max_batch_size, self.n_local_kv_heads, self.head_dim),
                (1, self.max_batch_size, 32, self.head_dim),
            )
            q_rot = q_rot[:, :, : self.n_local_heads]
            k_rot = k_rot[:, :, : self.n_local_kv_heads]
            return q_rot, k_rot

        # Partial rotation (global layers)
        rotary_dim = self.rotary_dim

        q_rot_part = q_heads[:, :, :, :rotary_dim]
        q_pass_part = q_heads[:, :, :, rotary_dim:]
        k_rot_part = k_heads[:, :, :, :rotary_dim]
        k_pass_part = k_heads[:, :, :, rotary_dim:]

        q_rotated = ttnn.experimental.rotary_embedding(q_rot_part, rot_mats[0], rot_mats[1], int_current_pos)
        k_rotated = ttnn.experimental.rotary_embedding(k_rot_part, rot_mats[0], rot_mats[1], int_current_pos)

        q_rotated = ttnn.reshape(
            q_rotated,
            (1, self.max_batch_size, self.n_local_heads, rotary_dim),
            (1, self.max_batch_size, 32, rotary_dim),
        )
        k_rotated = ttnn.reshape(
            k_rotated,
            (1, self.max_batch_size, self.n_local_kv_heads, rotary_dim),
            (1, self.max_batch_size, 32, rotary_dim),
        )
        q_rotated = q_rotated[:, :, : self.n_local_heads]
        k_rotated = k_rotated[:, :, : self.n_local_kv_heads]

        q_out = ttnn.concat([q_rotated, q_pass_part], dim=-1)
        k_out = ttnn.concat([k_rotated, k_pass_part], dim=-1)

        return q_out, k_out

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        """Dispatch to prefill or decode."""
        if mode == Mode.PREFILL:
            return self.forward_prefill(x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx, kv_cache)
        else:
            return self.forward_decode(x, current_pos, rot_mats, page_table, kv_cache)
