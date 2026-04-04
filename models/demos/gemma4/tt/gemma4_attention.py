# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 Attention Module

Custom attention implementation for Gemma 4 E4B that handles:
- Dual head_dim (256 for sliding, 512 for global attention layers)
- Partial rotary embeddings (25% of dims for global layers)
- V-norm (RMSNorm without learnable scale)
- Attention scale = 1.0 (QK norms replace sqrt(d) scaling)
- KV cache sharing: layers 24-41 share KV from source layers (22/23)
  - Shared layers skip K/V projection and cache update
  - Only Q is computed with norm + rotary
  - Source layer's processed K/V (with rotary) is reused for SDPA
"""

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import num_to_corerange


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

        # Attention scale: Gemma 4 uses QK norms, so attention scaling = 1.0 (no additional sqrt(d) scaling)
        # HF Gemma4TextAttention sets self.scaling = 1.0 (line 1143 in modeling_gemma4.py)
        self.scale = 1.0

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        # Per-layer optimizer-configured compute kernels and dtypes
        from models.tt_transformers.tt.model_config import OpGroup, TensorGroup

        decoders_optimizations = configuration.decoders_optimizations
        self.kv_cache_dtype = decoders_optimizations.get_tensor_dtype(decoder_id=layer_num, tensor=TensorGroup.KV_CACHE)
        self.li_qkv_prefill_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_PREFILL, configuration=configuration
        )
        self.li_o_prefill_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_PREFILL, configuration=configuration
        )
        self.li_qkv_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_DECODE, configuration=configuration
        )
        self.li_o_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_DECODE, configuration=configuration
        )
        self.sdpa_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_DECODE, configuration=configuration
        )

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
        # Gemma 4 V-norm has with_scale=False => no weight parameter in checkpoint
        # We use ones as weight (identity scale) since V-norm only normalizes, no learned scaling
        if getattr(args, "use_v_norm", False):
            v_norm_key = f"{layer_name}.v_norm"
            v_norm_state = {f"{v_norm_key}.weight": torch.ones(self.head_dim)}
            fn_v_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=v_norm_state,
                state_dict_prefix=None,
                weight_cache_path=None,  # Constant weights, no caching needed
                weight_dtype=ttnn.bfloat16,
                weight_key=v_norm_key,
                add_unit_offset=False,  # Gemma 4 uses direct scale, not offset
                is_distributed=False,
                tt_ccl=self.tt_ccl,
            )
            self.v_norm = lambda x, mode, norm_config: norm_reshard(x, fn_v_norm, mode, norm_config)
        else:
            self.v_norm = lambda x, mode, norm_config: x

        # KV sharing state: set by the model before forward for shared layers
        self.shared_kv = None  # Set to (k_heads, v_heads) for KV-shared layers
        self.last_k_heads = None  # Stored after forward for source layers
        self.last_v_heads = None

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
        # Use optimizer-configured KV cache dtype (BF16 for accuracy, BFP8 for performance)
        kv_cache_dtype = self.kv_cache_dtype or ttnn.bfloat8_b

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
        """
        Prefill forward pass with KV sharing support.

        If self.shared_kv is set (by the model), skips K/V projection and uses
        shared K/V from source layer. After forward, stores processed K/V in
        self.last_k_heads / self.last_v_heads for potential sharing with later layers.
        """
        batch_size = x_11SH.shape[0]
        if batch_size > 1:
            x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

        seq_len = x_11SH.shape[-2]
        original_seq_len = seq_len

        if self.shared_kv is not None:
            # KV-shared layer: only compute Q, reuse K/V from source layer
            k_heads, v_heads = self.shared_kv

            # Reshape for large sequences (same as full path)
            if seq_len > self.MAX_QKV_MM_SEQ_LEN and seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                padded_seq_len = (
                    (seq_len + self.MAX_QKV_MM_SEQ_LEN - 1) // self.MAX_QKV_MM_SEQ_LEN
                ) * self.MAX_QKV_MM_SEQ_LEN
                pad_len = padded_seq_len - seq_len
                x_11SH = ttnn.pad(x_11SH, padding=[(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)
                seq_len = padded_seq_len

            if seq_len > self.MAX_QKV_MM_SEQ_LEN:
                x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

            # Q-only matmul: compute full QKV but only use Q
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

            # Split to get Q heads (K/V from split are discarded)
            (
                q_heads_pre_rot,
                _k_unused,
                _v_unused,
            ) = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(xqkv_fused)
            ttnn.deallocate(_k_unused)
            ttnn.deallocate(_v_unused)

            # Q norm + rotary (Q only — shared K already has rotary from source layer)
            norm_config = self.args.get_norm_config("attn", Mode.PREFILL, None)
            q_heads_pre_rot = self.q_norm(q_heads_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)

            # Apply rotary to Q only (shared K already has correct rotary from source)
            if q_heads_pre_rot.dtype != ttnn.bfloat16:
                q_heads_pre_rot = ttnn.typecast(q_heads_pre_rot, dtype=ttnn.bfloat16)
            q_heads = ttnn.experimental.rotary_embedding(q_heads_pre_rot, rot_mats[0], rot_mats[1])
            ttnn.deallocate(q_heads_pre_rot)

            # Store for potential downstream sharing (though shared layers typically aren't sources)
            self.last_k_heads = k_heads
            self.last_v_heads = v_heads

        else:
            # Normal (non-shared) layer: compute full QKV

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

            # Store processed K/V for potential sharing with later layers
            self.last_k_heads = k_heads
            self.last_v_heads = v_heads

        # Update KV cache (skip for shared layers — source layer handles cache)
        is_shared = self.shared_kv is not None
        if not is_shared:
            if kv_cache:
                keys, values = kv_cache[0], kv_cache[1]
            else:
                keys, values = self.layer_past[0], self.layer_past[1]

            # Typecast K/V to cache dtype before filling
            k_fill = ttnn.typecast(k_heads, dtype=keys.dtype)
            v_fill = ttnn.typecast(v_heads, dtype=values.dtype)

            if page_table is not None:
                ttnn.experimental.paged_fill_cache(keys, k_fill, page_table, batch_idx=user_id)
                ttnn.experimental.paged_fill_cache(values, v_fill, page_table, batch_idx=user_id)
            else:
                ttnn.fill_cache(keys, k_fill, user_id)
                ttnn.fill_cache(values, v_fill, user_id)

            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

        # Use HiFi3+fp32 for SDPA (HiFi4+fp32 has known Wormhole accuracy bug)
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=True,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        ttnn.deallocate(q_heads)
        # Don't deallocate k_heads/v_heads - they may be needed for sharing
        # They'll be cleaned up when the next forward pass overwrites last_k_heads/last_v_heads

        # Concat heads and output projection
        attn_output = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        dense_out = ttnn.linear(
            attn_output,
            self.wo,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(attn_output)

        # Clear shared_kv after use (one-shot)
        self.shared_kv = None

        return dense_out

    def forward_decode(self, x, current_pos, rot_mats=None, page_table=None, kv_cache=None):
        """Decode forward pass with KV sharing support.

        When self.shared_kv is truthy, this is a KV-shared layer:
        - Computes Q only (discards K/V from QKV split)
        - Skips KV cache update (source layer handles it)
        - Reads from source layer's KV cache (passed via kv_cache)
        """
        is_shared = self.shared_kv is not None and self.shared_kv is not False

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

        if is_shared:
            # Shared layer: discard K/V from split, only use Q
            ttnn.deallocate(k_heads_pre_rot)
            ttnn.deallocate(v_heads)

            # Apply rotary to Q only (shared K already has correct rotary from source)
            int_current_pos = int(ttnn.to_torch(ttnn.get_device_tensors(current_pos)[0])[0])
            q_heads = ttnn.experimental.rotary_embedding(q_heads_pre_rot, rot_mats[0], rot_mats[1], int_current_pos)
            ttnn.deallocate(q_heads_pre_rot)
            q_heads = ttnn.reshape(
                q_heads,
                (1, self.max_batch_size, self.n_local_heads, self.head_dim),
                (1, self.max_batch_size, 32, self.head_dim),
            )
            q_heads = q_heads[:, :, : self.n_local_heads]
        else:
            # Normal layer: process K/V too
            k_heads_pre_rot = self.k_norm(k_heads_pre_rot, mode=Mode.DECODE, norm_config=norm_config)
            v_heads = self.v_norm(v_heads, mode=Mode.DECODE, norm_config=norm_config)

            q_heads, k_heads = self._apply_rotary_decode(q_heads_pre_rot, k_heads_pre_rot, rot_mats, current_pos)
            ttnn.deallocate(q_heads_pre_rot)
            ttnn.deallocate(k_heads_pre_rot)

        # KV cache: source layer's cache for shared layers, own cache otherwise
        if kv_cache:
            keys, values = kv_cache[0], kv_cache[1]
        else:
            keys, values = self.layer_past[0], self.layer_past[1]

        if not is_shared:
            # Only update cache for non-shared layers
            ttnn.experimental.paged_update_cache(keys, k_heads, update_idxs_tensor=current_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(values, v_heads, update_idxs_tensor=current_pos, page_table=page_table)
            ttnn.deallocate(k_heads)
            ttnn.deallocate(v_heads)

        # SDPA decode
        sdpa_decode_prog_cfg = self.args.get_attn_sdpa_decode_program_config(None)
        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            q_heads,
            keys,
            values,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            sliding_window_size=self.sliding_window,
            program_config=sdpa_decode_prog_cfg,
            compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_heads)

        # SDPA output must be sharded for nlp_concat_heads_decode
        sdpa_output_sharded_config = ttnn.create_sharded_memory_config(
            shape=(math.ceil(self.n_local_heads / ttnn.TILE_SIZE) * ttnn.TILE_SIZE, self.head_dim),
            core_grid=ttnn.CoreRangeSet({num_to_corerange(self.max_batch_size)}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_output = ttnn.to_memory_config(attn_output, sdpa_output_sharded_config)
        attn_output = ttnn.experimental.nlp_concat_heads_decode(attn_output, num_heads=self.n_local_heads)

        dense_out = ttnn.linear(
            attn_output,
            self.wo,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )
        ttnn.deallocate(attn_output)

        # Clear shared_kv after use (one-shot)
        self.shared_kv = None

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

        # Global layers: head_dim=512, cos/sin are also 512-dim
        # Apply rotation to the FULL 512-dim Q/K vector.
        # rotate_half pairs dim[i] with dim[256+i], matching HF exactly.
        # cos/sin have real values in first 64+192_zeros pattern (duplicated),
        # so only the first 64 dims of each half actually get rotated.
        if q_heads.dtype != ttnn.bfloat16:
            q_heads = ttnn.typecast(q_heads, dtype=ttnn.bfloat16)
        if k_heads.dtype != ttnn.bfloat16:
            k_heads = ttnn.typecast(k_heads, dtype=ttnn.bfloat16)

        q_rot = ttnn.experimental.rotary_embedding(q_heads, rot_mats[0], rot_mats[1])
        k_rot = ttnn.experimental.rotary_embedding(k_heads, rot_mats[0], rot_mats[1])
        return q_rot, k_rot

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

        # Global layers: full 512-dim rotation (matching HF rotate_half pairing)
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
