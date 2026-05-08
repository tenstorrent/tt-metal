# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 text decoder attention with fused att_proj and Qwen3-style QK-norm.

Phase 1 T3K layout (correctness-first, before performance optimization):
  - wqkv: column-parallel — each device holds its local Q+K+V head weights.
    No AllReduce needed (column-parallel QKV gives local heads directly).
  - After local attention: AllGather head outputs across T3K devices → full hidden.
  - wo: replicated — each device computes the full output projection independently.
    No AllReduce needed (all devices produce the same replicated output).
  - QK-norm applied AFTER nlp_create_qkv_heads (Qwen3-style, head_dim=128).

This produces [1, 1, S, 4096] replicated output on all devices, matching the
reference and enabling direct PCC verification.

Adapted from:
  - models/demos/qwen3_vl/tt/vision_attention.py  (q_norm/k_norm, column-parallel QKV)
  - models/demos/qwen3_vl/tt/attention.py          (prefill/decode flow, KV cache)
"""


import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode


class TtMolmo2TextAttention(LightweightModule):
    """Text decoder GQA with fused att_proj, QK-norm, RoPE, KV cache.

    Runs on T3K with column-parallel QKV + AllGather + replicated wo.
    Produces replicated [1, 1, S, 4096] output on all devices.
    """

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.padded_head_dim = configuration.padded_head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.n_local_heads = configuration.n_local_heads
        self.n_local_kv_heads = configuration.n_local_kv_heads
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size
        self.dtype = dtype
        self.transformation_mats = transformation_mats
        self.model_config = configuration.get_model_config()
        self.is_multichip = configuration.is_multichip
        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4
        self.scale = self.head_dim**-0.5

        layer_name = f"model.transformer.blocks.{layer_num}.self_attn"
        if configuration.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{layer_name}.{name}"

        # ------------------------------------------------------------------ #
        # Fused att_proj [6144, 4096] → column-parallel wqkv
        # Each device holds weights for its local Q+K+V heads (4+1+1 = 768 channels).
        # Input [S, 4096] (replicated) × wqkv_shard[4096, 768] → [S, 768] per device.
        # No AllReduce needed — each device independently computes its heads.
        # ------------------------------------------------------------------ #
        att_proj = state_dict[f"{layer_name}.att_proj.weight"]  # [6144, 4096]
        q_dim = self.n_heads * self.head_dim  # 4096
        kv_dim = self.n_kv_heads * self.head_dim  # 1024

        wq_full = att_proj[:q_dim]
        wk_full = att_proj[q_dim : q_dim + kv_dim]
        wv_full = att_proj[q_dim + kv_dim :]

        qkv_list = []
        for i in range(self.num_devices):
            wq_i = wq_full.chunk(self.num_devices, dim=0)[i]
            wk_i = wk_full.chunk(self.num_devices, dim=0)[i]
            wv_i = wv_full.chunk(self.num_devices, dim=0)[i]
            qkv_i = torch.cat([wq_i.T, wk_i.T, wv_i.T], dim=-1)
            qkv_list.append(qkv_i)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)
        # [1, 1, 4096, 6144] → ShardTensor2dMesh(dims=(2,3)) splits dim 3 per device

        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(2, 3), mesh_shape=configuration.cluster_shape),
            cache_file_name=cache_name("wqkv_sharded_2d"),
        )

        # ------------------------------------------------------------------ #
        # attn_out.weight [4096, 4096]:
        #   wo (prefill): REPLICATED — AllGather input is full [S,4096]
        #   wo_decode: ROW-PARALLEL — each device [512,4096], local matmul +
        #     ttnn.all_reduce (8x less compute for single-token decode)
        # Separate weights so prefill keeps its proven accuracy.
        # ------------------------------------------------------------------ #
        pt_wo = state_dict[f"{layer_name}.attn_out.weight"].T.unsqueeze(0).unsqueeze(0)

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=cache_name("wo_replicated"),
        )
        self.wo_decode = ttnn.as_tensor(
            pt_wo,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-1, -2), mesh_shape=configuration.cluster_shape),
            cache_file_name=cache_name("wo_row_parallel_v2"),
        )

        # ------------------------------------------------------------------ #
        # QK-norm (Qwen3-style: applied AFTER nlp_create_qkv_heads)
        # head_dim=128 is tile-aligned; no padding or special handling needed.
        # ------------------------------------------------------------------ #
        def _rmsnorm(key_suffix):
            return RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=f"{layer_name}.{key_suffix}",
                is_distributed=False,
                sharded_program_config=None,
                sharded_output_config=None,
            )

        self.q_norm = _rmsnorm("q_norm")
        self.k_norm = _rmsnorm("k_norm")

        # ------------------------------------------------------------------ #
        # KV cache — bfloat16 for precision: bfloat8_b causes logit flips for
        # long video sequences (S>2500) due to accumulated SDPA precision loss.
        # ------------------------------------------------------------------ #
        cache_k = torch.zeros(self.max_batch_size, self.n_local_kv_heads, self.max_seq_len, self.head_dim)
        self.layer_past = [
            ttnn.as_tensor(
                kv,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for kv in (cache_k, torch.zeros_like(cache_k))
        ]

    def _gather_and_project(self, attn_11SH, seq_len):
        """Attention output projection with mode-specific TP strategy.

        Decode (seq_len==1): row-parallel wo + ttnn.all_reduce — 8x less compute,
          L1 output, matches ign/Molmo2_8B_new decode path.
        Prefill (seq_len>1): AllGather → replicated wo — unchanged, preserves
          prefill accuracy (row-parallel AllReduce changed test 71).
        """
        chunked = seq_len > 1024 and seq_len % 1024 == 0
        if chunked:
            attn_11SH = ttnn.reshape(attn_11SH, [1, seq_len // 1024, 1024, -1])

        is_decode = seq_len == 1

        if is_decode:
            # Row-parallel wo_decode: [S, 512] × [512, 4096] → [S, 4096] partial, then AllReduce
            out = ttnn.linear(
                attn_11SH,
                self.wo_decode,
                compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(attn_11SH)
            if self.is_multichip:
                out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
                out = ttnn.all_reduce(out, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            # Prefill: AllGather → replicated wo (original path, preserves accuracy)
            if self.is_multichip:
                attn_gathered = ttnn.all_gather(attn_11SH, dim=3, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(attn_11SH)
            else:
                attn_gathered = attn_11SH
            out = ttnn.linear(
                attn_gathered,
                self.wo,
                compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(attn_gathered)

        if chunked:
            out = ttnn.reshape(out, [1, 1, seq_len, -1])
        return out

    # ------------------------------------------------------------------ #
    # Decode (single token)
    # ------------------------------------------------------------------ #

    def forward_decode(self, x, current_pos, rot_mats=None, kv_cache=None):
        """x: [1, 1, batch, dim] replicated on all devices."""
        # Column-parallel QKV — L1 output keeps tensor on-chip for subsequent ops
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(x)

        xqkv = ttnn.reshape(xqkv, (1, 1, self.max_batch_size, xqkv.shape[3]))

        q_pre, k_pre, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # nlp_create_qkv_heads_decode outputs HEIGHT_SHARDED.
        # Save KV memory configs for paged_update_cache; convert to L1 for QK-norm
        # (matches reference: ttnn.rms_norm without compute_kernel_config = default/HiFi4 precision)
        k_mem_cfg = k_pre.memory_config()
        v_mem_cfg = v.memory_config()
        q_pre = ttnn.to_memory_config(q_pre, ttnn.L1_MEMORY_CONFIG)
        k_pre = ttnn.to_memory_config(k_pre, ttnn.L1_MEMORY_CONFIG)
        q_pre = ttnn.rms_norm(q_pre, weight=self.q_norm.weight, epsilon=self.q_norm.eps)
        k_pre = ttnn.rms_norm(k_pre, weight=self.k_norm.weight, epsilon=self.k_norm.eps)

        if q_pre.dtype != ttnn.bfloat16:
            q_pre = ttnn.typecast(q_pre, dtype=ttnn.bfloat16)
        if k_pre.dtype != ttnn.bfloat16:
            k_pre = ttnn.typecast(k_pre, dtype=ttnn.bfloat16)

        # RoPE (HF-style). nlp_create_qkv_heads_decode output: [1, batch, heads, head_dim].
        # For single-token decode, we need ALL heads at the SAME position p.
        # Reshape to [1, heads, 1, head_dim] so Y=1 (seq) → rotary_embedding applies
        # position p (from rot_mats) uniformly to all heads. Then reshape back.
        q_pre_r = ttnn.reshape(q_pre, [1, self.n_local_heads, 1, self.padded_head_dim])
        k_pre_r = ttnn.reshape(k_pre, [1, self.n_local_kv_heads, 1, self.padded_head_dim])
        ttnn.deallocate(q_pre)
        ttnn.deallocate(k_pre)
        q_rotated = ttnn.experimental.rotary_embedding(q_pre_r, rot_mats[0], rot_mats[1])
        k_rotated = ttnn.experimental.rotary_embedding(k_pre_r, rot_mats[0], rot_mats[1])
        ttnn.deallocate(q_pre_r)
        ttnn.deallocate(k_pre_r)
        # Slice the padded seq dim back to 1, then reshape to [1, 1, heads, head_dim]
        q = ttnn.reshape(q_rotated[:, :, :1, :], [1, 1, self.n_local_heads, self.padded_head_dim])
        k = ttnn.reshape(k_rotated[:, :, :1, :], [1, 1, self.n_local_kv_heads, self.padded_head_dim])
        ttnn.deallocate(q_rotated)
        ttnn.deallocate(k_rotated)

        # SDPA decode requires Q in DRAM
        q = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)

        # Restore HEIGHT_SHARDED for paged_update_cache (requires sharded input)
        k_sharded = ttnn.to_memory_config(k, k_mem_cfg)
        v_sharded = ttnn.to_memory_config(v, v_mem_cfg)

        # KV cache update
        keys = kv_cache[0] if kv_cache else self.layer_past[0]
        values = kv_cache[1] if kv_cache else self.layer_past[1]
        ttnn.experimental.paged_update_cache(keys, k_sharded, update_idxs_tensor=current_pos)
        ttnn.experimental.paged_update_cache(values, v_sharded, update_idxs_tensor=current_pos)
        ttnn.deallocate(k_sharded)
        ttnn.deallocate(v_sharded)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            keys,
            values,
            cur_pos_tensor=current_pos,
            scale=self.scale,
            program_config=self.model_config["SDPA_DECODE_PROGCFG"],
            compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)

        # Convert SDPA output [1, batch, n_local_heads, head_dim] to HEIGHT_SHARDED
        # for nlp_concat_heads_decode (requires num_cores == batch_size).
        sdpa_batch = attn_out.shape[1]
        sdpa_grid_x = min(8, sdpa_batch)
        sdpa_grid_y = (sdpa_batch + sdpa_grid_x - 1) // sdpa_grid_x
        sdpa_shard_cfg = ttnn.create_sharded_memory_config(
            shape=(32, self.padded_head_dim),
            core_grid=ttnn.CoreGrid(y=sdpa_grid_y, x=sdpa_grid_x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn_out = ttnn.to_memory_config(attn_out, sdpa_shard_cfg)

        # [1, batch, n_local_heads, head_dim] → [1, 1, batch, n_local_heads * head_dim]
        attn_out = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=self.n_local_heads)

        # nlp_concat_heads_decode may tile-pad the batch dim; slice back to actual batch.
        if attn_out.shape[2] != self.max_batch_size:
            attn_out = ttnn.to_memory_config(attn_out, ttnn.DRAM_MEMORY_CONFIG)
            attn_out = attn_out[:, :, : self.max_batch_size, :]

        # AllGather local-head outputs across T3K devices, then apply wo.
        # Each device produced [1, 1, batch, n_local_heads*head_dim]; after gather → [1,1,batch,4096].
        return self._gather_and_project(attn_out, seq_len=1)

    # ------------------------------------------------------------------ #
    # Prefill (full sequence)
    # ------------------------------------------------------------------ #

    def forward_prefill(self, x_11SH, rot_mats, user_id=0, kv_cache=None, mask=None):
        """x_11SH: [1, 1, S, 4096] replicated on all T3K devices."""
        seq_len = x_11SH.shape[-2]

        # Only chunk if evenly divisible — arbitrary seq_lens (e.g. 2701) must not be chunked
        qkv_chunked = seq_len > self.MAX_QKV_MM_SEQ_LEN and seq_len % self.MAX_QKV_MM_SEQ_LEN == 0
        if qkv_chunked:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        # Column-parallel QKV: each device computes its local head outputs
        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

        if qkv_chunked:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

        # Split into Q, K, V head tensors
        q_pre, k_pre, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv_fused)

        # QK-norm AFTER head split (Qwen3-style, must come before RoPE)
        q_pre = self.q_norm(q_pre, mode=Mode.PREFILL)
        k_pre = self.k_norm(k_pre, mode=Mode.PREFILL)

        # RoPE: HF-style rotate_half via ttnn.experimental.rotary_embedding.
        # rot_mats = [cos, sin] shape [1, 1, max_seq_len, head_dim] (full pre-computed).
        # rotary_embedding pads the seq dim to tile=32 internally; we slice back to seq_len.
        if q_pre.dtype != ttnn.bfloat16:
            q_pre = ttnn.typecast(q_pre, dtype=ttnn.bfloat16)
        if k_pre.dtype != ttnn.bfloat16:
            k_pre = ttnn.typecast(k_pre, dtype=ttnn.bfloat16)

        q_rotated = ttnn.experimental.rotary_embedding(q_pre, rot_mats[0], rot_mats[1])
        ttnn.deallocate(q_pre)
        k_rotated = ttnn.experimental.rotary_embedding(k_pre, rot_mats[0], rot_mats[1])
        ttnn.deallocate(k_pre)

        # Slice back to original seq_len (rotary_embedding pads to tile multiple).
        # Do NOT deallocate q_rotated/k_rotated here: when seq_len is tile-aligned
        # (e.g. after power-of-2 padding), the slice IS the same tensor and an
        # explicit deallocate would free q/k while they're still in use.
        # They are freed implicitly when q and k go out of scope below.
        q = q_rotated[:, :, :seq_len, :]
        k = k_rotated[:, :, :seq_len, :]

        # KV cache fill: use bfloat16 directly (cache is bfloat16)
        keys = kv_cache[0] if kv_cache else self.layer_past[0]
        values = kv_cache[1] if kv_cache else self.layer_past[1]
        ttnn.fill_cache(keys, k, user_id % self.max_batch_size)
        ttnn.fill_cache(values, v, user_id % self.max_batch_size)

        # SDPA: is_causal and attn_mask are mutually exclusive in TTNN SDPA.
        # When a custom mask is provided (image-bidir override), it already encodes
        # the causal structure, so is_causal must be False.
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=(mask is None),
            attn_mask=mask,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_out = ttnn.reshape(attn_out, [1, self.n_local_heads, -1, self.padded_head_dim])
        attn_11SH = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        # AllGather + replicated wo → [1, 1, S, 4096] on all devices
        return self._gather_and_project(attn_11SH, seq_len)

    def forward(self, x, current_pos, rot_mats=None, user_id=0, mode="decode", kv_cache=None, attn_mask=None):
        if mode == "prefill":
            return self.forward_prefill(x, rot_mats, user_id, kv_cache=kv_cache, mask=attn_mask)
        return self.forward_decode(x, current_pos, rot_mats, kv_cache=kv_cache)
