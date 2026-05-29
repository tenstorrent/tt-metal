# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of the dots.ocr Qwen2 LM self-attention block.

Reference: models/demos/rednote_hilab_dots.ocr/reference/functional.py
           :func:`attention_forward` (+ ``_repeat_kv``, ``apply_rotary_pos_emb_lm``)

This is the Qwen2 decoder self-attention (distinct from the bias-free,
block-diagonal vision attention in ``tt/vision_attention.py``):

    GQA  : 12 query heads / 2 KV heads, head_dim 128, hidden 1536.
    bias : q_proj, k_proj, v_proj all carry BIAS; o_proj has NO bias.
    RoPE : 1D rotary (theta 1e6) on Q and K — apply_rotary_pos_emb_lm.
    mask : causal additive mask (0 / -inf), is_causal.

Pipeline (matching the eager reference exactly):

    q = x @ Wq^T + bq ; k = x @ Wk^T + bk ; v = x @ Wv^T + bv
    split via nlp_create_qkv_heads (12 q heads, 2 kv heads, transpose_k=False)
    q,k = rope_1d(q,k)                         # cos/sin broadcast over heads
    k,v = repeat_kv(2 -> 12)                    # GQA expansion
    attn = softmax(q k^T / sqrt(head_dim) + causal_mask) @ v
    out  = concat_heads(attn) @ Wo^T           # output proj, no bias

The fused QKV weight is built host-side as cat([Wq, Wk, Wv]) so a single
``ttnn.linear`` produces the [seq, (nh+2*nkv)*hd] tensor consumed by
``nlp_create_qkv_heads``. cos/sin tables and the causal mask are uploaded at
construction time (analogous to loading weights). forward() runs entirely in
ttnn ops (no host matmul / softmax / numpy / torch.nn.functional).

Reference TTNN technique: models/demos/rednote_hilab_dots.ocr/tt/vision_attention.py
"""
import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtAttention(LightweightModule):
    """dots.ocr Qwen2 LM self-attention (GQA + QKV bias + 1D RoPE, causal).

    Args:
        device: ttnn Device or MeshDevice.
        q_weight/k_weight/v_weight: torch [out, in] projection weights.
        q_bias/k_bias/v_bias: torch [out] projection biases.
        o_weight: torch [hidden, hidden] output projection weight (no bias).
        cos, sin: torch [seq, head_dim] 1D-RoPE tables (theta 1e6).
        attention_mask: torch additive causal mask [1, 1, seq, seq].
        seq_len: sequence length.
        num_heads: 12 query heads.
        num_kv_heads: 2 KV heads.
        head_dim: 128.
        dtype: activation/weight dtype (bf16).
    """

    def __init__(
        self,
        device,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        o_weight,
        cos,
        sin,
        attention_mask,
        seq_len,
        num_heads: int = 12,
        num_kv_heads: int = 2,
        head_dim: int = 128,
        dtype=ttnn.bfloat16,
        weight_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        decode_cos=None,
        decode_sin=None,
    ):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        self.n_rep = num_heads // num_kv_heads
        self.hidden = num_heads * head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self._dtype = dtype

        is_mesh_device = device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

        # Fused QKV weight: cat([Wq, Wk, Wv]) along output dim -> [hidden + 2*kv, in].
        # ttnn.linear computes x @ W^T when we store the torch weight transposed
        # (ttnn linear expects [in, out]).
        qkv_w = torch.cat([q_weight, k_weight, v_weight], dim=0)  # [(nh+2nkv)*hd, in]
        self.qkv_weight = ttnn.as_tensor(
            qkv_w.transpose(0, 1).contiguous(),  # [in, (nh+2nkv)*hd]
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )
        qkv_b = torch.cat([q_bias, k_bias, v_bias], dim=0)  # [(nh+2nkv)*hd]
        self.qkv_bias = ttnn.as_tensor(
            qkv_b.reshape(1, -1).contiguous(),
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        self.o_weight = ttnn.as_tensor(
            o_weight.transpose(0, 1).contiguous(),  # [hidden, hidden]
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # RoPE cos/sin tables: [seq, head_dim] -> [1, 1, seq, head_dim] broadcast
        # over heads (after q/k are [1, nh, seq, hd]).
        cos = cos.reshape(1, 1, seq_len, head_dim)
        sin = sin.reshape(1, 1, seq_len, head_dim)
        self.cos = ttnn.as_tensor(
            cos,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )
        self.sin = ttnn.as_tensor(
            sin,
            device=device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_memory_config,
            mesh_mapper=mesh_mapper,
        )

        # Additive causal mask [1, num_heads, seq, seq] -- only the short-prompt
        # prefill path materializes a dense score matrix and needs this mask. At
        # large seq_len the prefill uses causal flash SDPA (is_causal=True, no
        # dense mask), and a [1, nh, seq, seq] mask would itself be O(seq^2) and
        # OOM at construction, so it is skipped above the threshold.
        if seq_len <= 1024:
            mask = attention_mask.reshape(1, 1, seq_len, seq_len)
            mask = mask.expand(1, num_heads, seq_len, seq_len).contiguous()
            self.attn_mask = ttnn.as_tensor(
                mask,
                device=device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=weight_memory_config,
                mesh_mapper=mesh_mapper,
            )
        else:
            self.attn_mask = None

        # fp32 compute to match the reference float path.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # ---- Cached-decode RoPE tables (full max_seq_len, indexed per step) ----
        # The prefill path above uses the per-seq_len cos/sin baked at construction
        # for the FULL-causal forward. The cached AR decode path projects ONE token
        # at position cur_pos and needs cos/sin at THAT position only. We keep the
        # full [max_seq, head_dim] tables on host and slice the single row per step
        # (uploaded as a [1, 1, 1, head_dim] tile). decode_cos/decode_sin are passed
        # from the LM assembly (built once for the whole trunk at max_seq_len).
        self._decode_cos_host = decode_cos  # torch [max_seq, head_dim] or None
        self._decode_sin_host = decode_sin
        self._decode_max_seq = None if decode_cos is None else decode_cos.shape[0]

        # Persistent single-row cos/sin decode buffers (stable address for metal
        # trace replay). Updated in place each decode step via
        # copy_host_to_device_tensor before trace execution, so the captured
        # trace reads the correct per-position RoPE without rebaking kernel args.
        self._persistent_decode_cos = None
        self._persistent_decode_sin = None
        if decode_cos is not None:
            zero_row = torch.zeros(1, 1, 1, self.head_dim, dtype=torch.float32)
            self._persistent_decode_cos = ttnn.from_torch(
                zero_row,
                device=device,
                dtype=self._dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self._persistent_decode_sin = ttnn.from_torch(
                zero_row,
                device=device,
                dtype=self._dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Flash-decode SDPA program config. The default core allocation over-
        # subscribes the tree-reduction on this device for a single short cache
        # (TT_FATAL "got 65 cores/head"); cap the grid to an 8x8 region and let
        # the op pick chunk sizes (q/k_chunk_size=0). q_chunk is unused in decode.
        self._sdpa_decode_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=0,
        )

    def write_decode_rope(self, pos: int) -> None:
        """Stream the cos/sin row for ``pos`` into the persistent decode buffers."""
        cos_row = self._decode_cos_host[pos].reshape(1, 1, 1, self.head_dim)
        sin_row = self._decode_sin_host[pos].reshape(1, 1, 1, self.head_dim)
        cos_host = ttnn.from_torch(cos_row, dtype=self._dtype, layout=ttnn.TILE_LAYOUT)
        sin_host = ttnn.from_torch(sin_row, dtype=self._dtype, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(cos_host, self._persistent_decode_cos)
        ttnn.copy_host_to_device_tensor(sin_host, self._persistent_decode_sin)

    def _rotate_half(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2]) on the last dim."""
        d = self.head_dim
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        neg_x2 = ttnn.neg(x2)
        return ttnn.concat([neg_x2, x1], dim=-1)

    def _apply_rope(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, n_heads, seq, head_dim]; cos/sin broadcast over heads."""
        cos_term = ttnn.mul(x, self.cos)
        rot = self._rotate_half(x)
        sin_term = ttnn.mul(rot, self.sin)
        return ttnn.add(cos_term, sin_term)

    def _repeat_kv(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [1, num_kv_heads, seq, head_dim] -> [1, num_heads, seq, head_dim].

        Mirrors _repeat_kv: expand each KV head n_rep times along the head dim,
        interleaved as [kv0]*n_rep, [kv1]*n_rep, ... so head h maps to kv h//n_rep.
        """
        if self.n_rep == 1:
            return x
        seq = self.seq_len
        hd = self.head_dim
        nkv = self.num_kv_heads
        # [1, nkv, seq, hd] -> [1, nkv, 1, seq, hd] -> expand n_rep -> reshape.
        x = ttnn.reshape(x, (1, nkv, 1, seq, hd))
        x = ttnn.repeat(x, ttnn.Shape((1, 1, self.n_rep, 1, 1)))
        x = ttnn.reshape(x, (1, nkv * self.n_rep, seq, hd))
        return x

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """x: [seq, hidden] (TILE layout) -> [seq, hidden]."""
        seq = self.seq_len
        nh = self.num_heads
        nkv = self.num_kv_heads
        hd = self.head_dim

        # Fused QKV projection with bias: [seq, hidden] @ [hidden, (nh+2nkv)*hd].
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )  # [seq, (nh + 2*nkv)*hd]

        # GQA head split via the fused nlp_create_qkv_heads -- reshape + slice +
        # transpose in ONE op, returning q [1,nh,seq,hd] and k/v [1,nkv,seq,hd]
        # (the layout attention wants). Replaces a manual
        # slice->reshape->permute chain that was the block's top hotspot
        # (tracy: ~38% of kernel time, DRAM round-trips); the earlier
        # L1-pin only made that bad reshape faster instead of removing it. The
        # fused weight is cat([Wq,Wk,Wv]) -> contiguous q|k|v columns, exactly
        # what the op expects. head_dim 128 is tile-aligned (no sub-tile case).
        qkv = ttnn.reshape(qkv, (1, 1, seq, (nh + 2 * nkv) * hd))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=nh,
            num_kv_heads=nkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # q [1,nh,seq,hd]; k,v [1,nkv,seq,hd]

        # 1D RoPE on q and k (cos/sin broadcast over heads).
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # GQA expansion 2 -> 12.
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # attn = softmax(q k^T * scale + causal_mask) @ v.  k^T -> [1, nh, hd, seq]
        # Keep the QK^T scores in fp32: real-weight K reaches ±320 (large k_proj
        # bias), so the pre-softmax scores have a wide dynamic range and bf16
        # rounding of the scores is the dominant PCC loss (drops ~0.984 -> 0.998
        # when fp32). fp32_dest_acc already accumulates the dot product in fp32;
        # storing the matmul output fp32 preserves it through scale/mask/softmax.
        k_t = ttnn.permute(k, (0, 1, 3, 2))
        scores = ttnn.matmul(
            q,
            k_t,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.float32,
        )
        scores = ttnn.mul(scores, self.scale)
        scores = ttnn.add(scores, self.attn_mask)
        probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_kernel_config)
        attn = ttnn.matmul(
            probs,
            v,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )  # [1, nh, seq, hd]

        # Fused head-merge: nlp_concat_heads ([1,nh,seq,hd] -> [1,1,seq,nh*hd])
        # in one op instead of permute + reshape.
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, (seq, nh * hd), memory_config=ttnn.L1_MEMORY_CONFIG)

        # Output projection (no bias): [seq, hidden] @ [hidden, hidden].
        out = ttnn.linear(
            attn,
            self.o_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        return out

    # ------------------------------------------------------------------ #
    # Cached AR-decode path (KV cache + flash-decode SDPA).               #
    # ------------------------------------------------------------------ #
    def _qkv_proj_heads(self, x: ttnn.Tensor, n_tok: int):
        """Project x -> (q, k, v) head tensors.

        q -> [1, nh,  n_tok, hd]; k, v -> [1, nkv, n_tok, hd] (pre-GQA, pre-RoPE).
        Mirrors forward()'s QKV/split/permute exactly but for an arbitrary token
        count (n_tok=prompt_len in prefill, 1 in decode).
        """
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        qkv = ttnn.linear(
            x,
            self.qkv_weight,
            bias=self.qkv_bias,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
        )
        # Fused GQA head split (reshape+slice+transpose in ONE op) -> q
        # [1,nh,n_tok,hd], k/v [1,nkv,n_tok,hd]. Replaces a manual
        # slice->reshape->permute chain that round-tripped the QKV through DRAM
        # (the prefill hotspot at long prompt_len). Used by prefill_kv (n_tok=
        # prompt_len) and the decode paths (n_tok=1). head_dim 128 is tile-aligned.
        qkv = ttnn.reshape(qkv, (1, 1, n_tok, (nh + 2 * nkv) * hd))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=nh,
            num_kv_heads=nkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return q, k, v

    def _rope_with(self, x: ttnn.Tensor, cos_tt: ttnn.Tensor, sin_tt: ttnn.Tensor) -> ttnn.Tensor:
        """Apply 1D RoPE to x with explicit cos/sin (broadcast over heads/seq)."""
        cos_term = ttnn.mul(x, cos_tt)
        sin_term = ttnn.mul(self._rotate_half(x), sin_tt)
        return ttnn.add(cos_term, sin_term)

    def prefill_kv(self, x: ttnn.Tensor, kv_cache, layer_idx: int) -> ttnn.Tensor:
        """Run the full-causal prefill forward AND populate the KV cache.

        x: [prompt_len, hidden] TILE. Returns [prompt_len, hidden] (identical to
        forward(x)) and writes the per-layer K/V (post-RoPE, pre-GQA, shape
        [batch, nkv, prompt_len, hd]) into ``kv_cache`` at layer ``layer_idx``.
        """
        seq = self.seq_len
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q, k, v = self._qkv_proj_heads(x, seq)

        # RoPE on the full prompt using the prefill cos/sin baked at seq_len.
        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # Stash the post-RoPE K (shape [1, nkv, seq, hd]) and V into the cache so
        # the decode steps read consistent (already-rotated) K and raw V.
        kv_cache.fill_prefill(layer_idx, k, v)

        # GQA expansion 2 -> 12 for the prefill attention output.
        kq = self._repeat_kv(k)
        vq = self._repeat_kv(v)

        if seq <= 1024:
            # Short prompt: materialized fp32-score causal softmax (verified PCC
            # path; fp32 scores matter because real-weight K reaches ±320).
            k_t = ttnn.permute(kq, (0, 1, 3, 2))
            scores = ttnn.matmul(q, k_t, compute_kernel_config=self.compute_kernel_config, dtype=ttnn.float32)
            scores = ttnn.mul(scores, self.scale)
            scores = ttnn.add(scores, self.attn_mask)
            probs = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_kernel_config)
            attn = ttnn.matmul(probs, vq, compute_kernel_config=self.compute_kernel_config, dtype=ttnn.bfloat16)
            mem = ttnn.L1_MEMORY_CONFIG
        else:
            # Long prompt (e.g. a full-document vision prefill, seq in the
            # thousands): memory-efficient causal flash attention. Same
            # softmax(q k^T * scale + causal_mask) @ v math as the short path,
            # but the kernel never materializes the [1, nh, seq, seq] score
            # matrix -- it streams in O(seq) memory (flash-2 chunked), which is
            # what lets the full-resolution document prompt prefill without OOM.
            q_d = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            kq_d = ttnn.to_memory_config(kq, ttnn.DRAM_MEMORY_CONFIG)
            vq_d = ttnn.to_memory_config(vq, ttnn.DRAM_MEMORY_CONFIG)
            attn = ttnn.transformer.scaled_dot_product_attention(
                q_d,
                kq_d,
                vq_d,
                is_causal=True,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(
                        self.device.compute_with_storage_grid_size().x,
                        self.device.compute_with_storage_grid_size().y,
                    ),
                    exp_approx_mode=False,
                    q_chunk_size=128,
                    k_chunk_size=128,
                ),
            )
            mem = ttnn.DRAM_MEMORY_CONFIG

        # Fused head-merge: nlp_concat_heads ([1,nh,seq,hd] -> [1,1,seq,nh*hd]).
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.reshape(attn, (seq, nh * hd), memory_config=mem)
        return ttnn.linear(
            attn,
            self.o_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=mem,
        )

    def _decode_rope_tiles(self, pos: int):
        """Upload the single-position cos/sin row as [1, 1, 1, head_dim] tiles."""
        cos_row = self._decode_cos_host[pos].reshape(1, 1, 1, self.head_dim)
        sin_row = self._decode_sin_host[pos].reshape(1, 1, 1, self.head_dim)
        cos_tt = ttnn.from_torch(cos_row, device=self.device, dtype=self._dtype, layout=ttnn.TILE_LAYOUT)
        sin_tt = ttnn.from_torch(sin_row, device=self.device, dtype=self._dtype, layout=ttnn.TILE_LAYOUT)
        return cos_tt, sin_tt

    def forward_decode(self, x: ttnn.Tensor, pos: int, kv_cache, layer_idx: int) -> ttnn.Tensor:
        """Single-token cached decode at sequence position ``pos``.

        x: [1, hidden] TILE (the current token's hidden). Projects Q/K/V for one
        token, RoPE at ``pos``, writes K/V into ``kv_cache`` at ``pos``, then runs
        flash-decode SDPA against the full cache (GQA handled by the op). Returns
        [1, hidden].
        """
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q, k, v = self._qkv_proj_heads(x, 1)  # q [1,nh,1,hd], k/v [1,nkv,1,hd]

        cos_tt, sin_tt = self._decode_rope_tiles(pos)
        q = self._rope_with(q, cos_tt, sin_tt)
        k = self._rope_with(k, cos_tt, sin_tt)

        # Write this step's K/V (post-RoPE K, raw V) into the cache at pos.
        # update() expects [batch, nkv, 1, hd]; q/k/v are [1, n*, 1, hd] (batch=1).
        kv_cache.update(layer_idx, k, v, pos)

        k_cache, v_cache = kv_cache.read(layer_idx)  # [1, nkv, max_seq, hd]

        # Flash-decode SDPA: Q [1, b, nh, hd]; K/V [b, nkv, max_seq, hd]; cur_pos
        # masks slots > pos. The op performs the GQA 2->12 expansion internally.
        q_sdpa = ttnn.permute(q, (0, 2, 1, 3))  # [1, 1(=b), nh, hd]
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_sdpa,
            k_cache,
            v_cache,
            cur_pos=[pos],
            scale=self.scale,
            program_config=self._sdpa_decode_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )  # [1, b, nh, hd]
        attn = ttnn.reshape(attn, (1, nh * hd), memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.linear(attn, self.o_weight, compute_kernel_config=self.compute_kernel_config, dtype=ttnn.bfloat16)

    def forward_decode_traced(self, x: ttnn.Tensor, kv_cache, layer_idx: int) -> ttnn.Tensor:
        """Trace-capturable single-token decode.

        Identical maths to :meth:`forward_decode` but reads the decode position
        ENTIRELY from device memory (the KV-cache persistent ``pos_tt`` for both
        the cache update and SDPA ``cur_pos_tensor``) and the RoPE cos/sin from
        this layer's persistent decode buffers. No Python int is baked into any
        kernel arg, so one captured metal trace replays at every position once the
        host streams ``pos`` / cos / sin into the persistent buffers before replay.
        """
        nh, nkv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q, k, v = self._qkv_proj_heads(x, 1)  # q [1,nh,1,hd], k/v [1,nkv,1,hd]

        q = self._rope_with(q, self._persistent_decode_cos, self._persistent_decode_sin)
        k = self._rope_with(k, self._persistent_decode_cos, self._persistent_decode_sin)

        # Write K/V at the device-held position (persistent pos_tt), then read cache.
        pos_tt = kv_cache.get_persistent_pos_buffer()
        kv_cache.update(layer_idx, k, v, pos_tt)
        k_cache, v_cache = kv_cache.read(layer_idx)

        q_sdpa = ttnn.permute(q, (0, 2, 1, 3))  # [1, 1(=b), nh, hd]
        attn = ttnn.transformer.scaled_dot_product_attention_decode(
            q_sdpa,
            k_cache,
            v_cache,
            cur_pos_tensor=pos_tt,
            scale=self.scale,
            program_config=self._sdpa_decode_program_config,
            compute_kernel_config=self.compute_kernel_config,
        )
        attn = ttnn.reshape(attn, (1, nh * hd), memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.linear(attn, self.o_weight, compute_kernel_config=self.compute_kernel_config, dtype=ttnn.bfloat16)
