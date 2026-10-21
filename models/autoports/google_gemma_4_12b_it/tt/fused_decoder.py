# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused TTNN decoder layer for ``google/gemma-4-12B-it`` (text decoder, single device).

Same layer, constructor and runtime contract as ``functional_decoder.FunctionalDecoder``:
- HF reference ``Gemma4UnifiedTextDecoderLayer``, both ``sliding_attention`` and ``full_attention`` kinds;
- paged KV cache owned by the layer;
- any logical prefill length;
- trace-safe decode of 1..32 users, where ``-1`` skips a user.
Only the op graph differs. The rewrites relative to the functional decoder are listed below;
``doc/fused_decoder/README.md`` has the measurements.

Prefill
* Every projection (QKV, o_proj, gate, up, down) runs on ``ttnn.experimental.minimal_matmul``.
* GELU(tanh) is folded into the gate * up multiply as an input activation (no standalone GELU).
* ``layer_scalar`` is folded into the final residual add as a post-activation (no standalone multiply).
* ``attention_k_eq_v`` (full layers): K and V are the same raw projection, so
  ``k_norm(K) == v_norm(K) * w_k``. One weightless norm plus a row-broadcast multiply replace two norms.
* The default physical chunk is 8192 rows: fewer chunk slices and concats, and larger, more efficient matmuls.
* Sliding layers attend over a window-bounded, fixed-width slice of the page table. Each block of 2048 query
  rows starts its page-table row at the window, which is an exact origin shift. Without this, the
  FP32-accumulating chunked SDPA kernel walks and masks every key from position 0. The fixed width keeps one
  SDPA program for every position.
* Sliding layers (head_dim 256) use the interleaved (Meta) RoPE layout: Q/K weight columns, q/k norm weights
  and RoPE tables are permuted per head, and the K cache holds permuted K. The QKV matmul writes
  [Q half | Q half | K | V] (``minimal_matmul_split``), and ``dit_fused_distributed_rmsnorm`` does head
  split, per-head RMSNorm and RoPE in one op. That replaces nlp_create_qkv_heads, three norms and two
  RoPE ops. Full layers (head_dim 512) keep the HF layout: the decode interleaved-RoPE ops do not support
  head_dim 512 (``rotary_embedding_llama`` caps it at 256 and ``_fused_qk`` is wrong at 512), and the cache
  layout must match between prefill and decode.

Decode
* RoPE rows: each position is padded to one 32-row index block per user. ``ttnn.embedding`` gathers them,
  and its fused tilize writes straight into the RoPE op's height-sharded layout (``padding_idx=0`` turns
  the 31 pad rows into local copies). For each of cos and sin, this replaces the gather, a single-core
  tilize and an interleaved-to-sharded copy.
* Per-head q/k/v RMSNorm runs in place on the head-sharded tensors at batch 1, with no sharded <->
  interleaved round trip. Larger batches use interleaved copies: L1 on sliding layers, and DRAM on full
  layers, where L1 copies clash with the op CBs at head_dim 512 x 32 users.
* SDPA decode reads the head-sharded Q directly (no DRAM park). Sliding layers rotate Q/K with
  ``rotary_embedding_llama`` (interleaved layout, one shared 32-core transformation matrix).
* K and V go to the paged cache with one ``paged_fused_update_cache`` (V on a disjoint core range).
* Full layers (one KV head) write the SDPA output directly in the head-concat op's sharded layout.
* The same ``k_eq_v``, GELU-in-multiply and ``layer_scalar``-in-add rewrites as prefill.
"""

import math

import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import (
    HF_PREFIX,
    MAX_DECODE_BATCH,
    TILE,
    _round_up,
    rope_tables,
)
from models.common.lightweightmodule import LightweightModule

DEFAULT_PREFILL_CHUNK = 8192
SLIDING_Q_TILE = 256  # largest prefill SDPA Q tile on sliding layers
SLIDING_ATTN_BLOCK = 2048  # sliding-layer prefill attention runs per block of this many query rows


def interleave_perm(head_dim):
    """Column order mapping HF rotate-half pairs (j, j + hd/2) to interleaved pairs (2j, 2j + 1)."""
    half = head_dim // 2
    return torch.stack([torch.arange(half), torch.arange(half) + half], 1).flatten()


def rot_transformation_mat():
    """32x32 tile T with (x @ T)[2j] = -x[2j+1], (x @ T)[2j+1] = x[2j]: rotate_half in interleaved layout."""
    t = torch.zeros(1, 1, TILE, TILE)
    for i in range(0, TILE, 2):
        t[0, 0, i, i + 1] = 1
        t[0, 0, i + 1, i] = -1
    return t


class FusedDecoder(LightweightModule):
    """One Gemma4 text decoder layer (sliding or full attention) on a 1x1 mesh, fused op graph."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        num_blocks,
        block_size=64,
        prefill_chunk_size=DEFAULT_PREFILL_CHUNK,
        max_context=None,
        shared=None,
    ):
        """Build the layer from an HF state dict (same arguments as ``FunctionalDecoder.from_state_dict``).

        state_dict: full-checkpoint keys (``model.language_model.layers.{i}.*``) or layer-relative keys.
        hf_config: the HF text config (``Gemma4UnifiedTextConfig``).
        num_blocks / block_size: paged KV cache geometry owned by this layer.
        prefill_chunk_size: physical prefill chunk (multiple of ``block_size``).
        max_context: supported context; defaults to ``hf_config.max_position_embeddings``.
        shared: optional dict reused across layers to share RoPE tables per layer kind.
        """
        c = hf_config
        assert tuple(mesh_device.shape) == (1, 1), "fused decoder runs on a 1x1 mesh"
        assert not c.enable_moe_block and not c.hidden_size_per_layer_input and not c.num_kv_shared_layers
        assert not c.attention_bias and not c.use_double_wide_mlp
        assert c.hidden_activation == "gelu_pytorch_tanh"
        assert block_size % TILE == 0 and prefill_chunk_size % block_size == 0
        self = cls()
        self.device = mesh_device
        self.layer_idx = layer_idx
        self.layer_type = c.layer_types[layer_idx]
        assert self.layer_type in ("sliding_attention", "full_attention")
        self.is_sliding = self.layer_type == "sliding_attention"
        self.hidden_size = c.hidden_size
        self.num_heads = c.num_attention_heads
        self.k_eq_v = bool(c.attention_k_eq_v) and not self.is_sliding
        self.head_dim = c.head_dim if self.is_sliding else c.global_head_dim
        self.num_kv_heads = c.num_global_key_value_heads if self.k_eq_v else c.num_key_value_heads
        self.sliding_window = c.sliding_window if self.is_sliding else None
        self.eps = c.rms_norm_eps
        self.max_context = c.max_position_embeddings if max_context is None else max_context
        assert self.max_context % block_size == 0
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.prefill_chunk_size = prefill_chunk_size

        prefix = HF_PREFIX.format(layer_idx)
        sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)} or state_dict

        def as_tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(), dtype=dtype, layout=layout, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def norm_weight(name):
            # rms_norm consumes ROW_MAJOR weights folded into [1, 1, dim / 32, 32].
            return as_tt(sd[name + ".weight"].reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)

        q = sd["self_attn.q_proj.weight"]
        k = sd["self_attn.k_proj.weight"]
        # attention_k_eq_v: V is the same raw projection as K. Duplicating the K columns gives the
        # head-split ops an ordinary V slot; the decode head-split op has no tied-K/V mode.
        v = k if self.k_eq_v else sd["self_attn.v_proj.weight"]
        assert q.shape[0] == self.num_heads * self.head_dim and k.shape[0] == self.num_kv_heads * self.head_dim
        # Sliding layers use the interleaved (Meta) RoPE layout required by the fused norm+RoPE prefill op
        # and rotary_embedding_llama: every Q/K head's columns (and q_norm/k_norm, and the RoPE tables) are
        # permuted from HF rotate-half order to interleaved pairs. Q.K is invariant under the same
        # permutation of Q and K; V is untouched. The K cache holds permuted K.
        self.meta_rope = self.is_sliding
        perm = interleave_perm(self.head_dim) if self.meta_rope else None
        if self.meta_rope:
            q = q.reshape(self.num_heads, self.head_dim, -1)[:, perm, :].reshape(q.shape)
            k = k.reshape(self.num_kv_heads, self.head_dim, -1)[:, perm, :].reshape(k.shape)
            # Prefill splits [Q | K | V] into four equal column chunks: two Q halves, K, V.
            assert q.shape[0] == 2 * k.shape[0] == 2 * v.shape[0]
        self.wqkv = as_tt(torch.cat([q, k, v], dim=0).T)
        self.wo = as_tt(sd["self_attn.o_proj.weight"].T)
        self.w_gate = as_tt(sd["mlp.gate_proj.weight"].T)
        self.w_up = as_tt(sd["mlp.up_proj.weight"].T)
        self.w_down = as_tt(sd["mlp.down_proj.weight"].T)
        self.input_norm = norm_weight("input_layernorm")
        self.post_attn_norm = norm_weight("post_attention_layernorm")
        self.pre_ffn_norm = norm_weight("pre_feedforward_layernorm")
        self.post_ffn_norm = norm_weight("post_feedforward_layernorm")
        if self.meta_rope:
            q_w = sd["self_attn.q_norm.weight"][perm]
            k_w = sd["self_attn.k_norm.weight"][perm]
            # Decode per-head norms (ROW_MAJOR) and the prefill fused norm+RoPE op, whose weight spans all
            # heads of its input (8 heads: one Q half, or K).
            self.q_norm = as_tt(q_w.reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)
            self.k_norm = as_tt(k_w.reshape(1, 1, -1, TILE), layout=ttnn.ROW_MAJOR_LAYOUT)
            self.q_norm_half = as_tt(q_w.repeat(self.num_heads // 2).reshape(1, 1, 1, -1))
            self.k_norm_heads = as_tt(k_w.repeat(self.num_kv_heads).reshape(1, 1, 1, -1))
        else:
            self.q_norm = norm_weight("self_attn.q_norm")
        if self.k_eq_v:
            # k_norm(K) = v_norm(K) * w_k: the k_norm weight as a TILE row broadcast over heads and rows.
            self.k_norm_row = as_tt(sd["self_attn.k_norm.weight"].reshape(1, 1, 1, -1))
        elif not self.meta_rope:
            self.k_norm = norm_weight("self_attn.k_norm")
        self.layer_scalar = float(sd["layer_scalar"].float().item())

        shared = {} if shared is None else shared
        # The key names the column layout: a `shared` dict may also serve FunctionalDecoder layers, whose
        # sliding-kind tables are in HF order.
        key = ("rope", self.layer_type, self.max_context, "interleaved" if self.meta_rope else "hf")
        if key not in shared:
            cos, sin = rope_tables(c, self.layer_type, self.max_context)
            if self.meta_rope:
                cos, sin = cos[:, perm], sin[:, perm]
            shared[key] = {
                # Prefill slices [start, start + chunk) rows from the TILE tables.
                "cos": as_tt(cos.reshape(1, 1, self.max_context, self.head_dim)),
                "sin": as_tt(sin.reshape(1, 1, self.max_context, self.head_dim)),
                # Decode gathers rows with ttnn.embedding from ROW_MAJOR tables.
                "cos_rm": as_tt(cos, layout=ttnn.ROW_MAJOR_LAYOUT),
                "sin_rm": as_tt(sin, layout=ttnn.ROW_MAJOR_LAYOUT),
            }
        self.rope = shared[key]

        # Row index of each row within a physical chunk (float32 is exact), for masking pad rows.
        self.row_ids = as_tt(
            torch.arange(prefill_chunk_size, dtype=torch.float32).reshape(1, 1, -1, 1), dtype=ttnn.float32
        )

        cache_shape = [num_blocks, self.num_kv_heads, block_size, self.head_dim]
        self.kv_cache = [
            ttnn.zeros(
                cache_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for _ in range(2)
        ]

        self.compute_hifi4_fp32 = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Prefill SDPA keeps FP32 dest. With BF16 dest the op takes its streaming kernel (which also
        # skips out-of-window K tiles), but sliding-layer worst-row PCC drops from 0.9997 to 0.9963.
        self.compute_sdpa_prefill = self.compute_hifi4_fp32
        self.sliding_attn_block = SLIDING_ATTN_BLOCK
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.worker_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid.x - 1, self.grid.y - 1))}
        )
        if self.meta_rope:
            # Interleaved-pair rotation as a 32x32 matrix: one DRAM copy for prefill, and one L1 shard on
            # each of the first 32 user cores for decode. Every decode batch (users on cores 0..B-1) finds
            # its shard there, so nothing is allocated per batch size. Shared across layers.
            if "rope_trans" not in shared:
                trans = rot_transformation_mat()
                users = ttnn.num_cores_to_corerangeset(MAX_DECODE_BATCH, self.grid, row_wise=True)
                shared["rope_trans"] = {
                    "prefill": as_tt(trans),
                    "decode": ttnn.from_torch(
                        trans.repeat(1, 1, MAX_DECODE_BATCH, 1),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh_device,
                        memory_config=ttnn.create_sharded_memory_config(
                            (TILE, TILE),
                            core_grid=users,
                            strategy=ttnn.ShardStrategy.HEIGHT,
                            use_height_and_width_as_shard_shape=True,
                        ),
                    ),
                }
            self.rope_trans = shared["rope_trans"]
            # rotary_embedding_llama requires BF16 dest for head_dim > 128.
            self.compute_rope_decode = ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        # Decode SDPA uses HiFi4 + FP32 dest and one K tile per cache block. A K tile spanning two blocks
        # reads one page-table entry past an odd-width row (see FunctionalDecoder).
        self.sdpa_decode_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=0,
            k_chunk_size=block_size,
            exp_approx_mode=False,
        )
        self._decode_layouts = {}
        return self

    # ------------------------------------------------------------------ shared pieces
    def _rms(self, x, weight=None, **kwargs):
        return ttnn.rms_norm(
            x, epsilon=self.eps, weight=weight, compute_kernel_config=self.compute_hifi4_fp32, **kwargs
        )

    def _k_from_v(self, v_n):
        """k_eq_v layers: k_norm(K) = v_norm(K) * w_k (V is the raw K projection)."""
        return ttnn.multiply(v_n, self.k_norm_row)

    def _mlp_and_residuals(self, x, attn_out, matmul):
        """Post-attention norm + residual, GELU(tanh)-gated MLP, post-FFN norm + residual, layer scalar."""
        h = self._rms(attn_out, self.post_attn_norm)
        ttnn.deallocate(attn_out)
        x = ttnn.add(x, h)
        ttnn.deallocate(h)
        h = self._rms(x, self.pre_ffn_norm)
        gate = matmul(h, self.w_gate)
        up = matmul(h, self.w_up)
        ttnn.deallocate(h)
        prod = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.GELU_TANH])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        down = matmul(prod, self.w_down)
        ttnn.deallocate(prod)
        h = self._rms(down, self.post_ffn_norm)
        ttnn.deallocate(down)
        out = ttnn.add(x, h, activations=[(ttnn.UnaryOpType.MUL_UNARY_SFPU, self.layer_scalar)])
        ttnn.deallocate(x)
        ttnn.deallocate(h)
        return out

    # ------------------------------------------------------------------ prefill
    def _prefill_matmul(self, x, w):
        return ttnn.experimental.minimal_matmul(
            x, w, compute_kernel_config=self.compute_hifi4_fp32, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _sdpa_prefill_cfg(self, chunk_start, padded_len):
        """Chunked-SDPA program config for one prefill chunk (same policy as FunctionalDecoder).

        The chunk start must be a multiple of both tile sizes, and Q tiles divide the chunk length. Full
        attention keeps exact FP32 running accumulators (``fp32_output_accumulation``) for its unbounded
        prefix. K tiles never exceed one cache block.
        """
        full = self.sliding_window is None
        q_cap, k_cap = (128, min(64, self.block_size)) if full else (SLIDING_Q_TILE, self.block_size)

        def largest(cap, *values):
            size = cap
            while size > TILE and any(v % size for v in values):
                size //= 2
            return size

        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=largest(q_cap, chunk_start, padded_len),
            k_chunk_size=largest(k_cap, chunk_start),
            exp_approx_mode=False,
            fp32_output_accumulation=full,
        )

    def _window_shift(self, chunk_start):
        """Sliding layers: largest page- and Q-tile-aligned origin <= the first key the chunk can attend."""
        if self.sliding_window is None:
            return 0
        align = math.lcm(SLIDING_Q_TILE, self.block_size)
        return max(0, (chunk_start - self.sliding_window + 1) // align * align)

    def _window_table_blocks(self, rows):
        """Page-table blocks a sliding query block attends from its shifted origin.

        Keys run from the origin to the block's last row: at most window + Q-tile alignment + rows tokens.
        Using the same count for every block of up to ``SLIDING_ATTN_BLOCK`` rows keeps the slice width,
        which the SDPA program bakes in, constant at every position. One program then serves the whole
        context instead of one per 2048 rows of position.
        """
        align = math.lcm(SLIDING_Q_TILE, self.block_size)
        return -(-(self.sliding_window + align + max(rows, SLIDING_ATTN_BLOCK)) // self.block_size)

    def _sdpa(self, q, *, page_table, user_id, q_start):
        """Chunked SDPA of query rows ``q`` (absolute start ``q_start``) over the request's paged cache.

        Sliding layers start the attended page-table row at the window: keys before
        ``q_start - (window - 1)`` are never visited, whereas the FP32-accumulating SDPA kernel walks every
        K tile from row 0 of the table and only masks out-of-window ones. Both masks depend on
        ``q_pos - k_pos`` only, so moving the origin by ``shift`` (page- and Q-tile-aligned) is exact.
        """
        shift = self._window_shift(q_start)
        first = shift // self.block_size
        last = page_table.shape[-1]
        if self.sliding_window is not None:
            last = min(last, first + self._window_table_blocks(q.shape[-2]))
        table = ttnn.slice(page_table, [user_id, first], [user_id + 1, last])
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            q,
            self.kv_cache[0],
            self.kv_cache[1],
            table,
            q_start - shift,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=self._sdpa_prefill_cfg(q_start - shift, q.shape[-2]),
            compute_kernel_config=self.compute_sdpa_prefill,
        )

    def _prefill_attention(self, q_r, *, page_table, user_id, chunk_start):
        """Attention of a chunk's queries ``[1, heads, rows, hd]``; the chunk's K/V are already cached.

        Sliding layers split the queries into blocks of ``sliding_attn_block`` rows, each attending from
        its own window origin, so the per-chunk work is ~rows x (window + block) instead of ~rows^2 / 2.
        """
        rows = q_r.shape[-2]
        blk = self.sliding_attn_block
        if self.sliding_window is None or rows <= blk:
            return self._sdpa(q_r, page_table=page_table, user_id=user_id, q_start=chunk_start)
        outs = []
        for a in range(0, rows, blk):
            b = min(a + blk, rows)
            q_blk = ttnn.slice(q_r, [0, 0, a, 0], [1, self.num_heads, b, self.head_dim])
            outs.append(self._sdpa(q_blk, page_table=page_table, user_id=user_id, q_start=chunk_start + a))
            ttnn.deallocate(q_blk)
        attn = ttnn.concat(outs, dim=2)
        for t in outs:
            ttnn.deallocate(t)
        return attn

    def _prefill_heads_hf(self, h, cos, sin):
        """Full layers: QKV matmul, head split, per-head norms, HF rotate-half RoPE."""
        qkv = self._prefill_matmul(h, self.wqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q_n = self._rms(q, self.q_norm)
        v_n = self._rms(v)
        k_n = self._k_from_v(v_n) if self.k_eq_v else self._rms(k, self.k_norm)
        for t in (q, k, v):
            ttnn.deallocate(t)
        q_r = ttnn.experimental.rotary_embedding_hf(q_n, cos, sin)
        k_r = ttnn.experimental.rotary_embedding_hf(k_n, cos, sin)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)
        return q_r, k_r, v_n

    def _norm_rope(self, x, heads, weight=None, cos=None, sin=None):
        """Head split + per-head RMSNorm (+ interleaved RoPE) in one op: [1, 1, S, heads*hd] -> [1, heads, S, hd]."""
        rope = (
            dict(transformation_mat=self.rope_trans["prefill"], rope_cos=cos, rope_sin=sin) if cos is not None else {}
        )
        return ttnn.experimental.dit_fused_distributed_rmsnorm(
            x,
            0,
            self.device,
            [],
            epsilon=self.eps,
            num_heads_per_device=heads,
            per_head_norm=True,
            weight=weight,
            compute_kernel_config=self.compute_hifi4_fp32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **rope,
        )

    def _prefill_heads_meta(self, h, cos, sin):
        """Sliding layers: the QKV matmul writes [Q heads 0-7 | Q heads 8-15 | K | V] as four tensors, then the
        fused per-head norm + RoPE op replaces head split, three norms and two RoPE ops."""
        qa, qb, k_flat, v_flat = ttnn.experimental.minimal_matmul_split(
            h, self.wqkv, chunks=4, compute_kernel_config=self.compute_hifi4_fp32, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        half = self.num_heads // 2
        q_parts = [self._norm_rope(t, half, self.q_norm_half, cos, sin) for t in (qa, qb)]
        q_r = ttnn.concat(q_parts, dim=1)
        k_r = self._norm_rope(k_flat, self.num_kv_heads, self.k_norm_heads, cos, sin)
        v_n = self._norm_rope(v_flat, self.num_kv_heads)
        for t in (qa, qb, k_flat, v_flat, *q_parts):
            ttnn.deallocate(t)
        return q_r, k_r, v_n

    def _prefill_chunk(self, x, *, page_table, user_id, chunk_start, padded_len):
        """One physical chunk: x ``[1, 1, padded_len, H]`` at absolute ``chunk_start``."""
        h = self._rms(x, self.input_norm)
        end = chunk_start + padded_len
        # Slices of persistent tensors (RoPE tables, page table) may alias their source: never free them.
        cos = ttnn.slice(self.rope["cos"], [0, 0, chunk_start, 0], [1, 1, end, self.head_dim])
        sin = ttnn.slice(self.rope["sin"], [0, 0, chunk_start, 0], [1, 1, end, self.head_dim])
        heads = self._prefill_heads_meta if self.meta_rope else self._prefill_heads_hf
        q_r, k_r, v_n = heads(h, cos, sin)
        ttnn.deallocate(h)

        # Write this chunk's K/V at its absolute blocks, then attend over the request's cache.
        first_block = chunk_start // self.block_size
        num_chunk_blocks = padded_len // self.block_size
        chunk_table = ttnn.slice(page_table, [user_id, first_block], [user_id + 1, first_block + num_chunk_blocks])
        ttnn.experimental.paged_fill_cache(self.kv_cache[0], k_r, chunk_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(self.kv_cache[1], v_n, chunk_table, batch_idx=0)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_n)
        attn = self._prefill_attention(q_r, page_table=page_table, user_id=user_id, chunk_start=chunk_start)
        ttnn.deallocate(q_r)
        attn_flat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        o = self._prefill_matmul(attn_flat, self.wo)
        ttnn.deallocate(attn_flat)
        return self._mlp_and_residuals(x, o, self._prefill_matmul)

    def prefill_forward(self, hidden_states, *, page_table, start_pos=0, user_id=0):
        """``[1, 1, S, H]`` -> ``[1, 1, S, H]`` for one request (contract as FunctionalDecoder.prefill_forward)."""
        seq_len = hidden_states.shape[-2]
        assert hidden_states.shape[0] == 1 and hidden_states.shape[1] == 1, "prefill is one request per call"
        assert seq_len >= 1 and start_pos >= 0 and start_pos % self.block_size == 0
        assert start_pos + seq_len <= self.max_context
        chunk = self.prefill_chunk_size
        outputs = []
        for offset in range(0, seq_len, chunk):
            valid = min(chunk, seq_len - offset)
            padded = _round_up(valid, self.block_size)
            x = ttnn.slice(hidden_states, [0, 0, offset, 0], [1, 1, offset + valid, self.hidden_size])
            if padded != valid:
                # ttnn.pad may only relabel existing tile padding, whose contents are arbitrary for
                # device-produced activations; force the pad rows to zero so the cache never holds NaN.
                x = ttnn.pad(x, [(0, 0), (0, 0), (0, padded - valid), (0, 0)], 0.0)
                rows = ttnn.slice(self.row_ids, [0, 0, 0, 0], [1, 1, padded, 1])
                keep = ttnn.lt(rows, float(valid), dtype=ttnn.bfloat16)
                x = ttnn.where(keep, x, 0.0)
            outputs.append(
                self._prefill_chunk(
                    x, page_table=page_table, user_id=user_id, chunk_start=start_pos + offset, padded_len=padded
                )
            )
        out = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2)
        if len(outputs) > 1:
            for t in outputs:
                ttnn.deallocate(t)
        if out.shape[-2] != seq_len:
            # A row-prefix slice may be a view of ``out``; let Python own the buffer's lifetime.
            out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, seq_len, self.hidden_size])
        return out

    # ------------------------------------------------------------------ decode
    def _decode_matmul(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.compute_hifi4_fp32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _decode_layout(self, batch):
        """Height-sharded per-user ``[32, head_dim]`` layouts for the RoPE rows and the fused-update V."""
        if batch not in self._decode_layouts:
            users = ttnn.num_cores_to_corerangeset(batch, self.grid, row_wise=True)
            # paged_fused_update_cache needs V on as many cores as K, disjoint from K's (the first `batch`).
            start = ttnn.CoreCoord(batch % self.grid.x, batch // self.grid.x)
            v_cores = ttnn.num_cores_to_corerangeset_in_subcoregrids(start, batch, self.worker_grid, row_wise=True)

            def per_user(cores):
                return ttnn.create_sharded_memory_config(
                    (TILE, self.head_dim),
                    core_grid=cores,
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    use_height_and_width_as_shard_shape=True,
                )

            self._decode_layouts[batch] = {"rope": per_user(users), "v": per_user(v_cores)}
        return self._decode_layouts[batch]

    def _head_norm_in_place(self, t, weight):
        """Per-head RMSNorm of a one-user head-sharded ``[1, 1, heads, hd]`` tensor, kept in L1.

        The single ``[32, hd]`` shard is relabelled block-sharded (rms_norm rejects height sharding) and
        normalized by the sharded LayerNorm program on its own core. Returns the block-sharded result.
        """
        spec = t.memory_config().shard_spec
        block = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)
        tb = ttnn.to_memory_config(t, block)
        cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[1, 1],
            subblock_w=4,
            block_h=spec.shape[0] // TILE,
            block_w=self.head_dim // TILE,
            inplace=False,
        )
        out = self._rms(tb, weight, program_config=cfg, memory_config=block)
        ttnn.deallocate(tb)
        return out

    def _decode_head_norms(self, q, k, v):
        """q/k/v per-head RMSNorm. Returns (q_n, k_n, v_n); block-sharded at batch 1, DRAM otherwise."""
        if q.shape[1] == 1:
            q_n = self._head_norm_in_place(q, self.q_norm)
            v_n = self._head_norm_in_place(v, None)
            k_n = self._k_from_v(v_n) if self.k_eq_v else self._head_norm_in_place(k, self.k_norm)
            return q_n, k_n, v_n
        # Batches above one go through interleaved copies: their per-user shards do not form one
        # rectangular grid, so the block-sharded relabel is impossible. Sliding layers (head_dim 256) keep
        # the copies in L1 (6 us faster at B=32); at head_dim 512 x 32 users L1-interleaved copies clash
        # with the paged-cache CBs, so full layers use DRAM.
        mem = ttnn.L1_MEMORY_CONFIG if self.is_sliding else ttnn.DRAM_MEMORY_CONFIG

        def norm(t, weight=None):
            t_i = ttnn.to_memory_config(t, mem)
            out = self._rms(t_i, weight, memory_config=mem)
            ttnn.deallocate(t_i)
            return out

        q_n = norm(q, self.q_norm)
        v_n = norm(v)
        k_n = self._k_from_v(v_n) if self.k_eq_v else norm(k, self.k_norm)
        return q_n, k_n, v_n

    def _decode_rope_rows(self, current_pos, batch, rope_shard):
        """cos/sin ``[1, B, 1, hd]`` (one ``[32, hd]`` tile per user), height-sharded for the RoPE op.

        Each user's position becomes a 32-entry index row: the position, then 31 zeros. The embedding's
        fused tilize turns every index row into one tile, written straight into ``rope_shard``.
        ``padding_idx=0`` serves the zero entries from a local copy of table row 0, which is bit-identical
        to a gather of row 0. Inactive users (-1) gather row 0; the ops skip their cache write and attention.
        """
        idx = ttnn.typecast(ttnn.relu(current_pos), ttnn.uint32)
        idx = ttnn.pad(ttnn.reshape(idx, [batch, 1]), [(0, 0), (0, TILE - 1)], 0)
        rows = []
        for table in (self.rope["cos_rm"], self.rope["sin_rm"]):
            t = ttnn.embedding(idx, table, padding_idx=0, layout=ttnn.TILE_LAYOUT, memory_config=rope_shard)
            # [B, 32, hd] -> logical [1, B, 1, hd] over the same [1, B, 32, hd] tiles (metadata only).
            rows.append(ttnn.reshape(t, [1, batch, 1, self.head_dim], [1, batch, TILE, self.head_dim]))
        ttnn.deallocate(idx)
        return rows

    def decode_forward(self, hidden_states, *, current_pos, page_table):
        """``[1, 1, B, H]`` -> ``[1, 1, B, H]`` (contract as FunctionalDecoder.decode_forward)."""
        batch = hidden_states.shape[-2]
        assert 1 <= batch <= MAX_DECODE_BATCH
        layout = self._decode_layout(batch)
        x = hidden_states
        h = self._rms(x, self.input_norm)
        qkv = ttnn.linear(
            h, self.wqkv, compute_kernel_config=self.compute_hifi4_fp32, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        head_shard = q.memory_config()
        kv_shard = k.memory_config()
        q_n, k_n, v_n = self._decode_head_norms(q, k, v)
        for t in (q, k, v):
            ttnn.deallocate(t)

        cos, sin = self._decode_rope_rows(current_pos, batch, layout["rope"])
        q_s = ttnn.to_memory_config(q_n, head_shard)
        k_s = ttnn.to_memory_config(k_n, kv_shard)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)
        if self.meta_rope:
            rope = dict(is_decode_mode=True, compute_kernel_config=self.compute_rope_decode)
            q_rs = ttnn.experimental.rotary_embedding_llama(q_s, cos, sin, self.rope_trans["decode"], **rope)
            k_r = ttnn.experimental.rotary_embedding_llama(k_s, cos, sin, self.rope_trans["decode"], **rope)
        else:
            q_rs = ttnn.experimental.rotary_embedding_hf(q_s, cos, sin, is_decode_mode=True)
            k_r = ttnn.experimental.rotary_embedding_hf(k_s, cos, sin, is_decode_mode=True)
        for t in (q_s, k_s, cos, sin):
            ttnn.deallocate(t)
        q_r = q_rs  # SDPA decode reads the head-sharded Q directly (no DRAM park)

        v_s = ttnn.to_memory_config(v_n, layout["v"])
        ttnn.deallocate(v_n)
        ttnn.experimental.paged_fused_update_cache(
            self.kv_cache[0], k_r, self.kv_cache[1], v_s, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_s)
        # With a single KV head the op can write its output straight into the head-concat layout. With
        # GQA (sliding layers) it only produces interleaved output.
        direct = self.num_kv_heads == 1
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_r,
            self.kv_cache[0],
            self.kv_cache[1],
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=self.sdpa_decode_cfg,
            compute_kernel_config=self.compute_hifi4_fp32,
            memory_config=head_shard if direct else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_r)
        if not direct:
            attn_i = attn
            attn = ttnn.to_memory_config(attn_i, head_shard)
            ttnn.deallocate(attn_i)
        # Users are sharded row-wise over the grid; with more users than one grid row the op needs
        # the explicit worker grid.
        attn_flat = ttnn.experimental.nlp_concat_heads_decode(
            attn, num_heads=self.num_heads, sub_core_grids=self.worker_grid
        )
        ttnn.deallocate(attn)
        if attn_flat.shape[-2] != batch:
            # The head-concat output is tile-padded to 32 users; restore the logical batch (metadata only).
            width = attn_flat.shape[-1]
            attn_flat = ttnn.reshape(attn_flat, [1, 1, batch, width], [1, 1, attn_flat.padded_shape[-2], width])
        o = self._decode_matmul(attn_flat, self.wo)
        ttnn.deallocate(attn_flat)
        return self._mlp_and_residuals(x, o, self._decode_matmul)
