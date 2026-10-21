# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional TTNN decoder layer for ``google/gemma-4-12B-it`` (text decoder, single device).

HF reference: ``transformers.models.gemma4_unified.modeling_gemma4_unified.Gemma4UnifiedTextDecoderLayer``.

Layer kinds (selected by ``hf_config.layer_types[layer_idx]``):

* ``sliding_attention`` (40 of 48 layers): 16 Q heads, 8 KV heads, head_dim 256, separate K/V
  projections, default RoPE (theta 1e4) over the whole head, causal window of 1024 tokens
  (query at position p sees keys in ``(p - 1024, p]``).
* ``full_attention`` (8 of 48 layers): 16 Q heads, 1 KV head, head_dim 512, ``attention_k_eq_v``
  (V is the raw K projection, normalized without scale and not rotated), proportional RoPE
  (theta 1e6, first 64 of 256 frequency pairs rotate, the rest are identity), unbounded causal.

Both kinds share one implementation: pre-norm attention (q/k RMSNorm with scale, v RMSNorm without
scale, attention scale 1.0), post-attention norm, residual, pre-FFN norm, GELU(tanh)-gated MLP,
post-FFN norm, residual, and finally the whole hidden state is multiplied by ``layer_scalar``.

Runtime contract (all tensors are device tensors; nothing in either forward touches the host):

``prefill_forward(hidden_states, *, page_table, start_pos=0, user_id=0)``
    hidden_states: ``[1, 1, S, 3840]`` BF16 TILE, any logical ``S >= 1``.
    page_table: ``[num_users, max_pages]`` int32 ROW_MAJOR; row ``user_id`` maps this request's
        logical cache blocks to physical blocks of this layer's paged cache.
    start_pos: absolute position of row 0 (the number of tokens of this request already in the
        cache). ``0`` starts a fresh request; a positive value continues an explicit prefix and must
        be a multiple of ``block_size``. ``start_pos + S <= max_context``.
    Rows are processed in physical chunks of ``prefill_chunk_size`` tokens. The final chunk is
    zero-padded to a multiple of ``block_size``; padded rows are causally invisible to real rows,
    their K/V only land in the request's own last block beyond its logical length (overwritten by
    the next decode write), and their outputs are sliced away. Returns ``[1, 1, S, 3840]``.
    The page-table row must map blocks ``[0, ceil((start_pos + S) / block_size))``.

``decode_forward(hidden_states, *, current_pos, page_table)``
    hidden_states: ``[1, 1, B, 3840]`` BF16 TILE, ``1 <= B <= 32`` users (one token each).
    current_pos: ``[B]`` int32 ROW_MAJOR, absolute position of each user's token (its K/V is
        written there and it attends to ``[0, current_pos]``, windowed for sliding layers).
        ``-1`` skips a user's cache write and attention (its output row is undefined).
    page_table: ``[B, max_pages]`` int32 ROW_MAJOR, row b maps user b's blocks.
    Returns ``[1, 1, B, 3840]``. Trace-safe: no host reads/writes, positions and page tables are
    device tensors that the caller refreshes between replays.

The paged KV cache (``self.kv_cache = [k, v]``, each ``[num_blocks, kv_heads, block_size,
head_dim]`` BF16, zero-initialized) is owned by the layer and allocated in ``from_state_dict``.
Every page-table entry must name a valid block; entries past a request's length should name a block
holding finite data (e.g. a reserved zero block): attention reads the masked tail of the last K tile,
and non-finite keys there escape the additive mask. Attention K tiles never span more than one cache
block, so reads stay inside the row for any row width. Only text tokens are
supported: the multimodal bidirectional vision-token mask (``use_bidirectional_attention='vision'``)
is outside the text-decoder contract.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

TILE = 32
MAX_DECODE_BATCH = 32
HF_PREFIX = "model.language_model.layers.{}."


def _round_up(x, m):
    return -(-x // m) * m


def rope_inv_freq(hf_config, layer_type):
    """Inverse RoPE frequencies, mirroring HF ``default`` and ``proportional`` rope init."""
    params = hf_config.rope_parameters[layer_type]
    rope_type = params.get("rope_type", "default")
    base = params["rope_theta"]
    if rope_type == "default":
        dim = hf_config.head_dim
        return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    if rope_type == "proportional":
        head_dim = hf_config.global_head_dim if layer_type == "full_attention" else hf_config.head_dim
        rope_angles = int(params.get("partial_rotary_factor", 1.0) * head_dim // 2)
        rotated = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float() / head_dim))
        return torch.cat([rotated, torch.zeros(head_dim // 2 - rope_angles)])
    raise ValueError(f"unsupported rope_type {rope_type!r} for {layer_type}")


def rope_tables(hf_config, layer_type, max_context):
    """fp32 cos/sin ``[max_context, head_dim]`` computed exactly like HF's rotary embedding."""
    inv_freq = rope_inv_freq(hf_config, layer_type)
    pos = torch.arange(max_context, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :].float()
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


class FunctionalDecoder(LightweightModule):
    """One Gemma4 text decoder layer (sliding or full attention) on a 1x1 mesh."""

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
        prefill_chunk_size=2048,
        max_context=None,
        shared=None,
    ):
        """Build the layer from an HF state dict.

        state_dict: either full-checkpoint keys (``model.language_model.layers.{i}.*``) or keys
            relative to the layer (``self_attn.q_proj.weight``...).
        hf_config: the HF text config (``Gemma4UnifiedTextConfig``).
        num_blocks / block_size: paged KV cache geometry owned by this layer.
        prefill_chunk_size: physical prefill chunk (multiple of ``block_size``).
        max_context: supported context; defaults to ``hf_config.max_position_embeddings``.
        shared: optional dict reused across layers to share RoPE tables per layer kind.
        """
        c = hf_config
        assert tuple(mesh_device.shape) == (1, 1), "functional decoder runs on a 1x1 mesh"
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
        # attention_k_eq_v: V is the same raw projection as K; duplicating the K columns gives the
        # fused QKV head-split op an ordinary V slot (normalized differently and never rotated).
        v = k if self.k_eq_v else sd["self_attn.v_proj.weight"]
        assert q.shape[0] == self.num_heads * self.head_dim and k.shape[0] == self.num_kv_heads * self.head_dim
        self.wqkv = as_tt(torch.cat([q, k, v], dim=0).T)
        self.wo = as_tt(sd["self_attn.o_proj.weight"].T)
        self.w_gate = as_tt(sd["mlp.gate_proj.weight"].T)
        self.w_up = as_tt(sd["mlp.up_proj.weight"].T)
        self.w_down = as_tt(sd["mlp.down_proj.weight"].T)
        self.input_norm = norm_weight("input_layernorm")
        self.post_attn_norm = norm_weight("post_attention_layernorm")
        self.pre_ffn_norm = norm_weight("pre_feedforward_layernorm")
        self.post_ffn_norm = norm_weight("post_feedforward_layernorm")
        self.q_norm = norm_weight("self_attn.q_norm")
        self.k_norm = norm_weight("self_attn.k_norm")
        self.layer_scalar = float(sd["layer_scalar"].float().item())

        shared = {} if shared is None else shared
        key = ("rope", self.layer_type, self.max_context)
        if key not in shared:
            cos, sin = rope_tables(c, self.layer_type, self.max_context)
            shared[key] = {
                # Prefill slices [start, start + chunk) rows from the TILE tables.
                "cos": as_tt(cos.reshape(1, 1, self.max_context, self.head_dim)),
                "sin": as_tt(sin.reshape(1, 1, self.max_context, self.head_dim)),
                # Decode gathers one row per user with ttnn.embedding from ROW_MAJOR tables.
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

        arch = mesh_device.arch()
        self.compute_hifi4_fp32 = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Decode SDPA also uses HiFi4 + FP32 dest: BF16 dest left isolated decode rows at PCC ~0.9977 on
        # real layer-0 weights (doc/functional_decoder/logs/decode_precision_variants.log). With 16 BF16
        # Q heads the op takes its half-tile path, whose softmax exp is always approximate regardless of
        # math_approx_mode (FP32 Q, which would select full tiles, is rejected by the op).
        self.compute_sdpa_decode = self.compute_hifi4_fp32
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.worker_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.grid.x - 1, self.grid.y - 1))}
        )
        self.sdpa_decode_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=0,
            # One K tile per cache block: a K tile spanning two blocks reads one page-table entry past
            # the row when the row has an odd number of blocks, pulling an arbitrary DRAM block (possibly
            # Inf/NaN) into the masked tail of the last tile, which corrupts the softmax.
            k_chunk_size=block_size,
            exp_approx_mode=False,
        )
        return self

    # ------------------------------------------------------------------ shared pieces
    def _rms(self, x, weight=None):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=weight, compute_kernel_config=self.compute_hifi4_fp32)

    def _linear(self, x, w):
        return ttnn.linear(x, w, compute_kernel_config=self.compute_hifi4_fp32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _mlp_and_residuals(self, x, attn_out):
        """Post-attention norm + residual, gated GELU MLP, post-FFN norm + residual, layer scalar."""
        h = self._rms(attn_out, self.post_attn_norm)
        ttnn.deallocate(attn_out)
        x = ttnn.add(x, h)
        ttnn.deallocate(h)
        h = self._rms(x, self.pre_ffn_norm)
        gate = self._linear(h, self.w_gate)
        up = self._linear(h, self.w_up)
        ttnn.deallocate(h)
        gate_act = ttnn.gelu(gate, variant=ttnn.GeluVariant.Tanh)
        ttnn.deallocate(gate)
        h = ttnn.multiply(gate_act, up)
        ttnn.deallocate(gate_act)
        ttnn.deallocate(up)
        down = self._linear(h, self.w_down)
        ttnn.deallocate(h)
        h = self._rms(down, self.post_ffn_norm)
        ttnn.deallocate(down)
        out = ttnn.add(x, h)
        ttnn.deallocate(x)
        ttnn.deallocate(h)
        scaled = ttnn.multiply(out, self.layer_scalar)
        ttnn.deallocate(out)
        return scaled

    # ------------------------------------------------------------------ prefill
    def _sdpa_prefill_cfg(self, chunk_start, padded_len):
        """Chunked-SDPA program config for one prefill chunk.

        The op requires the chunk start to be a multiple of both tile sizes; Q tiles also divide the
        chunk length. Full attention keeps the running softmax @ V output and sum in exact FP32
        (``fp32_output_accumulation``): with BF16/TF32 accumulators the error grows with every K tile
        of an unbounded prefix (PCC 0.97 at 262144 tokens). Sliding layers visit at most
        ~1024 / 64 + 1 K tiles of the window and keep the default accumulators.
        """
        full = self.sliding_window is None
        # K tiles never exceed one cache block (see the decode config).
        q_cap, k_cap = (128, min(64, self.block_size)) if full else (256, self.block_size)

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

    def _prefill_chunk(self, x, *, page_table, user_id, chunk_start, padded_len):
        """One physical chunk: x ``[1, 1, padded_len, H]`` at absolute ``chunk_start``."""
        h = self._rms(x, self.input_norm)
        qkv = self._linear(h, self.wqkv)
        ttnn.deallocate(h)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        q_n = self._rms(q, self.q_norm)
        k_n = self._rms(k, self.k_norm)
        v_n = self._rms(v)
        for t in (q, k, v):
            ttnn.deallocate(t)
        end = chunk_start + padded_len
        cos = ttnn.slice(self.rope["cos"], [0, 0, chunk_start, 0], [1, 1, end, self.head_dim])
        sin = ttnn.slice(self.rope["sin"], [0, 0, chunk_start, 0], [1, 1, end, self.head_dim])
        q_r = ttnn.experimental.rotary_embedding_hf(q_n, cos, sin)
        k_r = ttnn.experimental.rotary_embedding_hf(k_n, cos, sin)
        # Slices of persistent tensors (RoPE tables, page table) may alias their source: never free them.
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)

        # Write this chunk's K/V at its absolute blocks, then attend over the request's cache.
        first_block = chunk_start // self.block_size
        num_chunk_blocks = padded_len // self.block_size
        chunk_table = ttnn.slice(page_table, [user_id, first_block], [user_id + 1, first_block + num_chunk_blocks])
        ttnn.experimental.paged_fill_cache(self.kv_cache[0], k_r, chunk_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(self.kv_cache[1], v_n, chunk_table, batch_idx=0)
        ttnn.deallocate(k_r)
        ttnn.deallocate(v_n)
        user_table = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[-1]])
        attn = ttnn.transformer.chunked_scaled_dot_product_attention(
            q_r,
            self.kv_cache[0],
            self.kv_cache[1],
            user_table,
            chunk_start,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=self._sdpa_prefill_cfg(chunk_start, padded_len),
            compute_kernel_config=self.compute_hifi4_fp32,
        )
        ttnn.deallocate(q_r)
        attn_flat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        o = self._linear(attn_flat, self.wo)
        ttnn.deallocate(attn_flat)
        return self._mlp_and_residuals(x, o)

    def prefill_forward(self, hidden_states, *, page_table, start_pos=0, user_id=0):
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
                keep = ttnn.typecast(ttnn.lt(rows, float(valid)), ttnn.bfloat16)
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
    def _decode_rope(self, x, cos, sin):
        return ttnn.experimental.rotary_embedding_hf(x, cos, sin, is_decode_mode=True)

    def decode_forward(self, hidden_states, *, current_pos, page_table):
        batch = hidden_states.shape[-2]
        assert 1 <= batch <= MAX_DECODE_BATCH
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
        # Per-head RMSNorm over head_dim on interleaved copies.
        q_i = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
        k_i = ttnn.to_memory_config(k, ttnn.DRAM_MEMORY_CONFIG)
        v_i = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
        for t in (q, k, v):
            ttnn.deallocate(t)
        q_n = self._rms(q_i, self.q_norm)
        k_n = self._rms(k_i, self.k_norm)
        v_n = self._rms(v_i)
        for t in (q_i, k_i, v_i):
            ttnn.deallocate(t)

        # Gather each user's cos/sin row on device: [1, B] uint32 -> [1, B, 1, head_dim].
        # Inactive users (-1) gather row 0; their cache write and attention are skipped by the ops.
        rope_idx = ttnn.reshape(ttnn.typecast(ttnn.relu(current_pos), ttnn.uint32), [1, batch])
        cos = ttnn.embedding(rope_idx, self.rope["cos_rm"], layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rope_idx, self.rope["sin_rm"], layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(rope_idx)
        cos = ttnn.reshape(cos, [1, batch, 1, self.head_dim])
        sin = ttnn.reshape(sin, [1, batch, 1, self.head_dim])
        rope_shard = ttnn.create_sharded_memory_config(
            (TILE, self.head_dim),
            core_grid=ttnn.num_cores_to_corerangeset(batch, self.grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        cos_s = ttnn.to_memory_config(cos, rope_shard)
        sin_s = ttnn.to_memory_config(sin, rope_shard)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        q_s = ttnn.to_memory_config(q_n, head_shard)
        k_s = ttnn.to_memory_config(k_n, kv_shard)
        ttnn.deallocate(q_n)
        ttnn.deallocate(k_n)
        q_rs = self._decode_rope(q_s, cos_s, sin_s)
        k_r = self._decode_rope(k_s, cos_s, sin_s)
        for t in (q_s, k_s, cos_s, sin_s):
            ttnn.deallocate(t)
        # Park Q in DRAM so the cache-update kernels (large CBs at head_dim 512 x 32 users) fit L1.
        q_r = ttnn.to_memory_config(q_rs, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_rs)

        ttnn.experimental.paged_update_cache(
            self.kv_cache[0], k_r, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.deallocate(k_r)
        # Shard V only now: at head_dim 512 x 32 users the update kernel's CBs leave room for a
        # single sharded K/V operand in L1.
        v_s = ttnn.to_memory_config(v_n, kv_shard)
        ttnn.deallocate(v_n)
        ttnn.experimental.paged_update_cache(
            self.kv_cache[1], v_s, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.deallocate(v_s)
        attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_r,
            self.kv_cache[0],
            self.kv_cache[1],
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
            scale=1.0,
            sliding_window_size=self.sliding_window,
            program_config=self.sdpa_decode_cfg,
            compute_kernel_config=self.compute_sdpa_decode,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_r)
        attn_s = ttnn.to_memory_config(attn, head_shard)
        ttnn.deallocate(attn)
        # Users are sharded row-wise over the grid; with more users than one grid row the op needs
        # the explicit worker grid.
        attn_flat = ttnn.experimental.nlp_concat_heads_decode(
            attn_s, num_heads=self.num_heads, sub_core_grids=self.worker_grid
        )
        ttnn.deallocate(attn_s)
        if attn_flat.shape[-2] != batch:
            # The head-concat output is tile-padded to 32 users; restore the logical batch (metadata only).
            width = attn_flat.shape[-1]
            attn_flat = ttnn.reshape(attn_flat, [1, 1, batch, width], [1, 1, attn_flat.padded_shape[-2], width])
        o = self._linear(attn_flat, self.wo)
        ttnn.deallocate(attn_flat)
        return self._mlp_and_residuals(x, o)
