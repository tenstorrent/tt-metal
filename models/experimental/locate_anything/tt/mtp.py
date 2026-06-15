# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device Parallel Box Decoding (MTP / multi-token prediction) for LocateAnything-3B.

Implements one ``mtp_step`` over a LATransformer (Qwen2.5-3B) that predicts a whole
structured unit (a box = 6 tokens) in a single forward, using the block-bidirectional
"generation window" attention from NVIDIA's reference (``modeling_qwen2.py`` SDPA path,
``update_causal_mask_for_one_gen_window_2d`` == ``build_magi_ranges``).

Per step the window fed to the LLM is::

    ids       = [ uncached_real_tokens..., last_real_tok, MASK, MASK, MASK, MASK, MASK ]
    positions = [ cached_len.. ,           cur_len-1,     cur_len, ..., cur_len+4       ]

The last ``n_future`` rows (the window) attend bidirectionally to each other and to all
real keys EXCEPT the blocked column ``kv_len - n_future - 1`` (the cached copy of the
duplicated last token). The ``n_future`` readout logits decode a box via the reference
``sample_tokens``/``handle_pattern`` (reused verbatim on host).

KV is committed lazily, exactly like the production ``batch_utils/engine_hybrid.py``: after
each forward only the K/V of the *real* tokens (cached + this step's uncached leading real
tokens) is kept; the duplicate-last + mask window K/V is dropped. Accepted box tokens become
the leading "uncached" real tokens of the next window forward.

This module closely reproduces the torch-CPU MTP reference (``reference/mtp_cpu_loop.py``):
the END-TO-END device-MTP vs torch-MTP logit PCC over every step/readout row is ~0.986
(first-step ~0.996); see ``tests/test_mtp.py`` for the validation and the forward-pass /
tok/s comparison against the AR bench.

PRECISION NOTE: ``ttnn.transformer.scaled_dot_product_attention`` corrupts the block-
bidirectional generation window when the Q/K sequence dims are not tile (32) multiples — with
a sub-tile ``q_len`` it silently degenerates the bidirectional last block toward causal
attention (SDPA output PCC ~0.94). ``_attn`` therefore tile-pads Q/K/V (and ``build_mask``
pads the additive mask with -inf in the pad region) so the bidirectional window is computed
correctly (per-layer attention PCC > 0.999), then slices the readout back to ``q_len``.

IMPORTANT (verified on the reference): greedy MTP does NOT reproduce greedy AR boxes — MTP
is an inherently approximate parallel decoder (see reference/mtp_cpu_loop.py). The device
correctness target here is therefore device-MTP == torch-CPU-MTP (same algorithm), not
MTP == AR. See the test docstring and the executor report for the full evidence.
"""

import torch

import ttnn
from models.tt_transformers.tt.common import Mode

NEG_INF = float("-inf")


class MTPDecoder:
    """Drives one or many MTP-window forwards over a built LATransformer.

    Reuses every weight/submodule of the model's TransformerBlocks (wqkv, wo, q_norm,
    k_norm, attention_norm, ff_norm, feed_forward) and the final norm + lm_head; only the
    attention *core* is reimplemented to (a) place K/V into a contiguous device cache slice
    at the committed length and (b) run a custom block-bidirectional masked SDPA.
    """

    def __init__(self, model, n_future=6):
        self.model = model
        self.args = model.args
        self.mesh_device = model.mesh_device
        self.n_future = n_future
        self.head_dim = self.args.head_dim
        self.n_kv_heads = self.args.n_kv_heads
        self.n_heads = self.args.n_heads
        # committed (real-token) K/V per layer: list of [k, v], each [1, n_kv, cached_len, hd]
        self.committed_k = [None] * len(model.layers)
        self.committed_v = [None] * len(model.layers)
        self.cached_len = 0

    # ---- KV management -------------------------------------------------------
    def reset_kv_from_prefill(self, tt_kv_cache, real_len):
        """Seed committed K/V from the prefill dense cache (first ``real_len`` positions).

        ``tt_kv_cache`` is the per-layer ``[k, v]`` dense cache used by the AR prefill
        (shape ``[batch, n_kv, max_seq, head_dim]``). We slice ``[0:real_len]`` on the seq
        dim for user 0 and store as the committed real-token K/V.
        """
        for i, (k, v) in enumerate(tt_kv_cache):
            # Slice [0:real_len] real tokens for user 0 and clone into a standalone DRAM
            # buffer (the source dense cache stays owned by the model / prefill path).
            kk = ttnn.slice(k, (0, 0, 0, 0), (1, self.n_kv_heads, real_len, self.head_dim))
            vv = ttnn.slice(v, (0, 0, 0, 0), (1, self.n_kv_heads, real_len, self.head_dim))
            self.committed_k[i] = ttnn.clone(kk, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            self.committed_v[i] = ttnn.clone(vv, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            ttnn.deallocate(kk)
            ttnn.deallocate(vv)
        self.cached_len = real_len

    def _commit(self, layer_idx, k_full, v_full, new_real_len):
        """Keep K/V for [0:new_real_len]; drop the dup-last + mask window rows.

        Clone the slice into a standalone DRAM buffer so the caller can safely
        deallocate ``k_full``/``v_full`` without aliasing the committed cache.
        """
        ks = ttnn.slice(k_full, (0, 0, 0, 0), (1, self.n_kv_heads, new_real_len, self.head_dim))
        vs = ttnn.slice(v_full, (0, 0, 0, 0), (1, self.n_kv_heads, new_real_len, self.head_dim))
        new_k = ttnn.clone(ks, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        new_v = ttnn.clone(vs, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(ks)
        ttnn.deallocate(vs)
        if self.committed_k[layer_idx] is not None:
            ttnn.deallocate(self.committed_k[layer_idx])
            ttnn.deallocate(self.committed_v[layer_idx])
        self.committed_k[layer_idx] = new_k
        self.committed_v[layer_idx] = new_v

    # ---- mask + rope ---------------------------------------------------------
    @staticmethod
    def _tile_pad(n, tile=32):
        return ((n + tile - 1) // tile) * tile

    def build_mask(self, cached_len, uncached_len):
        """Additive bf16 [1,1,q_pad,kv_pad] mask for one MTP-window forward (see module doc).

        ``ttnn.transformer.scaled_dot_product_attention`` corrupts the block-bidirectional
        window when the Q/K sequence dims are not tile (32) multiples: with a sub-tile q_len
        the op silently degenerates the bidirectional last block to causal-like attention
        (verified: SDPA output PCC drops to ~0.94, recovers to >0.999 once q/k are tile-padded).
        We therefore pad the logical [q_len,kv_len] mask out to 32-multiples with NEG_INF so
        the padded Q rows / K columns are fully masked, and ``_attn`` pads Q/K/V to match.
        """
        nf = self.n_future
        q_len = uncached_len + nf
        kv_len = cached_len + q_len
        window_start_k = kv_len - nf
        blocked_k = window_start_k - 1
        rows = []
        for i in range(uncached_len):
            rows.append(("causal", cached_len + i))
        for _ in range(uncached_len, q_len):
            rows.append(("window", blocked_k))
        return self._build_padded_mask(q_len, kv_len, rows)

    def _build_padded_mask(self, q_len, kv_len, rows):
        """Build an additive bf16 mask padded to 32-multiples in both seq dims.

        ``rows`` is a list of (kind, arg) of length q_len:
          ("causal", gpos)   -> attend keys [0:gpos+1]
          ("window", blocked)-> attend all kv keys except column ``blocked``
        The padded rows/cols (beyond q_len/kv_len) stay NEG_INF so SDPA ignores them.
        """
        q_pad = self._tile_pad(q_len)
        kv_pad = self._tile_pad(kv_len)
        m = torch.full((1, 1, q_pad, kv_pad), NEG_INF, dtype=torch.float32)
        for r, (kind, arg) in enumerate(rows):
            if kind == "causal":
                m[0, 0, r, : arg + 1] = 0.0
            else:  # window
                m[0, 0, r, :kv_len] = 0.0
                if 0 <= arg < kv_len:
                    m[0, 0, r, arg] = NEG_INF
        return ttnn.from_torch(
            m,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def window_rope(self, position_ids):
        """Gather per-position prefill cos/sin rows -> [1,1,q_len,head_dim] TILE tensors.

        position_ids: python list of length q_len (the window's positions, with the -1
        offset already applied to the n_future window per the reference).
        """
        rs = self.model.rope_setup
        cos = rs.cos_matrix_prefill  # [1,1,max_seq,head_dim] TILE
        sin = rs.sin_matrix_prefill
        idx = torch.tensor(position_ids, dtype=torch.int64)
        # gather rows on host from a cpu copy of the matrices (computed once, cheap)
        if not hasattr(self, "_cos_host"):
            self._cos_host = ttnn.to_torch(ttnn.get_device_tensors(cos)[0]).float()
            self._sin_host = ttnn.to_torch(ttnn.get_device_tensors(sin)[0]).float()
        cos_sel = self._cos_host[0, 0, idx, :].unsqueeze(0).unsqueeze(0)  # [1,1,q_len,hd]
        sin_sel = self._sin_host[0, 0, idx, :].unsqueeze(0).unsqueeze(0)
        tcos = ttnn.from_torch(
            cos_sel,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tsin = ttnn.from_torch(
            sin_sel,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return [tcos, tsin]

    # ---- one layer's MTP attention core -------------------------------------
    def _attn(self, layer_idx, attn_in, rot_mats, attn_mask, commit_real_len):
        """Prefill-style QKV/RoPE/heads + custom-mask SDPA + O-proj, on the q_len window.

        attn_in: [1,1,q_len,dim] (post attention_norm). Reuses the layer's Attention weights.
        Returns the attention output [1,1,q_len,dim]; updates committed K/V for this layer.
        """
        attn = self.model.layers[layer_idx].attention
        seq_len = attn_in.shape[-2]  # padded to a 128-multiple by the caller
        # QKV (prefill matmul path, single-device: no all_reduce needed but keep linear+bias).
        # Use the stock prefill program/mem configs so the DRAM-sharded wqkv weight gets a
        # valid (L1) circular buffer.
        xqkv = ttnn.linear(
            attn_in,
            attn.wqkv,
            dtype=attn.activation_dtype or ttnn.bfloat16,
            memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None),
            compute_kernel_config=attn.li_qkv_prefill_compute_kernel_cfg,
            program_config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None),
        )
        if attn.wqkv_bias_prefill is not None:
            xqkv = xqkv + attn.wqkv_bias_prefill

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=attn.n_local_heads,
            num_kv_heads=attn.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)
        norm_cfg = self.args.get_norm_config("attn", Mode.PREFILL, None)
        q = attn.q_norm(q, mode=Mode.PREFILL, norm_config=norm_cfg)
        k = attn.k_norm(k, mode=Mode.PREFILL, norm_config=norm_cfg)

        q, k = attn.rotary_embedding_prefill(q, k, rot_mats)  # [1,nqh,q_len,hd], [1,nkv,q_len,hd]

        # Build full K/V over committed real tokens + this window
        if self.committed_k[layer_idx] is not None:
            k_full = ttnn.concat([self.committed_k[layer_idx], k], dim=2)
            v_full = ttnn.concat([self.committed_v[layer_idx], v], dim=2)
        else:
            k_full = k
            v_full = v

        q8 = ttnn.typecast(q, dtype=ttnn.bfloat16)
        ttnn.deallocate(q)

        # Pad Q/K/V seq dims to tile (32) multiples so SDPA computes the bidirectional window
        # correctly (see build_mask). Padded Q rows / K cols are fully masked by attn_mask.
        # ttnn.pad CONSUMES its input, so pad COPIES of k_full/v_full and keep k_full/v_full
        # alive for the committed-KV update below.
        q_len = q8.shape[2]
        kv_len = k_full.shape[2]
        q_pad = self._tile_pad(q_len)
        kv_pad = self._tile_pad(kv_len)
        if q_pad != q_len:
            q8 = ttnn.pad(q8, padding=[(0, 0), (0, 0), (0, q_pad - q_len), (0, 0)], value=0.0)
        if kv_pad != kv_len:
            k_pad_src = ttnn.clone(k_full, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            v_pad_src = ttnn.clone(v_full, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
            k_pad_t = ttnn.pad(k_pad_src, padding=[(0, 0), (0, 0), (0, kv_pad - kv_len), (0, 0)], value=0.0)
            v_pad_t = ttnn.pad(v_pad_src, padding=[(0, 0), (0, 0), (0, kv_pad - kv_len), (0, 0)], value=0.0)
        else:
            k_pad_t = k_full
            v_pad_t = v_full

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q8,
            k_pad_t,
            v_pad_t,
            attn_mask=attn_mask,
            is_causal=False,
            scale=attn.scale,
            compute_kernel_config=attn.sdpa_prefill_compute_kernel_cfg,
        )
        if k_pad_t is not k_full:
            ttnn.deallocate(k_pad_t)
            ttnn.deallocate(v_pad_t)
        # drop the padded Q rows: keep the real [0:q_len] window rows
        if q_pad != q_len:
            attn_out_sliced = ttnn.slice(attn_out, (0, 0, 0, 0), (1, attn.n_local_heads, q_len, attn.head_dim))
            ttnn.deallocate(attn_out)
            attn_out = attn_out_sliced
        ttnn.deallocate(q8)

        # commit real-token K/V into fresh DRAM buffers; drop dup+mask window rows
        self._commit(layer_idx, k_full, v_full, commit_real_len)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v_full)
        if k_full is not k:
            ttnn.deallocate(k)
        if v_full is not v:
            ttnn.deallocate(v)

        attn_out = ttnn.reshape(attn_out, [1, attn.n_local_heads, -1, attn.head_dim])
        attn_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.linear(
            attn_out,
            attn.wo,
            compute_kernel_config=attn.li_o_prefill_compute_kernel_cfg,
            dtype=attn.activation_dtype or ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.args.get_attn_wo_program_config(Mode.PREFILL, seq_len, None),
        )
        ttnn.deallocate(attn_out)
        return out

    # ---- full MTP step over all layers --------------------------------------
    def mtp_step(self, window_embeds, position_ids, uncached_len):
        """Run one MTP-window forward; returns host logits [q_len, vocab].

        window_embeds: torch float [1, q_len, dim] (host) — the embeddings of
            [uncached_real..., dup_last, mask*5]. We upload + run the decoder stack.
        position_ids: python list, len q_len (window positions, with -1 offset on the last
            n_future already applied by the caller).
        uncached_len: number of leading real tokens in this window (their K/V is committed).
        """
        q_len = window_embeds.shape[1]
        commit_real_len = self.cached_len + uncached_len
        attn_mask = self.build_mask(self.cached_len, uncached_len)
        rot_mats = self.window_rope(position_ids)

        # upload embeds as [1,1,q_len,dim], replicated (single device) hidden state
        x = ttnn.from_torch(
            window_embeds.unsqueeze(1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device=self.mesh_device, dims=(None, 3), mesh_shape=self.args.cluster_shape
            ),
        )

        skip_mem_cfg = self.args.get_residual_mem_config(Mode.PREFILL, None)
        x = ttnn.to_memory_config(x, skip_mem_cfg)

        for li, layer in enumerate(self.model.layers):
            residual = x
            attn_norm_cfg = self.args.get_norm_config("attn", Mode.PREFILL, None)
            attn_in = layer.attention_norm(x, Mode.PREFILL, norm_config=attn_norm_cfg)
            attn_out = self._attn(li, attn_in, rot_mats, attn_mask, commit_real_len)
            attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
            hidden = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
            ttnn.deallocate(attn_out)
            residual2 = hidden
            ff_norm_cfg = self.args.get_norm_config("ff", Mode.PREFILL, None)
            ff_in = layer.ff_norm(hidden, Mode.PREFILL, norm_config=ff_norm_cfg)
            ff_out = layer.feed_forward.forward(ff_in, Mode.PREFILL)
            x = ttnn.add(residual2, ff_out, memory_config=skip_mem_cfg)
            ttnn.deallocate(ff_out)
            ttnn.deallocate(hidden)

        ttnn.deallocate(attn_mask)
        for t in rot_mats:
            ttnn.deallocate(t)

        # final norm + lm_head on all q_len rows
        x = self.model.norm(x, mode=Mode.PREFILL, norm_config=self.args.get_norm_config("lm_head", Mode.PREFILL, None))
        lm_in_cfg = self.args.get_lm_head_input_mem_config(Mode.PREFILL, None)
        if lm_in_cfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_in_cfg)
        logits = self.model.lm_head(x)
        logits = ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
        host = self.model.concat_host_output(logits.cpu())  # [1,1,q_len,vocab]
        ttnn.deallocate(logits)
        self.cached_len = commit_real_len
        return host[0, 0, :q_len, : self.model.vocab_size].to(torch.float32)
