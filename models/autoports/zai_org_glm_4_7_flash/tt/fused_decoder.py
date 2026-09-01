# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused TTNN decoder layer for zai-org/GLM-4.7-Flash (graph-fusing stage).

Same public contract as ``functional_decoder.FunctionalDecoder`` (see that
module for the full prefill/decode/paged-cache documentation); this subclass
rewrites the op graph for fewer, larger, more specialized ops while staying
numerically equivalent at the functional acceptance bar.

Graph rewrites relative to the functional decoder
=================================================

Attention (both modes):
- ``wq_a`` and ``wkv_a`` share the same LHS (the input-normed hidden states)
  and are fused into one ``wqkv_a`` matmul + two width slices (the
  deepseek_v3 mla1d decode idiom).
- For PREFILL, ``wq_b`` is additionally stored per-head as
  ``[1, nh, q_lora, qk_head]`` so its matmul broadcasts the LHS over the head
  batch and directly produces the ``[1, nh, S, qk_head]`` head layout,
  deleting the reshape -> permute head-split (measured ~2 ms + ~1.5 ms per
  1024-token chunk). Decode keeps the flat ``wq_b`` + untilize/reshape/tilize
  head split: the broadcast-batched matmul only runs through the default
  non-mcast path (explicit program configs and the batched DRAM-sharded
  config TT_FATAL on mismatched batch dims), and at M=1 tile it measured
  372 us vs 123 us for the whole functional q path. Folding ``W_UK`` into
  ``wq_b`` host-side (tried as ``fold_uk``) also measured slower in both
  modes (traced decode moe 1.652 vs 1.335 ms/tok) and was rejected.
- Decode output path: transpose -> untilize -> reshape -> tilize is replaced
  by ``ttnn.experimental.nlp_concat_heads_decode``.
- Prefill output path: permute -> reshape is replaced by
  ``ttnn.transformer.concatenate_heads``.
- Decode kvpe assembly transposes the two halves once each and concatenates
  in the update layout ([1, B, 1, 576]) instead of round-tripping through
  [1, 1, B, 576] (three transposes -> two). At batch 1 the (1, 2) transposes
  of [1, 1, 1, d] tensors (kv halves, rope cos/sin) are logical identities
  and are skipped entirely.

Measured on p150 (synthetic weights, bf16 + bf8 experts, same session,
functional -> fused): warmed prefill S=2048 moe 268 -> 210 ms, dense
19.3 -> 15.3 ms; traced decode batch 1 ctx 1024 moe 1.532 -> 1.035 ms/token,
dense 1.008 -> 0.969 ms/token. Full evidence: doc/fused_decoder/.

MoE:
- Router logits matmul takes the bf16 activations directly against the fp32
  gate weight (mixed-dtype matmul, fp32 accumulation and output); the
  explicit fp32 typecast of the activations is deleted.
- Decode at max_batch=1 uses the *indexed/gather* mode of
  ``ttnn.sparse_matmul``: the top-4 expert ids from ``ttnn.topk`` are handed
  to the kernels as a device-resident uint16 index list, the three expert
  matmuls compute ONLY those experts, and their outputs are compact
  ``[1, 4, B, N]`` instead of dense ``[1, 64, B, N]``. Routing weights for
  the combine are gathered (``ttnn.gather``) at the same indices, so the
  scatter-mask construction of the dense-weights path disappears. All
  shapes are static and the index values are read on device, so the path is
  trace-capturable.
- Decode at max_batch>1 keeps the functional union-sparsity path (the union
  of per-user top-4 sets is data-dependent in size, which the static compact
  output cannot express).
- The routed gate and up projections are packed into ONE weight tensor
  ``[1, E, hidden, 2*inter]`` and ONE sparse matmul (shared-LHS rewrite): the
  in0 multicast per (token-block, expert) pair happens once instead of twice.
  Measured on device: decode indexed gate/up 180 -> 144 us, prefill grouped
  gate/up stage 84.9 -> 68.1 ms per 1024-token chunk, both including the two
  post-matmul width slices the packing adds.
- SiLU is folded into the expert elementwise multiply
  (``input_tensor_a_activations=[SILU]``; the sparse matmul kernel hardcodes
  FUSE_ACTIVATION=0 so it cannot take the activation itself), and into the
  dense/shared-expert gate matmuls (``activation="silu"``).
- Routing weights are applied to the expert intermediate ``h`` (width 1536)
  instead of the down-projection output (width 2048): row scaling commutes
  with the right matmul, and the elementwise traffic drops 25%.
- ``routed_scaling_factor`` (x1.8) is folded host-side into the routed down
  weights (a constant output scale commutes with the matmul; block-fp
  quantization keeps the same relative error), so the routing-weight chain
  drops its trailing scalar multiply. The shared expert has separate,
  unscaled weights.
- Prefill gate/up sparse matmuls receive the *real* per-32-token-block union
  routing mask instead of all-ones, and the down matmul receives the
  per-chunk expert union, so never-selected (block, expert) pairs are
  skipped. Exact: un-selected experts have zero routing weight, so their
  (skipped, zero-filled) outputs contribute nothing either way.

Prefill driver:
- Chunk outputs are written into a preallocated output tensor with
  ``ttnn.experimental.slice_write`` instead of a rolling ``ttnn.concat``
  (which copied the accumulated prefix once per chunk).
"""

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import TILE, FunctionalDecoder


class FusedDecoder(FunctionalDecoder):
    """Graph-fused GLM-4.7-Flash decoder layer (same contract as functional)."""

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        max_batch_size=1,
        max_context=None,
        expert_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat16,
        prefill_chunk_size=2048,
        paged_config=None,
    ):
        import torch

        self = FunctionalDecoder.from_state_dict.__func__(
            cls,
            state_dict,
            hf_config=hf_config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_context=max_context,
            expert_dtype=expert_dtype,
            weight_dtype=weight_dtype,
            prefill_chunk_size=prefill_chunk_size,
            paged_config=paged_config,
        )
        dev = mesh_device

        def upload(w, dtype=weight_dtype):
            return ttnn.from_torch(
                w.contiguous(),
                device=dev,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Fused wq_a + wkv_a (shared LHS): [hidden, q_lora + kvpe_dim].
        wq_a = state_dict["self_attn.q_a_proj.weight"].to(torch.float32).T
        wkv_a = state_dict["self_attn.kv_a_proj_with_mqa.weight"].to(torch.float32).T
        self.wqkv_a = upload(torch.cat([wq_a, wkv_a], dim=-1))
        ttnn.deallocate(self.wq_a)
        ttnn.deallocate(self.wkv_a)
        del self.wq_a, self.wkv_a

        # Additional per-head wq_b layout for PREFILL: [1, nh, q_lora, qk_head]
        # (the broadcast-batched matmul emits the [1, nh, S, qk_head] head
        # layout directly, deleting the reshape + permute). Decode keeps the
        # inherited flat wq_b: at one M tile the broadcast-batched matmul is
        # 3x slower than flat + untilize/reshape/tilize (see module docstring).
        wq_b = state_dict["self_attn.q_b_proj.weight"].to(torch.float32).T  # [q_lora, nh*qk_head]
        wq_b_heads = wq_b.reshape(self.q_lora_rank, self.num_heads, self.qk_head_dim).permute(1, 0, 2)
        self.wq_b_heads = upload(wq_b_heads.unsqueeze(0))  # [1, nh, q_lora, qk_head]

        if self.layer_kind == "moe":
            # Packed routed gate+up: [1, E, hidden, 2*inter] (one sparse
            # matmul, one in0 multicast pass; replaces experts_gate/up).
            self.moe_inter = self.experts_gate.shape[-1]
            gate_up = torch.cat(
                [
                    torch.stack(
                        [
                            state_dict[f"mlp.experts.{e}.gate_proj.weight"].to(torch.float32).T
                            for e in range(self.n_experts)
                        ]
                    ),
                    torch.stack(
                        [
                            state_dict[f"mlp.experts.{e}.up_proj.weight"].to(torch.float32).T
                            for e in range(self.n_experts)
                        ]
                    ),
                ],
                dim=-1,
            ).unsqueeze(0)
            self.experts_gate_up = upload(gate_up, dtype=expert_dtype)
            ttnn.deallocate(self.experts_gate)
            ttnn.deallocate(self.experts_up)
            del self.experts_gate, self.experts_up

            # Fold routed_scaling_factor into the down weights (constant
            # output scale commutes with the matmul; routing weights then
            # stay plain normalized scores).
            down_stack = torch.stack(
                [state_dict[f"mlp.experts.{e}.down_proj.weight"].to(torch.float32).T for e in range(self.n_experts)]
            ).unsqueeze(0)
            scaled_down = upload(down_stack * self.routed_scaling, dtype=expert_dtype)
            ttnn.deallocate(self.experts_down)
            self.experts_down = scaled_down

            # Required-but-unread sparsity operand for the indexed expert matmuls.
            self.ones_e = ttnn.from_torch(
                torch.ones(1, 1, 1, self.n_experts),
                device=dev,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        return self

    # ------------------------------------------------------------------ shared mlp pieces

    def _swiglu_linear(self, x, w_gate, w_up, w_down, ck):
        # SiLU folded into the gate matmul (applied on the fp32 dest registers
        # before pack, so it is at least as accurate as the separate unary op).
        g = ttnn.linear(x, w_gate, compute_kernel_config=ck, activation="silu")
        u = ttnn.linear(x, w_up, compute_kernel_config=ck)
        h = ttnn.multiply(g, u)
        ttnn.deallocate(g)
        ttnn.deallocate(u)
        out = ttnn.linear(h, w_down, compute_kernel_config=ck)
        ttnn.deallocate(h)
        return out

    # ------------------------------------------------------------------ router

    def _router_scores(self, x_flat):
        """Biased fp32 scores + centered bf16 copy for topk.

        Returns (scores fp32 [1,1,T,E], centered bf16 [1,1,T,E]).
        """
        logits = ttnn.linear(
            x_flat, self.gate_w, compute_kernel_config=self.ck_hifi4, dtype=ttnn.float32
        )  # mixed-dtype matmul: bf16 activations x fp32 weight, fp32 acc/out
        scores = ttnn.sigmoid_accurate(logits)
        ttnn.deallocate(logits)
        biased = ttnn.add(scores, self.gate_bias)
        row_mean = ttnn.mean(biased, dim=-1, keepdim=True)
        centered = ttnn.subtract(biased, row_mean)
        ttnn.deallocate(row_mean)
        ttnn.deallocate(biased)
        centered_bf16 = ttnn.typecast(centered, ttnn.bfloat16)
        ttnn.deallocate(centered)
        return scores, centered_bf16

    def _routing_weights(self, x_flat):
        """Dense routing weights [1,1,T,E] (union/prefill path); same math as
        the functional decoder, minus the activation typecast."""
        scores, centered_bf16 = self._router_scores(x_flat)
        _, idx = ttnn.topk(centered_bf16, k=self.top_k, dim=-1, sorted=True)
        T = idx.shape[2]
        src = self.scatter_ones
        if src.shape[2] != T:
            src = ttnn.slice(self.scatter_ones, [0, 0, 0, 0], [1, 1, T, self.top_k])
        mask_bf16 = ttnn.scatter(ttnn.zeros_like(centered_bf16), dim=-1, index=idx, src=src)
        ttnn.deallocate(centered_bf16)
        ttnn.deallocate(idx)
        mask = ttnn.typecast(mask_bf16, ttnn.float32)
        ttnn.deallocate(mask_bf16)
        picked = ttnn.multiply(scores, mask)
        ttnn.deallocate(scores)
        ttnn.deallocate(mask)
        denom = ttnn.sum(picked, dim=-1, keepdim=True)
        denom = ttnn.add(denom, 1e-20)
        inv = ttnn.reciprocal(denom)
        ttnn.deallocate(denom)
        weights = ttnn.multiply(picked, inv)  # routed_scaling lives in the down weights
        ttnn.deallocate(picked)
        ttnn.deallocate(inv)
        weights_bf16 = ttnn.typecast(weights, ttnn.bfloat16)
        ttnn.deallocate(weights)
        return weights_bf16

    # ------------------------------------------------------------------ moe decode

    def _moe_decode(self, x):
        if self.max_batch == 1:
            return self._moe_decode_indexed(x)
        return self._moe_decode_union(x)

    def _moe_decode_indexed(self, x):
        """Batch-1 decode: compute only the token's top-4 experts.

        The topk expert ids are handed to ttnn.sparse_matmul as a device
        uint16 index list (indexed/gather mode): the kernels visit only those
        groups and the outputs are compact [1, k, B, N]. Routing weights are
        gathered at the same ids, so slot order matches by construction.
        """
        B = x.shape[2]  # == 1
        scores, centered_bf16 = self._router_scores(x)
        _, idx = ttnn.topk(centered_bf16, k=self.top_k, dim=-1, sorted=True)  # [1,1,1,k] uint16 TILE
        ttnn.deallocate(centered_bf16)

        # Compact routing weights: gather -> normalize -> scale. [1,1,1,k]
        picked = ttnn.gather(scores, dim=3, index=idx)
        ttnn.deallocate(scores)
        denom = ttnn.sum(picked, dim=-1, keepdim=True)
        denom = ttnn.add(denom, 1e-20)
        weights = ttnn.div(picked, denom)  # routed_scaling lives in the down weights
        ttnn.deallocate(picked)
        ttnn.deallocate(denom)
        w_bf16 = ttnn.typecast(weights, ttnn.bfloat16)
        ttnn.deallocate(weights)
        rw = ttnn.permute(w_bf16, (0, 3, 2, 1))  # [1, k, B, 1]
        ttnn.deallocate(w_bf16)

        # Index list for the kernels: single-stick ROW_MAJOR uint16 [1,1,1,k].
        idx_rm = ttnn.to_layout(idx, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(idx)

        inter = self.moe_inter
        gu = ttnn.sparse_matmul(
            x,
            self.experts_gate_up,
            sparsity=self.ones_e,  # required operand; not read in indexed mode
            indices=idx_rm,
            is_input_b_sparse=True,
            program_config=self._sparse_pc(B, 2 * inter, self.hidden),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ck_hifi2,
            dtype=ttnn.bfloat16,
        )  # [1, 1, 1, k, B, 2*inter] compact
        gu = ttnn.reshape(gu, (1, self.top_k, B, 2 * inter))
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, self.top_k, B, inter])
        up = ttnn.slice(gu, [0, 0, 0, inter], [1, self.top_k, B, 2 * inter])
        ttnn.deallocate(gu)
        h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        h = ttnn.multiply(h, rw)  # scale rows before down: commutes with the matmul
        ttnn.deallocate(rw)
        down = ttnn.sparse_matmul(
            h,
            self.experts_down,
            sparsity=self.ones_e,
            indices=idx_rm,
            is_input_a_sparse=True,
            is_input_b_sparse=True,
            program_config=self._sparse_pc(B, self.hidden, inter),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ck_hifi2,
            dtype=ttnn.bfloat16,
        )  # [1, k, B, hidden] compact
        ttnn.deallocate(h)
        ttnn.deallocate(idx_rm)
        routed = ttnn.sum(down, dim=1, keepdim=True)  # [1, 1, B, hidden]
        ttnn.deallocate(down)

        shared = self._swiglu_linear(x, self.shared_gate, self.shared_up, self.shared_down, self.ck_hifi4)
        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    def _moe_decode_union(self, x):
        """Batch>1 decode: union sparsity over the batch (functional path with
        the silu fold and rw-before-down rewrites)."""
        B = x.shape[2]
        routing = self._routing_weights(x)  # [1,1,B,E] bf16
        union = ttnn.max(routing, dim=2, keepdim=True)  # [1,1,1,E]
        sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(union)

        inter = self.moe_inter
        gu = ttnn.sparse_matmul(
            x,
            self.experts_gate_up,
            sparsity=sparsity,
            nnz=None,
            program_config=self._sparse_pc(B, 2 * inter, self.hidden),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ck_hifi2,
            dtype=ttnn.bfloat16,
        )
        gu = ttnn.reshape(gu, (1, self.n_experts, B, 2 * inter))
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, self.n_experts, B, inter])
        up = ttnn.slice(gu, [0, 0, 0, inter], [1, self.n_experts, B, 2 * inter])
        ttnn.deallocate(gu)
        h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        rw = ttnn.permute(routing, (0, 3, 2, 1))  # [1, E, B, 1]
        ttnn.deallocate(routing)
        h = ttnn.multiply(h, rw)
        ttnn.deallocate(rw)
        down = ttnn.sparse_matmul(
            h,
            self.experts_down,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=True,
            program_config=self._sparse_pc(B, self.hidden, inter),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ck_hifi2,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h)
        ttnn.deallocate(sparsity)
        routed = ttnn.sum(down, dim=1, keepdim=True)
        ttnn.deallocate(down)

        shared = self._swiglu_linear(x, self.shared_gate, self.shared_up, self.shared_down, self.ck_hifi4)
        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    # ------------------------------------------------------------------ moe prefill

    def _moe_prefill(self, x):
        """x: [1, 1, S, hidden] (S multiple of 32). All-expert compute, but
        with the real per-32-token-block union routing mask so never-selected
        (block, expert) pairs are skipped by the sparse matmuls (exact: their
        routing weight is zero)."""
        S = x.shape[2]
        G = S // TILE
        routing = self._routing_weights(x)  # [1,1,S,E]

        # Per-block union mask for gate/up, per-chunk expert union for down.
        routing_blocks = ttnn.reshape(routing, (1, G, TILE, self.n_experts))
        block_mask = ttnn.max(routing_blocks, dim=2, keepdim=True)  # [1,G,1,E]
        sparsity = ttnn.to_layout(block_mask, ttnn.ROW_MAJOR_LAYOUT)
        expert_union = ttnn.max(block_mask, dim=1, keepdim=True)  # [1,1,1,E]
        sparsity_e = ttnn.to_layout(expert_union, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(expert_union)
        ttnn.deallocate(block_mask)

        x_g = ttnn.reshape(x, (1, G, TILE, self.hidden))
        inter = self.moe_inter
        gu = ttnn.sparse_matmul(
            x_g,
            self.experts_gate_up,
            sparsity=sparsity,
            nnz=None,
            program_config=self._sparse_pc(TILE, 2 * inter, self.hidden),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.ck_hifi2,
            dtype=ttnn.bfloat16,
        )  # [1, G, 1, E, 32, 2*inter]
        gu = ttnn.transpose(gu, 1, 3)  # [1, E, 1, G, 32, 2*inter]
        gu = ttnn.reshape(gu, (1, self.n_experts, S, 2 * inter))
        gate = ttnn.slice(gu, [0, 0, 0, 0], [1, self.n_experts, S, inter])
        up = ttnn.slice(gu, [0, 0, 0, inter], [1, self.n_experts, S, 2 * inter])
        ttnn.deallocate(gu)
        h = ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        rw = ttnn.permute(routing, (0, 3, 2, 1))  # [1, E, S, 1]
        ttnn.deallocate(routing)
        h = ttnn.multiply(h, rw)  # scale rows before down: commutes with the matmul
        ttnn.deallocate(rw)

        split = 1024
        routed = None
        for s0 in range(0, S, split):
            s1 = min(s0 + split, S)
            h_s = ttnn.slice(h, [0, 0, s0, 0], [1, self.n_experts, s1, inter]) if S > split else h
            down = ttnn.sparse_matmul(
                h_s,
                self.experts_down,
                sparsity=sparsity_e,
                nnz=None,
                is_input_a_sparse=True,
                program_config=self._sparse_pc(s1 - s0, self.hidden, inter),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ck_hifi2,
                dtype=ttnn.bfloat16,
            )  # [1, E, s, hidden]
            if h_s is not h:
                ttnn.deallocate(h_s)
            part = ttnn.sum(down, dim=1, keepdim=True)  # [1, 1, s, hidden]
            ttnn.deallocate(down)
            if routed is None and s1 == S:
                routed = part
            else:
                if routed is None:
                    routed = ttnn.allocate_tensor_on_device(
                        ttnn.Shape((1, 1, S, self.hidden)),
                        ttnn.bfloat16,
                        ttnn.TILE_LAYOUT,
                        self.device,
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
                ttnn.experimental.slice_write(part, routed, [0, 0, s0, 0], [1, 1, s1, self.hidden], [1, 1, 1, 1])
                ttnn.deallocate(part)
        ttnn.deallocate(h)
        # sparsity / sparsity_e are to_layout results (fresh buffers); safe to free.
        ttnn.deallocate(sparsity)
        ttnn.deallocate(sparsity_e)

        shared = self._swiglu_linear(x, self.shared_gate, self.shared_up, self.shared_down, self.ck_hifi4)
        out = ttnn.add(routed, shared)
        ttnn.deallocate(routed)
        ttnn.deallocate(shared)
        return out

    # ------------------------------------------------------------------ attention: prefill

    def _attn_prefill_chunk(self, x, kv_cache, page_table, user_id, chunk_start):
        S_c = x.shape[2]
        cos, sin, trans = self.rope.prefill_mats(chunk_start, chunk_start + S_c)

        # --- fused QKV-A projection (shared LHS) ---
        qkv_a = ttnn.linear(x, self.wqkv_a, compute_kernel_config=self.ck_hifi4)  # [1,1,S,q_lora+576]
        q = ttnn.slice(qkv_a, [0, 0, 0, 0], [1, 1, S_c, self.q_lora_rank])
        kv_nope = ttnn.slice(qkv_a, [0, 0, 0, self.q_lora_rank], [1, 1, S_c, self.q_lora_rank + self.kv_lora_rank])
        kv_rope = ttnn.slice(
            qkv_a, [0, 0, 0, self.q_lora_rank + self.kv_lora_rank], [1, 1, S_c, self.q_lora_rank + self.kvpe_dim]
        )
        ttnn.deallocate(qkv_a)

        # --- KV path ---
        kv_nope = self._rms(kv_nope, self.kv_norm_w, self.lora_norm_eps)
        kv_rope = ttnn.experimental.rotary_embedding_llama(kv_rope, cos, sin, trans, is_decode_mode=False)
        kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1,1,S,576]
        ttnn.deallocate(kv_nope)
        ttnn.deallocate(kv_rope)

        block = self.paged_config.block_size
        start_block = chunk_start // block
        end_block = (chunk_start + S_c) // block
        # NB: page-table slices may alias the page table; never deallocate them.
        chunk_pt = ttnn.slice(page_table, [0, start_block], [page_table.shape[0], end_block])
        ttnn.experimental.paged_fill_cache(kv_cache, kvpe, page_table=chunk_pt, batch_idx=user_id)

        # --- Q path (batched per-head wq_b: output is already [1, nh, S, .]) ---
        q = self._rms(q, self.q_norm_w, self.lora_norm_eps)
        qh = ttnn.matmul(q, self.wq_b_heads, compute_kernel_config=self.ck_hifi4)  # [1, nh, S, qk_head]
        ttnn.deallocate(q)
        q_nope = ttnn.slice(qh, [0, 0, 0, 0], [1, self.num_heads, S_c, self.qk_nope])
        q_rope = ttnn.slice(qh, [0, 0, 0, self.qk_nope], [1, self.num_heads, S_c, self.qk_head_dim])
        ttnn.deallocate(qh)
        q_lat = ttnn.matmul(q_nope, self.w_uk, compute_kernel_config=self.ck_hifi4)  # [1, nh, S, kv_lora]
        ttnn.deallocate(q_nope)
        q_rope = ttnn.experimental.rotary_embedding_llama(q_rope, cos, sin, trans, is_decode_mode=False)
        q_abs = ttnn.concat([q_lat, q_rope], dim=-1)  # [1, nh, S, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        # cos/sin may alias the persistent rope tables; never deallocate.

        user_pt = ttnn.slice(page_table, [user_id, 0], [user_id + 1, page_table.shape[1]])
        attn_lat = ttnn.transformer.chunked_flash_mla_prefill(
            q_abs,
            kv_cache,
            self.kv_lora_rank,
            user_pt,
            chunk_start_idx=chunk_start,
            scale=self.scale,
            compute_kernel_config=self.ck_flash_prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, nh, S, kv_lora]
        ttnn.deallocate(q_abs)
        ttnn.deallocate(kvpe)

        v = ttnn.matmul(attn_lat, self.w_uv_t, compute_kernel_config=self.ck_hifi4)  # [1, nh, S, v_head]
        ttnn.deallocate(attn_lat)
        v = ttnn.transformer.concatenate_heads(v)  # [1, S, nh*v_head]
        v = ttnn.reshape(v, (1, 1, S_c, self.num_heads * self.v_head_dim))
        out = ttnn.linear(v, self.wo, compute_kernel_config=self.ck_hifi4)  # [1,1,S,hidden]
        ttnn.deallocate(v)
        return out

    # ------------------------------------------------------------------ attention: decode

    def _decode_rope_mats(self, rot_idxs, batch):
        """RopeSetup.decode_mats minus the [1,1,1,d] -> [1,1,1,d] identity
        transposes that batch 1 makes trivial."""
        if batch > 1:
            return self.rope.decode_mats(rot_idxs)
        rope = self.rope
        cos = ttnn.embedding(rot_idxs, rope.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(rot_idxs, rope.sin_matrix, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.unsqueeze_to_4D(cos)  # [1, 1, 1, dim] == its own (1, 2) transpose
        sin = ttnn.unsqueeze_to_4D(sin)
        cos = ttnn.to_memory_config(cos, rope.decode_cs_mem)
        sin = ttnn.to_memory_config(sin, rope.decode_cs_mem)
        trans = ttnn.to_memory_config(rope.trans_mat_decode_dram, rope.trans_mem_decode)
        return cos, sin, trans

    def _attn_decode(self, x, kv_cache, page_table, cur_pos_tensor, rot_idxs):
        B = x.shape[2]
        cos, sin, trans = self._decode_rope_mats(rot_idxs, B)

        # --- fused QKV-A projection (shared LHS) ---
        qkv_a = ttnn.linear(x, self.wqkv_a, compute_kernel_config=self.ck_hifi4)  # [1,1,B,q_lora+576]
        q = ttnn.slice(qkv_a, [0, 0, 0, 0], [1, 1, B, self.q_lora_rank])
        kv_nope = ttnn.slice(qkv_a, [0, 0, 0, self.q_lora_rank], [1, 1, B, self.q_lora_rank + self.kv_lora_rank])
        kv_rope = ttnn.slice(
            qkv_a, [0, 0, 0, self.q_lora_rank + self.kv_lora_rank], [1, 1, B, self.q_lora_rank + self.kvpe_dim]
        )
        ttnn.deallocate(qkv_a)

        # --- KV path: assemble kvpe directly in the [1, B, 1, 576] update layout ---
        kv_nope = self._rms(kv_nope, self.kv_norm_w, self.lora_norm_eps)
        if B > 1:  # at B == 1 the (1, 2) transpose is an identity
            kv_nope = ttnn.transpose(kv_nope, 1, 2)  # [1, B, 1(32), kv_lora]
            kv_rope = ttnn.transpose(kv_rope, 1, 2, memory_config=self.rope_in_decode_mem)  # [1, B, 1(32), 64]
        else:
            kv_rope = ttnn.to_memory_config(kv_rope, self.rope_in_decode_mem)
        kv_rope = ttnn.experimental.rotary_embedding_llama(kv_rope, cos, sin, trans, is_decode_mode=True)
        kv_rope = ttnn.to_memory_config(kv_rope, ttnn.DRAM_MEMORY_CONFIG)
        kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)  # [1, B, 1(32), 576]
        ttnn.deallocate(kv_nope)
        ttnn.deallocate(kv_rope)

        # --- Q path (flat wq_b; the broadcast-batched matmul is 3x slower at
        # one M tile, see module docstring) ---
        q = self._rms(q, self.q_norm_w, self.lora_norm_eps)
        q = ttnn.linear(q, self.wq_b, compute_kernel_config=self.ck_hifi4)  # [1,1,B,nh*qk_head]
        q = ttnn.untilize(q)
        q = ttnn.reshape(q, (1, B, self.num_heads, self.qk_head_dim))
        q = ttnn.tilize_with_zero_padding(q, use_multicore=True)  # [1, B, nh(32), qk_head]
        q_nope = ttnn.slice(q, [0, 0, 0, 0], [1, B, self.num_heads, self.qk_nope])
        q_rope = ttnn.slice(q, [0, 0, 0, self.qk_nope], [1, B, self.num_heads, self.qk_head_dim])
        ttnn.deallocate(q)
        q_nope = ttnn.transpose(q_nope, 1, 2)  # [1, nh, B, qk_nope]
        q_lat = ttnn.matmul(q_nope, self.w_uk, compute_kernel_config=self.ck_hifi4)  # [1, nh, B, kv_lora]
        ttnn.deallocate(q_nope)
        q_lat = ttnn.transpose(q_lat, 1, 2)  # [1, B, nh, kv_lora]
        q_rope = ttnn.to_memory_config(q_rope, self.rope_in_decode_mem)
        q_rope = ttnn.experimental.rotary_embedding_llama(q_rope, cos, sin, trans, is_decode_mode=True)
        q_rope = ttnn.to_memory_config(q_rope, ttnn.DRAM_MEMORY_CONFIG)
        q_abs = ttnn.concat([q_lat, q_rope], dim=-1)  # [1, B, nh, 576]
        ttnn.deallocate(q_lat)
        ttnn.deallocate(q_rope)
        # Free L1-resident rope tensors before paged_update_cache (its static
        # CBs scale with B*Wt; see the functional decoder note).
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(trans)

        # --- cache update (kvpe already [1, B, 1, 576]; shard per <=16-user group) ---
        for g0, g1, mem in self.kvpe_update_groups:
            single = g0 == 0 and g1 == B and len(self.kvpe_update_groups) == 1
            # NB: user-range slices can alias their input (view fast path);
            # never explicitly deallocate them (see the functional decoder).
            kvpe_g = kvpe if single else ttnn.slice(kvpe, [0, g0, 0, 0], [1, g1, kvpe.shape[2], self.kvpe_dim])
            pos_g = cur_pos_tensor if single else ttnn.slice(cur_pos_tensor, [g0], [g1])
            pt_g = page_table if single else ttnn.slice(page_table, [g0, 0], [g1, page_table.shape[1]])
            kvpe_sh = ttnn.to_memory_config(kvpe_g, mem)
            ttnn.experimental.paged_update_cache(kv_cache, kvpe_sh, update_idxs_tensor=pos_g, page_table=pt_g)
            ttnn.deallocate(kvpe_sh)
        ttnn.deallocate(kvpe)

        attn_lat = ttnn.transformer.paged_flash_multi_latent_attention_decode(
            q_abs,
            kv_cache,
            head_dim_v=self.kv_lora_rank,
            page_table_tensor=page_table,
            cur_pos_tensor=cur_pos_tensor,
            scale=self.scale,
            program_config=self.flash_decode_pc,
            compute_kernel_config=self.ck_flash_decode,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, B, nh(padded), kv_lora]
        ttnn.deallocate(q_abs)
        attn_lat = ttnn.slice(attn_lat, [0, 0, 0, 0], [1, B, self.num_heads, self.kv_lora_rank])
        attn_lat = ttnn.transpose(attn_lat, 1, 2)  # [1, nh, B, kv_lora]
        v = ttnn.matmul(attn_lat, self.w_uv_t, compute_kernel_config=self.ck_hifi4)  # [1, nh, B, v_head]
        ttnn.deallocate(attn_lat)
        v = ttnn.transpose(v, 1, 2)  # [1, B, nh, v_head]
        v = ttnn.to_memory_config(v, self.concat_heads_mem(B))
        v = ttnn.experimental.nlp_concat_heads_decode(v, num_heads=self.num_heads)  # [1, 1, 32, nh*v_head]
        v = ttnn.to_memory_config(v, ttnn.DRAM_MEMORY_CONFIG)
        if v.shape[2] != B:  # the concat-heads op pads the user dim to a full tile
            v = ttnn.slice(v, [0, 0, 0, 0], [1, 1, B, self.num_heads * self.v_head_dim])
        out = ttnn.linear(v, self.wo, compute_kernel_config=self.ck_hifi4)  # [1,1,B,hidden]
        ttnn.deallocate(v)
        return out

    def concat_heads_mem(self, batch):
        """One-user-per-core height shard on a SINGLE rectangular core range:
        nlp_concat_heads_decode treats a multi-range input grid as subcoregrid
        mode (which then demands an explicit sub_core_grids), and the row-wise
        corerangeset splits into several ranges on the 13-wide Blackhole grid
        for batches that do not fit one row."""
        if batch != self.max_batch:
            raise ValueError(f"decode batch {batch} != max_batch_size {self.max_batch}")
        mem = getattr(self, "_concat_heads_mem", None)
        if mem is None:
            grid = self.device.compute_with_storage_grid_size()
            w = next(w for w in range(min(batch, grid.x), 0, -1) if batch % w == 0 and batch // w <= grid.y)
            core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, batch // w - 1))})
            mem = ttnn.create_sharded_memory_config(
                shape=(TILE, self.v_head_dim),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self._concat_heads_mem = mem
        return mem

    # ------------------------------------------------------------------ public forwards

    def prefill_forward(self, x, *, kv_cache, page_table, user_id=0, seq_len=None, progress_cb=None):
        S = seq_len if seq_len is not None else x.shape[2]
        block = self.paged_config.block_size
        S_pad = -(-S // block) * block
        if x.shape[2] < S_pad:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, S_pad - x.shape[2]), (0, 0)], 0.0)
        elif x.shape[2] > S_pad:
            raise ValueError(f"input has {x.shape[2]} rows but logical seq_len is {S}")

        chunk = self.prefill_chunk_size
        n_chunks = -(-S_pad // chunk)
        single_chunk = n_chunks == 1
        out_acc = None
        if not single_chunk:
            out_acc = ttnn.allocate_tensor_on_device(
                ttnn.Shape((1, 1, S_pad, self.hidden)),
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                self.device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
        for chunk_idx, start in enumerate(range(0, S_pad, chunk)):
            if progress_cb is not None:
                progress_cb(chunk_idx, n_chunks)
            end = min(start + chunk, S_pad)
            x_c = ttnn.slice(x, [0, 0, start, 0], [1, 1, end, self.hidden]) if not single_chunk else x

            h = self._rms(x_c, self.input_norm_w, self.rms_eps)
            attn = self._attn_prefill_chunk(h, kv_cache, page_table, user_id, start)
            ttnn.deallocate(h)
            res = ttnn.add(x_c, attn)
            ttnn.deallocate(attn)
            if x_c is not x:
                ttnn.deallocate(x_c)
            h2 = self._rms(res, self.post_norm_w, self.rms_eps)
            mlp = self._mlp(h2, "prefill")
            ttnn.deallocate(h2)
            out_c = ttnn.add(res, mlp)
            ttnn.deallocate(res)
            ttnn.deallocate(mlp)

            if single_chunk:
                out_acc = out_c
            else:
                ttnn.experimental.slice_write(out_c, out_acc, [0, 0, start, 0], [1, 1, end, self.hidden], [1, 1, 1, 1])
                ttnn.deallocate(out_c)

        if out_acc.shape[2] != S:
            out = ttnn.slice(out_acc, [0, 0, 0, 0], [1, 1, S, self.hidden])
            ttnn.deallocate(out_acc)
            return out
        return out_acc

    forward = FunctionalDecoder.decode_forward
