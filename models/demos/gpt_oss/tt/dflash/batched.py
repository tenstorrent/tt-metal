# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy DFlash for 32 users at once (batch 32) on the batched gpt-oss-20b decode model (batched_decode.py, dense MoE).

A round, for every user u: the drafter proposes 7 tokens after the user's last token; the target checks the 8-row
block [last token, 7 drafts] of every user in one 256-row verify pass and keeps the drafts up to the first mismatch plus
its own next token. Every pass uses the same row order: row 32 i + u is row i of user u's block, at position
start_u + i, so a 32-row step i holds one row of each user (the batched decode's layout: one K/V cache write per step).

Passes (device tensors throughout; the host writes the per-round inputs and reads two 256-row argmax results):
  draft: the noise block [last token, mask x 7] through the drafter's 8 layers (256-row matmuls; Q / K norms and RoPE
      for all 256 rows at once with per-row cos / sin; attention in 8 steps of 32 rows: K/V of every step written first,
      then each step attends to its user's whole block, not causal), its LM head, a per-device argmax;
  verify: the block [last token, 7 drafts] through the target: norm, QKV, o_proj and the all-reduce over all 256 rows;
      the attention core in 8 steps of 32 rows (step i is the batched decode attention at positions start_u + i, so it
      sees the K/V rows of steps 0 .. i: causal within the block; Q and K rotated by one op, K and V written by one op),
      the MoE on all 256 rows at once (router, every expert's gate / up / down as 256-row matmuls in the dense layout of
      the batched decode, each expert weight read once per pass, the routing weights zeroing unrouted experts), the
      target's LM head, a per-device argmax; the outputs of target layers 1, 6, 11, 16, 21 kept for the drafter;
  context: those features of all 256 rows into the drafter's K/V cache at positions start_u + i (rows past the
      accepted ones are rewritten by the next round's block before they are read).
"""

import time

import torch

import ttnn
from models.tt_transformers.tt.load_checkpoints import reverse_permute, reverse_permute_1d

from .draft import draft_snapshot, load_draft_config

TILE = ttnn.TILE_SIZE
U32 = 32  # rows per attention step: one row per user


def _replicate(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh if device else None, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _copy(mesh, dst, t, dtype):
    """Write host tensor t into the persistent device tensor dst (a trace input)."""
    ttnn.copy_host_to_device_tensor(_replicate(mesh, t, dtype, device=False), dst)


def _all_reduce(x):
    """TP all-reduce over the mesh row (2 fabric links: [256, 2880] in 0.054 ms, 1 link 0.062; same sums)."""
    return ttnn.all_reduce(
        x, num_links=2, topology=ttnn.Topology.Ring, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


class _Argmax:
    """Row argmax over a vocabulary split over the mesh columns: each device's best (value, index) on device; the host
    combines the devices (read after the pass)."""

    def __init__(self, per_dev, vocab):
        self.per_dev, self.vocab = per_dev, vocab

    def device(self, logits):
        return ttnn.topk(logits, k=1, dim=-1, sorted=True)

    def host(self, out):
        vals, idxs = out
        v = torch.stack([ttnn.to_torch(t).float().reshape(-1) for t in ttnn.get_device_tensors(vals)])
        i = torch.stack([ttnn.to_torch(t).long().reshape(-1) for t in ttnn.get_device_tensors(idxs)])
        gi = i + torch.arange(v.shape[0])[:, None] * self.per_dev
        v = torch.where(gi < self.vocab, v, torch.full_like(v, -1e30))
        best = v.argmax(0)  # the first device with the maximum
        return gi[best, torch.arange(v.shape[1])].tolist()


class BatchedDraft:
    """The z-lab drafter (8 layers, TP like a dense target layer) for `users` sequences, each with its own paged BF16
    K/V cache; draft and context passes over all users' rows at once."""

    def __init__(self, mesh_device, target, max_positions, users, window=None, block_size=64):
        from safetensors.torch import load_file

        self.mesh_device = mesh_device
        self.target = target
        self.path = draft_snapshot()
        c = load_draft_config(self.path)
        D = mesh_device.shape[1]
        self.hidden, self.hd = c["hidden_size"], c["head_dim"]
        self.nq, self.nkv = c["num_attention_heads"] // D, c["num_key_value_heads"] // D
        self.inter = c["intermediate_size"] // D
        self.layers = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        self.block = c["block_size"]
        self.mask_token_id = c["dflash_config"]["mask_token_id"]
        self.target_layer_ids = c["dflash_config"]["target_layer_ids"]
        self.ctx_in = len(self.target_layer_ids) * self.hidden
        self.window = window
        self.users = users
        self.rows = users * self.block
        w = load_file(str(self.path / "model.safetensors"))
        mesh = mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def put(t, mapper=rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def norm_w(t):
            return put(t.float().reshape(1, 1, -1, TILE), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        cols = ttnn.ShardTensorToMesh(mesh, dim=-1)
        rows = ttnn.ShardTensorToMesh(mesh, dim=-2)
        hd, nq, nkv = self.hd, self.nq, self.nkv

        def per_device(parts):
            """[q | k | v]-style row blocks: each device's slice of every part, side by side."""
            return torch.cat(
                [torch.cat([p[d * (p.shape[0] // D) : (d + 1) * (p.shape[0] // D)] for p in parts]) for d in range(D)]
            )

        self.fc_full = put(w["fc.weight"].T)  # replicated: the features are replicated on every device
        self.hidden_norm = norm_w(w["hidden_norm.weight"])
        self.norm = norm_w(w["norm.weight"])
        self.L = []
        for i in range(self.layers):
            p, a, m = f"layers.{i}.", f"layers.{i}.self_attn.", f"layers.{i}.mlp."
            q = reverse_permute(w[a + "q_proj.weight"], D * nq, D * nq * hd, self.hidden)
            k = reverse_permute(w[a + "k_proj.weight"], D * nkv, D * nkv * hd, self.hidden)
            v = w[a + "v_proj.weight"]
            qb = reverse_permute(w[a + "q_proj.bias"][:, None], D * nq, D * nq * hd, 1)[:, 0]
            kb = reverse_permute(w[a + "k_proj.bias"][:, None], D * nkv, D * nkv * hd, 1)[:, 0]
            vb = w[a + "v_proj.bias"]
            g, u = w[m + "gate_proj.weight"], w[m + "up_proj.weight"]
            self.L.append(
                dict(
                    ln1=norm_w(w[p + "input_layernorm.weight"]),
                    ln2=norm_w(w[p + "post_attention_layernorm.weight"]),
                    qkv=put(per_device([q, k, v]).T, cols),
                    qkv_b=put(per_device([qb, kb, vb]).reshape(1, -1), cols),
                    q_norm=norm_w(reverse_permute_1d(w[a + "q_norm.weight"])),
                    k_norm=norm_w(reverse_permute_1d(w[a + "k_norm.weight"])),
                    o=put(w[a + "o_proj.weight"].T, rows),
                    o_b=put(w[a + "o_proj.bias"].reshape(1, -1)),
                    gu=put(per_device([g, u]).T, cols),
                    down=put(w[m + "down_proj.weight"].T, rows),
                )
            )
        self.block_size = block_size
        self.num_blocks = -(-(max_positions + self.block) // block_size)
        user_tables = torch.arange(users * self.num_blocks, dtype=torch.int32).reshape(users, self.num_blocks)
        self.user_tables = user_tables
        self.page_table = _replicate(mesh, user_tables, ttnn.int32)  # row u: user u (each 32-row step)
        heads_split = ttnn.ShardTensorToMesh(mesh, dim=1)
        self.caches = []
        for _ in range(self.layers):
            z = torch.zeros(users * self.num_blocks, D * nkv, block_size, hd)
            self.caches.append(
                tuple(
                    ttnn.from_torch(
                        z,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=heads_split,
                    )
                    for _ in range(2)
                )
            )
        from ..model import create_rope_setup

        self.rope = create_rope_setup(mesh, target.hf_config, max_local_batch_size=U32, datatype=ttnn.bfloat16)
        # all rows' Q / K norms and RoPE at once (prefill layout, per-row cos / sin), then per step only the head split
        # (Q heads on cores 0-31, K on 32-63) and one K / V cache write
        self.head64_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, hd),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        n_heads = nq + 2 * nkv
        self.qkv_grid = ttnn.CoreCoord(10, n_heads // 10) if n_heads % 10 == 0 else ttnn.CoreCoord(n_heads, 1)
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
        )
        self.ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.mm_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.argmax = _Argmax(target.lm_head_weight.shape[-1], target.vocab_size)
        # the drafter's LM head on a BFP4 copy of the target's with BF8 rows: 0.89 -> 0.52 ms per pass, live-like
        # acceptance on a fixed text 3.07 (BF16 rows x BF8 weight: 3.05 / 3.10 in two runs)
        self.lm_head = ttnn.typecast(target.lm_head_weight, ttnn.bfloat4_b)

    # -- pieces ----------------------------------------------------------------------------------------------------
    def _linear(self, x, w, bias=None):
        return ttnn.linear(
            x,
            w,
            bias=bias,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.mm_ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def rows_cos_sin(self, pos_rows):
        """Per-row cos / sin [1, 1, rows, hd] for positions pos_rows (device [1, rows] UINT32)."""
        out = []
        for m in (self.rope.cos_matrix, self.rope.sin_matrix):
            t = ttnn.embedding(pos_rows, m, layout=ttnn.TILE_LAYOUT)
            out.append(ttnn.reshape(t, [1, 1, self.rows, self.hd]))
        return out

    def _rows_qk(self, qkv, L, cos, sin, with_q):
        """[1, 1, rows, qkv] projection -> the same with K (and Q) head-normed and rotated, every row at its own
        position: a head split, the norms and RoPE in the prefill layout, the heads joined again. with_q: also
        returns the rotated Q heads [1, heads, rows, hd]."""
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.nq,
            num_kv_heads=self.nkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        trans = self.rope.get_both_trans_mats()["prefill"]
        parts = []
        q_rows = None
        for j, (t, w, rotate) in enumerate(((q, L["q_norm"], with_q), (k, L["k_norm"], True), (v, None, False))):
            if rotate:
                n = ttnn.rms_norm(t, epsilon=self.eps, weight=w, compute_kernel_config=self.ck)
                t.deallocate(True)
                t = ttnn.experimental.rotary_embedding_llama(n, cos, sin, trans, is_decode_mode=False)
                n.deallocate(True)
            parts.append(ttnn.experimental.nlp_concat_heads(t, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            if j == 0 and with_q:
                q_rows = t
            else:
                t.deallocate(True)
        out = ttnn.concat(parts, dim=-1)
        for t in parts:
            t.deallocate(True)
        return (out, q_rows) if with_q else out

    def _step_write(self, li, qkv, i, pos):
        """Step i of a prepared [1, 1, rows, qkv]: its heads (Q on cores 0-31) and the K / V cache write. Returns Q."""
        part = ttnn.interleaved_to_sharded_partial(
            qkv, self.qkv_grid, [U32, self.hd], self.block, i, ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )  # fmt: skip
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            part, num_heads=self.nq, num_kv_heads=self.nkv, overlap_qk_coregrid=False, memory_config=self.head64_mc
        )
        part.deallocate(True)
        kc, vc = self.caches[li]
        ttnn.experimental.paged_fused_update_cache(kc, k, vc, v, update_idxs_tensor=pos, page_table=self.page_table)
        k.deallocate(True)
        v.deallocate(True)
        return q

    # -- prompt context ---------------------------------------------------------------------------------------------
    def add_prompt_context(self, features, user, chunk=2048):
        """The prompt's context K / V (prefill-style, paged_fill_cache of block-aligned chunks)."""
        n = features.shape[0]
        mesh = self.mesh_device
        pt_host = self.user_tables[user][None]
        qw, kvw = self.nq * self.hd, self.nkv * self.hd
        for s0 in range(0, n, chunk):
            rows = min(chunk, -(-(n - s0) // self.block_size) * self.block_size)
            x = torch.zeros(1, 1, rows, self.ctx_in)
            x[0, 0, : min(rows, n - s0)] = features[s0 : s0 + rows].float()
            tx = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ctx = self._linear(tx, self.fc_full)
            tx.deallocate(True)
            ctx = ttnn.rms_norm(ctx, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
            cos = self.rope.cos_matrix_prefill[:, :, s0 : s0 + rows, :]
            sin = self.rope.sin_matrix_prefill[:, :, s0 : s0 + rows, :]
            trans = self.rope.get_both_trans_mats()["prefill"]
            b0 = s0 // self.block_size
            chunk_pt = _replicate(mesh, pt_host[:, b0 : b0 + rows // self.block_size].contiguous(), ttnn.int32)
            for i in range(self.layers):
                L = self.L[i]
                qkv = self._linear(ctx, L["qkv"], L["qkv_b"])
                for j, cache in enumerate(self.caches[i]):
                    t = ttnn.slice(qkv, (0, 0, 0, qw + j * kvw), (1, 1, rows, qw + (j + 1) * kvw))
                    t = ttnn.transpose(ttnn.reshape(t, [1, rows, self.nkv, self.hd]), 1, 2)
                    if j == 0:
                        t = ttnn.rms_norm(t, epsilon=self.eps, weight=L["k_norm"], compute_kernel_config=self.ck)
                        t = ttnn.experimental.rotary_embedding_llama(t, cos, sin, trans, is_decode_mode=False)
                    ttnn.experimental.paged_fill_cache(cache, t, chunk_pt, batch_idx=0)
                    t.deallocate(True)
                qkv.deallocate(True)
            chunk_pt.deallocate(True)
            ctx.deallocate(True)

    # -- passes -----------------------------------------------------------------------------------------------------
    def context(self, features, pos, rot):
        """features: device [1, 1, rows, 5 * hidden] (row 32 i + u); pos: per-step positions; rot: per-row (cos, sin)."""
        ctx = self._linear(features, self.fc_full)
        ctx2 = ttnn.rms_norm(ctx, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
        ctx.deallocate(True)
        for li in range(self.layers):
            L = self.L[li]
            qkv = self._linear(ctx2, L["qkv"], L["qkv_b"])
            qkv2 = self._rows_qk(qkv, L, *rot, with_q=False)
            qkv.deallocate(True)
            for i in range(self.block):
                self._step_write(li, qkv2, i, pos[i]).deallocate(True)
            qkv2.deallocate(True)
        ctx2.deallocate(True)

    def draft(self, tokens, pos, rot, bound):
        """tokens: device [1, rows] UINT32 noise block (row 32 i + u: user u's last token for i = 0, else the mask).
        Returns the per-device argmax of every row (rows 32 i + u, i >= 1: the draft of position start_u + i)."""
        emb = ttnn.embedding(tokens, self.target.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        h = ttnn.typecast(ttnn.reshape(emb, [1, 1, self.rows, self.hidden]), ttnn.float32)
        for li in range(self.layers):
            L = self.L[li]
            x = ttnn.rms_norm(h, epsilon=self.eps, weight=L["ln1"], compute_kernel_config=self.ck)
            qkv = self._linear(x, L["qkv"], L["qkv_b"])
            x.deallocate(True)
            qkv2, q_rows = self._rows_qk(qkv, L, *rot, with_q=True)
            qkv.deallocate(True)
            for i in range(self.block):  # every step's K/V first
                self._step_write(li, qkv2, i, pos[i]).deallocate(True)
            qkv2.deallocate(True)
            kc, vc = self.caches[li]
            # block attention: a user's 8 rows see the same positions (start .. start + 7, not causal), so they go in as
            # 8 x heads query heads of one decode SDPA (order [head, row]: each K/V head's queries stay contiguous) and
            # every user's K/V is read once per layer, not once per row
            t = ttnn.reshape(q_rows, [self.nq, self.block, U32, self.hd])
            packed = ttnn.permute(t, (2, 0, 1, 3))  # [user, head, row, hd]
            q_rows.deallocate(True)
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                ttnn.reshape(packed, [1, U32, self.nq * self.block, self.hd]),
                kc,
                vc,
                cur_pos_tensor=bound,
                page_table_tensor=self.page_table,
                scale=self.hd**-0.5,
                sliding_window_size=self.window,
                program_config=self.sdpa_pc,
                compute_kernel_config=self.ck,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            packed.deallocate(True)
            perm = ttnn.permute(
                ttnn.reshape(attn, [U32, self.nq, self.block, self.hd]), (1, 2, 0, 3)
            )  # [head, row, user]
            attn.deallocate(True)
            heads = ttnn.experimental.nlp_concat_heads(
                ttnn.reshape(perm, [1, self.nq, self.rows, self.hd]), memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            perm.deallocate(True)
            partial = self._linear(heads, L["o"])
            heads.deallocate(True)
            o = _all_reduce(partial)
            partial.deallocate(True)
            o = ttnn.add(o, L["o_b"], output_tensor=o)
            h2 = ttnn.add(h, o, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
            h.deallocate(True)
            o.deallocate(True)
            h = h2
            x = ttnn.rms_norm(h, epsilon=self.eps, weight=L["ln2"], compute_kernel_config=self.ck)
            gu = self._linear(x, L["gu"])
            x.deallocate(True)
            I = self.inter
            g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, self.rows, I))
            u = ttnn.slice(gu, (0, 0, 0, I), (1, 1, self.rows, 2 * I))
            gu.deallocate(True)
            act = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            g.deallocate(True)
            u.deallocate(True)
            partial = self._linear(act, L["down"])
            act.deallocate(True)
            mlp = _all_reduce(partial)
            partial.deallocate(True)
            h2 = ttnn.add(h, mlp, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
            h.deallocate(True)
            mlp.deallocate(True)
            h = h2
        x = ttnn.rms_norm(h, epsilon=self.eps, weight=self.norm, compute_kernel_config=self.ck)
        h.deallocate(True)
        x8 = ttnn.typecast(x, ttnn.bfloat8_b)
        x.deallocate(True)
        logits = ttnn.matmul(x8, self.lm_head, dtype=ttnn.bfloat16)
        x8.deallocate(True)
        out = self.argmax.device(logits)
        logits.deallocate(True)
        return out


class BatchedVerify:
    """The target's 256-row verify pass for 32 users on the batched decode model's layers (DecoderLayer.batched_decode):
    attention in 8 batched-decode steps, the MoE on all rows at once, norms on 32-row residual shards as the decode."""

    def __init__(self, model, page_table, kv_cache, users, block, capture_layers):
        self.model = model
        self.mesh = model.mesh_device
        self.users, self.block = users, block
        self.rows = users * block
        self.page_table = page_table
        self.kv_cache = kv_cache
        self.capture_layers = list(capture_layers)
        self.hidden = model.hf_config.hidden_size
        cfg = model.hf_config
        self.top_k, self.num_experts = cfg.num_experts_per_tok, cfg.num_local_experts
        self.swiglu_limit = cfg.swiglu_limit
        self.alpha = 1.702  # ExpertConfig.alpha
        self.moe = []
        for layer in model.layers:
            assert layer.batched_decode, "BatchedVerify runs on the batched decode model (batch 32, dense MoE)"
            dw = layer.mlp.experts.dense_decode_weights
            n_act = dw.gate.shape[-1]
            self.moe.append(
                dict(
                    gate=ttnn.to_memory_config(dw.gate, ttnn.DRAM_MEMORY_CONFIG),
                    up=ttnn.to_memory_config(dw.up, ttnn.DRAM_MEMORY_CONFIG),
                    down=ttnn.to_memory_config(dw.down, ttnn.DRAM_MEMORY_CONFIG),
                    gate_bias=ttnn.slice(dw.gate_bias, [0, 0, 0, 0], [1, 1, 1, n_act]),
                    up_bias=ttnn.slice(dw.up_bias, [0, 0, 0, 0], [1, 1, 1, n_act]),
                    down_bias=dw.down_bias,
                    expand=dw.expand,
                )
            )
        self.ck = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # down [rows, E * I_local] x [E * I_local, hidden]: a long K and a narrow N, which the auto-configured matmul
        # spreads poorly (0.75 ms per layer); 2D over 10 x 8 cores, 16-tile K blocks: 0.27 ms
        n_tiles, m_tiles = self.hidden // TILE, self.rows // TILE
        self.down_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(10, m_tiles),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=1,
            out_block_w=n_tiles // 10,
            per_core_M=1,
            per_core_N=n_tiles // 10,
            transpose_mcast=False,
            fused_activation=None,
        )
        self.down_ck = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,  # FP32 sums over the 23040-long K (4.5x smaller error, same time)
            packer_l1_acc=True,
        )
        # gate / up [rows, hidden] x [hidden, E * I_local] and the routing-weight expand [rows, E] x [E, E * I_local]:
        # 1D over 72 cores (all 8 row tiles x 10 output tiles each; gate / up in 2-tile K blocks, the auto config's
        # sums), outputs width-sharded in L1 ([rows, 320] per core), SwiGLU and the routing multiply there: MoE
        # 1.16 -> 1.00 ms per layer, identical output
        n_act = self.moe[0]["gate"].shape[-1]
        moe_cores = ttnn.CoreCoord(9, 8)
        per_n = n_act // TILE // (moe_cores.x * moe_cores.y)

        def moe_pc(in0_block_w):
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=moe_cores,
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=5,
                out_block_h=m_tiles,
                out_block_w=per_n,
                per_core_M=m_tiles,
                per_core_N=per_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

        self.gate_pc, self.expand_pc = moe_pc(2), moe_pc(1)
        self.moe_mc = ttnn.create_sharded_memory_config(
            (self.rows, per_n * TILE),
            ttnn.CoreGrid(y=moe_cores.y, x=moe_cores.x),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # router: the whole K in one block per output tile (TopKRouter.dense_decode's accumulation order: the same
        # logits), one core per 32-row tile
        k_tiles = self.hidden // TILE
        self.router_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.rows // TILE, 1),
            in0_block_w=k_tiles,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=-(-self.num_experts // TILE),
            per_core_M=1,
            per_core_N=-(-self.num_experts // TILE),
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        self.router_ck = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.argmax = _Argmax(model.lm_head_weight.shape[-1], model.vocab_size)
        from ..fused_decode.config import RESIDUAL_CORES, sdpa_decode_program_config, width_sharded_memory_config

        attn0 = model.layers[0].self_attn
        mc = model.layers[0].self_attn.mesh_config
        self.local_heads = mc.shard_size(attn0.config.num_heads)
        self.local_kv_heads = mc.shard_size(attn0.config.num_kv_heads)
        hd = attn0.config.head_dim
        self.residual_mc = width_sharded_memory_config(self.mesh, self.hidden, RESIDUAL_CORES, rows=self.rows)
        self.head_grid_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, hd),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # full-attention layers: SDPA on the whole grid (11 x 10; the decode's 8 x 8 holds one user per core), Q read
        # from DRAM (a sharded Q must sit on the SDPA grid's cores): at 4K context attention 22.2 -> 20.9 ms per pass,
        # same output; the 128-token sliding-window layers keep the decode's 8 x 8 config and sharded Q
        self.sdpa_pc = {
            False: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
                q_chunk_size=0,
                k_chunk_size=sdpa_decode_program_config(False).k_chunk_size,
                exp_approx_mode=False,
            ),
            True: sdpa_decode_program_config(True),
        }
        # the unpadded decode o_proj (batched_decode.BatchedOProj) is DRAM width-sharded: interleaved copies
        self.oproj_w = [
            ttnn.to_memory_config(layer.self_attn.batched_oproj.weight, ttnn.DRAM_MEMORY_CONFIG)
            for layer in model.layers
        ]
        # QKV / o_proj over 256 rows: the batched decode's compute config (HiFi2, BF16 accumulation in L1)
        self.qkv_ck = attn0.batched_qkv.compute_kernel_config
        # Q and K rotated by one op and K / V written by one op (the fused decode's layout: Q heads on cores 0-31, K on
        # 32-63; cos / sin rows for the Q users then the K users)
        from ..model import create_rope_setup

        self.rope_qk = create_rope_setup(
            self.mesh, cfg, max_local_batch_size=users, datatype=ttnn.bfloat16, use_qk_fused=True
        )
        self.trans_qk = self.rope_qk.get_both_trans_mats()["decode"]
        # the QKV head input's width-sharded cores (interleaved_to_sharded_partial takes a rectangle)
        self.qkv_grid = ttnn.CoreCoord(10, (self.local_heads + 2 * self.local_kv_heads) // 10)

    def _router(self, layer, x):
        """Dense [rows, experts] routing weights: each row's softmax over its top-k logits, zeros elsewhere (as
        TopKRouter.dense_decode, for any number of rows)."""
        r = layer.mlp.router
        logits = ttnn.linear(
            x,
            r.weight,
            bias=r.bias,
            program_config=self.router_pc,
            compute_kernel_config=self.router_ck,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        values, indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        indices.deallocate(True)
        kth = ttnn.min(values, dim=-1, keepdim=True)
        values.deallocate(True)
        top = ttnn.max(logits, dim=-1, keepdim=True)
        selected = ttnn.ge(logits, kth)
        kth.deallocate(True)
        e = ttnn.exp(ttnn.subtract(logits, top, output_tensor=logits), output_tensor=logits)
        top.deallocate(True)
        e = ttnn.multiply(e, selected, output_tensor=e)
        selected.deallocate(True)
        total = ttnn.sum(e, dim=-1, keepdim=True)
        w = ttnn.divide(e, total)
        total.deallocate(True)
        e.deallocate(True)
        return w

    def _moe(self, li, layer, x):
        """All experts for all rows ([rows, E * I_local] matmuls), unrouted experts zeroed by the routing weights; then
        the TP all-reduce. x: [1, 1, rows, hidden] normed (DRAM)."""
        UWP, U = ttnn.UnaryWithParam, ttnn.UnaryOpType
        W = self.moe[li]
        w = self._router(layer, x)
        x8 = ttnn.typecast(x, ttnn.bfloat8_b)
        # the biases inside the matmuls (with the FP32 down below: the fixed-text agreement of the separate adds)
        gate, up = (
            ttnn.linear(
                x8,
                W[n],
                bias=W[n + "_bias"],
                program_config=self.gate_pc,
                memory_config=self.moe_mc,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.ck,
            )
            for n in ("gate", "up")
        )
        x8.deallocate(True)
        w_cols = ttnn.matmul(
            w, W["expand"], program_config=self.expand_pc, memory_config=self.moe_mc, dtype=ttnn.bfloat16
        )
        limit, alpha = self.swiglu_limit, self.alpha
        act = ttnn.multiply(
            up,
            gate,
            input_tensor_a_activations=[UWP(U.CLAMP_TSS, -limit, limit), UWP(U.ADD_UNARY_SFPU, 1.0)],
            input_tensor_b_activations=[
                UWP(U.CLAMP_TSS, -3.0e38, limit),
                UWP(U.MUL_UNARY_SFPU, alpha),
                UWP(U.SILU),
                UWP(U.MUL_UNARY_SFPU, 1.0 / alpha),
            ],
            memory_config=self.moe_mc,
        )
        gate.deallocate(True)
        up.deallocate(True)
        # BF16 like the batched decode's down input, to DRAM for the 2D down
        act8 = ttnn.multiply(act, w_cols, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        act.deallocate(True)
        w_cols.deallocate(True)
        down = ttnn.matmul(
            act8, W["down"], program_config=self.down_pc, dtype=ttnn.bfloat16, compute_kernel_config=self.down_ck
        )
        act8.deallocate(True)
        bias = ttnn.matmul(w, W["down_bias"], dtype=ttnn.bfloat16)
        w.deallocate(True)
        down = ttnn.add(down, bias, output_tensor=down)
        bias.deallocate(True)
        out = _all_reduce(down)
        down.deallocate(True)
        return out

    def _attention(self, li, layer, res, pos, rot_mats):
        """The layer's attention for all rows: input norm, QKV and o_proj (+ all-reduce) once over the 256 rows (their
        weights read once per pass); per 32-row step the decode attention core: heads, RoPE, paged K/V write at
        positions start_u + i, paged SDPA decode (step i sees steps 0 .. i). res: [rows, hidden] width-sharded residual;
        rot_mats[i]: step i's fused-QK cos / sin. Returns res + attention (same layout)."""
        attn_mod, model = layer.self_attn, self.model
        w, cfg = attn_mod.weights, attn_mod.config
        nh, nkv, hd = self.local_heads, self.local_kv_heads, cfg.head_dim
        normed = layer.input_layernorm.forward_sharded(res)
        x = ttnn.to_memory_config(normed, ttnn.DRAM_MEMORY_CONFIG)
        normed.deallocate(True)
        qkv = ttnn.linear(x, w.wqkv, bias=w.wqkv_bias, dtype=ttnn.bfloat16, compute_kernel_config=self.qkv_ck)
        x.deallocate(True)
        k_cache, v_cache = self.kv_cache[li]
        outs = []
        for i in range(self.block):
            # step i's 32 rows straight from the interleaved QKV into the width-sharded head input
            part = ttnn.interleaved_to_sharded_partial(
                qkv, self.qkv_grid, [U32, hd], self.block, i, ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )  # fmt: skip
            # Q heads on cores 0-31, K on 32-63: Q and K rotated by one op, K and V written by one op
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                part, num_heads=nh, num_kv_heads=nkv, overlap_qk_coregrid=False, memory_config=self.head_grid_mc
            )
            part.deallocate(True)
            cos, sin = rot_mats[i]
            q2, k2 = ttnn.experimental.rotary_embedding_llama_fused_qk(q, k, cos, sin, self.trans_qk)
            q.deallocate(True)
            k.deallocate(True)
            ttnn.experimental.paged_fused_update_cache(
                k_cache, k2, v_cache, v, update_idxs_tensor=pos[i], page_table=self.page_table
            )
            k2.deallocate(True)
            v.deallocate(True)
            if not cfg.sliding_window:
                q3 = ttnn.to_memory_config(q2, ttnn.DRAM_MEMORY_CONFIG)
                q2.deallocate(True)
                q2 = q3
            outs.append(
                ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q2,
                    k_cache,
                    v_cache,
                    cur_pos_tensor=pos[i],
                    sliding_window_size=cfg.sliding_window,
                    attention_sink=w.decode_sinks,
                    page_table_tensor=self.page_table,
                    scale=cfg.scaling,
                    program_config=self.sdpa_pc[bool(cfg.sliding_window)],
                    compute_kernel_config=attn_mod.program_config.get_compute_kernel_config(),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            q2.deallocate(True)
        qkv.deallocate(True)
        # [1, rows, heads, hd] (row 32 i + u) -> [1, heads, rows, hd] -> [1, 1, rows, heads * hd]: one concat, one
        # permute and one head concat per layer (not a reshard, a head concat and a copy per step)
        sdpa = ttnn.concat(outs, dim=1)
        for t in outs:
            t.deallocate(True)
        perm = ttnn.permute(sdpa, (0, 2, 1, 3))
        sdpa.deallocate(True)
        heads = ttnn.experimental.nlp_concat_heads(perm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        perm.deallocate(True)
        op = attn_mod.batched_oproj
        o = ttnn.matmul(heads, self.oproj_w[li], dtype=ttnn.bfloat16, compute_kernel_config=self.qkv_ck)
        heads.deallocate(True)
        # the decode's BF8 partial sum (the bias inside the matmul with a BF8 output: top-1 vs HF 0.08)
        partial = ttnn.add(o, op.bias, dtype=ttnn.bfloat8_b)
        o.deallocate(True)
        partial16 = ttnn.typecast(partial, ttnn.bfloat16)
        partial.deallocate(True)
        summed = _all_reduce(partial16)
        partial16.deallocate(True)
        summed_s = ttnn.to_memory_config(summed, self.residual_mc)
        summed.deallocate(True)
        out = ttnn.add(res, summed_s, memory_config=self.residual_mc, dtype=ttnn.bfloat16)
        summed_s.deallocate(True)
        return out

    def _sharded_rows(self, x, norm, residual_mc):
        """Norm each 32-row tile of x on its width-sharded residual shards (the batched decode's norm)."""
        parts = []
        for i in range(self.block):
            xi = ttnn.slice(x, [0, 0, i * U32, 0], [1, 1, (i + 1) * U32, self.hidden])
            res = ttnn.to_memory_config(xi, residual_mc, dtype=ttnn.bfloat16)
            xi.deallocate(True)
            n = norm.forward_sharded(res)
            res.deallocate(True)
            parts.append(ttnn.to_memory_config(n, ttnn.DRAM_MEMORY_CONFIG))
            n.deallocate(True)
        out = ttnn.concat(parts, dim=2)
        for t in parts:
            t.deallocate(True)
        return out

    def forward(self, tokens, pos, rot_idx):
        """tokens: device [1, rows] UINT32 (row 32 i + u). pos[i] / rot_idx[i]: step i's [32] INT32 positions and
        [1, 32] UINT32 RoPE indices. Returns (per-device argmax of every row, device features [1, 1, rows, 5 hidden]).
        """
        model = self.model
        emb = ttnn.embedding(tokens, model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(emb, [1, 1, self.rows, self.hidden])
        feats = []
        res = ttnn.to_memory_config(x, self.residual_mc, dtype=ttnn.bfloat16)
        x.deallocate(True)
        rot_mats = []  # every layer's RoPE rows of step i (once per pass, not per layer)
        for r in rot_idx:
            r2 = ttnn.concat([r, r], dim=1)  # the Q users' rows, then the K users'
            rot_mats.append(self.rope_qk.get_rot_mats(r2))
            r2.deallocate(True)
        for li, layer in enumerate(model.layers):
            res2 = self._attention(li, layer, res, pos, rot_mats)
            res.deallocate(True)
            n2 = layer.post_attention_layernorm.forward_sharded(res2)
            normed = ttnn.to_memory_config(n2, ttnn.DRAM_MEMORY_CONFIG)
            n2.deallocate(True)
            x = ttnn.to_memory_config(res2, ttnn.DRAM_MEMORY_CONFIG)
            res2.deallocate(True)
            moe = self._moe(li, layer, normed)
            normed.deallocate(True)
            x2 = ttnn.add(x, moe, dtype=ttnn.bfloat16)
            x.deallocate(True)
            moe.deallocate(True)
            if li in self.capture_layers:
                feats.append(ttnn.clone(x2))
            res = ttnn.to_memory_config(x2, self.residual_mc)
            x = x2
            if li < len(model.layers) - 1:
                x2.deallocate(True)
        for t in (t for pair in rot_mats for t in pair):
            t.deallocate(True)
        features = ttnn.concat(feats, dim=-1)
        for t in feats:
            t.deallocate(True)
        normed_s = model.norm.forward_sharded(res)
        res.deallocate(True)
        x.deallocate(True)
        normed = ttnn.to_memory_config(normed_s, ttnn.DRAM_MEMORY_CONFIG)
        normed_s.deallocate(True)
        normed8 = ttnn.typecast(normed, ttnn.bfloat8_b)  # (to_memory_config to the same layout returns its input)
        normed.deallocate(True)
        logits8 = ttnn.matmul(normed8, model.lm_head_weight, dtype=ttnn.bfloat8_b)  # as the batched decode's LM head
        normed8.deallocate(True)
        logits = ttnn.typecast(logits8, ttnn.bfloat16)
        logits8.deallocate(True)
        out = self.argmax.device(logits)
        logits.deallocate(True)
        return out, features


class BatchedDFlashRunner:
    """Greedy DFlash for `users` sequences (each prefilled into its own KV slot) on the batched decode model."""

    def __init__(self, generator, model, page_table, kv_cache, users=32, window=1024, eos_ids=(200002, 200012)):
        from .capture import TargetCapture

        self.generator, self.model, self.kv_cache = generator, model, kv_cache
        mesh = model.mesh_device
        self.mesh = mesh
        self.users = users
        self.page_table_host = page_table
        self.page_table = _replicate(mesh, page_table, ttnn.int32)
        self.V = model.vocab_size
        self.eos_ids = set(eos_ids)
        max_pos = page_table.shape[-1] * 64
        self.draft = BatchedDraft(mesh, model, max_pos, users, window=window)
        self.block = B = self.draft.block
        self.rows = users * B
        self.capture = TargetCapture(model, self.draft.target_layer_ids)
        kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
        self.verify = BatchedVerify(model, self.page_table, kv, users, B, self.draft.target_layer_ids)
        # persistent per-round inputs (the traces' inputs)
        z = torch.zeros(users, dtype=torch.int32)
        self.d_tokens = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        self.v_tokens = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        self.pos = [_replicate(mesh, z, ttnn.int32) for _ in range(B)]
        self.rot = [_replicate(mesh, z.reshape(1, users), ttnn.uint32) for _ in range(B)]
        self.bound = _replicate(mesh, z, ttnn.int32)
        self.pos_rows = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        self.traces = None

    def prefill(self, prompts):
        firsts = []
        for u, p in enumerate(prompts):
            logits = self.generator.prefill_forward_text(
                torch.tensor([p]),
                page_table=self.page_table_host,
                kv_cache=self.kv_cache,
                prompt_lens=[len(p)],
                empty_slots=[u],
                enable_trace=False,
                warmup_prefill=False,
            )
            firsts.append(int(logits.reshape(-1)[: self.V].float().argmax()))
            self.draft.add_prompt_context(self.capture.prefill_features(len(p)), user=u)
        return firsts

    # -- passes (eager, or captured once and replayed) ----------------------------------------------------------------
    def _rot_mats(self):
        return self.draft.rows_cos_sin(self.pos_rows)

    def _draft_pass(self):
        return self.draft.draft(self.d_tokens, self.pos, self._rot_mats(), self.bound)

    def _verify_pass(self):
        out, features = self.verify.forward(self.v_tokens, self.pos, self.rot)
        self.draft.context(features, self.pos, self._rot_mats())
        features.deallocate(True)
        return out

    def capture_traces(self):
        """One trace for the draft pass, one for verify + context (after one eager run of each)."""
        mesh = self.mesh
        self.release_traces()
        for t in list(self._draft_pass()) + list(self._verify_pass()):
            t.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.draft_out = self._draft_pass()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        tid2 = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.verify_out = self._verify_pass()
        ttnn.end_trace_capture(mesh, tid2, cq_id=0)
        self.traces = (tid, tid2)

    def release_traces(self):
        if self.traces is not None:
            ttnn.synchronize_device(self.mesh)
            for t in self.traces:
                ttnn.release_trace(self.mesh, t)
        self.traces = None

    def _set_inputs(self, starts, lasts):
        mesh, U, B = self.mesh, self.users, self.block
        toks = torch.full((1, self.rows), self.draft.mask_token_id, dtype=torch.int32)
        toks[0, :U] = torch.tensor(lasts, dtype=torch.int32)
        _copy(mesh, self.d_tokens, toks, ttnn.uint32)
        st = torch.tensor(starts, dtype=torch.int32)
        for i in range(B):
            _copy(mesh, self.pos[i], st + i, ttnn.int32)
            _copy(mesh, self.rot[i], (st + i).reshape(1, U), ttnn.uint32)
        _copy(mesh, self.bound, st + B - 1, ttnn.int32)
        _copy(mesh, self.pos_rows, (st[None] + torch.arange(B)[:, None]).reshape(1, self.rows), ttnn.uint32)

    def _run(self, which):
        if self.traces is None:
            return (self._draft_pass if which == 0 else self._verify_pass)(), True
        ttnn.execute_trace(self.mesh, self.traces[which], cq_id=0, blocking=False)
        return (self.draft_out if which == 0 else self.verify_out), False

    def generate(self, prompts, max_new_tokens, firsts):
        """Returns (per-user new tokens, per-user accepted counts, per-round seconds)."""
        mesh = self.mesh
        U, B = self.users, self.block
        lens = [len(p) for p in prompts]
        ids = [list(p) + [f] for p, f in zip(prompts, firsts)]
        accept = [[] for _ in range(U)]
        times = []

        def done(u):
            return len(ids[u]) - lens[u] >= max_new_tokens or bool(set(ids[u][lens[u] :]) & self.eos_ids)

        while not all(done(u) for u in range(U)):
            t0 = time.perf_counter()
            starts = [len(ids[u]) - 1 for u in range(U)]
            lasts = [ids[u][-1] for u in range(U)]
            self._set_inputs(starts, lasts)
            t1 = time.perf_counter()
            d_out, owned = self._run(0)
            dr = self.draft.argmax.host(d_out)
            t2 = time.perf_counter()
            if owned:
                for t in d_out:
                    t.deallocate(True)
            vt = torch.tensor(lasts + dr[U:], dtype=torch.int32).reshape(1, self.rows)
            _copy(mesh, self.v_tokens, vt, ttnn.uint32)
            t3 = time.perf_counter()
            v_out, owned = self._run(1)
            post = self.verify.argmax.host(v_out)
            t4 = time.perf_counter()
            if owned:
                for t in v_out:
                    t.deallocate(True)
            for u in range(U):
                d = [dr[i * U + u] for i in range(1, B)]
                p = [post[i * U + u] for i in range(B)]
                k = 0
                while k < B - 1 and d[k] == p[k]:
                    k += 1
                if not done(u):
                    ids[u] += d[:k] + [p[k]]
                    accept[u].append(k + 1)
            t5 = time.perf_counter()
            times.append((t5 - t0, t2 - t1, t4 - t3, (t1 - t0) + (t3 - t2) + (t5 - t4)))  # round, draft, verify, host
        return [ids[u][lens[u] : lens[u] + max_new_tokens] for u in range(U)], accept, times
