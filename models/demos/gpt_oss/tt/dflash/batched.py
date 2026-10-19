# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy DFlash for 32 users at once (batch 32) on the batched gpt-oss-20b decode model (batched_decode.py, dense MoE).

A round, for every user u: the drafter proposes 7 tokens after the user's last token; the target checks the 8-row
block [last token, 7 drafts] of every user in one 256-row verify pass (row 32 i + u = row i of user u's block) and
keeps the drafts up to the first mismatch plus its own next token.

Verify pass, per target layer:
  attention: the 8 block rows in 8 steps of 32 rows (step i = row i of every user's block), each the batched decode
      attention (QKV, RoPE, paged K/V write, paged SDPA decode, o_proj, all-reduce) at positions start_u + i, so step i
      sees the K/V rows of steps 0 .. i (causal within the block);
  MoE: the 256 rows at once: router, every expert's gate / up / down as 256-row matmuls (the dense layout of the
      batched decode, each expert weight read once per pass), the routing weights zeroing the unrouted experts.
The drafter (BatchedDraft) is the ttnn drafter of the DFlash bring-up for several users: one paged BF16 K/V cache per
user, up to 4 users' 8-row blocks per 32-row call.
"""

import time

import torch

import ttnn
from models.tt_transformers.tt.load_checkpoints import reverse_permute, reverse_permute_1d

from ..fused_decode.config import residual_memory_config, width_sharded_memory_config
from .draft import draft_snapshot, load_draft_config

TILE = ttnn.TILE_SIZE
ROWS = 32


def _replicate(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))


class _AllReduce:
    """One-op BF16 all-reduce of a [1, 1, 32, hidden] partial over the mesh columns."""

    def __init__(self, mesh_device, hidden):
        from ..fused_decode.config import RESIDUAL_CORES

        self.mesh_device = mesh_device
        n = mesh_device.shape[1]
        self.memory_config = residual_memory_config(mesh_device, hidden)
        grid = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.slots = [
            (
                ttnn.empty(
                    [1, 1, ROWS, hidden * n],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=width_sharded_memory_config(mesh_device, hidden * n, RESIDUAL_CORES),
                ),
                ttnn.create_global_semaphore(mesh_device, cores, 0),
            )
            for _ in range(2)
        ]
        self.i = 0

    def __call__(self, partial):
        moved = ttnn.to_memory_config(partial, self.memory_config, dtype=ttnn.bfloat16)
        partial.deallocate(True)
        scratch, sem = self.slots[self.i]
        self.i ^= 1
        out = ttnn.experimental.all_reduce_async(
            moved,
            scratch,
            cluster_axis=1,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=sem,
            memory_config=self.memory_config,
            dtype=ttnn.bfloat16,
            topology=ttnn.Topology.Ring,
            num_links=1,
        )
        moved.deallocate(True)
        flat = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        return flat


class BatchedDraft:
    """The z-lab drafter (8 layers, TP like a dense target layer) for `users` sequences, each with its own paged BF16
    K/V cache. A call drafts up to 32 / 8 = 4 users (one 32-row tile)."""

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
        w = load_file(str(self.path / "model.safetensors"))
        mesh = mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def put(t, mapper=rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(), dtype=dtype, layout=layout, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )  # fmt: skip

        def norm_w(t):
            return put(t.float().reshape(1, 1, -1, TILE), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        cols = ttnn.ShardTensorToMesh(mesh, dim=-1)
        rows = ttnn.ShardTensorToMesh(mesh, dim=-2)
        hd, nq, nkv = self.hd, self.nq, self.nkv

        def per_device(parts):
            """[q | k | v]-style column blocks, each device's slice of every part side by side."""
            return torch.cat(
                [torch.cat([p[d * (p.shape[0] // D) : (d + 1) * (p.shape[0] // D)] for p in parts]) for d in range(D)]
            )

        self.fc = put(w["fc.weight"].T, rows)
        self.fc_full = put(w["fc.weight"].T)
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
                    kv=put(per_device([k, v]).T, cols),
                    kv_b=put(per_device([kb, vb]).reshape(1, -1), cols),
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
        self.users = users
        self.user_tables = torch.arange(users * self.num_blocks, dtype=torch.int32).reshape(users, self.num_blocks)
        heads_split = ttnn.ShardTensorToMesh(mesh, dim=1)
        self.caches = []
        for _ in range(self.layers):
            z = torch.zeros(users * self.num_blocks, D * nkv, block_size, hd)
            self.caches.append(
                tuple(
                    ttnn.from_torch(z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=heads_split)
                    for _ in range(2)
                )
            )  # fmt: skip
        from ..model import create_rope_setup

        self.rope = create_rope_setup(mesh, target.hf_config, max_local_batch_size=ROWS, datatype=ttnn.bfloat16)
        self.trans_mat = self.rope.get_both_trans_mats()["decode"]
        self.all_reduce = _AllReduce(mesh, self.hidden)
        self.head_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, hd), core_grid=ttnn.CoreGrid(y=4, x=8), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )  # fmt: skip
        self.sdpa_pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=0, k_chunk_size=128, exp_approx_mode=False
        )
        self.ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.mm_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    # -- helpers -------------------------------------------------------------------------------------------------
    def _page_table(self, row_users):
        rows = list(row_users) + [0] * (ROWS - len(row_users))
        return _replicate(self.mesh_device, self.user_tables[rows].contiguous(), ttnn.int32)

    def _rows(self, t, mapper=None):
        x = torch.zeros(1, 1, ROWS, t.shape[-1])
        x[0, 0, : t.shape[0]] = t.float()
        return ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper or ttnn.ReplicateTensorToMesh(self.mesh_device),
        )  # fmt: skip

    def _pos(self, positions):
        p = torch.full((ROWS,), -1, dtype=torch.int32)
        p[: len(positions)] = torch.as_tensor(list(positions), dtype=torch.int32)
        return _replicate(self.mesh_device, p, ttnn.int32), self.rope.get_rot_mats(torch.clamp(p, min=0))

    def _heads(self, x, layer):
        L = self.L[layer]
        xqkv = ttnn.linear(
            x, L["qkv"], bias=L["qkv_b"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )  # fmt: skip
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv, num_heads=self.nq, num_kv_heads=self.nkv, memory_config=self.head_mc
        )
        xqkv.deallocate(True)
        return q, k, v

    def _norm_rope(self, t, w, rot):
        flat = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        t.deallocate(True)
        normed = ttnn.rms_norm(flat, epsilon=self.eps, weight=w, compute_kernel_config=self.ck)
        flat.deallocate(True)
        sharded = ttnn.to_memory_config(normed, self.head_mc)
        normed.deallocate(True)
        out = ttnn.experimental.rotary_embedding_llama(sharded, rot[0], rot[1], self.trans_mat, is_decode_mode=True)
        sharded.deallocate(True)
        return out

    def _write_kv(self, layer, k, v, positions, page_table):
        """One paged_update_cache per row of a sequence: the op's read-modify-write of a cache tile races when several
        rows of one call share a tile (rows of different users never do)."""
        kc, vc = self.caches[layer]
        B = self.block
        n = len(positions)
        for r in range(min(n, B)):
            idx = torch.full((ROWS,), -1, dtype=torch.int32)
            for row in range(r, n, B):
                idx[row] = positions[row]
            tt = _replicate(self.mesh_device, idx, ttnn.int32)
            ttnn.experimental.paged_update_cache(kc, k, update_idxs_tensor=tt, page_table=page_table)
            ttnn.experimental.paged_update_cache(vc, v, update_idxs_tensor=tt, page_table=page_table)
            tt.deallocate(True)

    # -- context and drafts --------------------------------------------------------------------------------------
    def add_prompt_context(self, features, user, chunk=2048):
        """The prompt's context K / V (prefill-style, paged_fill_cache of block-aligned chunks)."""
        n = features.shape[0]
        mesh = self.mesh_device
        pt_host = self.user_tables[user][None]
        for s0 in range(0, n, chunk):
            rows = min(chunk, -(-(n - s0) // self.block_size) * self.block_size)
            x = torch.zeros(1, 1, rows, self.ctx_in)
            x[0, 0, : min(rows, n - s0)] = features[s0 : s0 + rows].float()
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
            ctx = ttnn.linear(tx, self.fc_full, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            tx.deallocate(True)
            ctx = ttnn.rms_norm(ctx, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
            cos = self.rope.cos_matrix_prefill[:, :, s0 : s0 + rows, :]
            sin = self.rope.sin_matrix_prefill[:, :, s0 : s0 + rows, :]
            trans = self.rope.get_both_trans_mats()["prefill"]
            b0 = s0 // self.block_size
            chunk_pt = _replicate(mesh, pt_host[:, b0 : b0 + rows // self.block_size].contiguous(), ttnn.int32)
            for i in range(self.layers):
                L = self.L[i]
                kv = ttnn.linear(ctx, L["kv"], bias=L["kv_b"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
                width = self.nkv * self.hd
                for j, cache in enumerate(self.caches[i]):
                    t = ttnn.slice(kv, (0, 0, 0, j * width), (1, 1, rows, (j + 1) * width))
                    t = ttnn.transpose(ttnn.reshape(t, [1, rows, self.nkv, self.hd]), 1, 2)
                    if j == 0:
                        t = ttnn.rms_norm(t, epsilon=self.eps, weight=L["k_norm"], compute_kernel_config=self.ck)
                        t = ttnn.experimental.rotary_embedding_llama(t, cos, sin, trans, is_decode_mode=False)
                    ttnn.experimental.paged_fill_cache(cache, t, chunk_pt, batch_idx=0)
                    t.deallocate(True)
                kv.deallocate(True)
            chunk_pt.deallocate(True)
            ctx.deallocate(True)

    def add_context(self, features, positions, users):
        """features: host [n, 5 * hidden] target hidden states; row r belongs to users[r] at positions[r] (up to 32 rows
        per call, at most 8 rows of one user)."""
        n = features.shape[0]
        for s in range(0, n, ROWS):
            chunk, pos_list, us = features[s : s + ROWS], list(positions)[s : s + ROWS], list(users)[s : s + ROWS]
            x = self._rows(chunk, ttnn.ShardTensorToMesh(self.mesh_device, dim=-1))
            partial = ttnn.linear(x, self.fc, dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            x.deallocate(True)
            summed = self.all_reduce(partial)
            ctx = ttnn.rms_norm(summed, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
            summed.deallocate(True)
            pos, rot = self._pos(pos_list)
            table = self._page_table(us)
            kc_rows = self._row_groups(us)
            for i in range(self.layers):
                q, k, v = self._heads(ctx, i)
                q.deallocate(True)
                k = self._norm_rope(k, self.L[i]["k_norm"], rot)
                self._write_rows(i, k, v, pos_list, kc_rows, table)
                k.deallocate(True)
                v.deallocate(True)
            ctx.deallocate(True)
            pos.deallocate(True)
            table.deallocate(True)

    def _row_groups(self, users):
        """Rows split into groups with at most one row per user (one paged_update_cache each)."""
        groups, seen = [], []
        for r, u in enumerate(users):
            for g, s in zip(groups, seen):
                if u not in s:
                    g.append(r)
                    s.add(u)
                    break
            else:
                groups.append([r])
                seen.append({u})
        return groups

    def _write_rows(self, layer, k, v, positions, groups, page_table):
        kc, vc = self.caches[layer]
        for g in groups:
            idx = torch.full((ROWS,), -1, dtype=torch.int32)
            for r in g:
                idx[r] = positions[r]
            tt = _replicate(self.mesh_device, idx, ttnn.int32)
            ttnn.experimental.paged_update_cache(kc, k, update_idxs_tensor=tt, page_table=page_table)
            ttnn.experimental.paged_update_cache(vc, v, update_idxs_tensor=tt, page_table=page_table)
            tt.deallocate(True)

    def draft_users(self, last_tokens, starts, users):
        """Up to 4 users at once (user j's block: rows 8 j .. 8 j + 7). Returns each user's 7 draft tokens."""
        mesh = self.mesh_device
        B = self.block
        n = len(users)
        assert n * B <= ROWS
        toks = torch.full((1, ROWS), self.mask_token_id, dtype=torch.int32)
        for j, t in enumerate(last_tokens):
            toks[0, j * B] = t
        tt_toks = _replicate(mesh, toks, ttnn.uint32)
        emb = ttnn.embedding(tt_toks, self.target.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        tt_toks.deallocate(True)
        h = ttnn.typecast(ttnn.reshape(emb, [1, 1, ROWS, self.hidden]), ttnn.float32)
        positions = [s0 + i for s0 in starts for i in range(B)]
        pos, rot = self._pos(positions)
        row_users = [u for u in users for _ in range(B)]
        table = self._page_table(row_users)
        groups = [list(range(r, n * B, B)) for r in range(B)]  # row r of every user's block: distinct users
        bound = torch.full((ROWS,), -1, dtype=torch.int32)
        for j, s0 in enumerate(starts):
            bound[j * B : (j + 1) * B] = s0 + B - 1  # every block row sees the whole block (not causal)
        tt_bound = _replicate(mesh, bound, ttnn.int32)
        for i in range(self.layers):
            L = self.L[i]
            x = ttnn.rms_norm(h, epsilon=self.eps, weight=L["ln1"], compute_kernel_config=self.ck)
            q, k, v = self._heads(x, i)
            x.deallocate(True)
            q = self._norm_rope(q, L["q_norm"], rot)
            k = self._norm_rope(k, L["k_norm"], rot)
            self._write_rows(i, k, v, positions, groups, table)
            k.deallocate(True)
            v.deallocate(True)
            kc, vc = self.caches[i]
            attn = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, kc, vc, cur_pos_tensor=tt_bound, page_table_tensor=table, scale=self.hd**-0.5,
                sliding_window_size=self.window, program_config=self.sdpa_pc, compute_kernel_config=self.ck,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # fmt: skip
            q.deallocate(True)
            attn_s = ttnn.to_memory_config(attn, self.head_mc)
            attn.deallocate(True)
            heads = ttnn.experimental.nlp_concat_heads_decode(attn_s, num_heads=self.nq)
            attn_s.deallocate(True)
            partial = ttnn.linear(heads, L["o"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            heads.deallocate(True)
            o = self.all_reduce(partial)
            o = ttnn.add(o, L["o_b"], output_tensor=o)
            h2 = ttnn.add(h, o, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
            h.deallocate(True)
            o.deallocate(True)
            h = h2
            x = ttnn.rms_norm(h, epsilon=self.eps, weight=L["ln2"], compute_kernel_config=self.ck)
            gu = ttnn.linear(x, L["gu"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            x.deallocate(True)
            I = self.inter
            g = ttnn.slice(gu, (0, 0, 0, 0), (1, 1, ROWS, I))
            u = ttnn.slice(gu, (0, 0, 0, I), (1, 1, ROWS, 2 * I))
            gu.deallocate(True)
            act = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            g.deallocate(True)
            u.deallocate(True)
            partial = ttnn.linear(act, L["down"], dtype=ttnn.bfloat16, compute_kernel_config=self.mm_ck)
            act.deallocate(True)
            mlp = self.all_reduce(partial)
            h2 = ttnn.add(h, mlp, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
            h.deallocate(True)
            mlp.deallocate(True)
            h = h2
        x = ttnn.rms_norm(h, epsilon=self.eps, weight=self.norm, compute_kernel_config=self.ck)
        h.deallocate(True)
        ids = _lm_argmax(self.target, x, mesh)
        x.deallocate(True)
        pos.deallocate(True)
        table.deallocate(True)
        tt_bound.deallocate(True)
        return [ids[j * B + 1 : (j + 1) * B] for j in range(n)]


def _lm_argmax(target, x, mesh, bf8=False):
    """Row argmax over the vocabulary of x @ the target's LM head ([1, 1, rows, hidden], vocab split over the mesh
    columns): each device's best (value, index), the host picks the best device per row. bf8: BF8 input and logits,
    as the batched decode's LM head (so near-ties resolve as in plain decode)."""
    if bf8:
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    logits = ttnn.matmul(x, target.lm_head_weight, dtype=ttnn.bfloat8_b if bf8 else ttnn.bfloat16)
    if bf8:
        x.deallocate(True)
        wide = ttnn.typecast(logits, ttnn.bfloat16)
        logits.deallocate(True)
        logits = wide
    rows = logits.shape[-2]
    per_dev = logits.shape[-1]
    V = target.vocab_size
    vals, idxs = ttnn.topk(logits, k=1, dim=-1, sorted=True)
    logits.deallocate(True)
    v = [ttnn.to_torch(t).float().reshape(rows) for t in ttnn.get_device_tensors(vals)]
    i = [ttnn.to_torch(t).long().reshape(rows) for t in ttnn.get_device_tensors(idxs)]
    vals.deallocate(True)
    idxs.deallocate(True)
    best = []
    for r in range(rows):
        cands = [
            (v[d][r].item(), d * per_dev + i[d][r].item()) for d in range(len(v)) if d * per_dev + i[d][r].item() < V
        ]
        best.append(max(cands)[1])
    return best


class BatchedVerify:
    """The target's 256-row verify pass for 32 users (row 32 i + u = row i of user u's 8-row block) on the batched decode
    model's layers (DecoderLayer.batched_decode)."""

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
        self.swiglu_limit, self.alpha = cfg.swiglu_limit, 1.702
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
            self.mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )  # fmt: skip

    def _router(self, layer, x):
        """Dense [rows, experts] routing weights: each row's softmax over its top-k logits, zeros elsewhere (as
        TopKRouter.dense_decode, for any number of rows)."""
        r = layer.mlp.router
        if not hasattr(self, "_router_pc"):
            # the whole K in one block per output tile (TopKRouter.dense_decode's accumulation order: the same logits),
            # one core per 32-row tile
            k_tiles = x.shape[-1] // TILE
            self._router_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(self.rows // TILE, 1), in0_block_w=k_tiles,
                out_subblock_h=1, out_subblock_w=1, out_block_h=1, out_block_w=-(-self.num_experts // TILE),
                per_core_M=1, per_core_N=-(-self.num_experts // TILE), fuse_batch=True, fused_activation=None,
                mcast_in0=False,
            )  # fmt: skip
            self._router_ck = ttnn.init_device_compute_kernel_config(
                self.mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                fp32_dest_acc_en=False, packer_l1_acc=True,
            )  # fmt: skip
        logits = ttnn.linear(
            x, r.weight, bias=r.bias, program_config=self._router_pc, compute_kernel_config=self._router_ck,
            memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
        )  # fmt: skip
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
        gate = ttnn.matmul(x8, W["gate"], dtype=ttnn.bfloat16, compute_kernel_config=self.ck)
        up = ttnn.matmul(x8, W["up"], dtype=ttnn.bfloat16, compute_kernel_config=self.ck)
        x8.deallocate(True)
        gate = ttnn.add(gate, W["gate_bias"], output_tensor=gate)
        up = ttnn.add(up, W["up_bias"], output_tensor=up)
        w_cols = ttnn.matmul(w, W["expand"], dtype=ttnn.bfloat16)
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
        )
        gate.deallocate(True)
        up.deallocate(True)
        act = ttnn.multiply(act, w_cols, output_tensor=act)
        w_cols.deallocate(True)
        down = ttnn.matmul(act, W["down"], dtype=ttnn.bfloat16, compute_kernel_config=self.ck)
        act.deallocate(True)
        bias = ttnn.matmul(w, W["down_bias"], dtype=ttnn.bfloat16)
        w.deallocate(True)
        down = ttnn.add(down, bias, output_tensor=down)
        bias.deallocate(True)
        out = ttnn.all_reduce(
            down, num_links=1, topology=ttnn.Topology.Ring, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        down.deallocate(True)
        return out

    def forward(self, tokens, positions, rot_idxs):
        """tokens: device [1, 1, 1, rows] UINT32 (row 32 i + u). positions[i] / rot_idxs[i]: step i's device [32] INT32
        positions and [1, 32] UINT32 RoPE indices. Returns (logits-argmax rows, {layer: [rows, hidden] host features}).
        """
        model, U = self.model, self.users
        emb = ttnn.embedding(tokens, model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(emb, [1, 1, self.rows, self.hidden])
        feats = {}
        for li, layer in enumerate(model.layers):
            rope = None
            chunks, normed_chunks = [], []
            for i in range(self.block):
                xi = ttnn.slice(x, [0, 0, i * U, 0], [1, 1, (i + 1) * U, self.hidden])
                res = ttnn.to_memory_config(xi, layer.residual_memory_config, dtype=ttnn.bfloat16)
                xi.deallocate(True)
                normed = layer.input_layernorm.forward_sharded(res)
                rope = model.rope_setup.get_rot_mats(rot_idxs[i])
                partial = layer.self_attn(
                    normed, rope_mats=rope, position_idx=positions[i], page_table=self.page_table,
                    kv_cache=self.kv_cache[li], is_decode=True,
                )  # fmt: skip
                normed.deallocate(True)
                attn = layer.decode_all_reduce(partial, "attn")
                out = ttnn.add(res, attn, memory_config=layer.residual_memory_config, dtype=ttnn.bfloat16)
                attn.deallocate(True)
                res.deallocate(True)
                # the post-attention norm on the residual's own shards, as the batched decode layer
                n2 = layer.post_attention_layernorm.forward_sharded(out)
                chunks.append(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))
                normed_chunks.append(ttnn.to_memory_config(n2, ttnn.DRAM_MEMORY_CONFIG))
                out.deallocate(True)
                n2.deallocate(True)
            x.deallocate(True)
            x = ttnn.concat(chunks, dim=2)
            normed = ttnn.concat(normed_chunks, dim=2)
            for t in chunks + normed_chunks:
                t.deallocate(True)
            moe = self._moe(li, layer, normed)
            normed.deallocate(True)
            x2 = ttnn.add(x, moe, dtype=ttnn.bfloat16)
            x.deallocate(True)
            moe.deallocate(True)
            x = x2
            if li in self.capture_layers:
                feats[li] = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float().reshape(self.rows, self.hidden)
        parts = []
        for i in range(self.block):  # the final norm per 32-row tile on sharded rows, as the batched decode
            xi = ttnn.slice(x, [0, 0, i * U, 0], [1, 1, (i + 1) * U, self.hidden])
            res = ttnn.to_memory_config(xi, model.layers[-1].residual_memory_config, dtype=ttnn.bfloat16)
            xi.deallocate(True)
            n = model.norm.forward_sharded(res)
            res.deallocate(True)
            parts.append(ttnn.to_memory_config(n, ttnn.DRAM_MEMORY_CONFIG))
            n.deallocate(True)
        x.deallocate(True)
        normed = ttnn.concat(parts, dim=2)
        for t in parts:
            t.deallocate(True)
        ids = _lm_argmax(model, normed, self.mesh, bf8=True)
        normed.deallocate(True)
        return ids, feats


class BatchedDFlashRunner:
    """Greedy DFlash for `users` sequences (each prefilled into its own KV slot) on the batched decode model."""

    def __init__(self, generator, model, page_table, kv_cache, users=32, window=1024, eos_ids=(200002, 200012)):
        from .capture import TargetCapture

        self.generator, self.model, self.kv_cache = generator, model, kv_cache
        self.users = users
        self.page_table_host = page_table
        self.page_table = _replicate(model.mesh_device, page_table, ttnn.int32)
        self.V = model.vocab_size
        self.eos_ids = set(eos_ids)
        max_pos = page_table.shape[-1] * 64
        self.draft = BatchedDraft(model.mesh_device, model, max_pos, users, window=window)
        self.block = self.draft.block
        self.capture = TargetCapture(model, self.draft.target_layer_ids)
        kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
        self.verify = BatchedVerify(model, self.page_table, kv, users, self.block, self.draft.target_layer_ids)

    def prefill(self, prompts):
        firsts = []
        for u, p in enumerate(prompts):
            logits = self.generator.prefill_forward_text(
                torch.tensor([p]), page_table=self.page_table_host, kv_cache=self.kv_cache, prompt_lens=[len(p)],
                empty_slots=[u], enable_trace=False, warmup_prefill=False,
            )  # fmt: skip
            firsts.append(int(logits.reshape(-1)[: self.V].float().argmax()))
            self.draft.add_prompt_context(self.capture.prefill_features(len(p)), user=u)
        return firsts

    def generate(self, prompts, max_new_tokens, firsts):
        """Returns (per-user new tokens, per-user accepted counts, per-round seconds)."""
        mesh = self.model.mesh_device
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
            drafts = [None] * U
            for g in range(0, U, 4):
                grp = list(range(g, min(g + 4, U)))
                for u, d in zip(grp, self.draft.draft_users([ids[u][-1] for u in grp], [starts[u] for u in grp], grp)):
                    drafts[u] = d
            toks = torch.zeros(1, 1, 1, U * B, dtype=torch.int32)
            for u in range(U):
                blk = [ids[u][-1]] + drafts[u]
                for i in range(B):
                    toks[0, 0, 0, i * U + u] = blk[i]
            tt_toks = _replicate(mesh, toks, ttnn.uint32)
            pos = [torch.tensor([starts[u] + i for u in range(U)], dtype=torch.int32) for i in range(B)]
            tt_pos = [_replicate(mesh, p, ttnn.int32) for p in pos]
            tt_rot = [_replicate(mesh, p.reshape(1, U).to(torch.int32), ttnn.uint32) for p in pos]
            post, feats = self.verify.forward(tt_toks, tt_pos, tt_rot)
            for t in [tt_toks] + tt_pos + tt_rot:
                t.deallocate(True)
            ctx_rows, ctx_pos, ctx_users = [], [], []
            fcat = torch.cat([feats[L] for L in self.draft.target_layer_ids], dim=-1)  # [rows, 5 * hidden]
            for u in range(U):
                d = drafts[u]
                p = [post[i * U + u] for i in range(B)]
                k = 0
                while k < B - 1 and d[k] == p[k]:
                    k += 1
                if not done(u):
                    ids[u] += d[:k] + [p[k]]
                    accept[u].append(k + 1)
                for i in range(k + 1):
                    ctx_rows.append(fcat[i * U + u])
                    ctx_pos.append(starts[u] + i)
                    ctx_users.append(u)
            order = sorted(range(len(ctx_rows)), key=lambda r: (ctx_pos[r] - starts[ctx_users[r]], ctx_users[r]))
            self.draft.add_context(
                torch.stack([ctx_rows[r] for r in order]), [ctx_pos[r] for r in order], [ctx_users[r] for r in order]
            )
            times.append(time.perf_counter() - t0)
        return [ids[u][lens[u] : lens[u] + max_new_tokens] for u in range(U)], accept, times
