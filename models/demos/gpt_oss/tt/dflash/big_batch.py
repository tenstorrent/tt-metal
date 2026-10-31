# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Plain greedy decode of gpt-oss-20b for more than 32 users (64 / 128 / 256) on the batched decode model's layers.

The batched decode (batched_decode.py) holds one 32-row tile of tokens per device. Here a step runs U = 32 G users as
U rows through the DFlash verify's row-generic pieces (dflash/batched.py, with a block of one row per user), so every
weight is read once per step for all U rows:

  QKV and o_proj: one matmul over the U rows each;
  attention: per 32-user group, the batched decode's decode ops on that group's rows (heads, RoPE, the K / V cache
      update, paged decode SDPA with the sinks and the sliding window, concat heads);
  residual add + RMSNorm (+ the router's logits): one program per add (_ResNorm) on the U rows;
  MoE: the grouped MoE (dflash/grouped_moe.py) on the U rows' routed (row, expert) pairs;
  LM head: one matmul over the U rows, the per-device argmax, the devices combined on device.

Row u is user u. The step's next tokens and positions are written on device, so traced steps replay back to back.
"""

import time

import torch

import ttnn

from .batched import (
    TILE,
    U32,
    _all_reduce,
    _Argmax,
    _even_lm_head,
    _matmul_config,
    _mm_config_256,
    _replicate,
    _ResNorm,
    _TPAllReduce,
    hold_prefill_l1,
)


def verify_moe_weights(model):
    """Every layer's MoE weights as BatchedVerify.moe: the dense decode weights in DRAM interleaved (GroupedMoE relays
    gate / up / down out at init; pop them after)."""
    moe = []
    for layer in model.layers:
        assert layer.batched_decode, "needs the batched decode model (batch 32, dense MoE)"
        dw = layer.mlp.experts.dense_decode_weights
        n_act = dw.gate.shape[-1]
        moe.append(
            dict(
                gate=ttnn.to_memory_config(dw.gate, ttnn.DRAM_MEMORY_CONFIG),
                up=ttnn.to_memory_config(dw.up, ttnn.DRAM_MEMORY_CONFIG),
                down=ttnn.to_memory_config(dw.down, ttnn.DRAM_MEMORY_CONFIG),
                gate_bias=ttnn.slice(dw.gate_bias, [0, 0, 0, 0], [1, 1, 1, n_act]),
                up_bias=ttnn.slice(dw.up_bias, [0, 0, 0, 0], [1, 1, 1, n_act]),
                down_bias=dw.down_bias,
            )
        )
    return moe


def grouped_moe_for(model, rows, share=None):
    """A GroupedMoE for `rows`-row passes: share's relaid weights, or relaid from the model's dense decode weights."""
    from .grouped_moe import GroupedMoE

    hidden = model.hf_config.hidden_size
    if share is not None:
        return GroupedMoE(model.mesh_device, None, hidden, rows=rows, share=share)
    moe = verify_moe_weights(model)
    g = GroupedMoE(model.mesh_device, moe, hidden, rows=rows)
    for W in moe:  # (the interleaved copies were only the relayout's source)
        for k in ("gate", "up", "down"):
            W.pop(k).deallocate(True)
    return g


class BigPlain:
    """Greedy decode of `users` (a multiple of 32, at most 256) sequences, each with its own row of the page table."""

    def __init__(self, model, page_table, kv_cache, users, grouped=None, verify=None):
        """page_table: host INT32 [users, blocks] (row u: user u's blocks of the shared K / V cache pool); kv_cache: the
        model's per-layer (k, v) caches; grouped: a GroupedMoE to share relaid weights with (one is built otherwise);
        verify: a BatchedVerify whose o_proj copies to share."""
        assert users % U32 == 0 and U32 <= users <= 256
        self.model, self.mesh = model, model.mesh_device
        mesh = self.mesh
        self.users = self.rows = users
        self.G = users // U32
        self.hidden = model.hf_config.hidden_size
        self.kv_cache = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
        attn0 = model.layers[0].self_attn
        mc = attn0.mesh_config
        self.nh, self.nkv = mc.shard_size(attn0.config.num_heads), mc.shard_size(attn0.config.num_kv_heads)
        self.hd = attn0.config.head_dim
        self.grouped = grouped_moe_for(model, users, share=grouped)
        self.lm_head = _even_lm_head(model)
        self.argmax = _Argmax(self.lm_head.shape[-1], model.vocab_size)
        from ..fused_decode.config import width_sharded_memory_config

        # the residual block sharded on 10 x (users / 32) cores, one 32-row tile x 9 column tiles each (the verify's)
        self.residual_mc = ttnn.create_sharded_memory_config(
            (TILE, self.hidden // 10), ttnn.CoreGrid(x=10, y=self.rows // TILE), ttnn.ShardStrategy.BLOCK,
            ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )  # fmt: skip
        self.resnorm = _ResNorm(mesh, self.rows, self.hidden, self.residual_mc,
                                width_sharded_memory_config(mesh, self.hidden, _TPAllReduce.CORES, rows=self.rows))  # fmt: skip
        for l in model.layers:  # (the weights' tiles made here, not during a trace capture)
            self.resnorm.gamma(l.input_layernorm)
            self.resnorm.gamma(l.post_attention_layernorm)
            self.resnorm.router_bias(l.mlp.router)
        self.resnorm.gamma(model.norm)
        # o_proj: the unpadded decode weight (batched_decode.BatchedOProj, DRAM width-sharded) as an interleaved copy
        self.oproj_w = verify.oproj_w if verify is not None else [
            ttnn.to_memory_config(layer.self_attn.batched_oproj.weight, ttnn.DRAM_MEMORY_CONFIG)
            for layer in model.layers
        ]  # fmt: skip
        self.qkv_ck = attn0.batched_qkv.compute_kernel_config
        self.trans = model.transformation_mats["decode"]
        # the decode ops' layouts for one 32-user group (attention/decode.py): one user per core of the 8 x 8 grid
        grid8 = ttnn.CoreCoord(8, 8)
        self.heads_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, self.hd), core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
        )  # fmt: skip
        self.sdpa_out_mc = ttnn.create_sharded_memory_config(
            shape=(-(-self.nh // TILE) * TILE, self.hd), core_grid=ttnn.num_cores_to_corerangeset(U32, grid8, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )  # fmt: skip
        # persistent per-step inputs (trace inputs): the tokens, and per group its positions (INT32 [32], the cache
        # update and SDPA), its RoPE indices (UINT32 [1, 32]) and its page table rows
        self.tokens = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        z = torch.zeros(U32, dtype=torch.int32)
        self.pos = [_replicate(mesh, z, ttnn.int32) for _ in range(self.G)]
        self.rot_idx = [_replicate(mesh, z.reshape(1, U32), ttnn.uint32) for _ in range(self.G)]
        self.page_table = [
            _replicate(mesh, page_table[g * U32 : (g + 1) * U32].contiguous(), ttnn.int32) for g in range(self.G)
        ]
        self.trace = None
        hold_prefill_l1(model)

    # -- the step ---------------------------------------------------------------------------------------------------
    def _attention(self, li, layer, x, rope):
        """QKV over all rows, the attention core per 32-user group (the batched decode's decode ops), o_proj over all
        rows, all-reduced (the all-reduce's width shards: _ResNorm adds them to the residual)."""
        attn = layer.self_attn
        w, cfg = attn.weights, attn.config
        k_cache, v_cache = self.kv_cache[li]
        qkv = ttnn.linear(x, w.wqkv, bias=w.wqkv_bias,
                          program_config=_mm_config_256(w.wqkv.shape[-2], w.wqkv.shape[-1], self.rows),
                          dtype=ttnn.bfloat16, compute_kernel_config=self.qkv_ck, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        x.deallocate(True)
        n = qkv.shape[-1]
        outs = []
        from ..fused_decode.config import sdpa_decode_program_config

        for g in range(self.G):
            qg = ttnn.slice(qkv, [0, 0, g * U32, 0], [1, 1, (g + 1) * U32, n], memory_config=ttnn.L1_MEMORY_CONFIG)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                qg, num_heads=self.nh, num_kv_heads=self.nkv, memory_config=self.heads_mc
            )
            qg.deallocate(True)
            q2 = ttnn.experimental.rotary_embedding_llama(q, rope[g][0], rope[g][1], self.trans, is_decode_mode=True)
            k2 = ttnn.experimental.rotary_embedding_llama(k, rope[g][0], rope[g][1], self.trans, is_decode_mode=True)
            q.deallocate(True)
            k.deallocate(True)
            for t, cache in ((k2, k_cache), (v, v_cache)):
                # (the heads layout is the cache update's: to_memory_config would return the same buffer)
                tc = t if t.memory_config() == attn.kv_mem_cfg else ttnn.to_memory_config(t, attn.kv_mem_cfg)
                ttnn.experimental.paged_update_cache(cache, tc, update_idxs_tensor=self.pos[g],
                                                     page_table=self.page_table[g])  # fmt: skip
                if tc is not t:
                    tc.deallocate(True)
                t.deallocate(True)
            sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q2, k_cache, v_cache, cur_pos_tensor=self.pos[g], sliding_window_size=cfg.sliding_window,
                attention_sink=w.decode_sinks, page_table_tensor=self.page_table[g], scale=cfg.scaling,
                program_config=sdpa_decode_program_config(cfg.sliding_window),
                compute_kernel_config=attn.program_config.get_compute_kernel_config(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # fmt: skip
            q2.deallocate(True)
            hs = ttnn.to_memory_config(sdpa, self.sdpa_out_mc)
            sdpa.deallocate(True)
            o = ttnn.experimental.nlp_concat_heads_decode(hs, num_heads=self.nh)
            hs.deallocate(True)
            outs.append(ttnn.to_memory_config(o, ttnn.L1_MEMORY_CONFIG))
            o.deallocate(True)
        qkv.deallocate(True)
        heads = ttnn.concat(outs, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG) if self.G > 1 else outs[0]
        if self.G > 1:
            for t in outs:
                t.deallocate(True)
        o = ttnn.linear(heads, self.oproj_w[li], bias=attn.batched_oproj.bias,
                        program_config=_mm_config_256(heads.shape[-1], self.oproj_w[li].shape[-1], self.rows),
                        dtype=ttnn.bfloat16, compute_kernel_config=self.qkv_ck, memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        heads.deallocate(True)
        out = _all_reduce(o)
        o.deallocate(True)
        return out

    def step(self):
        """One decode step for every user from the persistent inputs: returns the next tokens ([1, 1, rows, 1] FLOAT32,
        the same on every device) and writes them, and the advanced positions, as the next step's inputs."""
        _TPAllReduce.reset()
        model = self.model
        rope = [model.rope_setup.get_rot_mats(self.rot_idx[g]) for g in range(self.G)]
        emb = ttnn.embedding(self.tokens, model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(emb, [1, 1, self.rows, self.hidden])
        res = ttnn.to_memory_config(x, self.residual_mc, dtype=ttnn.bfloat16)
        x.deallocate(True)
        n0 = model.layers[0].input_layernorm.forward_sharded(res)
        xin = ttnn.to_memory_config(n0, ttnn.L1_MEMORY_CONFIG)
        n0.deallocate(True)
        L = len(model.layers)
        for li, layer in enumerate(model.layers):
            y = self._attention(li, layer, xin, rope)  # (deallocates xin)
            res2, x8, logits = self.resnorm(y, res, layer.post_attention_layernorm, out_bf16=False, x8=True,
                                            router=layer.mlp.router)  # fmt: skip
            y.deallocate(True)
            res.deallocate(True)
            self.grouped.route(logits, li)
            logits.deallocate(True)
            down = self.grouped.from_x8(li, x8)
            x8.deallocate(True)
            y = _all_reduce(down)
            down.deallocate(True)
            nxt = model.layers[li + 1].input_layernorm if li + 1 < L else model.norm
            res, xin = self.resnorm(y, res2, nxt)
            y.deallocate(True)
            res2.deallocate(True)
        for cs in rope:
            for t in cs:
                t.deallocate(True)
        res.deallocate(True)
        normed8 = ttnn.typecast(xin, ttnn.bfloat8_b)
        xin.deallocate(True)
        logits8 = ttnn.matmul(
            normed8, self.lm_head, dtype=ttnn.bfloat8_b, program_config=_matmul_config(normed8, self.lm_head, True)
        )
        normed8.deallocate(True)
        out = self.argmax.device(logits8)
        logits8.deallocate(True)
        tok = self.argmax.combine(out, self.mesh)
        for t in out:
            t.deallocate(True)
        # the next step's inputs
        row = ttnn.transpose(tok, -2, -1)  # [1, 1, 1, rows]
        nxt_tok = ttnn.reshape(ttnn.to_layout(ttnn.typecast(row, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, self.rows])
        row.deallocate(True)
        ttnn.copy(nxt_tok, self.tokens)
        nxt_tok.deallocate(True)
        for g in range(self.G):
            ttnn.plus_one(self.pos[g], skip_negative_entries=True)
            ttnn.plus_one(self.rot_idx[g])
        return tok

    # -- driving ----------------------------------------------------------------------------------------------------
    def set_inputs(self, tokens, positions):
        """tokens / positions: per user (lists of `users` ints): the token at each user's position."""
        mesh = self.mesh
        ttnn.copy_host_to_device_tensor(
            _replicate(mesh, torch.tensor([tokens], dtype=torch.int32), ttnn.uint32, device=False), self.tokens
        )
        for g in range(self.G):
            p = torch.tensor(positions[g * U32 : (g + 1) * U32], dtype=torch.int32)
            ttnn.copy_host_to_device_tensor(_replicate(mesh, p, ttnn.int32, device=False), self.pos[g])
            ttnn.copy_host_to_device_tensor(
                _replicate(mesh, p.reshape(1, U32), ttnn.uint32, device=False), self.rot_idx[g]
            )

    def capture(self):
        """Capture one step as a trace (after a compile step: the eager step writes its K / V rows and advances the
        inputs, so set the inputs again before replaying)."""
        self.release()
        self.step().deallocate(True)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        self.out = self.step()
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        self.trace = tid
        hold_prefill_l1(self.model)

    def release(self):
        if self.trace is not None:
            ttnn.synchronize_device(self.mesh)
            ttnn.release_trace(self.mesh, self.trace)
        self.trace = None

    def generate(self, tokens, positions, steps):
        """steps traced steps from (tokens, positions), each step's tokens read back without stalling the queue (step s
        + 1 is queued before step s is read). Returns (per user the steps' tokens, the median seconds between steps'
        completions)."""
        self.set_inputs(tokens, positions)
        out = [[] for _ in range(self.users)]
        done = []
        pending = []

        def drain():
            host, ev = pending.pop(0)
            ttnn.event_synchronize(ev)
            done.append(time.perf_counter())
            ids = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1).long().tolist()
            for u in range(self.users):
                out[u].append(ids[u])

        for _ in range(steps):
            ttnn.execute_trace(self.mesh, self.trace, cq_id=0, blocking=False)
            pending.append((self.out.cpu(blocking=False), ttnn.record_event(self.mesh, 0)))
            if len(pending) > 1:
                drain()
        while pending:
            drain()
        gaps = sorted(b - a for a, b in zip(done, done[1:]))
        return out, gaps[len(gaps) // 2] if gaps else float("nan")


class VerifyPlain:
    """Plain greedy decode of `users` (at most 256) sequences as the DFlash verify pass with a block of one row per user
    (BatchedVerify, block 1): QKV, the attention (block attention, one row per user: every user in a few programs),
    o_proj, the residual + norm programs, the grouped MoE and the LM head each once over the users' rows."""

    def __init__(self, model, page_table, kv_cache, users, grouped=None):
        """page_table: host INT32 [users, blocks]; grouped: a GroupedMoE to share relaid weights with."""
        from ..model import create_rope_setup
        from .batched import BatchedVerify

        assert users % U32 == 0 and U32 <= users <= 256
        self.model, self.mesh = model, model.mesh_device
        mesh = self.mesh
        self.users = self.rows = users
        kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
        self.page_table = _replicate(mesh, page_table, ttnn.int32)
        self.verify = BatchedVerify(model, self.page_table, kv, users, 1, [], moe_copies=grouped is None)
        if grouped is None:
            from .grouped_moe import GroupedMoE

            self.verify.grouped = GroupedMoE(mesh, self.verify.moe, self.verify.hidden, rows=users)
            for W in self.verify.moe:
                for k in ("gate", "up", "down"):
                    W.pop(k).deallocate(True)
        else:
            self.verify.grouped = grouped_moe_for(model, users, share=grouped)
        self.rope = create_rope_setup(mesh, model.hf_config, max_local_batch_size=U32, datatype=ttnn.bfloat16)
        self.verify.trans_rows = self.rope.get_both_trans_mats()["prefill"]
        self.hd = model.layers[0].self_attn.config.head_dim
        # persistent per-step inputs: the tokens, the positions (INT32 [users]: the cache writes and the attention;
        # UINT32 [1, users]: the RoPE rows)
        self.tokens = _replicate(mesh, torch.zeros(1, users, dtype=torch.int32), ttnn.uint32)
        self.pos = _replicate(mesh, torch.zeros(users, dtype=torch.int32), ttnn.int32)
        self.pos_rows = _replicate(mesh, torch.zeros(1, users, dtype=torch.int32), ttnn.uint32)
        self.trace = None
        hold_prefill_l1(model)

    def step(self):
        rot = []
        for m in (self.rope.cos_matrix, self.rope.sin_matrix):
            t = ttnn.embedding(self.pos_rows, m, layout=ttnn.TILE_LAYOUT)
            rot.append(ttnn.reshape(t, [1, 1, self.rows, self.hd]))
        out, _ = self.verify.forward(self.tokens, [self.pos, self.pos], rot)
        for t in rot:
            t.deallocate(True)
        tok = self.verify.argmax.combine(out, self.mesh)
        for t in out:
            t.deallocate(True)
        row = ttnn.transpose(tok, -2, -1)
        nxt = ttnn.reshape(ttnn.to_layout(ttnn.typecast(row, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, self.rows])
        row.deallocate(True)
        ttnn.copy(nxt, self.tokens)
        nxt.deallocate(True)
        ttnn.plus_one(self.pos, skip_negative_entries=True)
        ttnn.plus_one(self.pos_rows)
        return tok

    def set_inputs(self, tokens, positions):
        mesh = self.mesh
        p = torch.tensor(positions, dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            _replicate(mesh, torch.tensor([tokens], dtype=torch.int32), ttnn.uint32, device=False), self.tokens
        )
        ttnn.copy_host_to_device_tensor(_replicate(mesh, p, ttnn.int32, device=False), self.pos)
        ttnn.copy_host_to_device_tensor(_replicate(mesh, p.reshape(1, -1), ttnn.uint32, device=False), self.pos_rows)

    capture = BigPlain.capture
    release = BigPlain.release
    generate = BigPlain.generate
