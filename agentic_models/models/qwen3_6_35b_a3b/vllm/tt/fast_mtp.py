# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MTP drafter for speculative decoding (QWEN_SPEC=K): the checkpoint's `mtp.*` layer on device.

MTP step for a pair (h_{t-1}, token_t), stored at position t (a constant +1 relabeling of the usual (h_t, token_{t+1})
pairing; attention only sees relative positions):
    x = fc([rmsnorm_e(embed(token_t)), rmsnorm_h(h_{t-1})])   (fc: 4096 -> 2048)
    y = full-attention MoE decoder layer (own paged KV cache, same page tables as the target)
    draft for token_{t+1} = argmax(lm_head(rmsnorm_f(y)))       (shared LM head)
and the next draft feeds (y, draft) at position t+1.

KV filling: prefill runs the MTP attention over every prompt position (hidden of the previous position, pending row
kept per slot across prefill chunks); decode-time rows are planned on the host each step (see `plan_rows`): after a
verify the committed columns, after an ordinary decode the step's own row (+ the prefill's pending row once).
Row inputs are selected on device by exact 0/1 matmuls from [verify hidden; ordinary hidden; pending hidden] and
[embed(tokens just sampled); embed(tokens the ordinary step consumed)], so no token or hidden state leaves the device.
"""

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
ROWS = 32


class FastMTP:
    def __init__(self, model, gen, kv_shape, kv_dtype):
        from .fast_generator import attach_fast_layers

        self.model = model
        self.gen = gen
        mesh = model.mesh
        ck = model.checkpoint
        sd = ck.select("mtp.layers.0.")
        index = next(
            i for i in range(model.config.num_hidden_layers) if model.config.layer_types[i] == "full_attention"
        )
        self.layer = model._build_layer(sd, index)
        attach_fast_layers(model, self.layer, sd)
        for name in ("router_bias", "router_indices", "router_out", "router_out_ids", "inline_selector"):
            if hasattr(self.layer, name):
                delattr(self.layer, name)
        rep = lambda t, dtype, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=layout,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.fc = rep(ck.tensor("mtp.fc.weight").T.reshape(1, 1, 4096, 2048), ttnn.bfloat16)
        self.norm_e = rep((ck.tensor("mtp.pre_fc_norm_embedding.weight").float() + 1).reshape(1, 1, 2048), ttnn.float32)
        self.norm_h = rep((ck.tensor("mtp.pre_fc_norm_hidden.weight").float() + 1).reshape(1, 1, 2048), ttnn.float32)
        self.norm_f = rep((ck.tensor("mtp.norm.weight").float() + 1).reshape(1, 1, 2048), ttnn.float32)
        self.eps = model.config.rms_norm_eps
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        # own paged KV cache (same page geometry as a target attention layer; the target's page tables are reused)
        target = next(states for states in gen.cache if "page_table" in states[0])
        context = target[0].get("context_length")
        first = self.layer.new_kv_state(context_length=context, num_physical_pages=kv_shape[0])
        self.kv = [first] + [
            self.layer.new_kv_state(
                context_length=context, num_physical_pages=kv_shape[0], kv_cache=(first["k"], first["v"])
            )
            for _ in range(ROWS - 1)
        ]
        z = lambda shape, dtype=ttnn.bfloat16: ttnn.zeros(
            shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
        )
        up = gen._up
        self.pending = z((1, 1, ROWS, 2048))  # per slot: final hidden of its last prefilled position
        self.ord_hidden = z((1, ROWS, 2048))  # ordinary decode: final hidden per slot (written by the decode trace)
        self.ord_in_tokens = up(torch.zeros((1, ROWS), dtype=torch.int32), ttnn.uint32)  # tokens it consumed
        self.hsel = up(torch.zeros(1, 1, ROWS, 96), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.esel = up(torch.zeros(1, 1, ROWS, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.gsel = up(torch.zeros(1, 1, ROWS, ROWS), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.cur_pos = [up(torch.full((ROWS,), -1, dtype=torch.int32), ttnn.int32) for _ in range(gen.spec.T)]
        self.rope_idx = [up(torch.zeros((1, ROWS), dtype=torch.int32), ttnn.uint32) for _ in range(gen.spec.T)]
        # per pass: page table rows, route activity (MoE) and attention masks (pass 0 rows = planned pairs,
        # pass k >= 1 rows = user/slot rows)
        self.page_table = [
            up(torch.zeros((ROWS, gen.page_width), dtype=torch.int32), ttnn.int32) for _ in range(gen.spec.K)
        ]
        self.active = [
            model.upload(torch.zeros(32, dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            for _ in range(gen.spec.K)
        ]
        self.attn_mask = [up(torch.zeros(1, ROWS, 4, 256), ttnn.bfloat16, ttnn.TILE_LAYOUT) for _ in range(gen.spec.T)]
        self.drafts = [up(torch.zeros((1, ROWS), dtype=torch.int32), ttnn.uint32) for _ in range(gen.spec.K)]
        self.drafts_all = up(torch.zeros((1, gen.spec.K * ROWS), dtype=torch.int32), ttnn.uint32)  # one readback
        vocab_local = model.vocab_size // 4 if model.vocab_size % 4 == 0 else None
        self.vocab_local = vocab_local or (248320 // 4)
        self.chip_offset = ttnn.from_torch(
            torch.arange(4, dtype=torch.float32).mul(self.vocab_local).reshape(4, 1, 1, 1),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        self._host_pages = [None] * gen.spec.K
        self.zeros32 = ttnn.zeros(
            (1, 1, ROWS, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM
        )
        self.zero_row = ttnn.zeros(
            (1, 1, 2048),
            dtype=model.boundary_dtypes["residual_dtype"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=DRAM,
        )

    # ------------------------------------------------------------------ pieces
    def _rms(self, x, w):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=w, compute_kernel_config=self.compute)

    def _input(self, e, h):
        rows = e.shape[-2]
        e = ttnn.reshape(e, (1, rows, 2048))
        h = ttnn.reshape(h, (1, rows, 2048))
        x = ttnn.concat([self._rms(e, self.norm_e), self._rms(h, self.norm_h)], dim=-1)  # [1, R, 4096]
        x = ttnn.linear(x, self.fc, dtype=ttnn.bfloat16, compute_kernel_config=self.compute, memory_config=DRAM)
        return ttnn.reshape(x, (1, rows, 2048))

    def _layer(self, x, k):
        cur_pos, rope_idx, attn_mask = self.cur_pos[k], self.rope_idx[k], self.attn_mask[k]
        from .fast_generator import fast_norm

        g, m, layer = self.gen, self.model, self.layer
        cos = ttnn.reshape(ttnn.embedding(rope_idx, m.rope_cos, layout=ttnn.TILE_LAYOUT), (1, ROWS, 1, 64))
        sin = ttnn.reshape(ttnn.embedding(rope_idx, m.rope_sin, layout=ttnn.TILE_LAYOUT), (1, ROWS, 1, 64))
        norm = fast_norm(layer, x, "input_layernorm.weight", sharded=True)
        partial = layer.fast_attn.decode(
            norm, self.kv[0], cur_pos=cur_pos, page_table=self.page_table[k], cos=cos, sin=sin, attn_mask=attn_mask
        )
        x = ttnn.to_memory_config(ttnn.add(x, layer._reduce(partial, batched=True)), DRAM)
        norm = fast_norm(layer, x, "post_attention_layernorm.weight", sharded=True)
        moe = layer._moe(
            norm,
            batched=True,
            route_activity_mask=self.active[k],
            route_slots=tuple(range(ROWS)),
            fuse_moe_route_plan=m.fuse_moe_route_plan,
        )
        return ttnn.to_memory_config(ttnn.add(x, layer._reduce(moe, batched=True)), DRAM)

    def _argmax(self, y, out):
        """Greedy token of lm_head(norm_f(y)) per row, across the 4 vocab shards, into `out` [1, 32] uint32."""
        m = self.model
        norm = ttnn.typecast(
            self._rms(ttnn.typecast(y, ttnn.float32), self.norm_f), m.boundary_dtypes["activation_dtype"]
        )
        logits = (
            m.dram_head(ttnn.reshape(norm, (1, 1, ROWS, 2048)))
            if m.dram_head is not None
            else ttnn.linear(norm, m.head_weight, compute_kernel_config=m.head_compute, memory_config=DRAM)
        )
        logits = ttnn.reshape(logits, (1, 1, ROWS, logits.shape[-1]))
        v, i = ttnn.topk(logits, k=1, dim=-1, sorted=True)
        # broadcast each chip's best (value, global id) over a full 32-wide tile, gather the 4 chips' tiles
        # ([1,1,32,128], tile aligned), take the best of 128 and read its id (fp32 holds ids < 2^24 exactly)
        vals = ttnn.add(self.zeros32, ttnn.typecast(v, ttnn.float32))
        gidx = ttnn.add(self.zeros32, ttnn.add(ttnn.typecast(i, ttnn.float32), self.chip_offset))
        ccl = m.workspace.ccl

        def gather(t):
            return ttnn.experimental.all_gather_async(
                t,
                dim=-1,
                cluster_axis=1,
                mesh_device=m.mesh,
                topology=ttnn.Topology.Linear,
                num_links=1,
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(),
                barrier_semaphore=ccl.get_barrier_semaphore(),
                memory_config=DRAM,
            )

        _, which = ttnn.topk(gather(vals), k=1, dim=-1, sorted=True)
        tok = ttnn.gather(gather(gidx), -1, index=ttnn.typecast(which, ttnn.uint32))
        tok = ttnn.typecast(tok, ttnn.uint32)
        ttnn.copy(ttnn.reshape(ttnn.to_layout(tok, ttnn.ROW_MAJOR_LAYOUT), (1, ROWS)), out)

    # ------------------------------------------------------------------ propose graph (traced)
    def propose_forward(self):
        """Pass 1 over host-planned rows (committed pairs) -> gather each user's last row -> K-1 more passes.
        Drafts land in self.drafts[k] (row u = user/slot u)."""
        g, m = self.gen, self.model
        spec = g.spec
        e_all = ttnn.concat(
            [
                ttnn.reshape(m.embed(ttnn.reshape(g.tokens, (1, ROWS))), (1, 1, ROWS, 2048)),
                ttnn.reshape(m.embed(self.ord_in_tokens), (1, 1, ROWS, 2048)),
            ],
            dim=2,
        )  # [1,1,64,2048]
        h_all = ttnn.concat(
            [
                ttnn.reshape(spec.hidden, (1, 1, ROWS, 2048)),
                ttnn.reshape(self.ord_hidden, (1, 1, ROWS, 2048)),
                self.pending,
            ],
            dim=2,
        )  # [1,1,96,2048]
        e = ttnn.matmul(self.esel, e_all, dtype=ttnn.bfloat16, compute_kernel_config=self.compute)
        h = ttnn.matmul(self.hsel, h_all, dtype=ttnn.bfloat16, compute_kernel_config=self.compute)
        y = self._layer(ttnn.reshape(self._input(e, h), (1, ROWS, 2048)), 0)
        # each user's newest row -> user row u (exact one-hot gather)
        y = ttnn.reshape(
            ttnn.matmul(
                self.gsel, ttnn.reshape(y, (1, 1, ROWS, 2048)), dtype=ttnn.bfloat16, compute_kernel_config=self.compute
            ),
            (1, ROWS, 2048),
        )
        self._argmax(y, self.drafts[0])
        for k in range(1, spec.K):
            e = ttnn.reshape(m.embed(self.drafts[k - 1]), (1, ROWS, 2048))
            y = self._layer(self._input(e, y), k)
            self._argmax(y, self.drafts[k])
        ttnn.copy(ttnn.concat(self.drafts, dim=-1), self.drafts_all)

    # ------------------------------------------------------------------ prefill: fill MTP KV for prompt positions
    def prefill(self, executor, source, token_ids, slot, start, length):
        """After the target layers of a prefill chunk: MTP pairs (h_{t-1}, tok_t) for t in [start, start+length)
        (t = 0 uses a zero hidden), attention KV only; keep h_{start+length-1} pending for the next chunk/decode."""
        from .fast_gdn import write_row

        g, m = self.gen, self.model
        layer = self.layer
        state = self.kv[slot]
        target = next(states for states in g.cache if "page_table" in states[0])
        ttnn.copy(target[slot]["page_table"], state["page_table"])
        CH = 256
        scratch = executor.residual[2] if source is not executor.residual[2] else executor.residual[0]
        for offset in range(0, length, CH):
            n = min(CH, length - offset)
            n32 = (n + 31) // 32 * 32
            host = torch.zeros((1, n32), dtype=torch.int32)
            host[0, :n] = token_ids[offset : offset + n]
            tok = ttnn.from_torch(
                host,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=m.mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(m.mesh),
            )
            e = ttnn.reshape(m.embed(tok), (1, n32, 2048))
            if offset == 0:
                # row 0 = hidden of position start-1 (pending from the previous chunk; zeros at position 0)
                hbuf = executor._empty(n32)
                if start > 0:
                    first = ttnn.reshape(
                        ttnn.to_layout(self.pending[:, :, slot : slot + 1, :], ttnn.ROW_MAJOR_LAYOUT), (1, 1, 2048)
                    )
                    first = ttnn.typecast(first, hbuf.dtype) if first.dtype != hbuf.dtype else first
                else:
                    first = self.zero_row
                executor.transport.rows(first, hbuf, count=1)
                executor.transport.rows(source, hbuf, source_row=0, destination_row=1, count=n - 1, extent=n32 - 1)
                h = ttnn.to_layout(hbuf, ttnn.TILE_LAYOUT)
            else:
                h = executor._read(source, offset - 1, n32, valid_rows=n)
            x = self._input(e, h)  # [1, n32, 2048]
            executor._write(x, scratch, offset, n)
        from .prefill import attention_chunks

        for offset, n, absolute in attention_chunks("full_attention", length, start):
            executor._attention(layer, state, scratch, offset, n, absolute, length)
        last = executor._read(source, length - 1, 1)
        write_row(self.pending, ttnn.reshape(last, (1, 1, 1, 2048)), slot, executor.consts)

    # ------------------------------------------------------------------ host planning for one propose
    def plan(self, rows, users):
        """rows: list of (row, hidden_src_index_in_96, embed_src_index_in_64, position, user);
        users: {user: (newest_row, next_position)}. Uploads selections, positions, masks, pages."""
        g = self.gen
        K = g.spec.K
        hsel = torch.zeros(ROWS, 96)
        esel = torch.zeros(ROWS, 64)
        gsel = torch.zeros(ROWS, ROWS)
        cur = [torch.full((ROWS,), -1, dtype=torch.int32) for _ in range(K)]
        rope = [torch.zeros((1, ROWS), dtype=torch.int32) for _ in range(K)]
        mask = [torch.zeros(1, ROWS, 4, 256) for _ in range(K)]
        pages = [torch.zeros((ROWS, g.page_width), dtype=torch.int32) for _ in range(K)]
        act = [torch.zeros(32, dtype=torch.int32) for _ in range(K)]
        for r, hs, es, pos, u in rows:
            hsel[r, hs] = 1
            esel[r, es] = 1
            cur[0][r] = pos
            rope[0][0, r] = pos
            mask[0][0, r] = 1
            pages[0][r] = g.f_host_pages[u]
            act[0][r] = 1
        for u, (newest, pos) in users.items():
            gsel[u, newest] = 1
            for k in range(1, K):
                cur[k][u] = pos + k - 1
                rope[k][0, u] = pos + k - 1
                mask[k][0, u] = 1
                pages[k][u] = g.f_host_pages[u]
                act[k][u] = 1
        g._put(hsel.reshape(1, 1, ROWS, 96).bfloat16(), self.hsel)
        g._put(esel.reshape(1, 1, ROWS, 64).bfloat16(), self.esel)
        g._put(gsel.reshape(1, 1, ROWS, ROWS).bfloat16(), self.gsel)
        for k in range(K):
            g._put(cur[k], self.cur_pos[k])
            g._put(rope[k], self.rope_idx[k])
            g._put(mask[k].bfloat16(), self.attn_mask[k])
            g._put(act[k], self.active[k])
            if self._host_pages[k] is None or not torch.equal(pages[k], self._host_pages[k]):
                g._copy_host(pages[k], self.page_table[k])
                self._host_pages[k] = pages[k]
