# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP speculative decoding for Ornith serving (opt-in, ORNITH_MTP=K).

Drafter: the checkpoint's mtp.* head (one full-attention decoder layer built by the model's own MultichipDecoder from
mtp.layers.0.*, its own paged KV pool addressed by the request's vLLM block table, fc split into its embedding and
hidden halves, mtp.norm, the shared decode LM head), greedy drafts from a second device sampler pinned to greedy.

Verify of B users x T = 1+K candidates (rows u*T + t; user u = pool slot u): dense ops on the R = B*T rows of the decode
tile, attention as T rows per user sharing that user's page table (KV written one candidate column at a time:
`cache_update_positions`, see TTNN limit L12), GDN layers through the chained programs (tt/spec_chain.py) whose
per-candidate states stay in chain buffers until commit. Target tokens per row come from the generator's own device
sampler, so a greedy row uses the plain decode's greedy pick.

Per step (synchronous scheduling): verify -> [runner accept walk] -> propose = commit the accepted candidate per user,
MTP catch-up over the accepted rows (the verify's target hidden), K-1 chained MTP drafts. After an ordinary decode the
catch-up uses the plain decode's hidden (captured by the decode traces) and, right after a prefill, the prompt's last
hidden. The drafter's prompt KV is written right after every prefill segment from the prefill staging rows.
Every device phase is a trace per (bucket, T).
"""
from types import SimpleNamespace

import torch

import ttnn
from agentic_models.models.ornith_1_5_9b.vllm.tt.multichip_decoder import MultichipDecoder
from agentic_models.models.ornith_1_5_9b.vllm.tt.precision import layer_policy
from agentic_models.models.ornith_1_5_9b.vllm.tt.spec_chain import conv_gates_chain, gdn_chain, gdn_commit

BUCKETS = (1, 2, 4, 8, 16)


def bucket_of(users):
    return next(b for b in BUCKETS if b >= users)


class MTPDrafter:
    def __init__(self, model, cache_kwargs):
        m = self.model = model
        ck = m.checkpoint
        full = m.config.layer_types.index("full_attention")
        self.layer = MultichipDecoder.from_state_dict(
            ck.tensors("mtp.layers.0."),
            hf_config=m.config,
            layer_idx=full,
            mesh_device=m.device,
            collectives=m.collectives,
            precision_policy=layer_policy(m.precision_policy, full),
        )
        self.cache = self.layer.allocate_batch_state(**cache_kwargs)

        def norm(key):
            return m.upload((ck.tensor(key).float() + 1).reshape(1, 1, 1, -1))

        self.norm_e = norm("mtp.pre_fc_norm_embedding.weight")
        self.norm_h = norm("mtp.pre_fc_norm_hidden.weight")
        self.norm_out = norm("mtp.norm.weight")
        fc = ck.tensor("mtp.fc.weight").float()  # [4096 out, 8192 in] = [embedding half | hidden half]
        self.fc_e = m.upload(fc[:, :4096].T.contiguous(), dim=1)
        self.fc_h = m.upload(fc[:, 4096:].T.contiguous(), dim=1)
        self.fc_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def _norm(self, x, w, sharded=True):
        m = self.model
        if not sharded:
            return ttnn.rms_norm(x, weight=w, epsilon=m.config.rms_norm_eps, compute_kernel_config=m.norm_compute)
        return ttnn.rms_norm(
            ttnn.to_memory_config(x, m.collectives.ar_mem),
            weight=w,
            epsilon=m.config.rms_norm_eps,
            program_config=m.layers[0].norm_program,
            compute_kernel_config=m.norm_compute,
            memory_config=m.collectives.ar_mem,
        )

    def target_norm(self, hidden, sharded=True):
        """The target's final norm (model.logits applies it before the LM head)."""
        return self._norm(hidden, self.model.norm, sharded)

    def inputs(self, embedded, hidden_normed, rows, sharded=True):
        """fc([norm_e(embed(token)), norm_h(target hidden)]) -> replicated [1,1,rows,4096]."""
        e = self._norm(embedded, self.norm_e, sharded)
        h = self._norm(hidden_normed, self.norm_h, sharded)
        de = ttnn.to_memory_config(e, ttnn.DRAM_MEMORY_CONFIG)
        dh = ttnn.to_memory_config(h, ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.add(
            ttnn.linear(de, self.fc_e, compute_kernel_config=self.fc_compute),
            ttnn.linear(dh, self.fc_h, compute_kernel_config=self.fc_compute),
        )
        x = self.layer.gather(
            ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG if not sharded else ttnn.L1_MEMORY_CONFIG)
        )
        return x[:, :, :rows, :] if x.shape[2] != rows else x

    def decode(self, tokens, hidden, positions, tables, rows, update_positions=None):
        """rows R of drafts (several rows of one user share its table; KV written per column)."""
        m, layer = self.model, self.layer
        x = ttnn.to_memory_config(self.inputs(m.embed(tokens, batch=rows), hidden, rows), layer.ar_mem)
        packed = layer.decode_linear(layer.decode_norm(x, "input_layernorm"), "packed_input")
        local = layer.full_attention_decode_batch(
            packed, [self.cache[0]] * rows, [positions], tables, masked=True, cache_update_positions=update_positions
        )
        out = layer.decode_finish(x, layer.decode_reduce(layer.decode_linear(local, "self_attn.o_proj")))
        hn = self._norm(out, self.norm_out)
        return m.decode_head(hn), ttnn.to_memory_config(hn, ttnn.DRAM_MEMORY_CONFIG)

    def kv_rows(self, tokens, hidden_normed, positions, tables):
        """32 prompt rows (positions in 32 distinct KV pages, -1 = skip): write the drafter's K/V only."""
        m, layer = self.model, self.layer
        x = ttnn.to_memory_config(self.inputs(m.embed(tokens, batch=32), hidden_normed, 32), layer.ar_mem)
        packed = layer.decode_linear(layer.decode_norm(x, "input_layernorm"), "packed_input")
        layer.full_attention_decode_batch(packed, [self.cache[0]] * 32, [positions], tables, masked=True, kv_only=True)


class SpecEngine:
    def __init__(self, gen, state, cache_kwargs, K, hdec):
        from agentic_models.models.ornith_1_5_9b.vllm.tt.generator import (
            DEVICE_SAMPLING_MAX_TOP_K,
            SamplingCCL,
            SamplingGenerator,
            SamplingParams,
            format_sampling_params,
        )

        self.gen, self.state, self.K = gen, state, K
        self.model, self.cache, self.dev = gen.model, gen.cache, gen.device
        self.capacity = len(self.cache[0])
        self.drafter = MTPDrafter(self.model, cache_kwargs)
        self.linear = [i for i, layer in enumerate(self.model.layers) if layer.kind == "linear_attention"]
        self.chain = {}
        for i in self.linear:
            st = self.cache[i][0]
            rec = ttnn.zeros(
                (8 * 32, 128, 128),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.chain[i] = (rec, ttnn.zeros_like(st.conv))
        u32 = lambda shape: gen._integer(torch.zeros(shape, dtype=torch.int32))
        i32 = lambda shape: gen._integer(torch.full(shape, -1, dtype=torch.int32), signed=True)
        self.v_tok, self.d_tok, self.x_tok = u32((1, 1, 1, 32)), u32((1, 1, 1, 32)), u32((1, 1, 1, 32))
        self.v_out, self.d_out, self.x_out = u32((1, 1, 1, 32)), u32((1, 1, 1, 32)), u32((1, 1, 1, 32))
        self.v_pos, self.d_pos, self.x_pos = i32((32,)), i32((32,)), i32((32,))
        self.v_upd = [i32((32,)) for _ in range(K + 1)]
        self.d_upd = [i32((32,)) for _ in range(K + 1)]
        self.commit_index = ttnn.from_torch(
            torch.full((1, 32), -1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.dev),
        )
        self.g_index = gen._integer(torch.zeros((1, 32), dtype=torch.int32))
        z = lambda: ttnn.from_torch(
            torch.zeros(1, 1, 32, 4096),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.dev),
        )
        self.H = z()  # catch-up target hidden (normed), row u*T + j
        self.G = z()  # chained-draft MTP hidden, row u
        self.hdec = hdec  # plain decode's normed hidden, row = slot (model.hidden_sink, set before decode warmup)
        self.hpend = z()  # prompt's last normed hidden, row = slot
        # device tables for the drafter's prompt KV writes (one per slot, host-written at prefill)
        pages = state.tables[0].shape[-1]
        self.prefill_table = gen._integer(torch.zeros((1, pages), dtype=torch.int32), signed=True)
        self.p_tok, self.p_pos = u32((1, 1, 1, 32)), i32((32,))
        self.prefill_rows = gen._integer(torch.zeros((1, 32), dtype=torch.int32))
        mk = lambda: SamplingGenerator(
            args=SimpleNamespace(
                vocab_size=self.model.vocab_size,
                padded_vocab_size=self.model.vocab_size,
                max_batch_size=32,
                max_top_k=DEVICE_SAMPLING_MAX_TOP_K,
                cluster_shape=(1, 4),
                sampling_all_gather_axis=1,
            ),
            mesh_device=self.dev,
            tt_ccl=SamplingCCL(self.model),
        )
        # verify rows: lane r = candidate r % T of user r // T, under that user's own temperature / top-k / top-p,
        # an independent draw per column (spec_ids_are_sampled); its seed stream advances every verify
        self.verify_sampler = mk()
        self._format = lambda params: format_sampling_params(params, 32, max_top_k=DEVICE_SAMPLING_MAX_TOP_K)
        self._SamplingParams = SamplingParams
        self.verify_sampler.reset_sampling_params(self._format(SamplingParams(temperature=0.0, top_k=1, top_p=1.0)))
        import random

        self.verify_sampler.seed_manager.write_device_seed_values([random.randrange(1, 500_000) for _ in range(32)])
        self._verify_params_key = None
        self.draft_sampler = SamplingGenerator(
            args=SimpleNamespace(
                vocab_size=self.model.vocab_size,
                padded_vocab_size=self.model.vocab_size,
                max_batch_size=32,
                max_top_k=DEVICE_SAMPLING_MAX_TOP_K,
                cluster_shape=(1, 4),
                sampling_all_gather_axis=1,
            ),
            mesh_device=self.dev,
            tt_ccl=SamplingCCL(self.model),
        )
        self.draft_sampler.reset_sampling_params(
            format_sampling_params(
                SamplingParams(temperature=0.0, top_k=1, top_p=1.0), 32, max_top_k=DEVICE_SAMPLING_MAX_TOP_K
            )
        )
        self.traces = {}
        self.mtp_end = {}  # slot -> first position without drafter KV
        self.has_pend = set()  # slots whose hpend row holds the prompt's last hidden at mtp_end
        self.last_input = {}  # slot -> (token, position) of the last ordinary decode input
        self.pending = None  # last verify: (b, T, start positions per user)
        self.stats = dict(verifies=0, proposals=0, committed=0, drafts=0, accepted=0, plain_catchups=0)

    # ---------------- helpers
    def _write(self, tensor, values, signed):
        self.gen._copy(
            torch.as_tensor(values, dtype=torch.int32).reshape(tuple(tensor.shape)),
            tensor,
            "position_refreshes" if signed else "token_refreshes",
            signed=signed,
        )

    def _read(self, tensor, n):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1)[:n].to(torch.int64)

    def _tables(self, b, T):
        return [self.state.tables[u] for u in range(b) for _ in range(T)]

    def _pad32(self, logits):
        return (
            ttnn.pad(logits, [(0, 0), (0, 0), (0, 32 - logits.shape[2]), (0, 0)], 0.0)
            if logits.shape[2] < 32
            else logits
        )

    def _sample_target(self, logits, out):
        self.verify_sampler._run_sampling(
            self._pad32(logits), penalties_on=False, grammar_on=False, tt_out_tok=out, count_tokens=False
        )
        ttnn.plus_one(self.verify_sampler.tt_sampling.seeds_tt_tensor, skip_negative_entries=True)

    def set_verify_sampling(self, params, b, T):
        """Lay the per-slot request params out over the verify rows (row u*T + t -> slot u)."""

        def lanes(value, default):
            values = value if isinstance(value, (list, tuple)) else [value] * 32
            return [values[r // T] if r < b * T and r // T < len(values) else default for r in range(32)]

        if params is None:
            fields = dict(temperature=[0.0] * 32, top_k=[1] * 32, top_p=[1.0] * 32)
        else:
            fields = dict(
                temperature=lanes(params.temperature, 0.0), top_k=lanes(params.top_k, 1), top_p=lanes(params.top_p, 1.0)
            )
        key = tuple(tuple(v) for v in fields.values())
        if key != self._verify_params_key:
            self.verify_sampler.reset_sampling_params(self._format(self._SamplingParams(**fields)))
            self._verify_params_key = key

    def _sample_draft(self, logits, out):
        self.draft_sampler._run_sampling(
            self._pad32(logits), penalties_on=False, grammar_on=False, tt_out_tok=out, count_tokens=False
        )

    def _rm_rows(self, x, dst, rows):
        """copy [1,1,rows,4096] (any layout/memory) into rows 0..rows-1 of the RM DRAM buffer dst"""
        x = ttnn.to_layout(ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(x, dst, (0, 0, 0, 0), (1, 1, rows, 4096), (1, 1, 1, 1))

    def _tile_rows(self, src, rows):
        return ttnn.to_layout(src[:, :, :rows, :], ttnn.TILE_LAYOUT)

    # ---------------- device phases (traced per (bucket, T))
    def _verify_fn(self, b, T):
        m, R = self.model, b * T
        hidden = m.embed(self.v_tok, batch=R)
        for index, (layer, states) in enumerate(zip(m.layers, self.cache)):
            x = ttnn.to_memory_config(hidden, layer.ar_mem)
            packed = layer.decode_linear(layer.decode_norm(x, "input_layernorm"), "packed_input")
            if layer.kind == "full_attention":
                local = layer.full_attention_decode_batch(
                    packed,
                    [states[u] for u in range(b) for _ in range(T)],
                    [self.v_pos[:R]],
                    self._tables(b, T),
                    masked=True,
                    cache_update_positions=[t[:R] for t in self.v_upd[:T]],
                )
                name = "self_attn.o_proj"
            else:
                st = states[0]
                rec, hist = self.chain[index]
                conv, g, beta = conv_gates_chain(packed, st.conv, hist, layer.float_taps, layer.dt_bias, layer.decay, T)
                pool = ttnn.reshape(st.recurrent, (st.recurrent.shape[0] * 8, 128, 128))
                local = gdn_chain(
                    conv,
                    g,
                    beta,
                    pool,
                    rec,
                    packed,
                    layer.weights["linear_attn.norm.weight"],
                    layer.config.rms_norm_eps,
                    T,
                )
                name = "linear_attn.out_proj"
            hidden = layer.decode_finish(x, layer.decode_reduce(layer.decode_linear(local, name)))
        hn = self.drafter.target_norm(hidden)
        self._sample_target(m.decode_head(hn), self.v_out)
        self._rm_rows(hn, self.H, R)

    def _commit_fn(self, b, T):
        layers = []
        for i in self.linear:
            st = self.cache[i][0]
            rec, hist = self.chain[i]
            layers.append((rec, ttnn.reshape(st.recurrent, (st.recurrent.shape[0] * 8, 128, 128)), hist, st.conv))
        gdn_commit(self.commit_index, layers, T, b)

    def _catchup_fn(self, b, T):
        R = b * T
        logits, hn = self.drafter.decode(
            self.d_tok,
            self._tile_rows(self.H, R),
            self.d_pos[:R],
            self._tables(b, T),
            R,
            update_positions=[t[:R] for t in self.d_upd[:T]],
        )
        self._sample_draft(logits, self.d_out)
        g = ttnn.embedding(self.g_index[:, :b], ttnn.to_layout(hn, ttnn.ROW_MAJOR_LAYOUT), layout=ttnn.ROW_MAJOR_LAYOUT)
        self._rm_rows(ttnn.reshape(g, (1, 1, b, 4096)), self.G, b)

    def _chain_fn(self, b):
        logits, hn = self.drafter.decode(
            self.x_tok, self._tile_rows(self.G, b), self.x_pos[:b], [self.state.tables[u] for u in range(b)], b
        )
        self._sample_draft(logits, self.x_out)
        self._rm_rows(hn, self.G, b)

    def _run(self, key, fn):
        trace = self.traces.get(key)
        if trace is None:
            trace = ttnn.begin_trace_capture(self.dev, cq_id=0)
            fn()
            ttnn.end_trace_capture(self.dev, trace, cq_id=0)
            self.traces[key] = trace
        ttnn.execute_trace(self.dev, trace, cq_id=0, blocking=False)

    def warm(self, buckets_T, capture):
        """Phase 1 (capture=False): compile every phase eagerly with all rows inactive (positions -1: no KV / state
        writes), including the prompt K/V rows. Phase 2: capture the traces (program cache already warm)."""
        self._write(self.commit_index, [-1] * 32, signed=True)
        for b, T in buckets_T:
            for key, fn in (
                (("v", b, T), lambda: self._verify_fn(b, T)),
                (("c", b, T), lambda: self._commit_fn(b, T)),
                (("d", b, T), lambda: self._catchup_fn(b, T)),
                (("x", b), lambda: self._chain_fn(b)),
            ):
                if capture:
                    self._run(key, fn)
                else:
                    fn()
        if not capture:
            self._kv_call([], [self.prefill_table] * 32, source=self.gen.prefill_runtime.staging[0])
            self._kv_call([], [self.prefill_table] * 32, hpend=True)
            self._gather_plain_hidden([])
            self.remap(list(range(self.capacity)))
            zero = ttnn.to_layout(
                ttnn.to_memory_config(self.hpend[:, :, :1, :], ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT
            )
            for slot in range(self.capacity):
                ttnn.experimental.slice_write(zero, self.hpend, (0, 0, slot, 0), (1, 1, slot + 1, 4096), (1, 1, 1, 1))
        ttnn.synchronize_device(self.dev)

    # ---------------- page tables (verify/propose bypass the generator's decode reload)
    def refresh_tables(self, page_table):
        st = self.state
        for row in range(min(page_table.shape[0], self.capacity)):
            source = page_table[row].to(torch.int32)
            if not torch.equal(source, st.host_tables[row]):
                self.gen._copy(source[None], st.tables[row], "page_table_refreshes", signed=True)
                st.host_tables[row] = source.clone()

    # ---------------- serving API
    def verify(self, tokens, start_pos, page_table, sampling_params=None, num_valid=None):
        """tokens / start_pos [B, 1+K] (-1 = placeholder / padding). Returns target picks [B, 1+K] (int32)."""
        B, T_full = tokens.shape
        live = [u for u in range(B) if int(start_pos[u, 0]) >= 0]
        b = bucket_of(max(live) + 1 if live else 1)
        # the call is always 1+K wide; k_by_rows only caps each row's valid drafts, so verify T = min(1+K, 32/b)
        # columns (the decode tile holds b*T rows); a row with more valid drafts than that cannot be verified
        T = min(T_full, 32 // b)
        if num_valid is not None and any(int(num_valid[u]) > T - 1 for u in live):
            raise RuntimeError(f"MTP verify: {b} users x more than {T - 1} drafts exceed the 32-row decode tile")
        R = b * T
        self.refresh_tables(page_table)
        self.set_verify_sampling(sampling_params, b, T)
        tok = torch.zeros(32, dtype=torch.int32)
        pos = torch.full((32,), -1, dtype=torch.int32)
        for u in live:
            for t in range(T):
                if int(start_pos[u, t]) >= 0 and int(tokens[u, t]) >= 0:
                    tok[u * T + t] = int(tokens[u, t])
                    pos[u * T + t] = int(start_pos[u, t])
        self._write(self.v_tok, tok, signed=False)
        self._write(self.v_pos, pos, signed=True)
        for t in range(T):
            col = torch.full((32,), -1, dtype=torch.int32)
            col[t:R:T] = pos[t:R:T]
            self._write(self.v_upd[t], col, signed=True)
        self._run(("v", b, T), lambda: self._verify_fn(b, T))
        out = self._read(self.v_out, R)
        ids = torch.zeros((B, T_full), dtype=torch.int32)
        for u in range(min(B, b)):
            ids[u, :T] = out[u * T : (u + 1) * T].to(torch.int32)
        self.pending = (b, T, {u: int(start_pos[u, 0]) for u in live})
        self.stats["verifies"] += 1
        return ids

    def note_plain_inputs(self, tokens, start_pos):
        """An ordinary decode is about to run: remember each live slot's input token and position."""
        self.pending = None
        flat = tokens.reshape(tokens.shape[0], -1)[:, 0]
        pos = start_pos.reshape(start_pos.shape[0], -1)[:, 0]
        for u in range(min(len(flat), self.capacity)):
            if int(pos[u]) >= 0:
                self.last_input[u] = (int(flat[u]), int(pos[u]))

    def propose(self, K, committed_tokens, committed_positions, accepted_counts, T_default):
        B = committed_tokens.shape[0]
        live = [u for u in range(B) if int(committed_positions[u, 0]) >= 0]
        drafts = torch.zeros((B, K), dtype=torch.int32)
        num_valid = torch.zeros(B, dtype=torch.int32)
        if not live:
            return drafts, num_valid
        tok = torch.zeros(32, dtype=torch.int32)
        dpos = torch.full((32,), -1, dtype=torch.int32)
        last = {}
        if self.pending is not None:
            b, T, starts = self.pending
            index = torch.full((1, 32), -1, dtype=torch.int32)
            for u in live:
                if u not in starts:
                    continue
                a = int(accepted_counts[u]) - 1
                index[0, u] = a
                P = starts[u]
                for j in range(a + 1):
                    tok[u * T + j] = int(committed_tokens[u, j])
                    dpos[u * T + j] = P + j
                last[u] = (u * T + a, P + a)
                self.stats["committed"] += a + 1
                self.stats["accepted"] += a
            self._write(self.commit_index, index, signed=True)
            self._run(("c", b, T), lambda: self._commit_fn(b, T))
        else:
            b = bucket_of(max(live) + 1)
            T = min(T_default, 32 // b)
            rows = []  # (dst row, source table row (0..31 hdec, 32..63 hpend))
            for u in live:
                P = int(committed_positions[u, 0]) - 1  # the ordinary step's input position
                j = 0
                if u in self.has_pend and self.mtp_end.get(u) == P - 1 and u in self.last_input:
                    tok[u * T] = self.last_input[u][0]
                    dpos[u * T] = P - 1
                    rows.append((u * T, 32 + u))
                    j = 1
                tok[u * T + j] = int(committed_tokens[u, 0])
                dpos[u * T + j] = P
                rows.append((u * T + j, u))
                last[u] = (u * T + j, P)
                self.has_pend.discard(u)
            self._gather_plain_hidden(rows)
            self.stats["plain_catchups"] += 1
        R = b * T
        self._write(self.d_tok, tok, signed=False)
        self._write(self.d_pos, dpos, signed=True)
        for t in range(T):
            col = torch.full((32,), -1, dtype=torch.int32)
            col[t:R:T] = dpos[t:R:T]
            self._write(self.d_upd[t], col, signed=True)
        gidx = torch.zeros((1, 32), dtype=torch.int32)
        for u, (row, _) in last.items():
            gidx[0, u] = row
        self._write(self.g_index, gidx, signed=False)
        self._run(("d", b, T), lambda: self._catchup_fn(b, T))
        out = self._read(self.d_out, R)
        cur = {u: int(out[row]) for u, (row, _) in last.items()}
        for u in last:
            drafts[u, 0] = cur[u]
        k_eff = min(K, 32 // b - 1)  # the next verify's columns (k_by_rows caps drafts the same way)
        for k in range(1, k_eff):
            xt = torch.zeros(32, dtype=torch.int32)
            xp = torch.full((32,), -1, dtype=torch.int32)
            for u, (_, q) in last.items():
                xt[u] = cur[u]
                xp[u] = q + k
            self._write(self.x_tok, xt, signed=False)
            self._write(self.x_pos, xp, signed=True)
            self._run(("x", b), lambda: self._chain_fn(b))
            o = self._read(self.x_out, b)
            for u in last:
                cur[u] = int(o[u])
                drafts[u, k] = cur[u]
        for u, (_, q) in last.items():
            self.mtp_end[u] = q + 1
            num_valid[u] = k_eff
        self.pending = None
        self.stats["proposals"] += 1
        self.stats["drafts"] += k_eff * len(last)
        return drafts, num_valid

    def _gather_plain_hidden(self, rows):
        """H rows <- plain decode hidden (hdec, row slot) or the prompt's last hidden (hpend, row slot): eager."""
        table = ttnn.concat([ttnn.reshape(self.hdec, (32, 4096)), ttnn.reshape(self.hpend, (32, 4096))], dim=0)
        idx = torch.zeros((1, 32), dtype=torch.int32)
        for dst, src in rows:
            idx[0, dst] = src
        index = self.gen._integer(idx)
        g = ttnn.embedding(index, table, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, 32, 4096]
        ttnn.copy(ttnn.reshape(g, (1, 1, 32, 4096)), self.H)

    # ---------------- prompt: drafter KV + last hidden, after each prefill segment
    def after_prefill_segment(self, runtime, source, slot, ids, start, end, page_row):
        """ids = tokens of [start, end); source = staging with the final (pre-norm) hidden at absolute rows.
        Drafter K/V for prompt positions i (token i+1, hidden i) in fixed 32-row calls whose rows are 32 positions
        apart (distinct KV pages: one fused cache write per call is safe); the last row's hidden is kept (hpend)."""
        pages = self.prefill_table.shape[-1]
        row = torch.zeros(pages, dtype=torch.int32)
        n = min(pages, len(page_row))
        row[:n] = torch.as_tensor(page_row[:n], dtype=torch.int32)
        self._write(self.prefill_table, row[None], signed=True)
        tables = [self.prefill_table] * 32
        if start > 0 and slot in self.has_pend and self.mtp_end.get(slot) == start - 1:
            # pending row start-1 from the previous segment: hidden from hpend, token = first token of this segment
            self._kv_call([(start - 1, int(ids[0]), 32 + slot)], tables, hpend=True)
        positions = list(range(start, end - 1))
        for base in range(0, len(positions), 1024):
            block = positions[base : base + 1024]
            for c in range(32):
                rows = [(i, int(ids[i + 1 - start]), i) for i in block[c::32]]
                if rows:
                    self._kv_call(rows, tables, source=source)
        last = ttnn.reshape(
            ttnn.embedding(self._prefill_index([end - 1] * 32), source, layout=ttnn.TILE_LAYOUT), (1, 1, 32, 4096)
        )
        hn_last = self.drafter.target_norm(last, sharded=False)
        x = ttnn.to_layout(ttnn.to_memory_config(hn_last, ttnn.DRAM_MEMORY_CONFIG), ttnn.ROW_MAJOR_LAYOUT)[:, :, :1, :]
        ttnn.experimental.slice_write(x, self.hpend, (0, 0, slot, 0), (1, 1, slot + 1, 4096), (1, 1, 1, 1))
        self.mtp_end[slot] = end - 1
        self.has_pend.add(slot)

    def _kv_call(self, rows, tables, source=None, hpend=False):
        """rows: (position, token at position+1, hidden source row); <= 32 rows in distinct KV pages."""
        toks = torch.zeros((1, 1, 1, 32), dtype=torch.int32)
        pos = torch.full((32,), -1, dtype=torch.int32)
        src = [0] * 32
        for k, (p, t, r) in enumerate(rows):
            toks.reshape(-1)[k], pos[k], src[k] = t, p, r
        self._write(self.p_tok, toks, signed=False)
        self._write(self.p_pos, pos, signed=True)
        if hpend:
            table = ttnn.concat([ttnn.reshape(self.hdec, (32, 4096)), ttnn.reshape(self.hpend, (32, 4096))], dim=0)
            hid = ttnn.to_layout(
                ttnn.reshape(
                    ttnn.embedding(self._prefill_index(src), table, layout=ttnn.ROW_MAJOR_LAYOUT), (1, 1, 32, 4096)
                ),
                ttnn.TILE_LAYOUT,
            )
            hn = hid  # hpend already holds the normed hidden
        else:
            hid = ttnn.reshape(
                ttnn.embedding(self._prefill_index(src), source, layout=ttnn.TILE_LAYOUT), (1, 1, 32, 4096)
            )
            hn = self.drafter.target_norm(hid, sharded=False)
        self.drafter.kv_rows(self.p_tok, hn, self.p_pos, tables)

    def _prefill_index(self, rows):
        self._write(self.prefill_rows, torch.as_tensor(rows, dtype=torch.int32).reshape(1, 32), signed=False)
        return self.prefill_rows

    def remap(self, remap):
        """slot_remap: row i now holds what slot remap[i] held (the generator moves the GDN state the same way)."""
        remap = [int(x) for x in remap]
        self.mtp_end = {i: self.mtp_end[o] for i, o in enumerate(remap) if o in self.mtp_end}
        self.has_pend = {i for i, o in enumerate(remap) if o in self.has_pend}
        self.last_input = {i: self.last_input[o] for i, o in enumerate(remap) if o in self.last_input}
        idx = torch.arange(32, dtype=torch.int32)[None].clone()
        idx[0, : len(remap)] = torch.as_tensor(remap, dtype=torch.int32)
        g = ttnn.embedding(self.gen._integer(idx), ttnn.reshape(self.hpend, (32, 4096)), layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy(ttnn.reshape(g, (1, 1, 32, 4096)), self.hpend)

    def release(self, slot):
        self.mtp_end.pop(slot, None)
        self.has_pend.discard(slot)
        self.last_input.pop(slot, None)
