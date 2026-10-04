# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1-Flash on the 4x8 Blackhole galaxy as ONE model object with GPT-OSS-style entry points (``prefill_forward``,
``prepare_inputs_prefill``, ``decode_forward``, ``prepare_inputs_decode``), built once and used for prefill AND decode over the SAME paged KV pool.

Layering: demo/text_demo.py -> tt/generator.py (``Generator``) -> this ``Model`` -> tt/layer.py (``DSV41Layer``), tt/paged_attention.py, tt/prefill_*.py.
The hand-off prefill -> decode is in tt/prefill_handoff.py (ring rows, compressed latents, ratio-2 compressor state, first token) and here (Engram token
history, per-user positions, device token feedback).
"""

import functools
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, HostEngramRows
from models.demos.blackhole.deepseek_v41_flash.tt.engram_ragged import RaggedNgramHash
from models.demos.blackhole.deepseek_v41_flash.tt.layer import DSV41Layer
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import (
    DSV41PagedAttention,
    DSV41PagedCompressedAttention,
    DSV41PagedStepState,
)
from models.demos.blackhole.deepseek_v41_flash.tt.paged_ops import PAGE_TOKENS, PagedKVPool
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention, clear_chunk_caches
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_handoff import GenPrefillModel, PagedStateSink
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillLayer, DSV41PrefillMoE
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_model import T

GATE_CUTOFFS = json.loads((Path(__file__).resolve().parents[1] / "configs" / "gate_cutoffs.json").read_text())


class Model:
    def __init__(self, mesh_device, args, max_ctx, num_pages=None, kv_dtype=ttnn.bfloat16, log=print):
        """args: ``DSV41ModelArgs`` (layer ids, users per row). ``max_ctx``: longest context (prompt + generated) of any user. ``num_pages``: pages of
        128 tokens PER MESH ROW shared by the users of that row (default users_per_row * ceil(max_ctx / 128))."""
        self.md, self.args, self.log = mesh_device, args, log
        self.rows, self.cols = tuple(mesh_device.shape)
        self.U = args.users_per_row
        self.B = self.rows * self.U
        self.max_ctx = max_ctx
        self.layer_ids = list(args.layer_ids)
        self.timing = {}
        ratios = {R.model_args().compress_ratios[L] for L in self.layer_ids}
        dl = 512 if 1 in ratios else 1024 if 2 in ratios else 1 << 30
        ui = os.environ.get("DSV41_INDEXER", "auto")
        self.use_indexer = (
            (max_ctx > dl) if ui == "auto" else ui == "1"
        )  # indexer top-512 (decode) + sparse prefill: needed beyond 512 compressed entries
        self.dec_idx, self.index_owner = {}, {}
        pages_per_user = -(-(max_ctx + 128) // PAGE_TOKENS)
        self.num_pages = num_pages or self.U * pages_per_user
        t0 = time.time()
        chain = DSV41DecodeChain(mesh_device, users_per_row=self.U, log=log)  # mesh config, CCL, shared MoE buffers
        self.chain = chain
        self.mc, self.ccl = chain.mesh_config, chain.ccl
        self.pool = PagedKVPool(
            mesh_device,
            self.U,
            self.num_pages,
            len(self.layer_ids),
            max_ctx + 128,
            ring_rows=int(
                os.environ.get("DSV41_RING_ROWS", "128")
            ),  # 160 = window + speculative-decoding slack (tt/spec_paged.py RING_SPEC)
            dtype=kv_dtype,
        )
        self.sink = PagedStateSink(mesh_device, self.pool, self.U)
        self.sources, self.attns, self.built, self.step_groups, pls = {}, {}, [], {}, []
        sh = _Shards()
        pool = ThreadPoolExecutor(max_workers=2)
        futs = {}
        submit = (
            lambda L: futs.setdefault(L, pool.submit(load_layer, L, True, max_ctx + 128, self.use_indexer))
            if L in self.layer_ids
            else None
        )
        for L in self.layer_ids[:2]:
            submit(L)
        first_pmoe = None
        for i, L in enumerate(self.layer_ids):
            submit(L + 1), submit(L + 2)
            w = futs.pop(L).result()
            attn = self._build_attention(L, w, slot=i)
            layer = DSV41Layer(
                mesh_device,
                self.mc,
                self.ccl,
                attn,
                w["norms"],
                w["mhc"],
                w["moe"],
                gate_bias_shift=GATE_CUTOFFS[str(L)],
                users_per_row=self.U,
                moe_buffers=chain.moe_buffers,
            )
            if chain.moe_buffers is None:
                chain.moe_buffers = layer.moe.decode.buffers
            pa = DSV41PrefillAttention(attn, w["attn"]["attn_sink"].float())
            pa.state_sink = functools.partial(self.sink.write, attn)
            attn.prefill = pa
            pmoe = DSV41PrefillMoE(layer.moe, T=T, buffers=None if first_pmoe is None else first_pmoe.decode.buffers)
            first_pmoe = first_pmoe or pmoe
            pls.append((L, DSV41PrefillLayer(layer, pa, pmoe, T=T)))
            key = getattr(attn, "ratio", 0)
            if key not in self.step_groups:
                self.step_groups[key] = DSV41StepState_paged(attn, max_ctx + 64, self.use_indexer)
            self.built.append((L, layer, key))
            self.attns[L] = attn
            del w
            gc.collect()
            if i % 5 == 0 or i == len(self.layer_ids) - 1:
                log(f"built layer {L} ({time.time() - t0:.0f}s)")
        if os.environ.get("DSV41_PF_SPARSE") == "1" or self.use_indexer:
            self.enable_prefill_sparse(c_max=int(os.environ.get("DSV41_PF_CMAX", "2048")))
        engram_ids = [l for l in (1, 14) if l in self.layer_ids]
        self.engram_ids = engram_ids
        self.host_rows = (
            HostEngramRows(tuple(engram_ids), max_batch_size=self.B, max_seq_len=max_ctx + 1024) if engram_ids else None
        )
        if self.host_rows is not None:
            self.hasher = RaggedNgramHash(self.host_rows.engram.hash)
            if os.environ.get("DSV41_ENGRAM_RAM", "1") == "1":
                t1 = time.time()
                self.host_rows.load_ram()
                log(f"Engram tables in process memory ({time.time() - t1:.0f}s)")
        dev_engram = {l: DSV41DeviceEngram(mesh_device, l, sh, mesh_config=self.mc, ccl=self.ccl) for l in engram_ids}
        embedding = DSV41DeviceEmbedding(mesh_device, sh.get("embed.weight"), users_per_row=self.U)
        self.head = DSV41DeviceHead(
            mesh_device, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps
        )
        self.dec = DSV41Decoder(mesh_device, self.built, embedding, self.head, dev_engram, step_states=self.step_groups)
        self.dec.mesh_config, self.dec.ccl = self.mc, self.ccl
        self.prefill_model = GenPrefillModel(mesh_device, pls, embedding, self.head, dev_engram, self.host_rows, self.U)
        self.prefill_model.set_head_sampling(self.mc, self.ccl)
        self.prefill_model.sink = self.sink
        self.engram_kin = {l: e.kin for l, e in dev_engram.items()}
        self.rows_cat = None
        self.trace_id = None
        self.admitted = False
        self.pool_pages_free = None
        log(
            f"model built: {len(self.layer_ids)} layers, U={self.U} users/row (batch {self.B}), pool {self.num_pages} pages/row ({time.time() - t0:.0f}s)"
        )

    # ---- construction ---------------------------------------------------------------------------------------------------------------
    def _build_attention(self, L, w, slot):
        meta = w["meta"]
        kw = dict(users_per_row=self.U)
        a = (self.md, self.mc, self.ccl, w["attn"], w["freqs_cis"])
        if meta["ratio"] == 0:
            return DSV41PagedAttention(*a, self.pool, slot, **kw)
        idx = self._build_indexer(L, meta, w)
        if meta["is_kv_source"]:
            attn = DSV41PagedCompressedAttention(
                *a, meta["ratio"], w["compressor"], self.pool, slot, L, indexer=idx, **kw
            )
            self.sources[L] = attn
            if (
                meta["ratio"] > 1
            ):  # the decode compressor reads the previous token's [kv|score]; the prefill hand-off fills it per user
                attn.prev_cs = ttnn.zeros(
                    [1, 1, self.U, 1024], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.md
                )
            return attn
        return DSV41PagedCompressedAttention(
            *a,
            meta["ratio"],
            None,
            self.pool,
            slot,
            meta["kv_source"],
            source=self.sources[meta["kv_source"]],
            indexer=idx,
            **kw,
        )

    def _build_indexer(self, L, meta, w):
        """Decode indexer of an index-source layer (same construction as DSV41DecodeChain's paged builder); key owners get the key slab, the other index
        sources alias the slab of their kv source."""
        if not (self.use_indexer and "indexer" in w):
            return None
        from models.demos.blackhole.deepseek_v41_flash.tt.indexer import DSV41DecodeIndexer, default_backend

        iw = w["indexer"]
        n_alloc = -(-(self.max_ctx // meta["ratio"] + 32) // 32) * 32
        idx = DSV41DecodeIndexer(
            self.md,
            iw,
            w["freqs_cis"],
            users_per_row=self.U,
            n_alloc=n_alloc,
            ratio=meta["ratio"],
            key_dtype=ttnn.bfloat8_b,
            fp4_q=True,
            backend=default_backend(self.max_ctx // meta["ratio"]),
        )
        if meta["is_kv_source"]:
            idx.set_key_weights(iw["wk"], iw["k_norm"])
            idx.load_keys(torch.zeros(self.B, 0, 128))
            self.index_owner[L] = idx
        else:
            idx.k_cache = self.index_owner[meta["kv_source"]].k_cache
        self.dec_idx[L] = idx
        return idx

    def enable_prefill_sparse(self, c_max=2048, max_tokens=None, enable=True):
        """Prefill indexer top-512 + sparse_sdpa for the compressed layers (tt/prefill_sparse.py): exact CSA selection for prompts with > 512 compressed
        entries. Also by env DSV41_PF_SPARSE=1 at model build. ``c_max``: largest prefill chunk. enable=False detaches (dense prefill attention).
        """
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import attach_prefill_sparse

        pas = {L: self.attns[L].prefill for L in self.layer_ids}
        if not enable:
            for pa in pas.values():
                pa.sparse = None
            return {}
        sh = _Shards()
        idx_w = {}
        for L in self.layer_ids:
            if L in R.model_args().index_source_layers and L < 40:
                idx_w[L] = load_layer(L, with_moe=False, max_seq_len=8, with_indexer=True)["indexer"]
        sinks = {L: sh.get(f"layers.{L}.attn.attn_sink").float() for L in self.layer_ids}
        self.prefill_sparse = attach_prefill_sparse(
            pas, idx_w, self.U, max_tokens or self.max_ctx, c_max, sinks, decode_indexers=self.dec_idx, enable=True
        )
        return self.prefill_sparse

    def dense_limit(self):
        """Longest context (tokens) whose compressed attention is exact WITHOUT the indexer: every compressed entry is selected while there are <= 512 of them
        (ratio 1: ctx <= 512; ratio 2: ctx <= 1024)."""
        ratios = {getattr(self.attns[L], "ratio", 0) for L in self.layer_ids}
        return 512 if 1 in ratios else 1024 if 2 in ratios else 1 << 30

    def check_context_supported(self, ctx):
        if ctx > self.dense_limit() and not self.use_indexer and os.environ.get("DSV41_ALLOW_DENSE") != "1":
            raise NotImplementedError(
                f"context {ctx} > {self.dense_limit()}: more than 512 compressed entries need the indexer top-512 selection in prefill and decode "
                "(paged decode indexer exists in tt/indexer.py but its key-slab write from the prefill and the prefill-side selection are not wired; "
                "DSV41_ALLOW_DENSE=1 runs the dense approximation for experiments)"
            )

    # ---- paged users ----------------------------------------------------------------------------------------------------------------
    def admit_users(self, prompt_lens, max_new_tokens):
        """(Re)admit every user: pages for the prompt + the generated tokens (+1), page table uploaded into the persistent device tensor."""
        for b in range(self.B):
            r, k = self.pool.user_key(b)
            if k in self.pool.allocs[r].pages:
                self.pool.release(b)
        for b in range(self.B):
            self.pool.admit(b, int(prompt_lens[b]) + 1, reserve_tokens=max_new_tokens)
        self.pool.sync_page_table()
        self.admitted = True

    # ---- prefill ----------------------------------------------------------------------------------------------------------------------
    def prepare_inputs_prefill(self, tokens, prompt_lens, chunk=None):
        """tokens [B, L] (right padded, any pad value) + prompt_lens [B] -> (padded tokens [B, W] with every user's tail = its last real token,
        chunk plan [(s0, C)])."""
        B, L = tokens.shape
        lens = torch.as_tensor(prompt_lens).long()
        S = int(lens.max())
        plan = self.prefill_model.chunk_plan(S, chunk)
        W = plan[-1][0] + plan[-1][1]
        tp = torch.zeros(B, W, dtype=torch.long)
        for b in range(B):
            n = int(lens[b])
            tp[b, :n] = tokens[b, :n].long()
            tp[b, n:] = tokens[b, n - 1]
        return tp, plan

    def prefill_forward(
        self,
        tokens,
        prompt_lens,
        chunk=None,
        max_new_tokens=0,
        want_logits=False,
        hook=None,
        enable_trace=True,
        s_pad_max=None,
    ):
        """Traced-chunk prefill (default): ONE chunk of ``chunk`` tokens per user is captured once and REPLAYED for every chunk of the prompt, the per-chunk values
        (positions, rope rows, masks, page-table / write-index tensors of the hand-off) being persistent device tensors refreshed before each replay
        (tt/prefill_dyn.py of the prefill model + ``PagedStateSink.update``). ``enable_trace=False``: the same dynamic chunk driven eagerly (compile / reference).
        Traced-chunk prefill is the default (verified at 40 layers, ISL 128); DSV41_PREFILL_DYN=0 selects the eager per-chunk reference path.
        """
        if (
            hasattr(self.prefill_model, "run_traced_chunks") and os.environ.get("DSV41_PREFILL_DYN", "1") != "0"
        ):  # UNVALIDATED until h44p validates the traced-chunk path
            return self.prefill_forward_dyn(
                tokens, prompt_lens, chunk, max_new_tokens, want_logits, enable_trace, s_pad_max
            )
        return self.prefill_forward_legacy(tokens, prompt_lens, chunk, max_new_tokens, want_logits, hook)

    def prepare_for_traces(self, lens):
        """Allocate EVERY persistent device tensor and run every compile pass BEFORE the first trace capture: a persistent tensor created after a capture can
        land on the (freed) intermediate buffers of a captured trace and is then clobbered by the next replay (or corrupts it). Creates the head's column-id
        constant, the decode device-loop buffers (tokens / positions), the packed Engram-rows buffer, and compiles one decode step (state restored).
        """
        if getattr(self, "_warm", False):
            return
        head = self.head
        if getattr(head, "_col_ids", None) is None:
            head._col_ids = ttnn.from_torch(
                torch.arange(head.cols, dtype=torch.float32).reshape(1, 1, 1, head.cols),
                device=self.md,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.md),
            )
        zeros = torch.zeros(self.B, dtype=torch.long)
        self._set_loop_state(zeros, torch.as_tensor(lens).long())
        if self.engram_ids:
            k = sum(self.engram_kin[l] for l in self.engram_ids)
            host = ttnn.from_torch(
                torch.zeros(self.B, 1, 1, k, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self._mp(),
            )
            self._upload_rows(host)
        snaps = self.dec.snapshot_states()
        self.last_logits = (
            self.dec.forward()
        )  # compile pass (writes pool rows at position = prompt length: rewritten by the prefill / first real step)
        ttnn.synchronize_device(self.md)
        self.dec.restore_states(snaps)
        del snaps
        self._set_loop_state(zeros, torch.as_tensor(lens).long())
        self._warm = True

    def _export_index_keys(self, lens):
        """Hand-off of the index keys: the prefill indexers' key slabs (key owners 2/8/14/20) -> the decode indexers' key slabs ``k_cache``."""
        if not self.use_indexer or not getattr(self, "prefill_sparse", None):
            return
        for L, dec in self.dec_idx.items():
            if L not in self.index_owner:
                continue  # layers 24..36 alias layer 20's slab
            sp = self.attns[L].prefill.sparse
            ix = None if sp is None else sp.indexer
            if (
                ix is None
                or ix.key_owner is not None
                or not getattr(sp, "dyn_on", True)
                and os.environ.get("DSV41_PREFILL_DYN", "1") != "0"
            ):
                continue
            ix.export_keys(dec.k_cache, int(torch.as_tensor(lens).max()) // self.attns[L].ratio)
        ttnn.synchronize_device(self.md)

    def _post_chunk(self, s0, C):
        """Host-only (no device allocation): read the ragged-head trace outputs of this chunk for the users whose last prompt token is inside it."""
        pm, U, rows, cols = self.prefill_model, self.U, self.rows, self.cols
        K = C // 32
        for u in range(U):
            lg, tk = pm.head_out[u]
            todo = [
                (r, (int(self._last_pos[r * U + u]) - s0) % 32, r * U + u)
                for r in range(rows)
                if s0 <= int(self._last_pos[r * U + u]) < s0 + C
            ]
            if not todo:
                continue
            devs = ttnn.get_device_tensors(ttnn.from_device(tk))
            full = pm.head.gather_logits(lg).reshape(rows, 32, -1) if self._want_logits else None
            for r, off, b in todo:
                tok = int(ttnn.to_torch(devs[r * cols]).reshape(-1)[off])
                self._res[b] = (tok, None if full is None else full[r, off].clone())

    def prefill_forward_dyn(self, tokens, prompt_lens, chunk, max_new_tokens, want_logits, enable_trace, s_pad_max):
        B = self.B
        assert tokens.shape[0] == B
        lens = torch.as_tensor(prompt_lens).long()
        assert int(lens.max()) + max_new_tokens <= self.max_ctx, "prompt + generated tokens exceed the model's max_ctx"
        self.check_context_supported(int(lens.max()) + max_new_tokens)
        t_start = time.perf_counter()
        S = int(lens.max())
        C = chunk or -(-S // 128) * 128
        S_pad = max(-(-S // C) * C, s_pad_max or 0)
        self.log(f"  prefill_dyn: admit users")
        self.admit_users(lens, max_new_tokens)
        self.sink.set_lengths(lens)
        self.sink.bind(C)
        bis = os.environ.get("DSV41_BISECT", "")
        if "nosink" in bis and not getattr(self, "_nosink_done", False):
            for _, pl in self.prefill_model.layers:
                pl.pa.state_sink = None
            self._nosink_done = True
        self.prepare_for_traces(lens)
        pm = self.prefill_model
        pm.timing = {}
        tp = torch.zeros(B, S_pad, dtype=torch.long)
        for b in range(B):
            n = int(lens[b])
            tp[b, :n] = tokens[b, :n].long()
            tp[b, n:] = tokens[b, n - 1]
        hashes = self.hasher(tp, torch.zeros(B, dtype=torch.long)) if self.host_rows is not None else None
        self._res, self._last_pos, self._want_logits = {}, lens - 1, want_logits
        if getattr(self, "_hooks_set", None) is not pm:
            pm.pre_replay_hooks.append(lambda s0, C_: self.sink.update(s0, C_))
            pm.post_replay_hooks.append(self._post_chunk)
            self._hooks_set = pm
        if "nohead" in bis:
            pm.post_replay_hooks[:] = [h for h in pm.post_replay_hooks if h != self._post_chunk]
        self.log(f"  prefill_dyn: run chunks (trace={enable_trace}, C={C}, S_pad={S_pad}, bisect={bis!r})")
        if enable_trace:
            pm.run_traced_chunks(tp, C, hashes=hashes)
            self.log("  prefill_dyn: chunks done")
        else:
            pm.setup_dyn(C, S_pad)
            for _, pl in pm.layers:
                pl.pa.reset_dyn()
            for ci in range(S_pad // C):
                s0 = ci * C
                hs = None if hashes is None else hashes[:, s0 : s0 + C]
                bufs = pm.alloc_inputs(C)
                pm.upload_inputs(pm.prep_inputs(tp[:, s0 : s0 + C], hs), bufs)
                pm.begin_chunk(s0, C)
                pm.forward_device(bufs, S, s0, C, dyn=True)
                ttnn.synchronize_device(self.md)
                self._post_chunk(s0, C)
        ttnn.synchronize_device(self.md)
        self._export_index_keys(lens)
        self.timing = dict(pm.timing, total=time.perf_counter() - t_start)
        first = torch.tensor([self._res[b][0] for b in range(B)], dtype=torch.long)
        logits = torch.stack([self._res[b][1] for b in range(B)]) if want_logits else None
        return first, logits

    def prefill_forward_legacy(self, tokens, prompt_lens, chunk=None, max_new_tokens=0, want_logits=False, hook=None):
        """tokens [B, L] right-padded prompts, prompt_lens [B] -> (first generated token [B] (greedy), logits [B, vocab] fp32 or None). Leaves, for every
        user: KV pages + rings + compressor state of all layers in the decode pool, the Engram token history on the host.
        """
        B = self.B
        assert tokens.shape[0] == B
        lens = torch.as_tensor(prompt_lens).long()
        assert int(lens.max()) + max_new_tokens <= self.max_ctx, "prompt + generated tokens exceed the model's max_ctx"
        self.check_context_supported(int(lens.max()) + max_new_tokens)
        self.timing = {}
        t_start = time.perf_counter()
        self.admit_users(lens, max_new_tokens)
        self.sink.set_lengths(lens)
        tp, plan = self.prepare_inputs_prefill(tokens, lens, chunk)
        pm = self.prefill_model
        pm.timing = {}
        pm.S = int(lens.max())
        for _, pl in pm.layers:
            pl.pa.begin()
        last_pos = lens - 1
        S = int(lens.max())

        def prep(ci):
            s0, C = plan[ci]
            tk = tp[:, s0 : s0 + C]
            hs = self.hasher(tk, torch.full((B,), s0)) if self.host_rows is not None else None
            return pm.prep_inputs(tk, hs)

        res = {}
        ex = ThreadPoolExecutor(1)
        fut = ex.submit(prep, 0)
        for ci, (s0, C) in enumerate(plan):
            t0 = time.perf_counter()
            pre = fut.result()
            pm.timing["host_wait"] = pm.timing.get("host_wait", 0.0) + time.perf_counter() - t0
            if ci + 1 < len(plan):
                fut = ex.submit(prep, ci + 1)
            bufs = pm.alloc_inputs(C)
            pm.upload_inputs(pre, bufs)
            self.sink.update(s0, C)
            out = pm.forward_device_legacy(
                bufs,
                S,
                s0,
                C,
                hook=hook if len(plan) == 1 else None,
                profile=True,
                last_pos=last_pos,
                want_logits=want_logits,
            )
            res.update(out)
            if len(plan) > 1:
                ttnn.synchronize_device(self.md)
                self.log(
                    f"  prefill chunk {ci + 1}/{len(plan)} (s0={s0}, C={C}) done at {time.perf_counter() - t_start:.1f} s"
                )
                clear_chunk_caches()
        ex.shutdown()
        ttnn.synchronize_device(self.md)
        self._export_index_keys(lens)
        self.timing = dict(pm.timing, total=time.perf_counter() - t_start)
        first = torch.tensor([res[b][0] for b in range(B)], dtype=torch.long)
        logits = torch.stack([res[b][1] for b in range(B)]) if want_logits else None
        return first, logits

    # ---- decode -------------------------------------------------------------------------------------------------------------------------
    def prepare_inputs_decode(self, tokens, current_pos):
        """Engram rows of the step's input tokens at their per-user positions -> host tensor for the persistent rows buffer (None without Engram)."""
        if not self.engram_ids:
            return None
        hashes = self.hasher(tokens.reshape(self.B, 1).long(), current_pos.long())
        rows = self._rows_threads(hashes)
        cat = torch.cat([rows[l].reshape(self.B, 1, 1, -1) for l in self.engram_ids], dim=-1).to(torch.bfloat16)
        return ttnn.from_torch(
            cat,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols)),
        )

    _rpool = None

    def _rows_threads(self, hashes):
        if Model._rpool is None:
            Model._rpool = ThreadPoolExecutor(max(1, len(self.engram_ids)))
        fs = {l: Model._rpool.submit(self.host_rows.rows, l, hashes) for l in self.engram_ids}
        return {l: f.result() for l, f in fs.items()}

    def _engram_rows_fn(self, tokens, pos):
        """In-trace: slice the persistent packed rows buffer into the per-layer rows the Engram layers read."""
        Tn = self.U
        rc = ttnn.reshape(self.rows_cat, [1, 1, Tn, self.rows_cat.shape[-1]])
        out, off = {}, 0
        for l in self.engram_ids:
            k = self.engram_kin[l]
            out[l] = ttnn.to_layout(ttnn.slice(rc, [0, 0, 0, off], [1, 1, Tn, off + k]), ttnn.TILE_LAYOUT)
            off += k
        return out

    def _mp(self):
        return ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=(self.rows, self.cols))

    def _set_loop_state(self, tokens, current_pos):
        tok = ttnn.from_torch(
            tokens.reshape(-1, 1).to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mp(),
        )
        pos = ttnn.from_torch(
            current_pos.reshape(-1).to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mp(),
        )
        if getattr(self.dec, "tok_dev", None) is None:
            self.dec.enable_device_loop(self.mc, self.ccl, tokens.reshape(-1), current_pos.reshape(-1))
            self.dec.engram_rows_fn = self._engram_rows_fn if self.engram_ids else None
        else:
            ttnn.copy_host_to_device_tensor(tok, self.dec.tok_dev)
            ttnn.copy_host_to_device_tensor(pos, self.dec.pos_dev)

    def _upload_rows(self, host_rows):
        if host_rows is None:
            return
        if self.rows_cat is None:
            self.rows_cat = ttnn.to_device(host_rows, self.md, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy_host_to_device_tensor(host_rows, self.rows_cat)

    def _read_tokens(self):
        devs = ttnn.get_device_tensors(ttnn.from_device(self.dec.tok_dev))
        return torch.cat([ttnn.to_torch(devs[r * self.cols]).reshape(-1) for r in range(self.rows)]).long()

    def decode_forward(self, tokens, current_pos, enable_trace=True, reload_inputs=True):
        """One decode step of every user. tokens [B] = the token fed at position current_pos [B] (per user). Returns the next greedy tokens [B].
        ``reload_inputs=False`` (steady state): the device already holds the fed-back token / position (device loop); only the Engram rows of
        ``tokens`` are uploaded."""
        t0 = time.perf_counter()
        self.pool.ensure(
            current_pos, lookahead=16
        )  # pages for the next replays (no-op, no upload, unless a page boundary is near)
        if reload_inputs or getattr(self.dec, "tok_dev", None) is None:
            self._set_loop_state(tokens, current_pos)
        host_rows = self.prepare_inputs_decode(tokens, current_pos)
        t1 = time.perf_counter()
        self._upload_rows(host_rows)
        if enable_trace and self.trace_id is None:
            self._capture_decode(tokens, current_pos)
            self._set_loop_state(tokens, current_pos)
            self._upload_rows(host_rows)
        t2 = time.perf_counter()
        if enable_trace:
            ttnn.execute_trace(self.md, self.trace_id, cq_id=0, blocking=False)
        else:
            self.last_logits = self.dec.forward()
        out = self._read_tokens()  # blocking read: waits for the step
        t3 = time.perf_counter()
        self.timing["decode_host_prep"] = t1 - t0
        self.timing["decode_upload"] = t2 - t1
        self.timing["decode_device_read"] = t3 - t2
        return out

    def _capture_decode(self, tokens, current_pos):
        """Compile pass (restores the step-carried compressor state) + trace capture of one device-loop step."""
        snaps = self.dec.snapshot_states()
        self.dec.forward()
        ttnn.synchronize_device(self.md)
        self.dec.restore_states(snaps)
        self._set_loop_state(tokens, current_pos)
        self.trace_id = ttnn.begin_trace_capture(self.md, cq_id=0)
        self.last_logits = self.dec.forward()
        ttnn.end_trace_capture(self.md, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.md)
        self.dec.restore_states(snaps)

    def read_logits(self):
        """Host copy [B, vocab] fp32 of the logits of the last decode step (diagnostics; the loop itself only reads tokens)."""
        ttnn.synchronize_device(self.md)
        return self.head.gather_logits(self.last_logits)[: self.B]

    def release_trace(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.md, self.trace_id)
            self.trace_id = None


def DSV41StepState_paged(attn, max_pos, with_indexer=False):
    return DSV41PagedStepState(attn, max_pos=max_pos, with_indexer=with_indexer, per_user_valid=with_indexer)
