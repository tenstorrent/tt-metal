# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""DFlash2 speculative decoding round loop (batch 1): drafter proposes 15 tokens, the target verifies 16 rows."""

import time

import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.demos.muse_glimmer_30b.tt.common import replicate_mapper
from models.demos.muse_glimmer_30b.tt.dflash.sampling import rejection_sample
from models.demos.muse_glimmer_30b.tt.dflash.topk import TOPK_PER_DEVICE, device_topk, read_topk
from models.demos.muse_glimmer_30b.tt.dflash.verify import TargetVerifier
from models.demos.muse_glimmer_30b.tt.model import DECODE_B_PAD, prefill_buckets
from models.demos.muse_glimmer_30b.tt.sampling import draw, filtered_probs, normalize_sampling

_WARM_SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=32)  # any sampled request: compiles the sampled paths


def _mean_ms(xs):
    return 1e3 * sum(xs) / len(xs) if xs else 0.0


class DFlash2Generator:
    def __init__(self, model, drafter, selector, page_table, page_table_tt, stop_ids, enable_trace=False):
        """page_table: host int32 [1, nb] for user 0; page_table_tt: the device prefill page table.
        enable_trace: each device stage (draft, verify per mode) runs eager the first time (compile), is captured
        the second time and replayed after that."""
        self.model = model
        self.drafter = drafter
        self.selector = selector
        self.cfg = drafter.cfg
        self.rows = self.cfg.block_size  # anchor + (rows - 1) drafts, verified in one target pass
        self.page_table = page_table
        self.page_table_tt = page_table_tt
        self.stop_ids = set(stop_ids)
        self.enable_trace = enable_trace
        self.mesh = model.mesh_device
        self.mapper = replicate_mapper(self.mesh)
        # Persistent per-round inputs (updated with copy_host_to_device_tensor; never created inside a capture).
        self.verifier = TargetVerifier(model, self.cfg.target_layer_ids, page_table)
        self.block_tokens_tt = ttnn.from_torch(
            torch.zeros(1, DECODE_B_PAD, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            mesh_mapper=self.mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.block_rope_idx = drafter.make_rope_idx([])
        self.ctx_rope_idx = drafter.make_rope_idx([])
        self.block_write_rows = (
            self.verifier.kv_write_rows
        )  # shared with the verify pass (same positions; the verifier owns them)
        self.ctx_write_rows = drafter.make_write_idxs(0, self.rows, valid=0)
        self.sdpa_pos = drafter.make_sdpa_pos(0)
        self.p = 0
        self.anchor = None
        self.pending = None
        self._ctx_done = True  # the drafter context covers every committed row (False between commit() and propose())
        self.traces = {}  # "draft" / "verify_argmax" / "verify_topk" / "verify_logits" -> trace id
        self.trace_out = {}  # same keys -> the output tensor(s) of the captured run
        self._seen = set()  # trace keys that already ran eager once
        self._t_draft = self._t_select = self._t_verify = 0.0
        self._t_capture = 0.0  # trace capture time inside the current round (not counted in the round timers)
        self.sp = None  # normalised sampling (temperature, top_p, k) or None for greedy
        self.gen = None  # host torch.Generator (sampling mode)
        self.round_times = []

    def _sync(self):
        ttnn.synchronize_device(self.mesh)

    # ---- host -> persistent device tensor copies
    def _copy_i32(self, values, dev):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor(values, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.mapper,
            ),
            dev,
        )

    def _copy_rope(self, positions, dev):
        t = torch.zeros(1, DECODE_B_PAD, dtype=torch.int32)
        t[0, : len(positions)] = torch.tensor(list(positions), dtype=torch.int32)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper), dev
        )

    def _copy_write_rows(self, values, rows):
        for v, d in zip(values, rows):
            self._copy_i32([v], d)

    def _set_block_inputs(self):
        """Block pass inputs for anchor at position p (block write rows = p..p+15)."""
        p, cfg = self.p, self.cfg
        toks = torch.zeros(1, DECODE_B_PAD, dtype=torch.int32)
        toks[0, 0] = self.anchor
        toks[0, 1 : self.rows] = cfg.mask_token_id
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(toks, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper),
            self.block_tokens_tt,
        )
        self._copy_rope(range(p, p + self.rows), self.block_rope_idx)
        self.verifier.write_rows(list(range(p, p + self.rows)) + [-1] * (DECODE_B_PAD - self.rows))
        t = torch.full((DECODE_B_PAD,), -1, dtype=torch.int32)
        t[: self.rows] = p + self.rows - 1
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.mapper), self.sdpa_pos
        )

    def _set_ctx_inputs(self, p_prev, valid):
        """Context pass inputs: rows p_prev + i, -1 for i >= valid (valid=0: write nothing)."""
        self._copy_rope(range(p_prev, p_prev + self.rows), self.ctx_rope_idx)
        self._copy_write_rows([p_prev + i if i < valid else -1 for i in range(self.rows)], self.ctx_write_rows)

    def start(self, prompt_ids, sampling=None, seed=0, return_logits=False, warm=False):
        """Reset the request state, prefill user 0 with taps and fill the drafter context.

        Returns the first token (greedy or sampled), or with return_logits=True the last-position logits as torch
        float32 [vocab_size] (and self.anchor = None: the caller samples and passes the token to verify()).
        warm=True (warm_prefill): run every drafter context chunk of the bucket."""
        self.round_times = []
        self.sp = normalize_sampling(sampling, self.model.args.max_top_k)
        self.gen = torch.Generator().manual_seed(seed) if self.sp is not None else None
        self.pending = None
        self._ctx_done = True
        m, cfg = self.model, self.cfg
        prompt = torch.tensor(list(prompt_ids), dtype=torch.int64)
        ids = list(cfg.target_layer_ids)
        if self.traces and m.prefill_is_new(prompt.numel()):
            logger.info(f"prefill length {prompt.numel()} is new: releasing traces (recaptured on next use)")
            self.release_traces(keep_seen=True)
        logits_tt, taps, t0, s0, s_real = m.prefill_with_taps(
            prompt, 0, self.page_table_tt, ids, tap_window=cfg.sliding_window
        )
        if return_logits:
            first = m.logits_to_torch(logits_tt, 1)[0]
        elif self.sp is None:
            tok_tt = m.sample_on_device(logits_tt)
            first = int(m.tokens_to_torch(tok_tt, 1)[0])
            ttnn.deallocate(tok_tt)
        else:
            temperature, top_p, k = self.sp
            vals_tt, ids_tt = device_topk(logits_tt, self.verifier.vocab_bias)
            values, tids = read_topk(
                vals_tt,
                ids_tt,
                [0],
                vocab_local=m.args.vocab_local,
                vocab_size=m.args.vocab_size,
                k=TOPK_PER_DEVICE,
            )
            ttnn.deallocate(vals_tt)
            ttnn.deallocate(ids_tt)
            probs, pids = filtered_probs(values, tids, temperature, top_p, k)
            first = draw(probs[0], pids[0], self.gen)
        ttnn.deallocate(logits_tt)
        self._fill_context(taps, t0, s0, s_real, all_chunks=warm)
        self.p = s_real
        self.anchor = None if return_logits else first
        return first

    def _fill_context(self, taps, t0, s0, s_real, page_table=None, batch=None, all_chunks=False):
        """Drafter context pass over prompt rows s0..s_real, 32 rows at a time, from taps holding rows t0.. (see
        MuseGlimmerModel.prefill_with_taps); frees taps. A chunk's slice program depends only on the tap shape and the
        chunk offset. all_chunks=True (warmup) also runs the chunks outside s0..s_real (they write no rows), so every
        chunk program of the bucket gets compiled."""
        d, ids = self.drafter, list(self.cfg.target_layer_ids)
        rows, h = int(taps[ids[0]].shape[-2]), int(taps[ids[0]].shape[-1])
        for c0 in range(t0, t0 + rows, DECODE_B_PAD):
            valid = 0 if c0 < s0 else max(0, min(DECODE_B_PAD, s_real - c0))
            if not valid and not all_chunks:
                continue
            r = c0 - t0
            chunk = [ttnn.slice(taps[i], [0, 0, r, 0], [1, 1, r + DECODE_B_PAD, h]) for i in ids]
            rope_idx = d.make_rope_idx(range(c0, c0 + DECODE_B_PAD))
            write_idxs = d.make_write_idxs(c0, DECODE_B_PAD, valid=valid)
            d.context_pass(chunk, rope_idx, write_idxs, page_table, batch)
            for t in chunk + [rope_idx] + write_idxs:
                ttnn.deallocate(t)
        for t in taps.values():
            ttnn.deallocate(t)

    def _warm_start(self, prompt, sampling, return_logits):
        """One warmup prefill (warm_prefill) that leaves no request the caller depends on."""
        self.start(prompt, sampling=sampling, return_logits=return_logits, warm=True)

    def warm_prefill(self):
        """Compile every prefill program of every bucket up to min(max_prefill_len, PREFILL_COARSE_MAX) before the
        first trace capture. A program built after a capture allocates persistent device memory (its kernel binary,
        CCL semaphores) that the trace's replays can overwrite. One prompt per bucket; the smallest bucket also runs the
        sampled and the logits first-token paths. A longer prompt still releases the traces (prefill_is_new)."""
        assert not self.traces, "warm_prefill must run before the first trace capture"
        a, t = self.model.args, time.perf_counter()
        pad = a.pad_token_id if a.pad_token_id is not None else 0
        buckets = prefill_buckets(a.max_prefill_len)
        for i, s in enumerate(buckets):
            self._warm_start([pad] * s, None, False)
            if i == 0:
                self._warm_start([pad] * s, _WARM_SAMPLING, False)
                self._warm_start([pad] * s, None, True)
        logger.info(
            f"warm_prefill: {len(buckets)} prefill buckets up to {buckets[-1]} in {time.perf_counter() - t:.1f}s"
        )

    def _warm_decode_runs(self, prompt):
        """The eager runs of warm_decode: the single-user path, greedy and sampled."""
        self.generate(prompt, 4)
        self.generate(prompt, 4, sampling=_WARM_SAMPLING)

    def warm_decode(self):
        """Compile every decode-step program (draft, verify in both modes, drafter context) with eager runs before the
        first trace capture, for the reason in warm_prefill. Stop tokens are ignored meanwhile, so every run decodes."""
        assert not self.traces, "warm_decode must run before the first trace capture"
        a, t = self.model.args, time.perf_counter()
        trace, stop = self.enable_trace, self.stop_ids
        self.enable_trace, self.stop_ids = False, set()
        try:
            self._warm_decode_runs([a.pad_token_id if a.pad_token_id is not None else 0] * 128)
        finally:
            self.enable_trace, self.stop_ids = trace, stop
        logger.info(f"warm_decode: {time.perf_counter() - t:.1f}s")

    def warm(self):
        """warm_prefill and warm_decode: after this, serving builds no new device program while a trace exists."""
        self.warm_prefill()
        self.warm_decode()

    # ---- device parts (traceable: no tensor creation from host)
    def _draft_device(self):
        """context pass + block pass + draft head on the persistent inputs -> (vals, ids, hp)."""
        m, d = self.model, self.drafter
        d.context_pass(self.verifier.tap_list(), self.ctx_rope_idx, self.ctx_write_rows)
        x = d.embed_block(self.block_tokens_tt, m.embedding_weight)
        h = d.block_forward(x, self.block_rope_idx, self.block_write_rows, self.sdpa_pos)
        out = d.draft_head(h, m.lm_head)
        ttnn.deallocate(x)
        ttnn.deallocate(h)
        return out

    def _plan(self, key):
        """Trace policy: "trace" (replay), "capture" (capture now, then replay) or "eager"."""
        if not self.enable_trace:
            return "eager"
        if key in self.traces:
            return "trace"
        if key in self._seen:
            return "capture"
        self._seen.add(key)
        return "eager"

    def _capture(self, key, fn):
        """Capture fn() (device ops only) as trace `key`; the capture does not execute it. The program is compiled
        already (the key ran eager once)."""
        t0 = time.perf_counter()
        self._sync()
        # Each trace's outputs are read right after its replay, before any other trace runs, so the other traces'
        # replays may overwrite them in between.
        with trace_allocation_tracker.corruptible_allocation_scope(self.mesh):
            self.traces[key] = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            self.trace_out[key] = fn()
            ttnn.end_trace_capture(self.mesh, self.traces[key], cq_id=0)
        self._sync()
        self._t_capture += time.perf_counter() - t0

    def release_traces(self, keep_seen=False):
        for t in self.traces.values():
            ttnn.release_trace(self.mesh, t)
        self.traces = {}
        out = self.trace_out
        self.trace_out = {}
        if not keep_seen:
            self._seen = set()
        for o in out.values():
            for t in o if isinstance(o, (tuple, list)) else [o]:
                ttnn.deallocate(t)

    def close(self):
        """Release the traces and every persistent device tensor this generator owns."""
        self.release_traces()
        for t in [
            self.block_tokens_tt,
            self.block_rope_idx,
            self.ctx_rope_idx,
            self.sdpa_pos,
            *self.ctx_write_rows,
        ]:
            ttnn.deallocate(t)
        self.verifier.close()

    @staticmethod
    def _accept_greedy(drafts, ids16):
        acc = 0
        while acc < len(drafts) and drafts[acc] == ids16[acc]:
            acc += 1
        return acc, ids16[acc]

    def _emit(self, drafts, acc, bonus, budget):
        new = drafts[:acc] + [bonus]
        for i, t in enumerate(new):
            if t in self.stop_ids:
                new = new[: i + 1]
                break
        return new[:budget]

    def _select(self, scores, cands, hpr):
        """Draft path: greedy walk, or sampled walk (also returns the draft distributions q)."""
        if self.sp is None:
            return self.selector.select(scores, cands, hpr, self.anchor), None
        return self.selector.sample(scores, cands, hpr, self.anchor, self.sp[0], self.gen)

    def propose(self):
        """Draft from the committed state (self.p, self.anchor, self.pending) -> (drafts [rows-1], q or None, cands)."""
        assert self.anchor is not None, "propose() needs an anchor (verify and commit the first token first)"
        m, d = self.model, self.drafter
        self._set_block_inputs()
        plan = self._plan("draft")
        if plan == "eager":
            if self.pending is not None:
                p_prev, produced = self.pending
                rope_idx = d.make_rope_idx(range(p_prev, p_prev + self.rows))
                write_idxs = d.make_write_idxs(p_prev, self.rows, valid=produced)
                d.context_pass(self.verifier.tap_list(), rope_idx, write_idxs)
                for t in [rope_idx] + write_idxs:
                    ttnn.deallocate(t)
            ta = time.perf_counter()
            x = d.embed_block(self.block_tokens_tt, m.embedding_weight)
            h = d.block_forward(x, self.block_rope_idx, self.block_write_rows, self.sdpa_pos)
            vals, ids, hp = d.draft_head(h, m.lm_head)
            scores, cands, hpr = d.read_draft(vals, ids, hp)
            for t in [x, h, vals, ids, hp]:
                ttnn.deallocate(t)
        else:
            if self.pending is None:
                self._set_ctx_inputs(self.p, 0)
            else:  # all rows: the stale rows past the accepted ones are rewritten by the block pass right after
                self._set_ctx_inputs(self.pending[0], self.rows)
            if plan == "capture":
                self._capture("draft", self._draft_device)
            ta = time.perf_counter()
            ttnn.execute_trace(self.mesh, self.traces["draft"], cq_id=0, blocking=False)
            scores, cands, hpr = d.read_draft(*self.trace_out["draft"])
        tb = time.perf_counter()
        self._ctx_done = True
        drafts, q = self._select(scores, cands, hpr)
        self._t_draft = tb - ta
        self._t_select = time.perf_counter() - tb
        return drafts, q, cands

    def verify(self, tokens, mode):
        """Target verify of [anchor] + drafts (1 <= len <= rows; padded with the last token) at positions
        self.p .. self.p+rows-1. Returns the host output of `mode`: argmax -> list of `rows` ints; topk -> (values,
        ids); logits -> torch float32 [rows, vocab_size]. Does not move self.p."""
        tokens = list(tokens)
        assert 1 <= len(tokens) <= self.rows
        if self.pending is not None and not self._ctx_done:
            # No propose() ran after the last commit (the caller declined it): feed the pending rows to the drafter
            # context now, before this verify overwrites the verifier taps.
            p_prev, produced = self.pending
            rope_idx = self.drafter.make_rope_idx(range(p_prev, p_prev + self.rows))
            write_idxs = self.drafter.make_write_idxs(p_prev, self.rows, valid=produced)
            self.drafter.context_pass(self.verifier.tap_list(), rope_idx, write_idxs)
            for t in [rope_idx] + write_idxs:
                ttnn.deallocate(t)
            self._ctx_done = True
        tokens += [tokens[-1]] * (self.rows - len(tokens))
        self.anchor = tokens[0]
        v = self.verifier
        v.prepare_single(tokens, self.p)
        key = f"verify_{mode}"
        plan = self._plan(key)
        if plan == "capture":
            self._capture(key, lambda: v.forward(mode))
        td = time.perf_counter()
        if plan == "eager":
            out = v.forward(mode)
            res = v.read_out(out, mode)
            for t in out if isinstance(out, tuple) else [out]:
                ttnn.deallocate(t)
        else:
            ttnn.execute_trace(self.mesh, self.traces[key], cq_id=0, blocking=False)
            res = v.read_out(self.trace_out[key], mode)
        self._t_verify = time.perf_counter() - td
        return res

    def commit(self, count, last_token):
        """Commit `count` tokens (accepted drafts + the bonus) of the last verify; last_token is the new anchor."""
        assert 1 <= count <= self.rows
        self.pending = (self.p, count)
        self._ctx_done = False
        self.p += count
        self.anchor = last_token

    def round(self, budget):
        """One draft + verify round. Returns the list of new tokens (cut at a stop id and at `budget`)."""
        self._t_capture = 0.0
        t0 = time.perf_counter()
        sample = self.sp is not None
        drafts, q, cands = self.propose()
        ver = self.verify([self.anchor] + drafts, "topk" if sample else "argmax")
        te = time.perf_counter()
        if sample:
            temperature, top_p, k = self.sp
            tp, tids = filtered_probs(ver[0], ver[1], temperature, top_p, k)
            acc, bonus = rejection_sample(drafts, q, cands, tp, tids, self.gen)
        else:
            acc, bonus = self._accept_greedy(drafts, ver)
        new = self._emit(drafts, acc, bonus, budget)
        t_accept = time.perf_counter() - te
        self.commit(acc + 1, bonus)
        total = time.perf_counter() - t0 - self._t_capture
        t_draft, t_select, t_verify = self._t_draft, self._t_select, self._t_verify
        self.round_times.append(
            dict(
                total=total,
                draft=t_draft,
                select=t_select,
                verify=t_verify,
                accept=t_accept,
                other=total - t_draft - t_select - t_verify - t_accept,
                n=len(new),
            )
        )
        return new

    def generate(self, prompt_ids, max_new_tokens, sampling=None, seed=0):
        """sampling: None (greedy) or a dict with temperature / top_p / top_k; seed drives the host RNG."""
        t0 = time.perf_counter()
        first = self.start(prompt_ids, sampling, seed)
        self._sync()
        ttft = time.perf_counter() - t0
        tokens = [first]
        accept = []
        t1 = time.perf_counter()
        while len(tokens) < max_new_tokens and tokens[-1] not in self.stop_ids:
            if self.p + self.rows > self.model.args.max_seq_len:
                logger.warning(f"context limit: p={self.p} + {self.rows} > max_seq_len, ending generation")
                break
            new = self.round(max_new_tokens - len(tokens))
            accept.append(len(new))
            tokens += new
        self._sync()
        decode_s = time.perf_counter() - t1
        steady = self.round_times[1:]  # round 1 runs eager and includes program compilation; the trace capture
        # runs before round 2, outside the round timers
        steady_s = sum(r["total"] for r in steady)
        steady_n = sum(r["n"] for r in steady)
        stats = {
            "ttft_s": ttft,
            "decode_s": decode_s,
            "rounds": len(accept),
            "accept_lengths": accept,
            "mean_accept": sum(accept) / max(1, len(accept)),
            "decode_tok_s": steady_n / steady_s if steady_s > 0 else 0.0,
            "decode_tok_s_all": (len(tokens) - 1) / decode_s if decode_s > 0 else 0.0,
            "sampling": self.sp,
            "mean_ms": {
                k: _mean_ms([r[k] for r in steady]) for k in ("draft", "select", "verify", "accept", "other", "total")
            },
        }
        mm = stats["mean_ms"]
        tag = "greedy" if self.sp is None else f"T{self.sp[0]},p{self.sp[1]},k{self.sp[2]}"
        logger.info(
            f"SUMMARY: TTFT {ttft:.2f}s | decode {stats['decode_tok_s']:.2f} tok/s/user "
            f"({'trace' if self.enable_trace else 'eager'}, from round 2) | rounds={stats['rounds']} "
            f"mean_accept={stats['mean_accept']:.2f} isl={len(list(prompt_ids))} | round ms: draft {mm['draft']:.1f} "
            f"select {mm['select']:.1f} verify {mm['verify']:.1f} accept {mm['accept']:.1f} other {mm['other']:.1f} "
            f"total {mm['total']:.1f} | sampling={tag}"
        )
        return tokens, stats
