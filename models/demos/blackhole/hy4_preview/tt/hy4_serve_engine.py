# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4Engine (task product-integ): the model side of the product server, one object that owns the 78-layer chain, the sequence-sharded SCALED_FP8 pool, the direct-to-pool prefill and the decode loop.

  * prefill: ``Hy4PoolPrefill`` (chunked, any prompt length up to the pool, 4 users in lock step, the pool writes of the rows that are NOT being prefilled are masked: live decode users keep their pages). With the
    MTP layer built, the MTP prompt KV rows are written per chunk (``mtp_prompt_chunk``) and a SEED round writes the pair of the last prompt position and produces the first K drafts.
  * decode, SPEC mode (K >= 1, default): ``Hy4SpecRunner`` (one traced round = verify S = K + 1 rows -> accept -> MTP draft, all state resident), a GREEDY trace and a lossless SAMPLED trace
    (``hy4_spec_sample``, temperature only) sharing every persistent tensor. The host reads (a_0..a_k, m, P, drafts, next tokens) per round.
  * decode, PLAIN mode (K = 0): ``Hy4PoolDecoder`` S = 1 + ``DeviceLoop`` (device-resident loop and sampler, one trace per sampler variant).
Persistent tensors are allocated in ``__init__`` / ``warm`` BEFORE any trace capture (D75-1: a tensor allocated after a capture can be clobbered by the replays).

State mirror (host, per PHYSICAL mesh row): pending token, its position P, the K drafts, live flag. The device state of a round is authoritative; the mirror is refreshed from the read-back of every round and
re-uploaded (``Hy4SpecRunner.load``) only on a transition (prefill, a row freed, an idle row present, a mismatch with the plugin's block).
"""
import hashlib
import os
import time

import numpy as np
import torch
from loguru import logger

import ttnn

PAD = 120002
ROWS = 4
PAGE = 512


class _Seed:
    """host sampling of the FIRST token of a request from its last-prompt-position logits row (fp32 [V])."""

    @staticmethod
    def sample(row, temperature, top_k, top_p, seed, pos):
        if temperature is None or temperature <= 0 or top_k == 1:
            return int(row.argmax())
        g = torch.Generator().manual_seed((int(seed) * 1000003 + int(pos)) & 0x7FFFFFFFFFFF)
        x = row.double() / float(temperature)
        if top_k is not None and 0 < top_k < x.numel():
            kth = torch.topk(x, int(top_k)).values[-1]
            x = torch.where(x >= kth, x, torch.full_like(x, float("-inf")))
        p = torch.softmax(x, dim=-1)
        if top_p is not None and 0 < top_p < 1.0:
            sp, si = torch.sort(p, descending=True)
            cs = torch.cumsum(sp, 0)
            keep = (cs - sp) < top_p
            p = torch.zeros_like(p).scatter_(0, si[keep], sp[keep])
            p = p / p.sum()
        return int(torch.multinomial(p.float(), 1, generator=g))


class Hy4Engine:
    def __init__(self, md, max_num_seqs, max_seq_len, k=None, log=None):
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as pool_decode
        from models.demos.blackhole.hy4_preview.tt.hy4_attn_opt import Opt
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import Hy4Chain, dram_view
        from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
        from models.demos.blackhole.hy4_preview.tt.hy4_pool_prefill_model import Hy4PoolPrefill
        from models.demos.gpt_oss.config import mesh_4x8
        from models.demos.gpt_oss.tt.ccl import CCLManager

        assert (
            os.environ.get("HY4_POOL", "0") == "1" and os.environ.get("HY4_POOL_PF", "0") == "1"
        ), "the product engine needs HY4_POOL=1 HY4_POOL_PF=1 HY4_KV_FMT=scaled_fp8 (direct-to-pool prefill)"
        self.md, self.max_num_seqs, self.max_seq_len = md, max_num_seqs, max_seq_len
        self.K = int(os.environ.get("HY4_SPEC_K", "3")) if k is None else int(k)
        self.spec = self.K > 0
        self.sampled = self.spec and os.environ.get("HY4_SPEC_SAMPLED", "1") == "1"
        self.S = self.K + 1 if self.spec else 1
        self.pf_rows = int(os.environ.get("HY4_SERVE_PF_ROWS", "512"))
        # HY4_PF_HYBRID=<cap tokens> (default 0 = off): prompts of at most cap tokens (longest of the call) use the OLD chain prefill (replicated SCALED_FP8 cache, 512-row chunks) + masked hand-off into the pool
        # (clean on stray tokens / GPQA thinking, program product-race RESULT_RACE sec 8); longer prompts use the direct-to-pool prefill. The cap is bounded by the replicated cache (scratch DRAM).
        self.hybrid_cap = int(os.environ.get("HY4_PF_HYBRID", "0"))
        # HY4_PFX=1 (default 0): ROW-LOCAL PREFIX CACHE. The pool pages of a finished request stay resident in its mesh row; a later prompt whose first tokens equal the retained prompt prefix prefills only the
        # suffix (chunk-aligned start; the last chunk of the hit is recomputed, HY4_PFX_REDO=1, so the MTP pair of the boundary position is exact). Spec mode only (the parked rows of the spec round are moved
        # past the retained blocks). HY4_PFX_FILE: live switch (a file holding 0 / 1, read per prefill call; 0 = never hit, retention bookkeeping continues).
        self.pfx = os.environ.get("HY4_PFX", "0") == "1" and self.spec
        self.pfx_redo = os.environ.get("HY4_PFX_REDO", "1") == "1"
        self.ret = [
            None
        ] * ROWS  # per mesh row: np.int32 tokens whose pool pages (positions 0..len-1) are valid, len % 512 == 0; None = nothing retained
        self.ret_t = [0.0] * ROWS
        self.pfx_stats = dict(calls=0, hit_calls=0, hit_tokens=0, prompt_tokens=0, computed_tokens=0)
        # HY4_PFX_LOSS_LOG=1 (default 0): one ``pfx_loss`` line per prefilled request with the cause of the tokens that were NOT served from the cache + running totals in pfx_stats (loss_<cause>)
        self.loss_log = os.environ.get("HY4_PFX_LOSS_LOG", "0") == "1"
        self.hist = (
            []
        )  # recent prompts (np.int32, floored to the page) with the row they were prefilled on and the time: the 'was it ever here' memory of the loss classifier
        self.hist_max = int(os.environ.get("HY4_PFX_LOSS_HIST", "24"))
        # HY4_XROW=1 (default 0; needs HY4_PFX=1): device-to-device cross-row COPY of a retained chain (tt/hy4_xrow.py, B45x): a request whose prefix sits on another mesh row (busy decoding, or a free row that is not the one
        # chosen) is served on a free row after the first h tokens of pool pages were copied row -> row. Live switch HY4_XROW_FILE (0 / 1), cap HY4_XROW_MAX (tokens), floor HY4_XROW_MIN. Any error = cold recompute.
        self.xrow_on = os.environ.get("HY4_XROW", "0") == "1" and self.pfx
        self.xrow = None  # Hy4Xrow, built in warm() before the first capture
        self.xrow_plan = {}  # destination mesh row -> (source row, h tokens), set by assign_rows, consumed by prefill
        self.xrow_stats = dict(copies=0, failed=0, tokens=0, seconds=0.0)
        # HY4_BLOCKS=1 (default 0; needs HY4_PFX=1, spec mode): multi-chain SHADOW block store (tt/hy4_blocks_store.py + hy4_blocks_copy.py, B45b, program/b45/blocks/DESIGN.md): every pool tensor gets HY4_BLOCKS_PAGES extra
        # pages (clamped to the DRAM budget); the retained chain of a row is published there (page copies) before another prompt overwrites it, a prompt matching a shadow chain deeper than the retained one gets the blocks
        # copied back before the prefill. Live switch HY4_BLOCKS_FILE (0 / 1): 0 = row-local behaviour (nothing published, nothing copied in). Any copy error = cold recompute.
        self.blocks_on = os.environ.get("HY4_BLOCKS", "0") == "1" and self.pfx
        self.blocks = None  # ShadowStore
        self.bcopy = None  # Hy4PageCopy, built in warm() before the first capture
        self.blocks_pages = 0
        self.blocks_stats = dict(out_pages=0, in_pages=0, in_calls=0, failed=0, seconds=0.0)
        ckpt = os.environ.get("HY4_CKPT", "/ckpt")
        nl = int(os.environ.get("HY4_SERVE_LAYERS", "78"))
        t0 = time.time()
        links = hy4_fabric_selftest(md)
        pool_decode._POOL_AG_LINKS = links.pool_ag
        logger.info(
            f"Hy4 fabric self-test: num_links={links.num_links} pool_ag={links.pool_ag} ({links.seconds:.1f} s)"
        )
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        self.ch = Hy4Chain(md, mc, ccl, ckpt=ckpt)
        self.ch.add_layers(0, nl, log=lambda s: logger.info(s))
        if self.spec:
            self.ch.add_layers(78, 79, log=lambda s: logger.info(s))
        os.environ["HY4_HEAD_FP32"] = "1"  # fp32 logits out of the lm_head (D12-1)
        self.ch.build_head()
        self.vocab = int(self.ch.cfg.vocab_size)
        dv = dram_view(md)
        logger.info(f"Hy4 model built: {nl} layers (+MTP {self.spec}), {time.time() - t0:.0f} s, dram {dv}")
        # pool pages per mesh row: the model length + the verify rows beyond the committed length (K + 1) + the pad rows of a lock-step prefill chunk, rounded to the prefill chunk
        need = max_seq_len + self.K + 2
        self.pages = (-(-need // self.pf_rows)) * self.pf_rows // PAGE
        if os.environ.get("HY4_IDX_BLK", "0") == "1" and os.environ.get("HY4_B37_PAGES8", "1") == "1":
            # B37: the blocked indexer score layout needs n_local = pages * 64 to be a multiple of 32 * B (B = 16): pages % 8 == 0 (2049 pages at the 1M pool broke it: block width 8196, see RESULT_B37)
            self.pages = -(-self.pages // 8) * 8
        pool_bytes = (
            self.pages * 3_750_912 // 8
        )  # per bank: 78 (+MTP) kv pools + 22 key pools, 3,750,912 B per page per chip (CAPACITY_MEASURED.md)
        if pool_bytes > dv["free_per_bank"] - (96 << 20):
            raise RuntimeError(
                f"pool of {self.pages} pages/row needs {pool_bytes >> 20} MiB/bank, only {dv['free_per_bank'] >> 20} MiB/bank free after the weights: lower --max-model-len"
            )
        if self.blocks_on:
            page_b = 475_978  # B/bank per pool page incl. pool-attached tensors (RESULT_B43 calibration)
            want = int(os.environ.get("HY4_BLOCKS_PAGES", "256"))
            room = (
                dv["free_per_bank"] - pool_bytes - (int(os.environ.get("HY4_BLOCKS_GUARD_MB", "400")) << 20)
            ) // page_b
            self.blocks_pages = max(0, min(want, int(room)))
            logger.info(
                f"HY4 blocks: shadow pages/row {self.blocks_pages} (asked {want}, room {room}); free {dv['free_per_bank'] >> 20} MiB/bank before the pool"
            )
            if self.blocks_pages < 8:
                self.blocks_on, self.blocks_pages = False, 0
        self.dec = Hy4PoolDecoder(
            self.ch, self.pages, mode="topk", S=self.S, bucket_tokens=self.pages * PAGE, extra_pages=self.blocks_pages
        )  # S = 1 for plain
        self.pp = Hy4PoolPrefill(self.dec, pf_rows=self.pf_rows)
        self.runner = self.dl = self.mtp = None
        if self.spec:
            from models.demos.blackhole.hy4_preview.tt.hy4_mtp import Hy4MTP, greedy_tokens
            from models.demos.blackhole.hy4_preview.tt.hy4_spec_trace import Hy4SpecRunner

            ly = self.ch.mtp_layers[0]
            self.mtp = Hy4MTP(self.ch, ly, self.ch.rd.layer_sd(-1), Opt.from_env(), moe_fn=None)
            smp = None
            if self.sampled:
                from models.demos.blackhole.hy4_preview.tt.hy4_sampler import Hy4DeviceSampler
                from models.demos.blackhole.hy4_preview.tt.hy4_spec_sample import SpecSampling

                smp = SpecSampling(Hy4DeviceSampler(md, mc, ccl, users_per_row=self.S), self.K, invT=1.0, seed=0)
            self.runner = Hy4SpecRunner(
                self.dec,
                self.mtp,
                self.K,
                lambda lg: greedy_tokens(self.ch, lg),
                self.pages,
                sampling=smp,
                extra_pages=self.blocks_pages,
            )
            self.runner.default_variant = "G"
            self.samp = smp
        else:
            from models.demos.blackhole.hy4_preview.tt.hy4_device_loop import DeviceLoop

            self.dl = DeviceLoop(self.dec)
            self.loop_variants = os.environ.get("HY4_LOOP_VARIANTS", "G,Fp,F").split(",")
        dv = dram_view(md)
        logger.info(
            f"Hy4 pool built: pages/row={self.pages} (max_model_len {max_seq_len}), spec K={self.K} sampled={self.sampled}, pf_rows={self.pf_rows}, dram {dv}"
        )
        # host mirror of the per-row decode state (spec)
        K = max(self.K, 1)
        self.pend = [PAD] * ROWS
        self.pos = [0] * ROWS  # idle rows: ``park_pos(r)`` (set at the first load)
        self.drafts = [[-1] * K for _ in range(ROWS)]
        self.live = [False] * ROWS
        self.dirty = True  # the device state differs from the mirror: ``load`` before the next round
        self.q_valid = False  # the device q rows / draft column describe the drafts of the mirror (sampled trace)
        self.params_dirty = True
        self.temp = [0.0] * ROWS
        self.seeds = [0] * ROWS
        self.rounds = 0
        self.t_round = 0.0
        self.warm_done = False
        # round pipelining (HY4_SPEC_PIPE=1): the device state of a round is resident, so round N+1 is submitted right behind the read-back of round N (speculatively: same caps / variant / params, no transition).
        # The host work between two ``round`` calls (plugin accept walk, proposal, scheduling, streaming) then overlaps the device time of the next round. A mismatch (finished request, admission, other caps / variant,
        # mirror != plugin block) drains the speculative round, discards it and reloads the mirror (the junk KV rows it wrote lie at positions >= the pending position and are overwritten before being read).
        self.pipe = os.environ.get("HY4_SPEC_PIPE", "0") == "1"
        self.inflight = None
        self.pipe_idle = (
            os.environ.get("HY4_SPEC_PIPE_IDLE", "1") == "1"
        )  # speculate also with idle (parked) rows: their device state drifts instead of being re-parked every round
        self.pipe_discards = 0
        self.pipe_hits = 0
        self.pipe_resync = int(os.environ.get("HY4_SPEC_PIPE_RESYNC", "256"))
        self._since_load = 0
        self.tl = [] if os.environ.get("HY4_ROUND_TL", "0") == "1" else None

    # ------------------------------------------------------------------------------------------------ prefix cache (row local)
    def park_pos(self, r):
        """position an idle row keeps: just past its retained prefix (0 when nothing is retained or the cache is off)."""
        return int(len(self.ret[r])) if (self.pfx and self.ret[r] is not None) else 0

    def park_pos_all(self):
        return [self.park_pos(r) for r in range(ROWS)]

    def _pfx_on(self):
        f = os.environ.get("HY4_PFX_FILE")
        if f:
            try:
                with open(f) as fh:
                    return fh.read().strip() == "1"
            except (OSError, ValueError):
                pass
        return True

    def match_len(self, r, toks):
        """number of leading tokens of ``toks`` (np.int32) equal to the retained prefix of mesh row r."""
        a = self.ret[r]
        if a is None or len(a) == 0:
            return 0
        n = min(len(a), len(toks))
        ne = (a[:n] != toks[:n]).nonzero()[0]
        return int(ne[0]) if len(ne) else n

    def hit_len(self, r, toks):
        """reusable prefix length for a prompt on row r: matched, chunk aligned, at least one token left to prefill."""
        if not (self.pfx and self._pfx_on()):
            return 0
        A = self.pf_rows
        return (min(self.match_len(r, toks), len(toks) - 1) // A) * A

    def assign_rows(self, prompts, cur_rows):
        """row-match admission: prompts = list of token lists (one new request each), cur_rows = the mesh row the plugin's slot currently maps to. Returns the mesh row per request: a free (not live) row holding the
        longest matching retained prefix; with no match the free row with nothing retained / least recently used (so retained prefixes of other sessions survive). Never a live row, never twice.
        """
        if not self._pfx_on():  # live switch off: the plugin's own slot -> row mapping, exactly as without the cache
            return list(cur_rows)
        taken, out = set(), [None] * len(prompts)
        arrs = [torch.as_tensor(p, dtype=torch.int32).numpy() for p in prompts]
        self.xrow_plan = {}
        if self.xrow is not None and self.xrow_on:
            from models.demos.blackhole.hy4_preview.tt import hy4_xrow as XR

            if XR.enabled():
                rows, copies = XR.plan_copies(
                    self.ret,
                    self.live,
                    taken,
                    arrs,
                    cur_rows,
                    pf_rows=self.pf_rows,
                    own_hit=self.ret_t,
                    min_hit_tokens=XR.min_hit(),
                    cap=XR.xrow_max(),
                )
                self.xrow_plan = {rows[i]: sc for i, sc in copies.items()}
                if copies:
                    logger.info(f"HY4 xrow plan: rows={rows} copies={ {rows[i]: sc for i, sc in copies.items()} }")
                return rows
        cand = lambda: [r for r in range(ROWS) if not self.live[r] and r not in taken]
        order = []
        for i, a in enumerate(arrs):
            best = max(((self.hit_len(r, a), r) for r in cand()), default=(0, None))
            order.append((best[0], i))
        for _, i in sorted(order, reverse=True):
            c = cand()
            if not c:
                out[i] = cur_rows[i]
                continue
            scored = [(self.hit_len(r, arrs[i]), r) for r in c]
            h, r = max(scored, key=lambda x: x[0])
            if h <= 0:
                # no usable prefix: prefer to keep the plugin's choice unless it would destroy a retained prefix while an empty / older row exists
                r = min(
                    c, key=lambda q: (self.ret[q] is not None and len(self.ret[q]) > 0, self.ret_t[q], q != cur_rows[i])
                )
            out[i] = r
            taken.add(r)
        return out

    def classify_loss(self, prompts_by_row, hits, start, live_before, rids=None):
        """cause of the tokens a request did not get from the cache, per mesh row of the call: -> {row: dict(rid, prompt, hit, potential, cause, home)}.
        potential = the aligned hit the longest earlier prompt of the history would have given (never more than the prompt - 1). Causes: hit (nothing lost beyond alignment), first_seen (no earlier prompt shares a
        token), policy_off (HY4_PFX_FILE switch 0), busy_home (the prefix is still resident on a row that was live, so the request went elsewhere), lockstep (resident on the row it got but the call starts at the common
        minimum of its rows), evicted (a row overwrote it), partial (resident but the call recomputed it for another reason).
        """
        A, out = self.pf_rows, {}
        on = self._pfx_on()
        for r, p in prompts_by_row.items():
            toks = torch.as_tensor(p, dtype=torch.int32).numpy()
            n = len(toks)
            used = min(hits.get(r, 0), start) if (hits.get(r, 0) > 0) else 0
            pot, row0 = 0, None
            for chain, r0, _t in self.hist:
                m = (
                    min(
                        int(np.argmax(np.append(chain[: min(len(chain), n)] != toks[: min(len(chain), n)], True))),
                        n - 1,
                    )
                    // A
                ) * A
                if m > pot:
                    pot, row0 = m, r0
            if not on and self.pfx:
                cause = "policy_off" if pot > 0 else "first_seen"
            elif pot == 0:
                cause = "first_seen"
            elif used >= pot:
                cause = "hit"
            else:
                resident = [q for q in range(ROWS) if (min(self.match_len(q, toks), n - 1) // A) * A >= pot]
                if r in resident:
                    cause = "lockstep" if hits.get(r, 0) > used else "partial"
                elif any(live_before[q] for q in resident):
                    cause = "busy_home"
                elif resident:
                    cause = "partial"
                else:
                    cause = "evicted"
            lost = max(pot - used, 0)
            self.pfx_stats["loss_" + cause] = self.pfx_stats.get("loss_" + cause, 0) + (lost if cause != "hit" else 0)
            out[r] = dict(rid=(rids or {}).get(r), prompt=n, hit=used, potential=pot, cause=cause, home=row0)
        return out

    def remember_prompts(self, prompts_by_row, rows):
        now = time.time()
        for r in rows:
            n = (len(prompts_by_row[r]) // PAGE) * PAGE
            if n:
                self.hist.append((torch.as_tensor(prompts_by_row[r][:n], dtype=torch.int32).numpy().copy(), r, now))
        del self.hist[: max(0, len(self.hist) - self.hist_max)]

    def _xrow_copy(self, src, dst, h):
        """copy the first h tokens of the retained chain of mesh row src to mesh row dst (device to device); the destination chain is cleared BEFORE the copy (an aborted copy retains nothing); any error = cold."""
        a = self.ret[src]
        self.ret[dst] = None
        if a is None or len(a) < h or h % PAGE or h <= 0:
            logger.warning(f"HY4 xrow: copy {src}->{dst} h={h} refused (source chain {None if a is None else len(a)})")
            return
        try:
            dt = self.xrow.copy(src, dst, h)
        except Exception as e:  # noqa: BLE001
            self.xrow_stats["failed"] += 1
            logger.error(
                f"HY4 xrow: copy {src}->{dst} h={h} FAILED ({type(e).__name__}: {str(e)[:200]}): cold recompute"
            )
            return
        self.ret[dst] = a[:h].copy()
        self.ret_t[dst] = time.time()
        st = self.xrow_stats
        st["copies"] += 1
        st["tokens"] += h
        st["seconds"] += dt
        logger.info(f"HY4 xrow: copied {h} tokens row {src} -> {dst} in {dt * 1e3:.0f} ms (stats {st})")

    def _pool_tensors(self):
        ts = [(f"kv{i}", t) for i, t in enumerate(self.dec.kv_pools)] + [
            (f"key{i}", t) for i, t in enumerate(self.dec.key_pools) if t is not None
        ]
        if self.runner is not None:
            ts += [("mtp_kv", self.runner.mtp_kv), ("mtp_keys", self.runner.mtp_keys)]
        return ts

    def _blocks_build(self):
        from models.demos.blackhole.hy4_preview.tt.hy4_blocks_copy import Hy4PageCopy
        from models.demos.blackhole.hy4_preview.tt.hy4_blocks_store import ShadowStore

        self._dram("before blocks build")
        self.bcopy = Hy4PageCopy(self.md, self._pool_tensors())
        for r in range(ROWS):
            self.bcopy._ctl_row(r)  # persistent control pages, before the first capture
        self.blocks = ShadowStore(
            rows=ROWS,
            shadow_pages=self.blocks_pages,
            shadow_base=self.pages,
            tip_sessions=int(os.environ.get("HY4_BLOCKS_TIPS", "1")),
            depth_weight=float(os.environ.get("HY4_BLOCKS_DEPTH_W", "0")),
        )
        self._dram("after blocks build")
        logger.info(
            f"HY4 blocks: store built, {self.blocks_pages} shadow pages/row ({self.blocks_pages * PAGE} tokens), {len(self.bcopy.tensors)} pool tensors"
        )

    def _blocks_live(self):
        if self.blocks is None:
            return False
        f = os.environ.get("HY4_BLOCKS_FILE")
        if f:
            try:
                with open(f) as fh:
                    return fh.read().strip() == "1"
            except (OSError, ValueError):
                pass
        return True

    def _blocks_pre_prefill(self, r, toks):
        """row r is about to be prefilled with ``toks`` (np.int32): plan the shadow hit against the retained chain, publish the retained chain (copy-out, parents first), then copy the hit blocks in. On return
        ``self.ret[r]`` = the tokens whose pool pages are valid (the hit prefix, chunk aligned) so ``hit_len`` sees the deeper hit. Never raises: an error leaves nothing retained (cold).
        """
        res = self.ret[r]
        plan = self.blocks.plan_hit(r, toks, res)
        out = []
        try:
            if res is not None and len(res) >= PAGE:
                out = self.blocks.plan_out(r, res, session=f"row{r}")
            t0 = time.perf_counter()
            if out:
                self.bcopy.copy(r, [(a, b) for a, b in out])  # active -> shadow
            if plan.copy_in:
                self.bcopy.copy(
                    r, [(a, b) for a, b in plan.copy_in]
                )  # shadow -> active (never the pages just published: pinned)
            dt = time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001
            self.blocks_stats["failed"] += 1
            logger.error(f"HY4 blocks: row {r} copy FAILED ({type(e).__name__}: {str(e)[:200]}): cold recompute")
            self.blocks.unpin(plan)
            self.ret[r] = None
            return
        self.blocks.unpin(plan)
        st = self.blocks_stats
        st["out_pages"] += len(out)
        st["in_pages"] += len(plan.copy_in)
        st["in_calls"] += 1 if plan.copy_in else 0
        st["seconds"] += dt
        if plan.copy_in:
            self.ret[r] = toks[: plan.start].copy()
            self.ret_t[r] = time.time()
            logger.info(
                f"HY4 blocks: row {r} hit {plan.start} tokens ({plan.active_blocks} active + {plan.shadow_blocks} shadow blocks), out {len(out)} / in {len(plan.copy_in)} pages in {dt * 1e3:.0f} ms (stats {st})"
            )

    def _xrow_build(self):
        from models.demos.blackhole.hy4_preview.tt.hy4_xrow import Hy4Xrow

        ts = [(f"kv{i}", t) for i, t in enumerate(self.dec.kv_pools)] + [
            (f"key{i}", t) for i, t in enumerate(self.dec.key_pools) if t is not None
        ]
        if self.runner is not None:
            ts += [("mtp_kv", self.runner.mtp_kv), ("mtp_keys", self.runner.mtp_keys)]
        self._dram("before xrow build")
        self.xrow = Hy4Xrow(self.md, ts)
        w = self.xrow.warm()
        self._dram("after xrow build")
        logger.info(f"HY4 xrow: stage buffers built, {len(ts)} pool tensors, warm {w:.1f} s")

    def drain(self):
        """wait for an in-flight speculative round and drop it (state on device advanced past the mirror: reload before the next round). Called before anything else touches the device (prefill, seed)."""
        if self.inflight is not None:
            ttnn.synchronize_device(self.md)
            self.inflight = None
            self.dirty = True
            self.q_valid = False
            self.params_dirty = True
            self.pipe_discards += 1

    # ------------------------------------------------------------------------------------------------ prefill
    def prefill(self, prompts_by_row, sampling=None, rids=None):
        """prompts_by_row {mesh row: list[int]} -> {row: first token}. ``sampling`` {row: (temperature, top_k, top_p, seed)}. Direct-to-pool chunked prefill (+ MTP prompt KV + seed round in spec mode).
        The other mesh rows keep their pool pages (masked writes) and their decode state."""
        t0 = time.perf_counter()
        self.drain()
        rows = sorted(prompts_by_row)
        prompts = [prompts_by_row.get(r, [PAD]) for r in range(ROWS)]
        cap = self.hybrid_cap
        hf = os.environ.get(
            "HY4_PF_HYBRID_FILE"
        )  # live A/B switch: a file holding the cap (0 = pure pool prefill); read per prefill call, never above the boot value
        if hf:
            try:
                with open(hf) as fh:
                    cap = min(int(fh.read().strip()), self.hybrid_cap)
            except (OSError, ValueError):
                pass
        # prefix reuse: the common start of the prefilled rows (lock step); a row with a longer hit recomputes down to it (writes the same bytes)
        start, hits = 0, {}
        if self.pfx:
            A = self.pf_rows
            plan, self.xrow_plan = self.xrow_plan, {}
            for r in rows:
                sc = plan.get(r)
                if sc is not None and self.xrow is not None:
                    self._xrow_copy(sc[0], r, sc[1])
            if self.blocks is not None and self._blocks_live() and self._pfx_on():
                for r in rows:
                    self._blocks_pre_prefill(r, torch.as_tensor(prompts_by_row[r], dtype=torch.int32).numpy())
            for r in rows:
                hits[r] = self.hit_len(r, torch.as_tensor(prompts_by_row[r], dtype=torch.int32).numpy())
            st = [(h - A if (self.pfx_redo and h > 0) else h) for h in hits.values()]
            start = max(0, min(st)) if st else 0
            if self.loss_log:
                live_before = list(self.live)
                loss = self.classify_loss(prompts_by_row, hits, start, live_before, rids)
                for r in rows:
                    d = loss[r]
                    logger.info(
                        f"pfx_loss rid={d['rid']} row={r} prompt={d['prompt']} hit={d['hit']} potential={d['potential']} cause={d['cause']} home={d['home']} live={live_before}"
                    )
            for r in rows:
                self.ret[r] = None  # the pages of this row change from position ``start`` on
            self.pfx_stats["calls"] += 1
            self.pfx_stats["prompt_tokens"] += sum(len(prompts_by_row[r]) for r in rows)
            self.pfx_stats["computed_tokens"] += sum(len(prompts_by_row[r]) - start for r in rows)
            if start > 0:
                self.pfx_stats["hit_calls"] += 1
                self.pfx_stats["hit_tokens"] += start * len(rows)
        chain = cap > 0 and start == 0 and max(len(prompts_by_row[r]) for r in rows) <= cap
        res = self.pp.prefill(
            prompts,
            keep_hidden=self.spec,
            keep_logits=True,
            mtp=self.mtp if self.spec else None,
            keep_rows=rows,
            chain=chain,
            start=start,
        )
        if self.pfx:
            if self.loss_log:
                self.remember_prompts(prompts_by_row, rows)
            now = time.time()
            for r in rows:
                n = (len(prompts_by_row[r]) // PAGE) * PAGE
                self.ret[r] = torch.as_tensor(prompts_by_row[r][:n], dtype=torch.int32).numpy().copy() if n else None
                self.ret_t[r] = now
        if self.blocks_on:
            self._dram(
                "blocks: after prefill call"
            )  # steady-state free DRAM per bank (low-water proxy for the mega-bundle gate)
        first = {}
        for r in rows:
            sp = (sampling or {}).get(r, (0.0, -1, 1.0, 0))
            first[r] = _Seed.sample(res.logits[r], sp[0], sp[1], sp[2], sp[3], len(prompts_by_row[r]))
        t1 = time.perf_counter()
        if self.spec:
            self._seed_rows(rows, first, res, prompts_by_row, sampling or {})
        logger.info(
            f"HY4 prefill call: lg_md5={ {r: hashlib.md5(res.logits[r].numpy().tobytes()).hexdigest()[:12] for r in rows} } pfx_start={start} hits={hits} pfx_stats={self.pfx_stats} xrow={self.xrow_stats} blocks={self.blocks_stats} mode={'chain' if chain else 'pool'} handoff={res.handoff_sec:.2f}s rows={rows} lens={[len(prompts_by_row[r]) for r in rows]} chunks={len(res.chunk_sec)} prefill={t1 - t0:.2f}s "
            f"(chunk median {sorted(res.chunk_sec)[len(res.chunk_sec) // 2]:.2f}s) chunks_s={[round(x, 1) for x in res.chunk_sec[:6]]}..{[round(x, 1) for x in res.chunk_sec[-2:]]} seed={time.perf_counter() - t1:.2f}s"
        )
        return first

    def _upload_rows(self, host, dt, lay, tgt):
        mp = ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=tuple(self.md.shape))
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dt, layout=lay, mesh_mapper=mp), tgt)

    def _seed_rows(self, rows, first, res, prompts_by_row, sampling):
        """MTP seed round for the new rows: pair (h_{L-1}, x_L) at position L-1 gives the first K drafts (greedy). Live rows of the batch are protected: their state is re-loaded from the mirror afterwards and the junk
        the seed run writes lies at positions >= their pending position (overwritten by their next round before being read).
        """
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import HID
        from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

        md, rn, S, K = self.md, self.runner, self.S, self.K
        pend, pos = list(self.pend), list(self.pos)
        for r in range(ROWS):
            if r in rows:
                pend[r], pos[r] = first[r], len(prompts_by_row[r]) - 1
            elif not self.live[r]:
                pend[r], pos[r] = PAD, self.park_pos(r)
        rn.load(pend, pos, [[-1] * K for _ in range(ROWS)], cap=[0] * ROWS)
        rn.set_park([0] * ROWS, self.park_pos_all())  # the seed tail must not park the rows being seeded
        a_seed = ttnn.from_torch(
            torch.tensor([[pend[r]] * S for r in range(ROWS)], dtype=torch.int32).reshape(ROWS * S, 1),
            device=md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, None), mesh_shape=tuple(md.shape)),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h = torch.zeros(ROWS, HID)
        for r in rows:
            h[r] = res.hidden[r]
        h_seed = upload(
            md, h.reshape(ROWS, 1, 1, HID).repeat(1, 1, S, 1).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT, row_dim=0
        )
        rn._set_variant("G")  # the seed draft is greedy (the last capture left the sampled variant armed)
        rn.sd.derive(rn.tail.pos)
        rn.tail.run(a_seed, h_seed, derived=True)
        ttnn.synchronize_device(md)
        ttnn.deallocate(a_seed)
        ttnn.deallocate(h_seed)
        st = rn.read_state()
        for r in rows:
            self.pend[r], self.pos[r] = st[r]["tok"][0], st[r]["pos"]
            self.drafts[r] = list(st[r]["drafts"])
            self.live[r] = True
            sp = sampling.get(r, (0.0, -1, 1.0, 0))
            self.temp[r], self.seeds[r] = float(sp[0] or 0.0), int(sp[3]) & 0x7FFFFFFF
            assert self.pend[r] == first[r], (r, self.pend[r], first[r])
        self.dirty = True  # live rows' device state was overwritten by the seed run: reload the mirror
        self.q_valid = False
        self.params_dirty = True

    def release(self, row):
        """request on this mesh row finished: park the row (pos 0, pad token) at the next round."""
        self.live[row] = False
        self.pend[row], self.pos[row] = PAD, self.park_pos(row)
        self.drafts[row] = [-1] * max(self.K, 1)
        self.dirty = True

    # ------------------------------------------------------------------------------------------------ spec rounds
    def round(self, caps, want_sampled, temps=None, seeds=None, plugin_block=None):
        """one speculative round for all four mesh rows. caps[r] = max drafts that may be accepted (0 for idle / draftless rows); want_sampled -> the lossless sampled trace (temps / seeds per row);
        plugin_block {row: (tokens [S], pos)} overrides the mirror when the plugin's candidate block differs. Returns per row dict(a, m, pos, drafts, tok).
        """
        rn = self.runner
        tA = time.perf_counter()
        pf = os.environ.get("HY4_SPEC_PIPE_FILE")
        if pf and self.rounds % 16 == 0:  # live A/B switch (a file holding 0 / 1) without a reboot
            try:
                with open(pf) as fh:
                    want = fh.read().strip() == "1"
                if want != self.pipe:
                    self.drain()
                    self.pipe = want
            except OSError:
                pass
        for r in range(ROWS):
            if not self.live[r]:
                caps[r] = 0
        idle = not all(self.live)
        if plugin_block:
            for r, (tk, p) in plugin_block.items():
                dr = [int(x) for x in tk[1 : 1 + self.K]]
                c = caps[r]
                if self.pend[r] != int(tk[0]) or self.pos[r] != int(p) or self.drafts[r][:c] != dr[:c]:
                    logger.warning(
                        f"HY4 round: row {r} mirror (tok {self.pend[r]}, pos {self.pos[r]}, drafts {self.drafts[r]}) != plugin block (tok {int(tk[0])}, pos {int(p)}, drafts {dr}): reload from the plugin's block"
                    )
                    self.pend[r], self.pos[r] = int(tk[0]), int(p)
                    self.drafts[r] = dr[:c] + [-1] * (self.K - c)
                    self.dirty = True
                    self.q_valid = False
        v = "S" if (want_sampled and self.sampled) else "G"
        park = tuple(0.0 if lv else 1.0 for lv in self.live)
        key = (
            park,
            tuple(self.park_pos_all()),
            tuple(caps),
            v,
            tuple(temps) if (v == "S" and temps is not None) else None,
            tuple(seeds) if (v == "S" and seeds is not None) else None,
        )
        hit = self.inflight is not None and not self.dirty and self.inflight == key
        if self.inflight is not None and not hit:
            self.drain()
        tB = time.perf_counter()
        if hit:
            self.pipe_hits += 1
        else:
            if self.dirty or (idle and not self.pipe) or (self.pipe and self._since_load >= self.pipe_resync):
                rn.load(self.pend, self.pos, self.drafts, cap=caps)
                self.dirty = False
                self._since_load = 0
                if idle:
                    for r in range(ROWS):
                        if not self.live[r]:
                            self.pend[r], self.pos[r] = PAD, self.park_pos(r)
            else:
                rn.set_cap(caps)
            rn.set_park(park, self.park_pos_all())
            if v == "S":
                if temps is not None:
                    new = (list(temps), list(seeds))
                    if new != (self.temp, self.seeds) or self.params_dirty:
                        self.temp, self.seeds = new
                        self.params_dirty = True
                if self.params_dirty:
                    self.samp.set_params([1e4 if t <= 0 else 1.0 / t for t in self.temp], self.seeds)
                    self.params_dirty = False
                if not self.q_valid:
                    self.samp.load_onehot(self.drafts)
                    self.q_valid = True
            else:
                self.q_valid = False
            rn.submit(v)
        tC = time.perf_counter()
        rd = rn.read_enqueue()  # behind the (possibly speculative) round, before the next one
        self.inflight = None
        if self.pipe and self._since_load < self.pipe_resync and (not idle or self.pipe_idle):
            rn.submit(v)  # speculative next round: same caps / variant / params
            self.inflight = key
            self._since_load += 1
        tD = time.perf_counter()
        st = rn.read_finish(rd)
        tE = time.perf_counter()
        self.t_round += tE - tB
        self.rounds += 1
        if self.tl is not None:
            self.tl.append(
                dict(
                    t=tA, prep=tB - tA, upload_submit=tC - tB, read_enq_spec_submit=tD - tC, wait_read=tE - tD, hit=hit
                )
            )
        for r in range(ROWS):
            if self.live[r]:
                self.pend[r], self.pos[r] = st[r]["tok"][0], st[r]["pos"]
                self.drafts[r] = list(st[r]["drafts"])
        if idle and not self.pipe:
            for r in range(ROWS):
                if not self.live[r]:
                    self.pend[r], self.pos[r] = PAD, self.park_pos(r)
        return st

    # ------------------------------------------------------------------------------------------------ warm-up (before any request: all persistent tensors exist before the traces)
    def _dram(self, tag):
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

        v = dram_view(self.md)
        logger.info(
            f"HY4 dram[{tag}]: alloc {v['alloc_per_bank'] / 1e6:.0f} MB/bank, free {v['free_per_bank'] / 1e6:.0f} MB/bank, largest free {v['largest_free_per_bank'] / 1e6:.0f} MB"
        )

    # B22fix (root cause of the HY4_NORM_FUSE=1 wedge, see program/product-integ/RESULT_B22FIX.md): L1 buffers are allocated TOP-DOWN and the L1 region that is legal for buffers next to the static CB region of the
    # big programs is only the top ~7.7 KB/bank (a program's CBs may reach address 1494400 of a 1502144 top: a guard / stash block below that is rejected: "Statically allocated circular buffers clash with L1 buffers",
    # or hangs a capture). The trace scratch (decode L1 intermediates) and every persistent L1 buffer created LAZILY after the captures (the GlobalSemaphores a new all_gather / all_broadcast workload allocates for a
    # prompt shape the warm-up did not run, alive as long as the program cache entry) therefore share the same few KB: the next trace replay rewrites such a semaphore and the next prefill using that workload hangs on it
    # (first seen at the first prompt chunk past the 3072-token warm-up, after one decode). HY4_NORM_OUT=dram keeps the B22 fused-norm outputs (6 KB/bank, written at the very top) out of that region. This only logs
    # where the next lazily created persistent L1 buffer would land.
    def _l1_report(self):
        self._l1_log("after captures")
        pr = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.md, ttnn.L1_MEMORY_CONFIG
        )
        logger.info(
            f"HY4 L1 probe after captures: a new persistent L1 buffer would be placed at addr {pr.buffer_address()}"
        )
        ttnn.deallocate(pr)

    # B39 (RESULT_B39.md): the Fabric Mux V2 of selective_reduce_combine / moe_compute carves RAW L1 from the allocator base upward and fits its buffers-per-channel to mesh.lowest_occupied_compute_l1_address() AT PROGRAM-BUILD
    # time (the program cache does not key on it). Any L1 buffer that is created later and sits BELOW that address (lazily created persistent global semaphores, trace scratch moved down by a stash) shifts every later
    # eager transient down by its size and, when it is larger than the slack of that program's mux map, the transients overwrite the map tail: first long prompt hangs in the MoE combine (mux free slots).
    # (superseded, negative result: HY4_L1_GUARD_HOLD=N) hold N bytes/bank of L1 from the start of warm() (before any program is built) until after the captures, then free it: every build sees a floor N lower than the later steady state, so a later
    # persistent buffer of up to N bytes lands in the freed hole and no later transient is lower than the build snapshot. HY4_L1_MAP=1: log the L1 allocator probe at every warm-up milestone (default off).
    def _l1_probe(self, tag):
        if os.environ.get("HY4_L1_MAP", "0") != "1":
            return
        try:
            pr = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.md, ttnn.L1_MEMORY_CONFIG
            )
            a = pr.buffer_address()
            ttnn.deallocate(pr)
            v = ttnn.get_memory_view(self.md, ttnn.BufferType.L1)
            logger.info(
                f"HY4 L1MAP[{tag}]: probe {a} alloc/bank {v.total_bytes_allocated_per_bank} free/bank {v.total_bytes_free_per_bank} largest_free {v.largest_contiguous_bytes_free_per_bank} guard {getattr(self, '_l1_guard_addr', None)}"
            )
        except Exception as e:  # noqa: BLE001
            logger.info(f"HY4 L1MAP[{tag}]: failed {e!r}")

    def _l1_wrap_moe(self):
        """HY4_L1_GUARD_PAD=N (bytes per bank): the first call of every distinct moe_compute signature (= the program build, where the fabric-mux map is fitted under the lowest occupied L1 address, which at that moment is
        the bottom of the op's own transient L1 outputs) runs with an N-byte L1 pad held on top, so the build sees a floor N lower than every later call: N bytes of slack for persistent L1 buffers created after the build
        (global semaphores, eager-path constants) - the failure mode of RESULT_B39.md. The pad is freed right after the call (never persistent). HY4_L1_MAP=1: also log the allocator probe at every distinct call.
        """
        pad = int(os.environ.get("HY4_L1_GUARD_PAD", "0"))
        mp = os.environ.get("HY4_L1_MAP", "0") == "1"
        if (pad <= 0 and not mp) or getattr(self, "_moe_wrapped", False):
            return
        self._moe_wrapped = True
        pad = (pad + 31) // 32 * 32
        orig = ttnn.experimental.moe_compute
        seen = {}
        eng = self
        nb = ttnn.get_memory_view(self.md, ttnn.BufferType.L1).num_banks

        def key_of(a, k):
            sh = tuple(tuple(x.shape) if hasattr(x, "shape") else repr(x) for x in a)
            kk = tuple(sorted((n, tuple(v.shape) if hasattr(v, "shape") else repr(v)) for n, v in k.items()))
            return (sh, kk)

        def probe():
            pr = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, eng.md, ttnn.L1_MEMORY_CONFIG
            )
            addr = pr.buffer_address()
            ttnn.deallocate(pr)
            return addr

        def wrapped(*a, **k):
            if getattr(eng, "_capturing", False):
                return orig(*a, **k)
            key = key_of(a, k)
            first = key not in seen
            seen[key] = seen.get(key, 0) + 1
            padt = None
            try:
                if mp and (first or seen[key] <= 2):
                    logger.info(
                        f"HY4 L1MAP moe_compute call #{sum(seen.values())}: in0 {tuple(a[0].shape)} layer {k.get('layer_id')} first={first} probe {probe()} (distinct keys {len(seen)})"
                    )
                if first and pad > 0:
                    padt = ttnn.allocate_tensor_on_device(
                        ttnn.Shape([1, 1, nb, pad // 2]),
                        ttnn.bfloat16,
                        ttnn.ROW_MAJOR_LAYOUT,
                        eng.md,
                        ttnn.L1_MEMORY_CONFIG,
                    )
            except Exception as e:  # noqa: BLE001
                logger.info(f"HY4 L1 guard moe wrap failed {e!r}")
            try:
                return orig(*a, **k)
            finally:
                if padt is not None:
                    ttnn.deallocate(padt)

        ttnn.experimental.moe_compute = wrapped

    def _l1_guard_hold(self):
        n = int(
            os.environ.get("HY4_L1_GUARD_HOLD", "0")
        )  # NEGATIVE RESULT (b39 d1: hangs the first warm prefill); kept for the record, default off
        self._l1_guard = None
        if n <= 0:
            return
        n = (n + 31) // 32 * 32
        nb = ttnn.get_memory_view(self.md, ttnn.BufferType.L1).num_banks
        self._l1_guard = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, nb, n // 2]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self.md, ttnn.L1_MEMORY_CONFIG
        )
        self._l1_guard_addr = self._l1_guard.buffer_address()
        logger.info(f"HY4 L1 guard pad: holding {n} B/bank x {nb} banks at addr {self._l1_guard_addr}")

    def _l1_guard_release(self):
        if getattr(self, "_l1_guard", None) is not None and os.environ.get("HY4_L1_GUARD_KEEP", "0") != "1":
            ttnn.deallocate(self._l1_guard)
            self._l1_guard = None
            logger.info("HY4 L1 guard pad: released (after the captures)")

    def _l1_peak(self, fn, tag):
        """run fn (an eager round body) while a sampler thread polls the L1 allocator: the peak per-bank L1 use of the body (= the extent of its trace scratch, a lower bound: the polling can miss short peaks)."""
        import threading

        if os.environ.get("HY4_L1_PEAK", "0") != "1":
            fn()
            return
        peak = [0, 1 << 60]
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                try:
                    v = ttnn.get_memory_view(self.md, ttnn.BufferType.L1)
                    peak[0] = max(peak[0], v.total_bytes_allocated_per_bank)
                    peak[1] = min(peak[1], v.largest_contiguous_bytes_free_per_bank)
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.001)

        th = threading.Thread(target=poll, daemon=True)
        th.start()
        try:
            fn()
            ttnn.synchronize_device(self.md)
        finally:
            stop.set()
            th.join()
        logger.info(
            f"HY4 L1 peak during eager body {tag}: allocated {peak[0]} B/bank (sampled max), smallest largest-free {peak[1]} B/bank"
        )

    def _l1_log(self, tag):
        try:
            v = ttnn.get_memory_view(self.md, ttnn.BufferType.L1)
            logger.info(
                f"HY4 L1[{tag}]: banks {v.num_banks} total {v.total_bytes_per_bank} alloc {v.total_bytes_allocated_per_bank} free {v.total_bytes_free_per_bank} largest_free {v.largest_contiguous_bytes_free_per_bank}"
            )
        except Exception as e:  # noqa: BLE001
            logger.info(f"HY4 L1[{tag}]: view failed {e!r}")

    def warm(self):
        """synthetic 4 x 3072-token prefill (compiles every PF / MTP chunk program, creates every lazily cached persistent tensor: PF constants, dense-id tables) -> seed -> eager round -> trace capture(s)."""
        if self.warm_done:
            return
        self.warm_done = True
        t0 = time.time()
        self._l1_guard_hold()
        self._l1_wrap_moe()
        self._l1_probe("warm start")
        g = torch.Generator().manual_seed(0)
        n = min(3072, self.max_seq_len - 8)
        prompts = {r: torch.randint(1000, 100000, (n,), generator=g).tolist() for r in range(ROWS)}
        self._dram("before warm prefill")
        cap, self.hybrid_cap = (
            self.hybrid_cap,
            0,
        )  # the first warm prefill goes through the direct-to-pool path (creates its persistent tensors)
        first = self.prefill(prompts)
        self.hybrid_cap = cap
        self._dram("after warm prefill")
        self._l1_probe("after warm prefill")
        if (
            self.hybrid_cap > 0
        ):  # chain prefill + hand-off once before any capture: every lazily created persistent tensor of that path exists before the traces (D75-1)
            tw = time.time()
            ch_prompts = {r: prompts[r][:1024] for r in range(ROWS)}
            self.prefill(ch_prompts)
            self._dram("after warm chain prefill")
            logger.info(f"HY4 warm-up: chain prefill 4 x 1024 {time.time() - tw:.1f} s (hybrid cap {self.hybrid_cap})")
            # B38: HY4_WARM_CHAIN_LENS=n1,n2,..: extra chain prefills (4 rows each) BEFORE the captures, so the persistent L1 buffers (CCL global semaphores) that new chunk shapes create lazily exist before the trace
            # scratch is laid out (see _l1_report / RESULT_B22FIX.md: a buffer created after the captures lands in the trace scratch and a replay clobbers it)
            for ln in [int(x) for x in os.environ.get("HY4_WARM_CHAIN_LENS", "").split(",") if x.strip()]:
                tw = time.time()
                g2 = torch.Generator().manual_seed(ln)
                self.prefill(
                    {
                        r: torch.randint(1000, 100000, (min(ln, self.hybrid_cap),), generator=g2).tolist()
                        for r in range(ROWS)
                    }
                )
                self._dram(f"after warm chain prefill {ln}")
                logger.info(f"HY4 warm-up: chain prefill 4 x {ln} {time.time() - tw:.1f} s")
        if self.xrow_on:
            self._xrow_build()  # persistent stage tensors + programs before the first capture (D75-1)
        if self.blocks_on:
            self._blocks_build()
        t1 = time.time()
        if self.spec:
            # D75-1: EVERY lazily created persistent tensor of EVERY variant must exist before the FIRST capture (one allocated between two captures can sit in the first trace's scratch and is clobbered by its
            # replays): run both round bodies eagerly first, then capture. (p4 / p5: the first sampled round hung at 78 layers after greedy rounds had replayed.)
            cp = os.environ.get("HY4_WARM_CAPPOS", "")
            if (
                cp
            ):  # old serve-long path captured its trace at the LAST pool position (a trace captured at a short position produced garbage above it, serve-long RESULT.md)
                cpos = self.pages * PAGE - self.K - 4 if cp == "last" else int(cp)
                self.runner.load([PAD] * ROWS, [cpos] * ROWS, [[-1] * self.K for _ in range(ROWS)], cap=[0] * ROWS)
                self.runner.set_park([0] * ROWS, self.park_pos_all())
                logger.info(f"HY4 warm-up: capture position {cpos}")
            self._l1_peak(lambda: self.runner.body("G"), "G")
            ttnn.synchronize_device(self.md)
            if self.sampled:
                self._l1_peak(lambda: self.runner.body("S"), "S")
                ttnn.synchronize_device(self.md)
            self._dram("after eager bodies")
            self._l1_probe("after eager bodies")
            self._capturing = True
            self.runner.capture("G")
            self._dram("after G capture")
            if self.sampled:
                self.runner.capture("S")
                self._dram("after S capture")
            self._capturing = False
            self._l1_probe("after captures (guard held)")
            self._l1_guard_release()
            self._l1_report()
            self._l1_probe("after captures (guard released)")
            logger.info(f"HY4 warm-up: spec traces captured ({list(self.runner.traces)})")
        else:
            self.dl.capture(self.loop_variants)
        for r in range(ROWS):
            self.release(r)
        self.pp.set_keep(None)
        self.ret = [None] * ROWS  # the warm-up prompts are synthetic: nothing retained
        self.pfx_stats = dict(calls=0, hit_calls=0, hit_tokens=0, prompt_tokens=0, computed_tokens=0)
        self._b50_chain_guard()
        self._b50_dram_log()
        probe = os.environ.get("HY4_PF_HYBRID_PROBE", "")
        if probe:
            self._probe_chain_cap([int(x) for x in probe.split(",")])
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

        logger.info(
            f"HY4 warm-up: prefill 4 x {n} {t1 - t0:.1f} s (compile incl.), captures {time.time() - t1:.1f} s, dram {dram_view(self.md)}; first tokens {first}"
        )

    def _b50_dram_log(self):
        """HY4_B50_DRAMLOG_S=<seconds> (default 0 = off): daemon thread; polls the host-side allocator view every HY4_B50_DRAMLOG_POLL_S (default 1 s) and logs every <seconds> one line
        'HY4 DRAMLOG free=<now> min_free=<low-water since the last line> largest=<now> min_largest=<low-water> alloc_max=<peak>' (MiB per bank) for the long-uptime soak (oom15).
        """
        every = float(os.environ.get("HY4_B50_DRAMLOG_S", "0"))
        if every <= 0:
            return
        import threading

        from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

        poll = float(os.environ.get("HY4_B50_DRAMLOG_POLL_S", "1"))

        def run():
            mn_f = mn_l = 1 << 62
            mx_a = 0
            t_last = time.time()
            while True:
                v = dram_view(self.md)
                if v.get("alloc_per_bank", -1) >= 0:
                    mn_f, mn_l = min(mn_f, v["free_per_bank"]), min(mn_l, v["largest_free_per_bank"])
                    mx_a = max(mx_a, v["alloc_per_bank"])
                    if time.time() - t_last >= every:
                        logger.info(
                            f"HY4 DRAMLOG free={v['free_per_bank'] / 1048576:.1f} min_free={mn_f / 1048576:.1f} largest={v['largest_free_per_bank'] / 1048576:.1f} min_largest={mn_l / 1048576:.1f} alloc_max={mx_a / 1048576:.1f} MiB/bank"
                        )
                        mn_f = mn_l = 1 << 62
                        mx_a = 0
                        t_last = time.time()
                time.sleep(poll)

        threading.Thread(target=run, daemon=True, name="b50-dramlog").start()

    def _b50_chain_guard(self):
        """HY4_B50_CHAIN_GUARD=1 (default off): size the chain (hybrid) prefill cap from the DRAM that is really free after the captures. The chain prefill keeps a replicated cache + indexer keys (+ per-layer masks) that grow
        with the prompt (measured per token at warm-up: ``pp.chain_bpt``, B per bank) and needs HY4_B50_CHAIN_RESERVE_MB (default 128) of headroom for its transients (iHC collapse 50 MB/bank with the stock sum, MoE / attention
        temporaries; oom15 RESULT_OOM15: e15 died at a ~12k chain prompt with 95 MB/bank free and a 87 KB-too-small largest block). cap = min(hybrid_cap, (free - reserve) / bpt) rounded down to 512; 0 = the pool prefill only.
        A larger bpt estimate can be forced with HY4_B50_CHAIN_BPT (B/bank/token). Prompts above the cap use the direct-to-pool prefill (slower, memory-flat).
        """
        if os.environ.get("HY4_B50_CHAIN_GUARD", "0") != "1" or self.hybrid_cap <= 0:
            return
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

        dv = dram_view(self.md)
        bpt = float(os.environ.get("HY4_B50_CHAIN_BPT", "0")) or float(getattr(self.pp, "chain_bpt", 0.0)) or 9600.0
        reserve = int(float(os.environ.get("HY4_B50_CHAIN_RESERVE_MB", "128")) * (1 << 20))
        room = (
            min(dv["free_per_bank"], dv["largest_free_per_bank"] * 2)
            - reserve
            - float(getattr(self.pp, "chain_fixed", 0.0))
        )
        cap = max(0, int(room / bpt) // 512 * 512)
        new = min(self.hybrid_cap, cap)
        logger.info(
            f"HY4 B50 chain guard: free {dv['free_per_bank'] >> 20} MiB/bank (largest {dv['largest_free_per_bank'] >> 20}), reserve {reserve >> 20} MiB, bpt {bpt:.0f} B/bank/token -> cap {cap}; hybrid cap {self.hybrid_cap} -> {new}"
        )
        self.hybrid_cap = new

    def _probe_chain_cap(self, lens):
        """HY4_PF_HYBRID_PROBE=n1,n2,...: after the captures, chain prefill of ONE n-token prompt (row 0; the cache size does not depend on the number of rows) per length, ascending; logs the DRAM low-water mark
        and stops at the first failure (the state after a failure is not trusted: probe-only boot, TERM and reset afterwards).
        """
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

        g = torch.Generator().manual_seed(1)
        for n in lens:
            toks = torch.randint(1000, 100000, (n,), generator=g).tolist()
            t0 = time.time()
            try:
                self.pp.mem = None
                res = self.pp.prefill([toks], keep_rows=[0], chain=True)
                ttnn.synchronize_device(self.md)
                logger.info(
                    f"HY4 PROBE chain n={n}: OK {time.time() - t0:.1f} s (handoff {res.handoff_sec:.1f} s), dram {dram_view(self.md)}"
                )
            except Exception as e:  # noqa: BLE001
                logger.error(f"HY4 PROBE chain n={n}: FAIL {type(e).__name__}: {str(e)[:300]}")
                break
        self.pp.set_keep(None)
