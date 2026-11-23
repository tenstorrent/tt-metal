# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4PoolPrefill: model-level DIRECT-TO-POOL chunked prefill of the 78-layer chain (task pf-integrate).

Same build contract as ``Hy4PoolDecoder`` (INTERFACE.md of integ): the chain must be built with HY4_POOL=1 HY4_POOL_PF=1 HY4_KV_FMT=scaled_fp8 (every layer's attention is ``Hy4AttentionPF``), and the pool
(``Hy4PoolDecoder``: kv pools + key pools + PoolCtx) must exist BEFORE the first prefill. ``prefill(prompts)`` writes every prompt row into the sequence-sharded pool (no replicated cache, no hand-off)
and returns the first generated token per user; decode then continues with the unchanged ``Hy4PoolDecoder`` (``prepare_decode`` + capture / run / read_argmax).

One user per mesh row (4 on the 4x8 mesh; 1 on a 1x8 slot for wiring tests). Chunks of ``pf_rows`` (512 or 1024) rows per user advance in lock step (SPMD): a user whose prompt is shorter than the longest
one computes pad tokens (id PAD) in the remaining chunks; those pool rows lie at positions >= its length, are masked by the decode (ids < pos + 1) and overwritten as it generates.
MTP prompt KV (layer 78) is NOT written (see INTERFACE.md of this task): a spec round at the first step reads an empty MTP prompt cache (lower acceptance only; the target verification stays exact).
"""
import threading
import time
from dataclasses import dataclass, field

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_chain import dram_view

PAD = 120002  # same pad id as the needle runners


@dataclass
class PrefillResult:
    first: list  # first generated token per user (greedy argmax of the last prompt token)
    lens: list
    chunk_sec: list = field(default_factory=list)
    total_sec: float = 0.0
    hidden: list = (
        None  # optional: final normed hidden of the last prompt token per user ([6144] fp32), keep_hidden=True
    )
    handoff_sec: float = 0.0
    logits: list = None  # optional: logits row of the last prompt token per user (fp32 [V]), keep_logits=True


def decode_bucket(max_len, pages):
    """bucket_tokens for ``dec.ctx.set_bucket(bucket, "topk")`` after a prefill of prompts of at most max_len tokens (as in the needle runners)."""
    return min(pages * 512 // 256 * 256, (-(-max_len // 4096)) * 4096)


@dataclass
class PfState:
    """carried state of one stepwise prefill call (HY4_CHUNKED_PF): everything the monolithic loop kept in locals."""

    prompts: list
    lens: list
    PF: int
    nch: int
    c: int
    start: int
    res: PrefillResult
    mtp: object
    keep_hidden: bool
    keep_logits: bool
    chain: bool
    keep_rows: list = None
    t0: float = 0.0
    ids_by_layer: dict = field(default_factory=dict)
    finished: set = field(default_factory=set)

    @property
    def done(self):
        return self.c >= self.nch


class MemSampler:
    """background sampler of the mesh DRAM allocator (host-side state, updated at op dispatch): records the peak allocated bytes per bank (= low-water mark of the free DRAM) between reset() calls.
    Opt-in (mem_sample_s > 0): get_memory_view from a second thread every mem_sample_s seconds (misses allocations shorter than the interval; chunk-boundary values are a lower bound of use).
    """

    def __init__(self, md, interval):
        self.md, self.interval, self.stop, self.active = md, interval, False, True
        self.peak, self.min_free, self.n = 0, 1 << 62, 0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self.stop:
            if not self.active:
                time.sleep(0.01)
                continue
            v = dram_view(self.md)
            a = v.get("alloc_per_bank", -1)
            if a >= 0:
                self.peak = max(self.peak, a)
                self.min_free = min(self.min_free, v["free_per_bank"])
                self.n += 1
            time.sleep(self.interval)

    def reset(self):
        r = (self.peak, self.min_free, self.n)
        self.peak, self.min_free, self.n = 0, 1 << 62, 0
        return r


class Hy4PoolPrefill:
    def __init__(self, dec, pf_rows=512, log=None, mem_sample_s=0.0, mem_every=1):
        assert pf_rows in (512, 1024)
        self.dec, self.ch, self.md = dec, dec.ch, dec.ch.md
        self.rows = int(self.md.shape[0])
        self.cols = int(self.md.shape[1])
        self.pf_rows = pf_rows
        self.log = log or (lambda **kw: None)
        self.mem_every = max(1, int(mem_every))
        self.mem = MemSampler(self.md, mem_sample_s) if mem_sample_s > 0 else None
        self.mem_peak_per_bank = 0  # max over all prefill chunks of the sampled peak allocation
        self.mem_min_free_per_bank = 1 << 62
        for ly in self.ch.layers:
            assert hasattr(ly.attn, "_pf_chunk512"), "chain not built with HY4_POOL=1 HY4_POOL_PF=1"
        # serving: a prefill may run while OTHER mesh rows hold live decode users. All four rows always compute (lock step); the pool writes of the rows that are not being prefilled are suppressed with the masked
        # slice_copy kernel (colid element 2 = 1 on a skipped row). The tensor is persistent: allocated here, before any decode trace is captured, and rewritten on the host (set_keep).
        from models.demos.blackhole.hy4_preview.tt import hy4_pool as HP
        from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

        self._HP = HP
        self.wr_mask = upload(
            self.md, HP.colid_masked_host([]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, shard_dim=1, row_dim=0
        )
        self.set_keep(None)
        self.mesh_map = ttnn.ShardTensor2dMesh(self.md, dims=(0, None), mesh_shape=tuple(self.md.shape))

    def set_keep(self, keep_rows):
        """rows whose pool pages this prefill may write (None = all). The masked write path is active from the first call on (``ctx.pf_wr_colid``)."""
        skip = [] if keep_rows is None else [r for r in range(self.rows) if r not in set(keep_rows)]
        self.keep_cur = (
            None if keep_rows is None else tuple(sorted(keep_rows))
        )  # the mask now on the device (stepwise prefill compares)
        host = ttnn.from_torch(
            self._HP.colid_masked_host(skip),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.md, dims=(0, 1), mesh_shape=tuple(self.md.shape)),
        )
        ttnn.copy_host_to_device_tensor(host, self.wr_mask)
        self.dec.ctx.pf_wr_colid = self.wr_mask

    def pages_needed(self, max_len):
        return (-(-max_len // self.pf_rows)) * self.pf_rows // 512

    def _host_rows(self, t, per_row_cols):
        """mesh tensor with one distinct value per mesh row (replicated over columns) -> torch [rows, ...] from device (r, 0)."""
        devs = ttnn.get_device_tensors(t)
        return torch.stack([ttnn.to_torch(devs[r * self.cols]).float() for r in range(self.rows)])

    def mtp_prompt_chunk(self, mtp, hn, c, prompts, PF):
        """MTP layer prompt KV of chunk c: row t of user r = pair (h_t [post model.norm hidden of the main model], token t + 1) at position c * PF + t, written into the MTP layer's pool pair by the layer's own
        PF attention (outputs discarded: only the KV / key rows matter; +1/79 of a prefill chunk). The pair of the last prompt position (token = the first generated token) is NOT known yet: its row is written by the
        seed round (the next token of a row beyond the prompt is PAD: junk rows at positions >= the prompt length are overwritten before they are read).
        """
        from models.demos.blackhole.hy4_preview.tt.hy4_chain import rms_exact

        ly = mtp.ly
        nxt = torch.full((self.rows, PF), PAD, dtype=torch.long)
        for r in range(self.rows):
            seg = prompts[r][c * PF + 1 : (c + 1) * PF + 1]
            if seg:
                nxt[r, : len(seg)] = torch.tensor(seg)
        tt_tok = ttnn.from_torch(
            nxt.to(torch.int32),
            device=self.md,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.mesh_map,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        e = mtp.embed(tt_tok)
        ttnn.deallocate(tt_tok)
        x = mtp.pre(e, hn)
        ttnn.deallocate(e)
        h1 = rms_exact(x, ly.w_in32, mtp.eps)
        ttnn.deallocate(x)
        o, ids = ly.attn.prefill_chunk(h1, c * PF)
        for t in (h1, o):
            ttnn.deallocate(t)
        if ids is not None:
            ttnn.deallocate(ids)
        from models.demos.blackhole.hy4_preview.tt.hy4_attention import clear_rope_cache

        clear_rope_cache()  # B49 rope tables must not outlive the chunk (D75-1)

    def _chain_mode(self, on):
        for ly in self.ch.layers:  # the MTP layer (ch.mtp_layers) always writes the pool directly
            ly.attn.chain_mode = on

    def _handoff(self, nrows):
        """chain prefill caches (replicated SCALED_FP8 rows + indexer keys) -> pool, only the mesh rows of the current mask (``wr_mask``, set_keep); then free the chain caches (``Hy4ServePool.handoff_rows``)."""
        HP = self._HP
        assert nrows % HP.CHUNK == 0 and nrows // HP.CHUNK <= self.dec.pages
        npg = nrows // HP.CHUNK
        for ly, kv, kp in zip(self.ch.layers, self.dec.kv_pools, self.dec.key_pools):
            rows = ly.attn.cache.rows
            assert int(rows.shape[2]) >= nrows and rows.dtype == ttnn.fp8_e4m3, (rows.shape, rows.dtype)
            HP.slice_copy(rows, kv, self.wr_mask, 64, HP.ROWB, npg, kernel="slice_copy_masked.cpp")
            if kp is not None:
                k = ly.attn.indexer.keys
                kb = ttnn.typecast(k, ttnn.bfloat8_b) if k.dtype != ttnn.bfloat8_b else k
                kv4 = ttnn.reshape(kp, [1, 1, self.dec.alloc_pages * 64, 128])
                HP.slice_copy(kb, kv4, self.wr_mask, 8, HP.KEYTILE, npg, kernel="slice_copy_masked.cpp")
                if kb is not k:
                    ttnn.deallocate(kb)
        ttnn.synchronize_device(self.md)
        self.ch.reset()
        from models.demos.blackhole.hy4_preview.tt.hy4_indexer import drop_shared_masks

        drop_shared_masks()  # B50 shared causal mask (no-op when HY4_B50_MASKSHARE is off)
        for ly in self.ch.layers:  # lazily cached indexer tensors grow with the prompt: drop them (serve-long D75-1)
            ix = getattr(ly.attn, "indexer", None)
            if ix is None:
                continue
            for d_ in (getattr(ix, "_sent", None), getattr(ix, "_mask_full", None)):
                if d_:
                    for t in d_.values():
                        ttnn.deallocate(t)
                    d_.clear()
        ttnn.synchronize_device(self.md)

    def prefill(
        self,
        prompts,
        keep_hidden=False,
        keep_logits=False,
        sync_every_chunk=True,
        mtp=None,
        keep_rows=None,
        chain=False,
        start=0,
    ):
        """prompts: list (<= rows) of token-id lists (a missing / empty user is a pad prompt). Returns PrefillResult.
        mtp: an Hy4MTP -> the MTP layer prompt KV is written per chunk (``mtp_prompt_chunk``). keep_rows: the mesh rows being prefilled (the pool writes of the others are suppressed: live decode users).
        """
        ch, md, PF = self.ch, self.md, self.pf_rows
        assert start == 0 or not chain, "prefix reuse (start > 0) needs the direct-to-pool path"
        assert start % PF == 0, (start, PF)
        if chain:
            PF = 512  # chain prefill: chunks of 512 rows, replicated cache, hand-off at the end (HY4_PF_HYBRID)
            self._chain_mode(True)
        try:
            return self._prefill(prompts, keep_hidden, keep_logits, sync_every_chunk, mtp, keep_rows, chain, PF, start)
        finally:
            if chain:
                self._chain_mode(False)

    # ---- stepwise API (HY4_CHUNKED_PF): begin / chunk / finish over the SAME chunk body as the monolithic loop ----------------------------------------------------------------
    def _begin(self, prompts, keep_hidden, keep_logits, mtp, keep_rows, chain, PF, start=0):
        if keep_rows is not None:
            self.set_keep(keep_rows)
        prompts = [list(p) for p in prompts] + [[PAD]] * (self.rows - len(prompts))
        assert len(prompts) == self.rows
        lens = [max(len(p), 1) for p in prompts]
        L = max(lens)
        nch = -(-L // PF)
        assert self.dec.pages >= self.pages_needed(L), (self.dec.pages, self.pages_needed(L))
        res = PrefillResult(first=[None] * self.rows, lens=lens)
        if keep_hidden:
            res.hidden = [None] * self.rows
        if keep_logits:
            res.logits = [None] * self.rows
        return PfState(
            prompts=prompts,
            lens=lens,
            PF=PF,
            nch=nch,
            c=start // PF,
            start=start,
            res=res,
            mtp=mtp,
            keep_hidden=keep_hidden,
            keep_logits=keep_logits,
            chain=chain,
            keep_rows=None if keep_rows is None else sorted(keep_rows),
            t0=time.time(),
        )

    def _chunk(self, st, sync_every_chunk=True):
        """run chunk st.c; returns the rows whose last prompt token lay in this chunk (their res.first / logits / hidden are filled). The loop body of the monolithic ``_prefill``, unchanged."""
        ch, md, PF, c = self.ch, self.md, st.PF, st.c
        prompts, lens, res, mtp, chain = st.prompts, st.lens, st.res, st.mtp, st.chain
        keep_logits, keep_hidden = st.keep_logits, st.keep_hidden
        ids = torch.full((self.rows, PF), PAD, dtype=torch.long)
        for r in range(self.rows):
            seg = prompts[r][c * PF : (c + 1) * PF]
            if seg:
                ids[r, : len(seg)] = torch.tensor(seg)
        tc = time.time()
        if self.mem:
            self.mem.reset()
            self.mem.active = (
                c % self.mem_every
            ) == 0  # sampling slows the host ~5x: only every mem_every-th chunk is sampled (its time is not representative)
        x4, st.ids_by_layer = ch.prefill(ids, start=c * PF, ids_by_layer=st.ids_by_layer)
        enders = [r for r in range(self.rows) if c * PF <= lens[r] - 1 < (c + 1) * PF]
        if mtp is not None and not enders:
            hn, logits = ch.head(x4)
            ttnn.deallocate(logits)
            self.mtp_prompt_chunk(mtp, hn, c, prompts, PF)
            ttnn.deallocate(hn)
        if enders:
            hn, logits = ch.head(x4)
            if mtp is not None:
                self.mtp_prompt_chunk(mtp, hn, c, prompts, PF)
            for r in enders:
                # only the last-prompt-position row of every finishing user crosses PCIe (the full [4, PF, V] logits of a chunk are ~1 GB: ~4 s per prefill call); the slice is taken per user at its own position
                last = lens[r] - 1 - c * PF
                ls = ttnn.slice(logits, [0, 0, last, 0], [1, 1, last + 1, int(logits.shape[3])])
                lg = ttnn.to_torch(
                    ls, mesh_composer=ttnn.ConcatMesh2dToTensor(md, dims=(0, 3), mesh_shape=tuple(md.shape))
                ).reshape(self.rows, -1)
                ttnn.deallocate(ls)
                row = lg[r].float()
                res.first[r] = int(row.argmax())
                if keep_logits:
                    res.logits[r] = row
                if keep_hidden:
                    hs = ttnn.slice(hn, [0, 0, last, 0], [1, 1, last + 1, int(hn.shape[3])])
                    res.hidden[r] = self._host_rows(hs, True).reshape(self.rows, -1)[r]
                    ttnn.deallocate(hs)
            ttnn.deallocate(hn)
            ttnn.deallocate(logits)
        if sync_every_chunk:
            ttnn.synchronize_device(md)
        ttnn.deallocate(x4)
        dt = time.time() - tc
        res.chunk_sec.append(dt)
        mem = {}
        if self.mem:
            pk, mf, n = self.mem.reset()
            if n:
                self.mem_peak_per_bank = max(self.mem_peak_per_bank, pk)
                self.mem_min_free_per_bank = min(self.mem_min_free_per_bank, mf)
            mem = dict(peak_alloc_per_bank=pk, min_free_per_bank=mf, mem_samples=n)
        self.log(
            event="prefill_chunk", c=c, of=st.nch, chunk_sec=round(dt, 3), sec=round(time.time() - st.t0, 1), **mem
        )
        st.c += 1
        return enders

    def _release_state(self, st):
        for t in st.ids_by_layer.values():
            ttnn.deallocate(t)
        st.ids_by_layer = {}

    def prefill_begin(self, prompts, keep_hidden=False, keep_logits=False, mtp=None, keep_rows=None, start=0):
        """stepwise pool prefill (never the chain path: the replicated chain cache cannot be carried across decode rounds). Returns a PfState; run ``prefill_step`` until ``st.done``, then ``prefill_finish``."""
        assert start % self.pf_rows == 0, (start, self.pf_rows)
        return self._begin(prompts, keep_hidden, keep_logits, mtp, keep_rows, False, self.pf_rows, start)

    def prefill_step(self, st, n=1, sync_every_chunk=True):
        """advance up to n chunks; returns the rows that finished in them (in order). Chunks after a row finished do not write its pool pages (the mask drops it: it may be decoding by then)."""
        done_rows = []
        for _ in range(n):
            if st.c >= st.nch:
                break
            if st.keep_rows is not None:
                keep = [r for r in st.keep_rows if r not in st.finished]
                if tuple(keep) != getattr(
                    self, "keep_cur", None
                ):  # another stepwise call (or a decode-time seed) may have changed the mask
                    self.set_keep(keep)
            fin = self._chunk(st, sync_every_chunk)
            st.finished.update(fin)
            done_rows += fin
        return done_rows

    def prefill_finish(self, st):
        assert st.c >= st.nch, "prefill_finish before the last chunk"
        self._release_state(st)
        st.res.total_sec = time.time() - st.t0
        return st.res

    def prefill_abort(self, st):
        self._release_state(st)

    def _prefill(self, prompts, keep_hidden, keep_logits, sync_every_chunk, mtp, keep_rows, chain, PF, start=0):
        md = self.md
        st = self._begin(prompts, keep_hidden, keep_logits, mtp, keep_rows, chain, PF, start)
        chain_alloc = []
        a0 = (
            dram_view(md).get("alloc_per_bank", -1) if chain else -1
        )  # B50: persistent growth of a chain prefill (replicated cache + keys + per-layer lazies), measured per token
        while st.c < st.nch:
            self._chunk(st, sync_every_chunk)
            if chain:
                chain_alloc.append(dram_view(md).get("alloc_per_bank", -1))
        res = st.res
        self._release_state(st)
        if chain:
            a1 = dram_view(md).get("alloc_per_bank", -1)
            if a0 >= 0 and len(chain_alloc) >= 2 and chain_alloc[-1] >= chain_alloc[0]:
                # per-bank bytes that stay allocated per prompt token (slope between the first and the last chunk) and the chunk independent part (ids carried between layers etc.)
                self.chain_bpt = (chain_alloc[-1] - chain_alloc[0]) / float((len(chain_alloc) - 1) * PF)
                self.chain_fixed = max(0.0, chain_alloc[0] - a0 - self.chain_bpt * PF)
                self.chain_peak_bytes = max(getattr(self, "chain_peak_bytes", 0), a1 - a0)
            th = time.time()
            self._handoff(st.nch * 512)
            res.handoff_sec = time.time() - th
        res.total_sec = time.time() - st.t0
        return res

    def prepare_decode(self, res, set_bucket=True):
        """after prefill: bucket + first decode inputs (tokens = res.first, pos = prompt lengths). The caller then calls dec.capture() once and dec.set_inputs / dec.run / dec.read_argmax per step."""
        dec = self.dec
        if set_bucket:
            dec.ctx.set_bucket(decode_bucket(max(res.lens), dec.pages), "topk")
        dec.set_inputs(res.first, res.lens)
