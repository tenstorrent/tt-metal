# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Device-resident decode loop ("Plan B"): one captured trace per decode step; the host only enqueues.

    trace = [ model.ttnn_decode_forward(on_device_logits=True)   # embedding(tokens) ... lm_head, plus_one(pos), plus_one(rot)
              ttnn.untilize(logits) -> ROW_MAJOR bf16 DRAM
              push (rows 0..B-1 -> the firmware's logits zone; streamed at B > 1: doorbell first, landed count per row)
              notify (non-streamed only)                          # req_seq += 1, doorbell
              wait                                                # x280 wrote tokens[] in place, then done_seq ]

The wait is LAST: iteration k consumes the token tensor (host-written before iteration 0, x280-written afterwards);
when iteration k ends the next token is already in place, so the first iteration needs no special request and the
last one leaves no request in flight (req_seq == done_seq at the end of every iteration).
The x280 side is the sampling firmware (tools/l2cpu sampling firmware, stream-capable build).

Several L2CPU tiles (session from l2cpu_sampler.l2cpu_bootstrap(tiles=N), batch > 1): tile i serves the contiguous
users split_users(B, N)[i] with its own firmware, region and link block (ctrl.user_base = its first user, so the
draw and the token write use the global user index). The trace's push is one program with one core per tile
(streamed: each core rings its own tile first), the wait covers every tile (X280_PLANB_WAIT=all: one kernel
polling every done_seq under one bound, default; each: one wait program per tile). Batch 1 always uses tile 0 only.
"""
from __future__ import annotations

import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deps  # noqa: E402,F401  (bindings to the other l2cpu components, see deps.py)

from l2cpu.sampling import layout as _S  # noqa: E402
from l2cpu_ops import LINK_BLOCK_OFF, L2cpuMultiOps, L2cpuOps  # noqa: E402
from tensix import ops as link_ops  # noqa: E402  (link block offsets, kernels/l2cpu_link.h)

ROW = 303_872  # Qwen3 vocab 151,936 x bf16
# region offsets: link block words (the firmware embeds l2cpu_link.h at LINK_BLOCK_OFF) and firmware control words
OFF_REQ = LINK_BLOCK_OFF + link_ops.LINK_OFF_REQ_SEQ
OFF_DONE = LINK_BLOCK_OFF + link_ops.LINK_OFF_DONE_SEQ
OFF_WAIT_STATUS = LINK_BLOCK_OFF + link_ops.LINK_OFF_WAIT_STATUS
OFF_DIAG = LINK_BLOCK_OFF + link_ops.LINK_OFF_DIAG
OFF_STEP_BASE, OFF_RING_WR, OFF_RING = _S.L2S_OFF_STEP_SEQ_BASE, _S.L2S_OFF_RING_WR, _S.L2S_OFF_RING
RING_SLOT, RS_TOK = _S.L2S_RING_SLOT_SIZE, _S.L2S_RS_TOK
RAW_DRAM_XY = {
    0: (0, 11),
    1: (0, 10),
    2: (0, 9),
    3: (0, 5),
    4: (9, 11),
    5: (9, 3),
    6: (9, 9),
    7: (9, 5),
}  # raw NoC0 of DRAM Dn


class StepTimeout(RuntimeError):
    """A queued step's wait op hit its bound (wait status word set) or the firmware reported an error."""


class PlanB:
    """Sampling firmware session: booted here (l2cpu.sampling.fw.boot) or attached to the decode harness's session
    (harness option x280=True boots it before any trace capture)."""

    def __init__(self, mesh, log=print, uncached=False, session=None, streamed=None):
        self.uncached = uncached  # 32-row batches: push to the uncached zone, x280 reads it uncached (RVV)
        # streamed: stream protocol ROWS (req_seq + doorbell at the start of the push, landed count per row);
        # needs an image with L2S_BUILD_STREAM. Default: streamed at batch > 1 (uncached zone, 32 rows: push and x280 overlap: 0.44 ms/token faster at batch 32 on P300),
        # off at batch 1 (no gain); override with X280_PLANB_STREAMED=0/1 or the constructor flag.
        env = os.environ.get("X280_PLANB_STREAMED")
        self.streamed = (env == "1" if env is not None else uncached) if streamed is None else streamed
        self.stream_timeout_us = 0  # 0 -> firmware default 100 ms
        import ttnn

        self.counts = {"enqueues": 0, "arena_polls": 0}
        if session is None:
            from l2cpu.sampling.fw import DEFAULT_IMAGE, boot as boot_firmware

            fw, arena, _ = boot_firmware(mesh, DEFAULT_IMAGE, log=log)
            session = {}
        else:
            fw, arena = session["fw"], session["arena"]
        self.ttnn, self.mesh, self.log = ttnn, mesh, log
        self.fw, self.hw, self.arena, self.A = fw, fw.hw, arena, fw.base
        self.fws = session.get("fws") or [fw]  # one sampling firmware per L2CPU tile (tile 0 first)
        self.regions = session.get("regions") or [arena]
        self.active, self.blocks, self.mops = [fw], [(0, 1)], None  # set by configure()
        self.wait_mode = os.environ.get("X280_PLANB_WAIT", "all")  # several tiles: "all" (one kernel) or "each"
        self.ops = L2cpuOps(mesh, arena)
        self.ops.hw = self.hw
        self.wait_timeout_us = int(os.environ.get("L2CPU_WAIT_TIMEOUT_US", "50000"))  # bound of every wait op
        self.max_step_ms = float(os.environ.get("L2CPU_MAX_STEP_MS", "200"))  # host bound per step (model + x280)

    # ---- firmware configuration (host writes, after READY, before the decode loop only) ----
    def tokens_desc(self, tokens):
        from l2cpu.sampling import layout as A
        from l2cpu.sampling.fw import pack_desc

        page = tokens.buffer_aligned_page_size()
        cols = list(tokens.shape)[-1]
        banks = [
            (RAW_DRAM_XY[b][0], RAW_DRAM_XY[b][1], 0, tokens.buffer_address()) for b in range(8)
        ]  # x280 TLB: raw coords
        return pack_desc(
            A.L2S_DTYPE_UINT32,
            rows=1,
            cols=cols,
            page_size=cols * 4,
            page_stride=page,
            banks=banks,
            first_bank=0,
            location=A.L2S_LOC_NOC,
        )

    def configure(self, tokens, batch, vocab, vpad, params, seeds):
        """params: [{temperature, top_k, top_p}] per user, seeds per user."""
        from l2cpu.sampling import layout as A
        from l2cpu.sampling.fw import local_desc

        from l2cpu.sampling import split_users

        fw = self.fw
        self.wait_idle()
        blocks = [b for b in split_users(batch, len(self.fws)) if b[1] > 0] if batch > 1 else [(0, batch)]
        if len(blocks) > 1:
            if self.mops is None or self.mops.blocks != blocks:
                assert getattr(self, "tid", None) is None, "cannot change the tile split after the trace was captured"
                self.mops = L2cpuMultiOps(self.mesh, self.fws[: len(blocks)], self.regions, blocks)
        self.blocks, self.active = blocks, self.fws[: len(blocks)]
        flags = A.L2S_FLAG_TOKENS_REMOTE
        if self.streamed:
            bf = fw.build_flags()
            if not bf & A.L2S_BUILD_STREAM:
                self.log(
                    f"Plan B: firmware build_flags 0x{bf:x} lack L2S_BUILD_STREAM (image without stream support): streamed mode OFF, "
                    "falling back to push-then-notify"
                )
                self.streamed = False
                assert getattr(self, "tid", None) is None, "cannot switch the stream mode after the trace was captured"
        if self.streamed:
            flags |= A.L2S_FLAG_STREAMED
        for t, (base, n) in zip(self.active, blocks):
            t.set_logits_desc(local_desc(A.L2S_DTYPE_BF16, vpad, rows=32, uncached=self.uncached, row_stride=ROW))
            t.set_tokens_desc(self.tokens_desc(tokens))  # one token tensor: tile t writes elements base .. base+n-1
            t.set_params(
                [
                    (p["temperature"], p["top_k"], p["top_p"], sd)
                    for p, sd in zip(params[base : base + n], seeds[base : base + n])
                ]
            )
            if self.streamed:
                t.set_stream(A.L2S_STREAM_ROWS, self.stream_timeout_us)
            t.set_ctrl(n, vocab, vpad, flags=flags, user_base=base)
        if not getattr(self, "_mode_logged", False):
            self.log(
                f"Plan B mode: batch {batch}, streamed={'on' if self.streamed else 'off'}, "
                f"zone={'uncached' if self.uncached else 'coherent'}, tiles {[t.hw.tile for t in self.active]} "
                f"users {blocks}" + (f", wait {self.wait_mode}" if len(blocks) > 1 else "")
            )
            self._mode_logged = True

    def wait_idle(self, timeout=10.0):
        t0 = time.time()
        for fw in self.fws:
            while fw.r32(OFF_REQ) != fw.r32(OFF_DONE):
                if time.time() - t0 > timeout:
                    raise RuntimeError(
                        f"tile {fw.hw.tile}: request in flight: req {fw.r32(OFF_REQ)} done {fw.r32(OFF_DONE)} "
                        f"error {fw.error()}"
                    )

    def check_fw(self):
        for fw in self.active:
            err = fw.error()
            if err:
                raise RuntimeError(f"tile {fw.hw.tile}: firmware error {err}\n{fw.ctl.log_text()[0][-4000:]}")

    def wait_status(self):
        """Wait status word of every active tile (non-zero: a wait op hit its bound for that tile)."""
        return [fw.r32(OFF_WAIT_STATUS) for fw in self.active]

    def timing(self, n):
        """Per step, the timing record of the slowest active tile (largest wake -> publish) of the last n requests;
        one tile: that tile's records (cycles at 1750 MHz, mtime at 50 MHz). timing_tiles(n): every tile's."""
        per = self.timing_tiles(n)
        return [max(recs, key=lambda r: r["cyc_total"]) for recs in zip(*per)]

    def timing_tiles(self, n):
        return [self._timing(fw, n) for fw in self.active]

    def _timing(self, fw, n):
        from l2cpu.sampling import layout as A

        cnt = fw.r32(A.L2S_OFF_TIMING_COUNT)
        out = []
        for i in range(cnt - n, cnt):
            d = fw.read(A.L2S_OFF_TIMING + (i % A.L2S_TIMING_SLOTS) * A.L2S_TIMING_SIZE, 56)
            f = struct.unpack("<IIQQIIIII3I", d)
            out.append(
                dict(
                    req_seq=f[0],
                    batch=f[1],
                    mtime_wake=f[2],
                    mtime_publish=f[3],
                    cyc_read=f[4],
                    cyc_sample=f[5],
                    cyc_write=f[6],
                    cyc_wait=f[7],
                    cyc_total=f[8],
                )
            )
        return out

    def diag(self):
        return struct.unpack("<6I", self.hw.pa_read(self.A + OFF_DIAG + 64, 24))

    def clear_diag(self):
        for fw in self.active:
            fw.write(OFF_DIAG, bytes(128))

    def rd(self, off):
        self.counts["arena_polls"] += 1
        return self.hw.pa_read32(self.A + off)

    def set_step_base(self, step=0):
        """Steps are numbered from each tile's current req_seq (host write, before the decode loop only): the next
        request of every tile samples with step `step` + 1."""
        self.wait_idle()
        out = []
        for fw in self.active:
            r = fw.r32(OFF_REQ)
            fw.w32(
                OFF_STEP_BASE, (r - step) & 0xFFFFFFFF
            )  # decode step s -> req r+s+1 -> firmware step s+1 (prefill = 0)
            out.append(r)
        return out[0]

    def step_ops(self, fwd, batch, tokens, capture=False, diag=False):
        """Ops of one decode step. fwd() returns TILE logits; returns (tile, rm)."""
        ttnn = self.ttnn
        tile = fwd()
        if len(self.active) > 1:  # several tiles: split push (+ notify per tile), wait for every tile
            m = self.mops
            rm = ttnn.untilize(tile, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if not capture:
                m.set_push_source(rm)
            m.run(m.push_program(uncached=self.uncached, streamed=self.streamed, diag=diag))
            if not self.streamed:
                m.run(m.notify_program(diag=diag))
            for w in m.wait_programs(timeout_us=self.wait_timeout_us, diag=diag, mode=self.wait_mode):
                m.run(w, out=tokens)
            return tile, rm
        if self.streamed:
            rm = ttnn.untilize(tile, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if not capture:
                self.ops.set_push_source(rm)
            self.ops.run(self.ops.push_stream_program(batch, uncached=self.uncached, diag=diag))
            self.ops.run(
                self.ops.wait_program(tokens, 1, batch, timeout_us=self.wait_timeout_us, diag=diag), out=tokens
            )
            return tile, rm
        rm = ttnn.untilize(tile, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if not capture:  # no host writes to the device inside a capture
            self.ops.set_push_source(rm)
        self.ops.run(self.ops.push_program(batch, ncores=1, uncached=self.uncached))
        self.ops.run(self.ops.notify_program(diag=diag))
        self.ops.run(self.ops.wait_program(tokens, 1, batch, timeout_us=self.wait_timeout_us, diag=diag), out=tokens)
        return tile, rm

    def capture(self, fwd, batch, tokens, diag=False):
        """Eager warm run (compiles every op; needs the firmware serving), then capture. The logits buffer of the capture
        must land where the eager one did, because push_program bakes the address into its (cached) program."""
        ttnn = self.ttnn
        tile, rm = self.step_ops(fwd, batch, tokens, diag=diag)
        ttnn.synchronize_device(self.mesh)
        self.check_fw()
        ttnn.deallocate(rm)
        ttnn.deallocate(tile)
        tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
        tile, rm = self.step_ops(fwd, batch, tokens, capture=True, diag=diag)
        ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
        if len(self.active) > 1:
            self.mops.set_push_source(rm)  # every tile's link block
        else:
            self.ops.set_push_source(rm)  # the captured logits buffer (address may differ from the eager one)
        self.tid, self.tile, self.rm = tid, tile, rm
        return tid

    def run(self, n, poll=True, poll_sleep=0.0, poll_cb=None):
        """Enqueue n replays back to back (no host device access in between), then wait for the ring."""
        ttnn = self.ttnn
        w0s = [fw.r32(OFF_RING_WR) for fw in self.active]
        self.last_w0s = w0s
        self.last_w0 = w0s[0]

        def published():  # steps every active tile has published
            self.counts["arena_polls"] += len(self.active)
            return min(fw.r32(OFF_RING_WR) - w for fw, w in zip(self.active, w0s))

        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(self.mesh, self.tid, cq_id=0, blocking=False)
            self.counts["enqueues"] += 1
        t_enq = time.perf_counter() - t0
        if poll:
            # Bound: every queued step ends within (model step + wait bound), even when the x280 is dead (each wait op
            # returns at its bound). Past that the device itself is stuck: raise instead of polling forever.
            deadline = time.perf_counter() + n * (self.max_step_ms * 1e-3 + self.wait_timeout_us * 1e-6) + 5.0
            while True:
                done = published()
                if done >= n:
                    break
                if any(self.wait_status()):  # a wait op hit its bound (some tile)
                    break
                if poll_cb is not None:
                    poll_cb(done)
                if poll_sleep:
                    time.sleep(poll_sleep)
                if time.perf_counter() > deadline:
                    raise StepTimeout(f"ring stuck at {done}/{n} past the {deadline - t0:.1f} s bound")
        ttnn.synchronize_device(self.mesh)  # bounded by the device-side wait bounds
        dt = time.perf_counter() - t0
        return dt, t_enq, published()

    def _ring(self, fw, i0, i1, n):
        out = []
        for i in range(i0, i1):
            s = fw.read(OFF_RING + RING_SLOT * (i % _S.L2S_RING_SLOTS), RS_TOK + 4 * n)
            out.append(list(struct.unpack(f"<{n}I", s[RS_TOK : RS_TOK + 4 * n])))
        return out

    def ring_tokens_range(self, i0, i1, batch):
        """Ring slots of steps i0..i1-1 of the current/last run (relative to its start), every tile's users joined
        in global user order."""
        per = [self._ring(fw, w + i0, w + i1, n) for fw, w, (_, n) in zip(self.active, self.last_w0s, self.blocks)]
        return [sum(rows, []) for rows in zip(*per)]

    def ring_tokens(self, n, batch):
        """Drain the output ring (slots written during the last run of n steps), users in global order."""
        per = []
        for fw, (_, nu) in zip(self.active, self.blocks):
            w = fw.r32(OFF_RING_WR)
            per.append(self._ring(fw, w - n, w, nu))
        return [sum(rows, []) for rows in zip(*per)]


# ---------------------------------------------------------------- decode-loop driver (accept + bench)
def first_tokens(h, plog, params, seeds):
    """Prefill token = sampling step 0 on the host C library, exactly as decode_harness.run_group does."""
    from x280s_ref import X280S, bf16_bits

    B = len(params)
    prow = bf16_bits(plog.reshape(B, -1)[:, : h.vocab].float().numpy())
    lib = X280S()
    out = []
    for b in range(B):
        p = params[b]
        t, _ = lib.sample(prow[b], p["temperature"], p["top_k"], p["top_p"], seeds[b], user=b, step=0, vocab=h.vocab)
        out.append(int(t))
    return out


def prepare_group_planb(h, pb, prompts, params, seeds, diag=False, x280_seeds=None):
    """Everything before the decode loop: prefill (host), first token (host C library, step 0), firmware
    configuration, capture on first use, inputs, step base. x280_seeds: seeds written to the firmware's params
    (default = seeds; only the negative control differs). Returns the prefill info."""
    import torch

    ttnn = h.ttnn
    B = len(params)
    ids, lens, used, plog, pre_s = h.prefill(prompts)
    first = first_tokens(h, plog, params, seeds)
    if h.inputs is None:
        h.write_inputs(torch.tensor(first, dtype=torch.int64), torch.tensor(lens, dtype=torch.int64), full=True)
    tokens_t = h.inputs[0]
    pb.configure(tokens_t, B, h.vocab, h.vocab, params, x280_seeds if x280_seeds is not None else seeds)
    if getattr(pb, "tid", None) is None:
        h.write_inputs(torch.tensor(first, dtype=torch.int64), torch.tensor(lens, dtype=torch.int64), full=True)
        h.model.switch_mode(h.Mode.DECODE)
        pb.capture(h._fwd, B, tokens_t, diag=diag)
    h.write_inputs(torch.tensor(first, dtype=torch.int64), torch.tensor(lens, dtype=torch.int64), full=True)
    ttnn.synchronize_device(h.mesh)
    base = pb.set_step_base()
    if diag:
        pb.clear_diag()
    return dict(first=first, lens=lens, prefill_s=pre_s, ids=ids, base=base)


def run_group_planb(h, pb, prompts, params, seeds, n, poll=True, diag=False, x280_seeds=None, poll_cb=None):
    """Prefill (host), configure the firmware for these users, then n Plan B steps with the host only enqueueing.
    Step convention (same as decode_harness + Plan A): prefill token = step 0 (host); decode step s is firmware request
    req = step_seq_base + s + 1, sampled with step index s + 1 and user index b."""
    B = len(params)
    g = prepare_group_planb(h, pb, prompts, params, seeds, diag=diag, x280_seeds=x280_seeds)
    first, pre_s = g["first"], g["prefill_s"]
    c0 = dict(pb.counts)
    dt, t_enq, nring = pb.run(n, poll=poll, poll_cb=poll_cb)
    c1 = dict(pb.counts)
    pb.check_fw()
    ring = pb.ring_tokens(n, B)
    toks = [[first[b]] + [ring[s][b] for s in range(n)] for b in range(B)]
    res = {
        "steps": [{"wall": dt / n}] * n,
        "tokens": toks,
        "prefill_s": pre_s,
        "decode_s": dt,
        "enqueue_s": t_enq,
        "ring_slots": nring,
        "enqueues": c1["enqueues"] - c0["enqueues"],
        "arena_reads": c1["arena_polls"] - c0["arena_polls"],
    }
    res["timing"] = pb.timing(n)
    if len(pb.active) > 1:
        res["timing_tiles"] = pb.timing_tiles(n)
    if diag:
        res["diag"] = pb.diag()
    return res


def run_group_planb_retry(
    h,
    pb,
    prompts,
    params,
    seeds,
    n,
    chunk=16,
    ctl=None,
    inject_at=None,
    inject_hart=0,
    inject_tile=0,
    diag=False,
    x280_seeds=None,
    log=print,
):
    """Serving-style recovery: steps are enqueued in chunks of `chunk` (one bounded ring drain per chunk, still no
    per-step host access). If a wait op hits its bound or the firmware reports an error inside a chunk:
      1. the queued steps of the chunk finish on their own (each later wait returns at its bound);
      2. the completed steps' tokens are kept (ring);
      3. done_seq := req_seq (nothing in flight), wait status cleared, firmware WARM-restarted through the bring-up
         component's control (`ctl.restart(warm=True)`: L1 mailbox park, then L2 RNMI; the arena and the
         configuration survive); with several tiles EVERY active tile is restarted (a healthy tile costs one
         cooperative restart, ~1 ms) and the kept steps are those every tile published;
      4. inputs rewritten for the first failed step (last good token, its position) and step_seq_base set so that the
         next request is sampled with the same step index: the retried step reproduces the same token;
      5. continue. Positions after the failed step were overwritten with stale tokens; re-running them rewrites
         their KV entries before they are read (attention never reads ahead).
    inject_at: test hook, hang `inject_hart` of tile `inject_tile` (index into the active tiles) through the firmware
    mailbox when the run reaches that step.
    Returns the run_group_planb result plus "retries": [{step, recover_s, restart_s}]."""
    import torch

    B = len(params)
    g = prepare_group_planb(h, pb, prompts, params, seeds, diag=diag, x280_seeds=x280_seeds)
    first, lens = g["first"], g["lens"]
    toks = [[t] for t in first]
    retries, s, injected = [], 0, inject_at is None
    t_all = time.perf_counter()
    while s < n:
        k = min(chunk, n - s)

        def cb(done, s=s):
            nonlocal injected
            if not injected and s + done >= inject_at:
                injected = True
                from l2cpu import layout as L

                # the hart stops in wfi with interrupts off (a hung hart): its request is never published, the wait op
                # hits its bound, and recovery needs the restart's RNMI path
                st, _ = pb.active[inject_tile].ctl.inject(inject_hart, L.L2CPU_INJECT_WFIPARK)
                log(
                    f"test hook: tile {pb.active[inject_tile].hw.tile} hart {inject_hart} hung (wfi, interrupts off) at "
                    f"decode step {s + done}, mailbox status {st}"
                )

        try:
            pb.run(k, poll=True, poll_cb=cb)
        except StepTimeout as e:
            log(f"step bound exceeded: {e}")
        cs = [fw.r32(OFF_RING_WR) - w for fw, w in zip(pb.active, pb.last_w0s)]
        c = min(cs)  # steps every tile published
        wss, errs = pb.wait_status(), [fw.error() for fw in pb.active]
        ws, err = (wss[0], errs[0]) if len(pb.active) == 1 else (wss, errs)
        if injected and inject_at is not None and s <= inject_at < s + k:
            log(
                f"chunk at step {s}: {cs}/{k} steps published per tile, wait status {[hex(x) for x in wss]}, "
                f"firmware error {errs}"
            )
        rows = pb.ring_tokens_range(0, min(c, k), B)
        for row in rows:
            for b in range(B):
                toks[b].append(row[b])
        if c >= k and not any(wss) and not any(errs):
            s += k
            continue
        s += min(c, k)
        t0 = time.perf_counter()
        pb.ttnn.synchronize_device(pb.mesh)  # the rest of the chunk drains (bounded waits)
        ctls = [ctl] if (ctl is not None and len(pb.active) == 1) else [fw.ctl for fw in pb.active]
        if any(c_ is None for c_ in ctls):
            raise StepTimeout(f"step {s}: wait status {wss}, firmware error {errs}; no restart control given")
        for fw in pb.active:  # nothing in flight: a restarted firmware must not serve stale rows
            fw.w32(OFF_DONE, fw.r32(OFF_REQ))
        t1 = time.perf_counter()
        rt = []
        for fw, c_ in zip(pb.active, ctls):
            ta = time.perf_counter()
            c_.restart(None, warm=True)
            fw.sync()  # picks up req_seq again (WARM keeps the region)
            rt.append(round((time.perf_counter() - ta) * 1e3, 2))
        t2 = time.perf_counter()
        for fw in pb.active:
            fw.w32(OFF_WAIT_STATUS, 0)
        last = [toks[b][-1] for b in range(B)]
        h.write_inputs(torch.tensor(last, dtype=torch.int64), torch.tensor([l + s for l in lens], dtype=torch.int64))
        pb.ttnn.synchronize_device(pb.mesh)
        pb.set_step_base(s)  # the next request of every tile -> step s + 1
        retries.append(
            dict(
                step=s,
                recover_s=time.perf_counter() - t0,
                restart_s=t2 - t1,
                restart_tiles_ms=rt,
                error=str(err),
                wait_status=hex(ws) if isinstance(ws, int) else [hex(x) for x in ws],
            )
        )
        log(
            f"recovered at decode step {s}: warm restart {retries[-1]['restart_s'] * 1e3:.1f} ms, "
            f"total recovery {retries[-1]['recover_s'] * 1e3:.1f} ms; retrying"
        )
    dt = time.perf_counter() - t_all
    return {
        "steps": [{"wall": dt / n}] * n,
        "tokens": toks,
        "prefill_s": g["prefill_s"],
        "decode_s": dt,
        "retries": retries,
    }


_BENCH = {}


def planb_bench_streamed(harness, prompts, params, seeds, n_tokens, run_idx):
    """bench.py x280-planB arm with the stream protocol (push and x280 overlap)."""
    if "pbs" not in _BENCH:
        from l2cpu_sampler import session

        _BENCH["pbs"] = PlanB(harness.mesh, session=session(), uncached=len(params) > 1, streamed=True)
    return run_group_planb(harness, _BENCH["pbs"], prompts, params, seeds, n_tokens)


def planb_bench_nonstreamed(harness, prompts, params, seeds, n_tokens, run_idx):
    """bench.py x280-planB arm forced to push-then-notify at every batch size."""
    if "pbn" not in _BENCH:
        from l2cpu_sampler import session

        _BENCH["pbn"] = PlanB(harness.mesh, session=session(), uncached=len(params) > 1, streamed=False)
    return run_group_planb(harness, _BENCH["pbn"], prompts, params, seeds, n_tokens)


def planb_bench(harness, prompts, params, seeds, n_tokens, run_idx):
    """bench.py x280-planB arm: callable(harness, prompts, params, seeds, n_tokens, run_idx) -> dict.
    Per-step walls are the run's decode time / n (the host does not touch the device per step)."""
    if "pb" not in _BENCH:
        from l2cpu_sampler import session

        _BENCH["pb"] = PlanB(harness.mesh, session=session(), uncached=len(params) > 1)  # defaults (streamed at b>1)
    return run_group_planb(harness, _BENCH["pb"], prompts, params, seeds, n_tokens)
