# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""On-chip replay: recorded logits rows -> region -> firmware request -> token, compared with the host build of the
sampling library (libx280s_host.so, bit-exact reference) run here on the same row and parameters. No recorded
tokens are needed: the expectation is generated at run time.

    batch 1 : rows 0..N-1 written into the COHERENT zone; per row 4 requests (greedy and the three sampled settings),
              seed fixed, user 0, step = row index.
    batch 32: R requests of 32 different rows written into the UNCACHED zone, per-user settings cycling over the
              4 settings ("mix") or all T0.7/k50/p0.9 ("bench"), per-user seeds, step = request index; 4 harts.
A restart can be injected after a given request (L1 mailbox park or L2 RNMI park + WARM restart through L2cpuCtl);
the replay must stay identical across it.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import numpy as np

from . import layout as S
from .fw import local_desc

SETTINGS = [
    ("greedy", 0.0, 0, 1.0),
    ("T0.7/k50/p0.9", 0.7, 50, 0.9),
    ("T1.0/k0/p1.0", 1.0, 0, 1.0),
    ("T0.6/k20/p0.95", 0.6, 20, 0.95),
]
V_QWEN = 151936
L2CPU_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def host_library():
    """The ctypes wrapper of the host reference (tools/l2cpu/sampling/lib; build it with `make -C .../lib`)."""
    d = os.environ.get("L2S_LIB_DIR") or os.path.join(L2CPU_DIR, "sampling", "lib")
    if d not in sys.path:
        sys.path.insert(0, d)
    from x280s_ref import X280S  # noqa: E402

    return X280S()


def restart(fw, level):
    """Park (L1 mailbox, or L2 RNMI for every hart) and WARM-restart the same image; returns the restart time."""
    from .. import layout as L

    if level == "L2":
        fw.ctl.rnmi(0xF, L.L2CPU_RNMI_PARK, timeout=0.5)
    out = fw.ctl.restart(None, warm=True)
    fw.sync()
    return out["t_total"]


class Split:
    """Per-request time split from the firmware's timing records (core clock cycles -> us)."""

    def __init__(self, mhz):
        self.mhz, self.t, self.host, self.wake = mhz, [], [], []

    def add(self, tm, host_s, wake_us):
        self.t.append(tm)
        self.host.append(host_s)
        self.wake.append(wake_us)

    def med(self, key):
        return statistics.median(t[key] for t in self.t) / self.mhz

    def row(self, name):
        if not self.t:
            return dict(name=name, n=0)
        other = (
            statistics.median(
                t["cyc_total"] - t["cyc_read"] - t["cyc_sample"] - t["cyc_write"] - t["cyc_wait"] for t in self.t
            )
            / self.mhz
        )
        return dict(
            name=name,
            n=len(self.t),
            total_us=self.med("cyc_total"),
            read_us=self.med("cyc_read"),
            sample_us=self.med("cyc_sample"),
            write_us=self.med("cyc_write"),
            wait_us=self.med("cyc_wait"),
            other_us=other,
            wake_us=statistics.median(self.wake),
            host_rt_us=statistics.median(self.host) * 1e6,
            worker_us=[statistics.median(t["cyc_worker"][h] for t in self.t) / self.mhz for h in range(3)],
        )


def replay_b1(fw, lib, rows, n, seed=1234, restart_at=None, restart_level="L1", log=print, mhz=1750.0):
    """n rows x 4 settings, batch 1, coherent zone. Returns dict(ok, total, mismatches, split rows, restart)."""
    fw.set_logits_desc(local_desc(S.L2S_DTYPE_BF16, V_QWEN))
    fw.set_ctrl(1, V_QWEN, V_QWEN, flags=0)
    hz = fw.mtime_hz()
    mism = {s[0]: 0 for s in SETTINGS}
    splits = {s[0]: Split(mhz) for s in SETTINGS}
    rst = None
    t0 = time.time()
    for i in range(n):
        if restart_at is not None and i == restart_at:
            rst = dict(at_row=i, level=restart_level, t_s=restart(fw, restart_level))
            fw.set_logits_desc(local_desc(S.L2S_DTYPE_BF16, V_QWEN))
            fw.set_ctrl(1, V_QWEN, V_QWEN, flags=0)
            log(f"  restart ({restart_level}, WARM) before row {i}: {rst['t_s'] * 1e3:.2f} ms")
        row = np.ascontiguousarray(rows[i])
        fw.write(S.L2S_OFF_LOGITS, row.tobytes())
        for name, T, k, p in SETTINGS:
            fw.set_params([(T, k, p, seed)])
            fw.set_step(i)
            ref, _ = lib.sample(row, T, k, p, seed, user=0, step=i)
            th = time.perf_counter()
            m0 = fw.mtime()
            fw.issue()
            fw.wait_done()
            host = time.perf_counter() - th
            tok = fw.next_tokens(1)[0]
            tm = fw.timing_last()
            ring = fw.ring_last(1)
            if tm["req_seq"] != fw.req or ring["req_seq"] != fw.req or ring["tok"][0] != tok or ring["step"] != i:
                raise RuntimeError(f"row {i} {name}: timing {tm} ring {ring} token {tok}")
            splits[name].add(tm, host, (tm["mtime_wake"] - m0) * 1e6 / hz)
            if tok != ref:
                mism[name] += 1
                if mism[name] <= 5:
                    log(f"MISMATCH b1 row {i} {name}: x280 {tok} host library {ref}")
        if (i + 1) % 250 == 0:
            log(f"  batch 1: {i + 1}/{n} rows, {time.time() - t0:.0f} s")
    total = n * len(SETTINGS)
    bad = sum(mism.values())
    return dict(
        ok=bad == 0,
        identical=total - bad,
        total=total,
        mismatches=mism,
        wall_s=time.time() - t0,
        restart=rst,
        split=[splits[s[0]].row(s[0]) for s in SETTINGS],
    )


def replay_b32(fw, lib, rows, nreq, n_rows, mode="mix", restart_at=None, restart_level="L2", log=print, mhz=1750.0):
    """nreq requests x 32 users, uncached zone, 4 harts. Row of user u in request r: (r * 32 + u) % n_rows."""
    params = [(*SETTINGS[1 if mode == "bench" else u % 4][1:], 1000 + 7 * u) for u in range(32)]
    hz = fw.mtime_hz()

    def configure():
        fw.set_logits_desc(local_desc(S.L2S_DTYPE_BF16, V_QWEN, uncached=True))
        fw.set_ctrl(32, V_QWEN, V_QWEN, flags=0)
        fw.set_params(params)

    configure()
    bad, n, split, rst = 0, 0, Split(mhz), None
    t0 = time.time()
    for r in range(nreq):
        if restart_at is not None and r == restart_at:
            rst = dict(at_request=r, level=restart_level, t_s=restart(fw, restart_level))
            configure()
            log(f"  restart ({restart_level}, WARM) before batch-32 request {r}: {rst['t_s'] * 1e3:.2f} ms")
        idx = [(r * 32 + u) % n_rows for u in range(32)]
        block = np.ascontiguousarray(rows[idx])  # [32, V] uint16; row stride = V * 2 = 303,872 B
        fw.write_uncached(S.L2S_OFF_LOGITS_UC, block.tobytes())
        refs = [lib.sample(np.ascontiguousarray(rows[idx[u]]), *params[u], user=u, step=r)[0] for u in range(32)]
        fw.set_step(r)
        th = time.perf_counter()
        m0 = fw.mtime()
        fw.issue()
        fw.wait_done(timeout=10)
        host = time.perf_counter() - th
        toks = fw.next_tokens(32)
        tm = fw.timing_last()
        ring = fw.ring_last(32)
        if ring["req_seq"] != fw.req or ring["tok"] != toks:
            raise RuntimeError(f"request {r}: ring {ring} tokens {toks}")
        split.add(tm, host, (tm["mtime_wake"] - m0) * 1e6 / hz)
        for u in range(32):
            n += 1
            if toks[u] != refs[u]:
                bad += 1
                if bad <= 5:
                    log(f"MISMATCH b32 request {r} user {u}: x280 {toks[u]} host library {refs[u]}")
    return dict(
        ok=bad == 0, identical=n - bad, total=n, wall_s=time.time() - t0, restart=rst, split=[split.row(f"b32 {mode}")]
    )


def format_split(s):
    if not s.get("n"):
        return f"  {s['name']:16s} n=0"
    return (
        f"  {s['name']:16s} n={s['n']:5d}  x280 total {s['total_us']:8.1f} us = read {s['read_us']:7.1f} + sample "
        f"{s['sample_us']:8.1f} + write-back {s['write_us']:5.1f} + wait-workers {s['wait_us']:7.1f} + other "
        f"{s['other_us']:5.1f} | wake-up {s['wake_us']:5.1f} us | host round trip {s['host_rt_us']:8.1f} us (medians)"
    )


def streamed_request(fw, lib, rows, idx, params, step, timeout=2.0):
    """One streamed batch-32 request with the host as the producer (req_seq + doorbell first, then the rows into the
    uncached zone, then landed): returns (served, tokens, expected)."""
    fw.set_step(step)
    fw.issue()
    block = np.ascontiguousarray(rows[idx])
    fw.write_uncached(S.L2S_OFF_LOGITS_UC, block.tobytes())
    fw.w32(S.L2S_OFF_LANDED, ((fw.req & 0xFFFF) << 16) | len(idx))  # one aligned u32 store, after the rows
    exp = [lib.sample(np.ascontiguousarray(rows[idx[u]]), *params[u], user=u, step=step)[0] for u in range(len(idx))]
    try:
        fw.wait_done(timeout)
    except Exception:  # noqa: BLE001
        return False, None, exp
    return True, fw.next_tokens(len(idx)), exp


def stream_after_hang(fw, lib, rows, n_rows, log=print, wait_s=0.05):
    """The serving recovery path: hart 0 hangs (wfi, interrupts off) during streamed batch-32 serving, a request is
    not served within the wait bound; recovery = done_seq := req_seq, ctl.restart(warm) (L1 mailbox park attempt,
    then L2 RNMI), then the next streamed request must be served within the bound with the expected tokens."""
    from .. import layout as L

    params = [(*SETTINGS[u % 4][1:], 1000 + 7 * u) for u in range(32)]
    fw.set_logits_desc(local_desc(S.L2S_DTYPE_BF16, V_QWEN, uncached=True))
    fw.set_ctrl(32, V_QWEN, V_QWEN, flags=S.L2S_FLAG_STREAMED)
    fw.set_stream(S.L2S_STREAM_ROWS)
    fw.set_params(params)
    out = dict(steps=[])
    for step in range(3):  # healthy streamed requests
        ok, toks, exp = streamed_request(fw, lib, rows, [(step * 32 + u) % n_rows for u in range(32)], params, step)
        out["steps"].append(dict(phase="before", served=ok, identical=ok and toks == exp))
    st, _ = fw.ctl.inject(0, L.L2CPU_INJECT_WFIPARK)
    time.sleep(0.01)
    ok, _, _ = streamed_request(fw, lib, rows, [(96 + u) % n_rows for u in range(32)], params, 3, timeout=wait_s)
    out["hung_request_served"] = ok
    fw.w32(S.L2S_OFF_DONE_SEQ, fw.r32(S.L2S_OFF_REQ_SEQ))  # nothing in flight (as the serving harness does)
    t0 = time.perf_counter()
    r = fw.ctl.restart(None, warm=True)
    fw.sync()
    out["restart"] = dict(levels=r["levels"], ms=(time.perf_counter() - t0) * 1e3, inject_status=st)
    for step in range(4, 8):
        ok, toks, exp = streamed_request(
            fw, lib, rows, [(step * 32 + u) % n_rows for u in range(32)], params, step, timeout=wait_s
        )
        out["steps"].append(dict(phase="after", served=ok, identical=ok and toks == exp))
        if not ok:
            out["records"] = fw.ctl.records()
            out["log_tail"] = fw.ctl.log_text()[0][-1500:]
            break
    fw.set_ctrl(32, V_QWEN, V_QWEN, flags=0)
    out["ok"] = (
        (not out["hung_request_served"])
        and all(s["served"] and s["identical"] for s in out["steps"])
        and len(out["steps"]) == 7
    )
    return out
