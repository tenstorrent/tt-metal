# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Failure-rate probe of the top-p certificate (phase 2, step S8; work/research_1026/67 sections 5.1, 5.6). Debug only.

With ``QWEN36_SPEC_STATS_SHADOW_EVERY=N`` (N > 0) every N-th fused sampled step without a logprobs or certificate row also replays the
stats trace and reads the union + stats (the reads of a logprobs step). No row uses the result: the host only measures the candidate
mass fraction the certificate would see, per examined verify position of every live sampled row:

    q_T = Zc_T / max(L_T, Zc_T),  Zc_T = sum over the top-M candidates of exp((x_i - x_0) / T)   (the cert body, SFS body(cert=True))
    q_1 = the same at T = 1 with L_raw

The certificate passes at top_p p when q >= p + CERT_EPS. Counters are summed in ``StatsShadow``; ``line()`` is one log line with the
stable prefix ``SPEC_SHADOW`` and key=value pairs (list values joined with "/"), parsed by tests/spec_shadow_parse.py.
Pure torch, no ttnn.
"""

import os
import threading

import torch
from loguru import logger

from models.demos.blackhole.qwen36.tt.spec_fused_sampled import CERT_EPS

PREFIX = "SPEC_SHADOW"
P_LIST = (0.8, 0.9, 0.95, 0.99)  # top_p values the fail counts are kept for
EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0.999)  # histogram bins [e_i, e_i+1); the last bin is [0.999, 1.0]
LOG_EVERY = 200  # probe steps between two log lines
MAX_T_BUCKETS = 24  # distinct temperatures (1 decimal) kept; later ones go to "other"


def candidate_mass_fraction(union_vals, norm, temperature, M):
    """q [n] (float32) for n verify rows: ``union_vals`` [n, W] raw logits (the union of the per-shard top-M; its M largest values are the
    global top-M), ``norm`` [n] = L_T (or L_raw at ``temperature`` 1), ``temperature`` [n] > 0. The arithmetic of the certificate body:
    v = x / T in fp32, e = exp(v - v_0), q = sum_M e / max(norm, sum_M e)."""
    top = torch.topk(torch.as_tensor(union_vals).to(torch.float32), M, dim=1).values
    t = torch.as_tensor(temperature).to(torch.float32).reshape(-1, 1)
    v = top / t
    zc = torch.exp(v - v[:, :1]).sum(dim=1)
    return zc / torch.maximum(torch.as_tensor(norm).to(torch.float32).reshape(-1), zc)


def _bucket(edges, q):
    """Histogram bin of each q (values past the last edge go to the last bin)."""
    return torch.bucketize(q, torch.tensor(edges, dtype=q.dtype), right=True).sub(1).clamp(0, len(edges) - 1)


class StatsShadow:
    """Counters of the probe. ``record`` is called once per probe step (engine thread on a sync step, reader thread on a deferred one)."""

    def __init__(self, every, pid=None):
        self.every = int(every)
        self.pid = os.getpid() if pid is None else int(pid)
        self._lock = threading.Lock()
        n, np_ = len(EDGES), len(P_LIST)
        z = lambda k: [0] * k  # noqa: E731
        self.steps = self.rows = self.pos = self.allpos = 0
        self.sum_q = {"T": 0.0, "1": 0.0}
        self.hist = {"T": z(n), "1": z(n)}
        self.failpos = {"T": z(np_), "1": z(np_)}  # examined positions with q < p + eps
        self.failrow = {"T": z(np_), "1": z(np_)}  # (row, step) pairs with an examined failing position
        self.failstep = {"T": z(np_), "1": z(np_)}  # steps with any failing examined position of any row
        self.allfail = {"T": z(np_)}  # every position of a live row (examined or not), q_T only
        self.bytemp = {}  # temperature bucket -> [positions, failpos qT list, failpos q1 list, rows]
        self.M = 0
        self.last_logged = 0

    def record(self, union_vals, stats, counts, live, temperature, B, T, M):
        """One probe step. ``union_vals`` [>= B*T, W] raw logits, ``stats`` [>= 32 + B*T] fp32 (rows 0..31 L_raw, 32.. L_T), ``counts`` [B]
        committed tokens per user (the walk examined columns 0..count-1), ``live`` [B] bool (active, not declined, sampled),
        ``temperature`` [B]. Rows of users that are not live are not read. Logs a line every LOG_EVERY probe steps."""
        live = torch.as_tensor(live).bool().reshape(B)
        users = live.nonzero().reshape(-1)
        if users.numel() == 0:
            return
        rows = (users[:, None] * T + torch.arange(T)[None, :]).reshape(-1)
        vals = torch.as_tensor(union_vals)[rows]
        st = torch.as_tensor(stats).reshape(-1).to(torch.float32)
        temp = torch.as_tensor(temperature).to(torch.float32).reshape(B)[users].clamp(min=1e-6)
        trow = temp.repeat_interleave(T)
        q = {
            "T": candidate_mass_fraction(vals, st[32 + rows], trow, M).reshape(-1, T),
            "1": candidate_mass_fraction(vals, st[rows], torch.ones_like(trow), M).reshape(-1, T),
        }
        exam = torch.arange(T)[None, :] < torch.as_tensor(counts).reshape(B)[users][:, None]  # [n, T]
        tb = [f"{float(t):.1f}" for t in temp]
        with self._lock:
            self.M = M
            self.steps += 1
            self.rows += len(users)
            self.pos += int(exam.sum())
            self.allpos += exam.numel()
            thr = torch.tensor(P_LIST, dtype=torch.float32)[:, None, None] + CERT_EPS  # [P, 1, 1]
            fails = {}
            for kind in ("T", "1"):
                qk = q[kind]
                self.sum_q[kind] += float(qk[exam].sum())
                h = torch.bincount(_bucket(EDGES, qk[exam]), minlength=len(EDGES))
                self.hist[kind] = [a + int(b) for a, b in zip(self.hist[kind], h.tolist())]
                low = qk[None] < thr  # [P, n, T]
                fail = low & exam[None]
                fails[kind] = fail.sum(dim=2)  # [P, n] failing examined positions per row
                for field, vals in (
                    (self.failpos[kind], fails[kind].sum(dim=1)),
                    (self.failrow[kind], fail.any(dim=2).sum(dim=1)),
                    (self.failstep[kind], fail.any(dim=2).any(dim=1)),
                ):
                    field[:] = [a + int(b) for a, b in zip(field, vals.tolist())]
                if kind == "T":
                    self.allfail["T"][:] = [a + int(b) for a, b in zip(self.allfail["T"], low.sum(dim=(1, 2)).tolist())]
            n_exam = exam.sum(dim=1).tolist()
            fT, f1 = fails["T"].tolist(), fails["1"].tolist()  # [P][n]
            for j, name in enumerate(tb):
                key = name if name in self.bytemp or len(self.bytemp) < MAX_T_BUCKETS else "other"
                e = self.bytemp.setdefault(key, [0, [0] * len(P_LIST), [0] * len(P_LIST), 0])
                e[0] += n_exam[j]
                e[3] += 1
                for i in range(len(P_LIST)):
                    e[1][i] += fT[i][j]
                    e[2][i] += f1[i][j]
            due = self.steps - self.last_logged >= LOG_EVERY
            if due:
                self.last_logged = self.steps
        if due:
            logger.info(self.line())

    def line(self):
        """The log line: ``SPEC_SHADOW key=value ...`` (one line, no spaces inside a value)."""
        j = lambda xs: "/".join(str(x) for x in xs)  # noqa: E731
        with self._lock:
            f = lambda x: f"{x:.6f}"  # noqa: E731
            n = max(self.pos, 1)
            parts = [
                f"pid={self.pid}",
                f"every={self.every}",
                f"M={self.M}",
                f"steps={self.steps}",
                f"rows={self.rows}",
                f"pos={self.pos}",
                f"allpos={self.allpos}",
                f"qT_mean={f(self.sum_q['T'] / n)}",
                f"q1_mean={f(self.sum_q['1'] / n)}",
                f"edges={j(EDGES)}",
                f"ps={j(P_LIST)}",
                f"eps={CERT_EPS}",
                f"hist_qT={j(self.hist['T'])}",
                f"hist_q1={j(self.hist['1'])}",
                f"failpos_qT={j(self.failpos['T'])}",
                f"failpos_q1={j(self.failpos['1'])}",
                f"failrow_qT={j(self.failrow['T'])}",
                f"failrow_q1={j(self.failrow['1'])}",
                f"failstep_qT={j(self.failstep['T'])}",
                f"failstep_q1={j(self.failstep['1'])}",
                f"allfail_qT={j(self.allfail['T'])}",
            ]
            for name in sorted(self.bytemp):
                e = self.bytemp[name]
                parts += [
                    f"T{name}_pos={e[0]}",
                    f"T{name}_rows={e[3]}",
                    f"T{name}_failpos_qT={j(e[1])}",
                    f"T{name}_failpos_q1={j(e[2])}",
                ]
        return PREFIX + " " + " ".join(parts)
