# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Top-p with top-k off on the verify sampler (vllm-tt-plugin ``supports_spec_device_top_p``) for
Qwen36MTPContractForCausalLM.

A sampled row with top-k off and top_p < 1 (a *certificate row*) draws from vLLM's nucleus over the whole vocabulary:
the largest tokens, in (value descending, id ascending) order, while the mass of the tokens before them is below top_p.
``ttnn.sampling`` draws from at most 32 candidates (its top-k is in [1, 32]), so the device serves such a row only
when the nucleus lies inside the row's top-32. On a verify with a certificate row the certificate sampler trace
(``TopPCertificate.body``) replays in place of the sampler trace:

1. the sampler's candidates (spec_sample.VerifySampler.candidates) and their top-32 v_0 >= ... >= v_31;
2. with the row's inverse temperature t (the sampler's own bfloat16 value): Z, the sum over the whole vocabulary of
   exp((x - v_0) t) (float32, spec_logprobs.vocab_exp_sum), Zc, the same sum over the top-32, and q = Zc / max(Z, Zc),
   the mass the top-32 hold;
3. the nucleus cut inside the top-32, in (value descending, id ascending) order, the verify sampler's tie order:
   candidate i > 0 is dropped when the top-32's mass from i on, over Zc, is at most 1 - top_p / q, which is vLLM's rule
   over the whole vocabulary when q > top_p (``certificate_cut``). Every candidate after the last one kept in that order
   is masked out of the sampler's input (in a group of equal values at the cut, the later ids), and the row draws with
   top-k 32 and top-p 1;
4. q is written for the host. A row with q - top_p < CERT_EPS fails the certificate: its nucleus may reach past the
   top-32, and its draw is redone on the host from the step's full logits (``redo_draws``), from vLLM's distribution
   in float64 (``row_distribution``), with the row's device seed.

Every other row of the step draws as on the sampler trace.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.spec_logprobs import vocab_exp_sum
from models.demos.blackhole.qwen36.tt.spec_sample import PAD_LOGIT, SAMPLER_ROWS, SAMPLER_TOP_K, row_sampling

# A certificate row fails when q - top_p < CERT_EPS: the float32 error of Z, Zc and q is about 1e-6, so a row near the
# margin is redone rather than drawn from a nucleus that may be cut one token short.
CERT_EPS = 1e-5
_BIG = 1e30


def is_certificate_row(temperature, top_k, top_p, vocab):
    """Whether a row with these sampling settings is a certificate row: sampled, top-k off (below 1 or at least the
    vocabulary) and top_p below 1."""
    return temperature > 0.0 and (top_k < 1 or top_k >= vocab) and top_p < 1.0


@dataclass
class CertificateRows:
    """The certificate rows of one verify of B users by T columns: per user, whether it is one, its top_p and its
    temperature (unused entries 1.0)."""

    rows: list
    top_p: list
    temperature: list

    @property
    def any(self):
        return any(self.rows)


def certificate_rows(sampling_params, B, vocab):
    """The ``CertificateRows`` of a step's ``sampling_params`` over B users, or None when there is none."""
    rows, top_p, temperature = [False] * B, [1.0] * B, [1.0] * B
    for u in range(B):
        temp, k, p, _ = row_sampling(sampling_params, u)
        if is_certificate_row(temp, k, p, vocab):
            rows[u], top_p[u], temperature[u] = True, p, temp
    cert = CertificateRows(rows, top_p, temperature)
    return cert if cert.any else None


class _Rows:
    """Per-row values of a plugin-style ``sampling_params`` with a certificate row's top-k and top-p replaced."""

    def __init__(self, sampling_params, cert):
        self._params, self._cert = sampling_params, cert

    def __getattr__(self, name):
        value = getattr(self._params, name)
        if name not in ("top_k", "top_p"):
            return value
        n = len(self._cert.rows)
        out = [row_sampling(self._params, u)[1 if name == "top_k" else 2] for u in range(n)]
        for u in range(n):
            if self._cert.rows[u]:
                out[u] = SAMPLER_TOP_K if name == "top_k" else 1.0
        return out


def sampler_params(sampling_params, cert):
    """``sampling_params`` as the sampler draws a certificate step: a certificate row with top-k 32 and top-p 1 (its
    nucleus is the mask the certificate trace applies), every other row as given."""
    return sampling_params if cert is None else _Rows(sampling_params, cert)


def certificate_cut(zc, full_norm, top_p, eps=CERT_EPS):
    """Top-p certificate of a full-vocabulary row. ``zc`` = sum over the first M candidates of exp((v_i - v_0) t),
    ``full_norm`` = the same sum over the whole vocabulary, ``top_p``. q = zc / max(full_norm, zc) in (0, 1]. When
    q >= p + eps every position past the candidates is dropped by the nucleus rule and, inside the candidates, position
    i is dropped iff S_i / zc <= 1 - p / q (S_i = candidate mass from i on). Returns (q, omp_eff = 1 - p / q,
    fail = q - p < eps); a failing row can only be a false "fail", never a false "pass"."""
    zc = torch.as_tensor(zc).to(torch.float32)
    q = zc / torch.maximum(torch.as_tensor(full_norm).to(torch.float32), zc)
    p = torch.as_tensor(top_p).to(torch.float32)
    return q, 1.0 - p / q, (q - p) < eps


def row_distribution(logit_row, temp, k, p):
    """One row's sampling distribution on the FULL logits, float64: temperature, top-k with ties kept, top-p (tail mass
    <= 1 - p dropped, never the first). Returns (ids int64 [V], prob float64 [V]) in (value desc, id asc) order (entries
    cut by top-k / top-p have probability 0). A top-k <= 0 or >= V means "off"."""
    v = torch.as_tensor(logit_row).to(torch.float64) / float(temp)
    order = torch.sort(v, descending=True, stable=True).indices  # ties: ascending id (stable on the id-ordered row)
    vs = v[order]
    k = vs.numel() if int(k) <= 0 else min(int(k), vs.numel())
    keep = vs >= vs[k - 1]
    vs = torch.where(keep, vs, torch.full_like(vs, float("-inf")))
    e = torch.exp(vs - vs[0])
    p0 = e / e.sum()
    if float(p) < 1.0:
        cum = p0.flip(0).cumsum(0).flip(0)
        drop = cum <= (1.0 - float(p))
        drop[0] = False
        e = torch.where(drop, torch.zeros_like(e), e)
    return order, e / e.sum()


def failed_rows(q, cert, T):
    """[B*T] bool: the verify rows u*T + j of certificate users whose q is below top_p + CERT_EPS."""
    n = len(cert.rows) * T
    q = torch.as_tensor(q).reshape(-1)[:n].to(torch.float32)
    p = torch.tensor(cert.top_p, dtype=torch.float32).repeat_interleave(T)
    rows = torch.tensor(cert.rows, dtype=torch.bool).repeat_interleave(T)
    return rows & (q - p < CERT_EPS)


def redo_draws(ids, logits, fail, cert, seeds, T):
    """``ids`` [B, T] int64 with each failed row's token drawn again on the host: from ``row_distribution`` of its
    full logits ``logits`` [>= B*T, V] at the user's temperature and top_p (top-k off), with a torch generator seeded by
    the row's device seed ``seeds[r]``, so a seeded row redoes reproducibly. Returns a new tensor."""
    out = torch.as_tensor(ids).to(torch.int64, copy=True)
    B = out.shape[0]
    for r in torch.as_tensor(fail).reshape(-1)[: B * T].nonzero().reshape(-1).tolist():
        u, j = divmod(r, T)
        order, prob = row_distribution(logits[r], cert.temperature[u], 0, cert.top_p[u])
        g = torch.Generator().manual_seed(int(seeds[r]))
        out[u, j] = int(order[torch.multinomial(prob, 1, generator=g)])
    return out


class TopPCertificate:
    """The device half: the per-row inputs (``flag`` 1 for a certificate row, ``top_p``, the inverse temperature
    ``inv_t``, float32 TILE [1, 1, B*T, 1]), the output ``q`` (float32 TILE [1, 1, B*T, 1]) and the trace-capturable
    ``body``, which draws into the sampler's ``draws`` buffer as the sampler trace does."""

    def __init__(self, sampler):
        self.sampler, self.mesh, self.rows = sampler, sampler.mesh, sampler.rows
        self.vocab, self.shards = sampler.vocab, sampler.shards
        rep = ttnn.ReplicateTensorToMesh(self.mesh)

        def persistent(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=rep,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        col = torch.ones(1, 1, self.rows, SAMPLER_TOP_K)
        col[..., 0] = 0.0
        self.notfirst = persistent(col)
        self.flag = persistent(torch.zeros(1, 1, self.rows, 1))
        self.top_p = persistent(torch.ones(1, 1, self.rows, 1))
        self.inv_t = persistent(torch.ones(1, 1, self.rows, 1))
        self.q = persistent(torch.ones(1, 1, self.rows, 1))
        self.floor = persistent(torch.full((1, 1, self.rows, 1), -_BIG))
        self.ceiling = persistent(torch.full((1, 1, self.rows, 1), _BIG))
        cols = sampler.candidates_per_row
        self.pad = persistent(torch.full((1, 1, self.rows, cols), PAD_LOGIT))
        self.iota = persistent(torch.arange(cols, dtype=torch.float32).expand(1, 1, self.rows, cols).contiguous())
        self._staged = None
        self.compute = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def stage(self, cert, plan, T):
        """Host -> device copies of the certificate rows' flags, top_p and inverse temperatures (the sampler's bfloat16
        encoding ``plan.temp``, so the certificate and the draw use the same temperature)."""
        n = len(cert.rows) * T
        flag = torch.zeros(self.rows)
        top_p = torch.ones(self.rows)
        flag[:n] = torch.tensor(cert.rows, dtype=torch.float32).repeat_interleave(T)
        top_p[:n] = torch.tensor(cert.top_p, dtype=torch.float32).repeat_interleave(T)
        inv_t = torch.tensor(plan.temp, dtype=torch.float32)[: self.rows].to(torch.bfloat16).to(torch.float32)
        host = torch.stack([flag, top_p, inv_t])
        if self._staged is not None and torch.equal(self._staged, host):
            return
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for values, dev in zip(host, (self.flag, self.top_p, self.inv_t)):
            h = ttnn.from_torch(
                values.reshape(1, 1, -1, 1), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep
            )
            ttnn.copy_host_to_device_tensor(h, dev)
        self._staged = host

    def body(self, logits):
        """``logits`` as ``VerifySampler.body`` takes them. Writes q into ``q`` and one draw per row into the sampler's
        ``draws``; a certificate row draws from its nucleus inside the top-32."""
        R = self.rows
        vals, gidx = self.sampler.candidates(logits)
        rows = int(logits.shape[2])
        # Stable: among equal values the earlier candidate first, which is the lower token id (the candidates sit in
        # shard order, each shard's in (value descending, id ascending) order).
        v32, i32 = ttnn.topk(vals, k=SAMPLER_TOP_K, dim=-1, largest=True, sorted=True, stable=True)
        top = ttnn.typecast(v32, ttnn.float32)
        ttnn.deallocate(v32)
        idx = ttnn.typecast(i32, ttnn.int32)
        ttnn.deallocate(i32)
        pos = ttnn.typecast(idx, ttnn.float32)
        ttnn.deallocate(idx)
        m = ttnn.slice(top, (0, 0, 0, 0), (1, 1, R, 1))
        diff = ttnn.subtract(top, m)
        d = ttnn.multiply(diff, self.inv_t)
        ttnn.deallocate(diff)
        e = ttnn.exp(d, fast_and_approximate_mode=False)
        ttnn.deallocate(d)
        zc = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=self.compute)
        m_rows, t_rows = m, self.inv_t
        if rows < R:
            m_rows = ttnn.slice(m, (0, 0, 0, 0), (1, 1, rows, 1))
            t_rows = ttnn.slice(self.inv_t, (0, 0, 0, 0), (1, 1, rows, 1))
        z = vocab_exp_sum(logits, m_rows, self.shards, self.vocab, self.compute, scale=t_rows)
        if rows < R:
            ttnn.deallocate(m_rows)
            ttnn.deallocate(t_rows)
        ttnn.deallocate(m)
        if R < SAMPLER_ROWS:
            full = z
            z = ttnn.slice(full, (0, 0, 0, 0), (1, 1, R, 1))
            ttnn.deallocate(full)
        zmax = ttnn.maximum(z, zc)
        ttnn.deallocate(z)
        q = ttnn.divide(zc, zmax)
        ttnn.deallocate(zmax)
        ttnn.copy(q, self.q)
        # Candidate i > 0 goes when the top-32's mass from i on, over Zc, is at most 1 - top_p / q.
        share = ttnn.divide(e, zc)
        cum = ttnn.cumsum(share, dim=3, reverse_order=True)
        ratio = ttnn.divide(self.top_p, q)
        omp = ttnn.rsub(ratio, 1.0)
        below = ttnn.le(cum, omp)
        drop = ttnn.multiply(below, self.notfirst)
        for t in (e, zc, q, share, cum, ratio, omp, below):
            ttnn.deallocate(t)
        # The last value kept: dropped candidates are lifted out of the minimum, kept ones add exactly 0.
        lifted = ttnn.multiply(drop, _BIG)
        raised = ttnn.add(top, lifted)
        ttnn.deallocate(lifted)
        cut = ttnn.min(raised, dim=-1, keepdim=True)
        ttnn.deallocate(raised)
        # The kept prefix may end inside a group of equal values: of that group only the candidates up to the last
        # one kept stay, the last one's position among the candidates (positions < 256 are exact in float32).
        same = ttnn.eq(top, cut)
        ttnn.deallocate(top)
        kept_flag = ttnn.rsub(drop, 1.0)
        kept_same = ttnn.multiply(same, kept_flag)
        for t in (same, drop, kept_flag):
            ttnn.deallocate(t)
        pos1 = ttnn.add(pos, 1.0)
        ttnn.deallocate(pos)
        marked = ttnn.multiply(kept_same, pos1)
        for t in (kept_same, pos1):
            ttnn.deallocate(t)
        last1 = ttnn.max(marked, dim=-1, keepdim=True)
        ttnn.deallocate(marked)
        last = ttnn.subtract(last1, 1.0)
        ttnn.deallocate(last1)
        thr = ttnn.where(self.flag, cut, self.floor)
        last_pos = ttnn.where(self.flag, last, self.ceiling)
        for t in (cut, last):
            ttnn.deallocate(t)
        x = ttnn.typecast(vals, ttnn.float32)
        ttnn.deallocate(vals)
        above = ttnn.gt(x, thr)
        equal = ttnn.eq(x, thr)
        early = ttnn.le(self.iota, last_pos)
        tie = ttnn.multiply(equal, early)
        inside = ttnn.add(above, tie)
        kept = ttnn.where(inside, x, self.pad)
        for t in (x, thr, last_pos, above, equal, early, tie, inside):
            ttnn.deallocate(t)
        masked = ttnn.typecast(kept, ttnn.bfloat16)
        ttnn.deallocate(kept)
        return self.sampler.draw(masked, gidx)

    def release(self):
        for t in (
            self.notfirst,
            self.flag,
            self.top_p,
            self.inv_t,
            self.q,
            self.floor,
            self.ceiling,
            self.pad,
            self.iota,
        ):
            ttnn.deallocate(t)
