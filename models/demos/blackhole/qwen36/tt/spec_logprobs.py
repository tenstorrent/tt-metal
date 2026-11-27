# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Logprobs of an argmax_ids verify (vllm-tt-plugin ``supports_spec_device_logprobs``) for
Qwen36MTPContractForCausalLM.

The plugin sends ``spec_num_logprobs`` with a verify whose rows ask for logprobs. The verify then also replays the
logprobs trace (``VerifyLogprobs.body``) right after the verify trace (and the sampler trace on a device-sampled step),
over the same logits, and the host reads its outputs with the ids:

  * the candidates, as the verify sampler builds them (``spec_sample.VerifySampler.candidates``): each vocabulary
    shard's stable top-32 with global ids, gathered in shard order, ``shards * 32`` per row;
  * the sum ``L = sum_v exp(x_v - M)`` over the whole vocabulary in float32, with ``M`` the row's largest candidate,
    which is its global maximum: each device sums its shard (float32 exp, HiFi4 with float32 destination accumulation;
    bfloat16 accumulation of the 31040 terms of a shard moves the log-sum-exp by about 0.25), and the shards' sums are
    gathered and added.

``spec_logprobs_payload`` turns them into the plugin's payload, per row and column t: the raw logprob of the token the
verify answers at that column (``ids[u, t]``, the argmax or the device sampler's draw) as its candidate value minus
``M + log L``, its rank as the count of candidates at or above that value, and the top-``TOP_LOGPROBS`` ids and logprobs.
Raw means before temperature: vLLM's default ``raw_logprobs``.

The token is always among the candidates: the argmax is the global maximum, and a draw is among the vocabulary's
top-32 under (value descending, id ascending), each of which is among its shard's candidates (spec_sample). The rank
is exact unless the token's value equals the smallest candidate of a shard, whose further tokens of that value are not
candidates: then it is a lower bound, as vLLM's rank of a token tied with tokens it never saw.
"""

import torch

import ttnn
from models.demos.blackhole.qwen36.tt.spec_argmax_fused import all_gather_dim3
from models.demos.blackhole.qwen36.tt.spec_sample import SAMPLER_ROWS, VerifySampler

TOP_LOGPROBS = 20  # vllm_tt_plugin.spec_decode.MAX_SPEC_DEVICE_LOGPROBS: the payload's top-N


class VerifyLogprobs:
    """The device half: the candidate source, the persistent outputs the host reads (``vals`` bfloat16 TILE and ``ids``
    uint32 ROW_MAJOR [1, 1, B*T, shards*32], ``sums`` float32 TILE [1, 1, 32, 1], one entry per row) and the
    trace-capturable ``body``. ``shards`` and ``gather`` as for ``VerifySampler``; ``source`` a ``VerifySampler`` of
    the same verify to take the candidates from (its ``candidates`` reads only its offsets), else one of its own."""

    def __init__(self, mesh, B, T, vocab, shards=1, gather=all_gather_dim3, source=None):
        self.mesh, self.rows, self.shards = mesh, B * T, int(shards)
        self.source = source if source is not None else VerifySampler(mesh, B, T, vocab, shards=shards, gather=gather)
        self.vocab, self.width = self.source.vocab, self.source.width
        cols = self.source.candidates_per_row
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def persistent(t, dtype, layout):
            return ttnn.from_torch(
                t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        self.vals = persistent(torch.zeros(1, 1, self.rows, cols), ttnn.bfloat16, ttnn.TILE_LAYOUT)
        self.ids = persistent(torch.zeros(1, 1, self.rows, cols, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        self.sums = persistent(torch.ones(1, 1, SAMPLER_ROWS, 1), ttnn.float32, ttnn.TILE_LAYOUT)
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    @property
    def outputs(self):
        """The tensors a verify reads after the replay, in ``payload_from_reads`` order."""
        return [self.vals, self.ids, self.sums]

    def body(self, logits):
        """``logits`` as ``VerifySampler.body`` takes them: each device's vocabulary shard [1, 1, rows, vocab / shards]
        of the first ``rows`` <= B*T rows, or the whole vocabulary on every device. Writes the candidates into ``vals``
        and ``ids`` and the rows' sums into ``sums`` (rows past ``rows`` hold 1)."""
        vals, gidx = self.source.candidates(logits)
        ttnn.copy(vals, self.vals)
        ttnn.copy(gidx, self.ids)
        ttnn.deallocate(gidx)
        rows = int(logits.shape[2])
        top = ttnn.max(vals, dim=-1, keepdim=True)
        ttnn.deallocate(vals)
        if rows < self.rows:
            full = top
            top = ttnn.slice(full, (0, 0, 0, 0), (1, 1, rows, 1))
            ttnn.deallocate(full)
        m = ttnn.typecast(top, ttnn.float32)
        ttnn.deallocate(top)
        s = vocab_exp_sum(logits, m, self.shards, self.vocab, self.compute)
        ttnn.deallocate(m)
        ttnn.copy(s, self.sums)
        ttnn.deallocate(s)

    def release(self):
        for t in (self.vals, self.ids, self.sums):
            ttnn.deallocate(t)


def vocab_exp_sum(logits, m, shards, vocab, compute, scale=None):
    """float32 TILE [1, 1, 32, 1]: for each row of ``logits`` (each device's vocabulary shard [1, 1, rows,
    vocab / shards], or the whole vocabulary on every device), the sum over the whole vocabulary of exp((x - m) * scale)
    with ``m`` and ``scale`` float32 [1, 1, rows, 1] (``scale`` None: 1). Each device sums its shard in float32 (HiFi4,
    float32 destination accumulation); the shards' sums are gathered and added. Rows past ``rows`` hold 1."""
    rows, width = int(logits.shape[2]), int(logits.shape[3])
    x = ttnn.typecast(logits, ttnn.float32)
    d = ttnn.subtract(x, m)
    ttnn.deallocate(x)
    if scale is not None:
        unscaled = d
        d = ttnn.multiply(unscaled, scale)
        ttnn.deallocate(unscaled)
    e = ttnn.exp(d, fast_and_approximate_mode=False)
    ttnn.deallocate(d)
    s = ttnn.sum(e, dim=-1, keepdim=True, compute_kernel_config=compute)
    ttnn.deallocate(e)
    if rows < SAMPLER_ROWS:
        short = s
        s = ttnn.pad(short, [(0, 0), (0, 0), (0, SAMPLER_ROWS - rows), (0, 0)], value=1.0)
        if not _aliases(s, short):
            ttnn.deallocate(short)
    if shards > 1 and width != vocab:
        g = ttnn.all_gather(s, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(s)
        g4 = ttnn.reshape(g, (1, shards, SAMPLER_ROWS, 1))
        s = ttnn.sum(g4, dim=1, keepdim=True, compute_kernel_config=compute)
        if not _aliases(g4, g):
            ttnn.deallocate(g4)
        ttnn.deallocate(g)
    return s


def _aliases(a, b):
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:  # pragma: no cover
        return True


def lse_from_stats(max_val, l_raw):
    """Raw logsumexp in float64 from the row maximum M [R] and L_raw [R] = sum exp(x - M): M + log(L_raw)."""
    return torch.as_tensor(max_val).to(torch.float64) + torch.as_tensor(l_raw).to(torch.float64).log()


def stats_from_logits(logits, temperature):
    """Float64 reference of the device sums on FULL logits [R, V]: (M [R], L_raw [R], L_T [R]); L_T uses the row's
    temperature [R] (<= 0 or greedy rows pass 1.0)."""
    x = torch.as_tensor(logits).to(torch.float64)
    t = torch.as_tensor(temperature).reshape(-1, 1).to(torch.float64)
    m = x.max(dim=-1, keepdim=True).values
    d = x - m
    return m.squeeze(-1), d.exp().sum(-1), (d / t).exp().sum(-1)


def stable_order(values, ids):
    """Per row, the column order (value descending, id ascending) of ``values`` and ``ids`` [R, C]."""
    by_id = torch.argsort(ids, dim=1, stable=True)
    by_value = torch.argsort(torch.gather(values, 1, by_id), dim=1, descending=True, stable=True)
    return torch.gather(by_id, 1, by_value)


def gather_logprobs_from_candidates(cand_ids, cand_values, lse, chosen, num_logprobs):
    """vLLM ``Sampler.gather_logprobs`` layout from the payload, for N rows: column 0 = the chosen token, columns
    1..num_logprobs = the global top-N. cand_ids [N, M], cand_values [N, M] raw logits in (value desc, id asc) order,
    lse [N], chosen [N] token ids. Returns (token_ids int64 [N, n+1], logprobs fp32 [N, n+1], rank int64 [N],
    lower_bound bool [N], missing bool [N]). rank = number of candidates whose value is >= the chosen one (vLLM). It
    is a lower bound when the M-th candidate value is not strictly below the chosen value (``lower_bound``). A chosen
    id outside the M candidates is ``missing``: logprob = the M-th candidate value (an upper bound), rank M + 1."""
    ids = torch.as_tensor(cand_ids).to(torch.int64)
    vals = torch.as_tensor(cand_values).to(torch.float32)
    lse = torch.as_tensor(lse).to(torch.float32).reshape(-1, 1)
    chosen = torch.as_tensor(chosen).to(torch.int64).reshape(-1, 1)
    N, M = ids.shape
    if not 0 <= num_logprobs <= M:
        raise ValueError(f"num_logprobs {num_logprobs} outside [0, {M}]")
    hit = ids == chosen
    missing = ~hit.any(dim=1)
    pos = hit.to(torch.int8).argmax(dim=1)
    cval = torch.where(missing, vals[:, M - 1], vals.gather(1, pos[:, None]).reshape(-1))
    rank = (vals >= cval[:, None]).sum(dim=1)
    rank = torch.where(missing, torch.full_like(rank, M + 1), rank)
    lower_bound = ~missing & (vals[:, M - 1] >= cval)
    tok = torch.cat([chosen, ids[:, :num_logprobs]], dim=1)
    lp = torch.cat([cval[:, None], vals[:, :num_logprobs]], dim=1) - lse
    return tok, lp, rank.to(torch.int64), lower_bound, missing


def spec_logprobs_payload(values, ids, sums, chosen, live=None, top=TOP_LOGPROBS):
    """The plugin's per-row payload from the logprobs trace's outputs.

    ``values`` [R, C] float (the candidates' raw logits, any order), ``ids`` [R, C] their token ids, ``sums`` [R] the
    rows' sums L, ``chosen`` [R] the token each row answers. ``live`` [R] bool: rows whose ``chosen`` must be a
    candidate (RuntimeError otherwise, with the rows); the others are padding rows, whose entries nothing reads.
    Returns (sampled_logprobs float32 [R], selected_token_ranks int32 [R], top_logprobs float32 [R, top],
    top_token_ids int32 [R, top])."""
    values = torch.as_tensor(values).to(torch.float32)
    ids = torch.as_tensor(ids).to(torch.int64)
    order = stable_order(values, ids)
    values, ids = torch.gather(values, 1, order), torch.gather(ids, 1, order)
    lse = lse_from_stats(values[:, 0], sums)
    tok, lp, rank, _, missing = gather_logprobs_from_candidates(ids, values, lse, chosen, top)
    live = torch.ones_like(missing) if live is None else torch.as_tensor(live, dtype=torch.bool).reshape(-1)
    bad = (live & missing).nonzero().reshape(-1).tolist()
    if bad:
        raise RuntimeError(
            f"verify rows {bad} answer tokens {tok[bad, 0].tolist()} that are not among their candidates; the "
            "argmax and every device draw are"
        )
    return lp[:, 0].contiguous(), rank.to(torch.int32), lp[:, 1:].contiguous(), tok[:, 1:].to(torch.int32)


def exact_logprobs(logits_row, chosen, top=TOP_LOGPROBS):
    """The payload entries of one row from its full logits [V]: (raw logprob of ``chosen`` float32, rank int32, top
    logprobs float32 [top], top ids int32 [top]), the top in (value descending, id ascending) order."""
    x = torch.as_tensor(logits_row).to(torch.float32)
    lsm = torch.log_softmax(x.to(torch.float64), dim=-1)
    order = torch.sort(x, descending=True, stable=True).indices[:top]
    return (
        lsm[int(chosen)].to(torch.float32),
        (x >= x[int(chosen)]).sum().to(torch.int32),
        lsm[order].to(torch.float32),
        order.to(torch.int32),
    )


def payload_from_reads(reads, chosen, live, B, T, full_logits=None, exact=None):
    """The payload of a B x T verify from the host copies of ``VerifyLogprobs.outputs`` (device-0 replicas) and the
    verify's answered ids ``chosen`` [B, T]; ``live`` [B] the rows a request owns. ``exact`` [B*T] bool (with
    ``full_logits`` [>= B*T, V]): rows whose token the host drew from the full logits, which may lie outside the
    candidates; their entries come from the full logits (``exact_logprobs``). Returns the four tensors of
    ``spec_logprobs_payload`` shaped [B, T] and [B, T, top]."""
    rows = B * T
    vals, ids, sums = (torch.as_tensor(t) for t in reads)
    vals = vals.reshape(-1, vals.shape[-1])[:rows]
    ids = ids.reshape(-1, ids.shape[-1])[:rows]
    sums = sums.reshape(-1)[:rows]
    chosen = torch.as_tensor(chosen).reshape(-1)
    live_rows = torch.as_tensor(live, dtype=torch.bool).reshape(B, 1).expand(B, T).reshape(-1)
    exact = torch.zeros(rows, dtype=torch.bool) if exact is None else torch.as_tensor(exact).reshape(-1)[:rows]
    out = spec_logprobs_payload(vals, ids, sums, chosen, live_rows & ~exact)
    for r in exact.nonzero().reshape(-1).tolist():
        for t, value in zip(out, exact_logprobs(full_logits[r], chosen[r])):
            t[r] = value
    return tuple(t.reshape(B, T, *t.shape[1:]) for t in out)
