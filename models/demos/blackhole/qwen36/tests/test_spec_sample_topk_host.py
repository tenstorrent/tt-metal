# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the candidates the verify sampler (tt/spec_sample.py VerifySampler) hands ttnn.sampling.

VerifySampler takes the top-32 of each vocabulary chunk of the verify's logits (4 chunks of 62080 columns at
V = 248320) and draws from the chunks' candidates in chunk order. ttnn.sampling ranks equal values by their position
among the candidates, so the order ttnn.topk gives equal values decides the token a draw lands on. The routed top-k
of these widths orders equal values in a deterministic but unspecified sequence of its own merges.

The sampler stacks the chunks as rows of one [1, chunks, rows, 62080] tensor and takes one ttnn.topk call;
ChunkedVerifySampler (tests/spec_op_graphs.py) is the sampler with one call per chunk. Here the ops of both run on a
torch model of each op (``Ops``), whose top-k orders equal values by a function of the row's content and width only,
as a row-independent top-k on the device does: the stacked sampler hands ttnn.sampling the same values and ids in the
same order as the per-chunk one, for every such tie order, for tie-heavy logits, at every row count, and draws the
same tokens. Whether the device top-k is row-independent is test_spec_sample.py's to show.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.spec_op_graphs import ChunkedVerifySampler
from models.demos.blackhole.qwen36.tt.spec_sample import SAMPLER_TOP_K, VerifySampler, sampling_plan, vocab_chunks

VOCAB = 248320


class FT:
    """A device tensor of the torch model: its value, dtype, layout, and whether it was freed."""

    def __init__(self, t, dtype=None, layout=None):
        self.t, self.dtype, self.layout, self.freed = t, dtype, layout, False

    @property
    def shape(self):
        return tuple(self.t.shape)


def _tie_key(order):
    """Per row of x [n, W], the key that orders equal values: lowest index first, highest first, or a hash of the
    index with the row's content and width (deterministic, independent of the row's place in the tensor)."""

    def key(x):
        W = x.shape[-1]
        idx = torch.arange(W, dtype=torch.int64).expand_as(x)
        if order == "lowest":
            return idx
        if order == "highest":
            return -idx
        digest = (x.to(torch.float64) * 1024).round().to(torch.int64).sum(dim=-1, keepdim=True) + W
        return (idx * 2654435761 + digest * 40503) % (1 << 31)

    return key


class Ops:
    """The sampler's ttnn ops on FT tensors. ``topk`` sorts each row by value, descending, equal values by ``tie_key``;
    ``sampling`` draws, per row, the candidate at rank (seed mod k) of the candidates sorted by value, descending, equal
    values by position, as ttnn.sampling's stable sort ranks them. Every op refuses a freed input."""

    def __init__(self, tie_key):
        self.tie_key, self.topk_inputs, self.sampled = tie_key, [], []

    @staticmethod
    def _live(*ts):
        for t in ts:
            assert not t.freed, "a freed tensor is read"

    def install(self, monkeypatch):
        for name in ("split", "concat", "topk", "typecast", "add", "untilize", "slice", "deallocate", "manual_seed",
                     "sampling", "from_torch", "to_device", "copy_host_to_device_tensor"):  # fmt: skip
            monkeypatch.setattr(ttnn, name, getattr(self, name))

    def from_torch(self, t, dtype=None, layout=None, mesh_mapper=None, **kw):
        return FT(torch.as_tensor(t).clone(), dtype, layout)

    def to_device(self, t, mesh, memory_config=None):
        return t

    def copy_host_to_device_tensor(self, host, dev):
        dev.t = host.t.reshape(dev.t.shape).to(dev.t.dtype).clone()

    def split(self, x, size, dim):
        self._live(x)
        return [FT(p.clone(), x.dtype, x.layout) for p in torch.split(x.t, size, dim=dim)]

    def concat(self, ts, dim, memory_config=None):
        self._live(*ts)
        return FT(torch.cat([t.t for t in ts], dim=dim), ts[0].dtype, ts[0].layout)

    def topk(self, x, k, dim=-1):
        self._live(x)
        self.topk_inputs.append(x.t.clone())
        rows = x.t.reshape(-1, x.shape[-1])
        by_key = torch.sort(self.tie_key(rows), dim=-1, stable=True).indices
        by_value = torch.sort(-rows.gather(1, by_key), dim=-1, stable=True).indices
        order = by_key.gather(1, by_value)[:, :k]
        shape = (*x.shape[:-1], k)
        return FT(rows.gather(1, order).reshape(shape), x.dtype, x.layout), FT(order.reshape(shape), "uint16")

    def typecast(self, x, dtype):
        self._live(x)
        return FT(x.t.clone(), dtype, x.layout)

    def add(self, a, b, dtype=None):
        self._live(a, b)
        return FT(a.t + b.t, dtype, a.layout)

    def untilize(self, x, use_multicore=True):
        self._live(x)
        return FT(x.t.clone(), x.dtype, "row_major")

    def slice(self, x, starts, ends):
        self._live(x)
        return FT(x.t[tuple(slice(s, e) for s, e in zip(starts, ends))].clone(), x.dtype, x.layout)

    def deallocate(self, x):
        self._live(x)
        x.freed = True

    def manual_seed(self, seeds, user_ids):
        self._live(seeds, user_ids)
        self.seeds = seeds.t.clone()

    def sampling(self, vals, idx, k, p, temp, output_tensor):
        self._live(vals, idx, k, output_tensor)
        v, i = vals.t.reshape(-1, vals.shape[-1]), idx.t.reshape(-1, idx.shape[-1])
        self.sampled.append((v.clone(), i.clone()))
        ranked = torch.sort(-v, dim=-1, stable=True).indices
        rank = self.seeds[: v.shape[0]].to(torch.int64) % k.t[: v.shape[0]].to(torch.int64).clamp(min=1)
        draws = i.gather(1, ranked.gather(1, rank.reshape(-1, 1))).reshape(-1)
        output_tensor.t = output_tensor.t.clone()
        output_tensor.t.reshape(-1)[: draws.numel()] = draws.to(output_tensor.t.dtype)
        return output_tensor


def _bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def _logits(kind, rows, vocab, seed):
    """[rows, vocab] logits with ties: ``coarse`` integers in [-8, 8] (thousands of equal maxima per chunk),
    ``plateau`` 40 equal maxima in the first chunk and 40 in the third, at chunk ends too, then a second plateau,
    ``bf16`` random logits rounded to bfloat16 (ties at bfloat16 spacing), ``zeros`` +0.0 and -0.0 as the maximum."""
    g = torch.Generator().manual_seed(seed)
    if kind == "coarse":
        return torch.randint(-8, 9, (rows, vocab), generator=g).to(torch.float32)
    if kind == "bf16":
        return _bf16(torch.randn(rows, vocab, generator=g) * 3)
    x = torch.full((rows, vocab), -30.0)
    width = vocab // vocab_chunks(vocab)
    for r in range(rows):
        if kind == "plateau":
            top = torch.cat(
                [torch.randperm(width, generator=g)[:38], torch.tensor([0, width - 1])]
                + ([torch.randperm(width, generator=g)[:38] + 2 * width, torch.tensor([2 * width, 3 * width - 1])]
                   if vocab >= 3 * width else [])  # fmt: skip
            )
            x[r, top] = 5.0
            x[r, torch.randperm(vocab, generator=g)[:50]] = 4.0
        else:
            hits = torch.randperm(vocab, generator=g)[:60]
            x[r, hits[:30]], x[r, hits[30:]] = 0.0, -0.0
    return x


def _run(sampler, logits, plan):
    """The sampler's candidates and draws of ``logits`` [rows, vocab] under ``plan``."""
    rows = logits.shape[0]
    sampler.stage(plan)
    vals, gidx = sampler.candidates(FT(logits.reshape(1, 1, rows, -1).clone(), "bf16", "tile"))
    cand = (vals.t.clone(), gidx.t.clone())
    sampler.draw(vals, gidx)
    return cand, sampler.draws.t.reshape(-1)[:rows].clone()


def _plan(B, T, step):
    params = SimpleNamespace(
        temperature=[1.0] * B, top_k=[SAMPLER_TOP_K] * B, top_p=[1.0] * B, seed=[1000 + u for u in range(B)]
    )
    return sampling_plan(params, [50 + 9 * u + step for u in range(B)], T)


@pytest.mark.parametrize("order", ["lowest", "highest", "hashed"])
@pytest.mark.parametrize("kind", ["coarse", "plateau", "bf16", "zeros"])
@pytest.mark.parametrize("B, T, vocab", [(8, 4, VOCAB), (1, 4, VOCAB), (2, 12, 131072), (4, 4, 8192)])
def test_the_stacked_top_k_hands_the_sampler_the_per_chunk_candidates(monkeypatch, order, kind, B, T, vocab):
    """Values and ids equal bit for bit, in the same positions, and the same draws for 3 seeded plans."""
    ops = Ops(_tie_key(order))
    ops.install(monkeypatch)
    stacked, chunked = VerifySampler("mesh", B, T, vocab), ChunkedVerifySampler("mesh", B, T, vocab)
    logits = _logits(kind, B * T, vocab, seed=B * T + len(kind))
    for step in range(3):
        plan = _plan(B, T, step)
        (v1, i1), d1 = _run(stacked, logits, plan)
        (v0, i0), d0 = _run(chunked, logits, plan)
        assert torch.equal(v1, v0) and torch.equal(i1, i0), f"step {step}: the candidates differ"
        assert all(torch.equal(a, b) for a, b in zip(ops.sampled[-1], ops.sampled[-2])), "ttnn.sampling inputs differ"
        assert torch.equal(d1, d0), f"step {step}: the draws differ"
        signs = torch.signbit(v1).flatten().tolist()
        assert kind != "zeros" or (any(signs) and not all(signs)), "the zeros case keeps both signs of zero"


@pytest.mark.parametrize("B, T", [(8, 4), (1, 4)])
def test_one_top_k_call_takes_every_chunk_as_rows_of_its_input(monkeypatch, B, T):
    """The stacked sampler calls ttnn.topk once, over [1, chunks, rows, width], whose row (c, r) is chunk c of row r;
    the per-chunk sampler calls it once per chunk over [1, 1, rows, width]."""
    ops = Ops(_tie_key("hashed"))
    ops.install(monkeypatch)
    rows, chunks = B * T, vocab_chunks(VOCAB)
    width = VOCAB // chunks
    logits = _logits("bf16", rows, VOCAB, seed=3)
    _run(VerifySampler("mesh", B, T, VOCAB), logits, _plan(B, T, 0))
    (stacked,) = ops.topk_inputs
    assert stacked.shape == (1, chunks, rows, width)
    for c in range(chunks):
        assert torch.equal(stacked[0, c], logits[:, c * width : (c + 1) * width])
    ops.topk_inputs.clear()
    _run(ChunkedVerifySampler("mesh", B, T, VOCAB), logits, _plan(B, T, 0))
    assert [tuple(t.shape) for t in ops.topk_inputs] == [(1, 1, rows, width)] * chunks


def test_the_sampler_frees_every_intermediate(monkeypatch):
    """Every tensor made while drawing is freed, the logits and the sampler's own buffers are not."""
    made = []
    ops = Ops(_tie_key("hashed"))
    ops.install(monkeypatch)
    sampler = VerifySampler("mesh", 8, 4, VOCAB)
    own = {id(t) for t in vars(sampler).values() if isinstance(t, FT)}
    real_init = FT.__init__

    def tracked(self, *a, **kw):
        real_init(self, *a, **kw)
        made.append(self)

    monkeypatch.setattr(FT, "__init__", tracked)
    logits = FT(_logits("bf16", 32, VOCAB, seed=5).reshape(1, 1, 32, -1), "bf16", "tile")
    sampler.stage(_plan(8, 4, 0))
    made.clear()
    sampler.body(logits)
    left = [t.shape for t in made if not t.freed and id(t) not in own]
    assert left == [] and not logits.freed
