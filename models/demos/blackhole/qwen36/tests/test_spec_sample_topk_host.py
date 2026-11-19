# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device): the candidates the verify sampler (tt/spec_sample.py VerifySampler) hands ttnn.sampling.

VerifySampler takes, on each device, the stable top-32 of each chunk of the device's vocabulary shard (one chunk of
31040 columns at V = 248320 and TP = 8, padded to 32768), adds the chunk's first vocabulary id and all-gathers the
candidates in shard order. ttnn.sampling ranks equal values by their position among the candidates, so a draw is a
function of the candidates sorted by (value descending, position ascending), the row's top-k, top-p and temperature,
and its core's seed.

Here the sampler's ops run on a torch model of each op over every device of a 1 x shards mesh (``Ops``): the top-k is
a stable sort that takes -0.0 as +0.0 (as the stable ttnn.topk does), the all-gather concatenates the devices' tensors
in mesh order, and ``sampling`` draws, per row, the candidate at rank (seed mod k) of the candidates sorted by value,
equal values by position. The reference is a stable torch sort of the whole vocabulary row. On tie-heavy logits, at
every row count and shard count:

  * every device holds each shard chunk's top-32 in value order, equal values lowest id first, in shard order;
  * sorted as ttnn.sampling sorts them, the candidates begin with the reference's top-32: the same values and ids;
  * the draws are the reference's: the token at rank (seed mod k) of the stable sort of the whole row.

A verify row bucket's sampler runs the same graph over the bucket's first b*T rows of the shard; the pad fills the
rows up to B*T, and the bucket rows' candidates and draws are the full-width sampler's.
"""

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.spec_sample import (
    NO_CORE,
    PAD_LOGIT,
    SAMPLER_TOP_K,
    TOPK_MAX_WIDTH,
    VerifySampler,
    sampling_plan,
    shard_chunks,
)

VOCAB = 248320


class Mesh:
    """A 1 x n mesh of the torch model."""

    def __init__(self, n):
        self.n = n


class MT:
    """A mesh tensor of the torch model: one torch tensor per device, its dtype and layout, whether it was freed."""

    def __init__(self, parts, dtype=None, layout=None):
        self.parts, self.dtype, self.layout, self.freed = [p.clone() for p in parts], dtype, layout, False

    @property
    def shape(self):
        return tuple(self.parts[0].shape)


def _sharded(t, mesh, dtype="bf16"):
    """``t`` [1, 1, rows, V] as each device's vocabulary shard."""
    return MT(torch.chunk(t, mesh.n, dim=3), dtype, "tile")


def _replicated(t, mesh, dtype="bf16"):
    return MT([t] * mesh.n, dtype, "tile")


class Ops:
    """The sampler's ttnn ops on MT tensors, device by device. Every op refuses a freed input."""

    def __init__(self):
        self.topk_calls, self.sampled, self.gathers = [], [], 0

    @staticmethod
    def _live(*ts):
        for t in ts:
            assert not t.freed, "a freed tensor is read"

    @staticmethod
    def _each(x, fn, dtype=None, layout=None):
        return MT([fn(p) for p in x.parts], x.dtype if dtype is None else dtype, x.layout if layout is None else layout)

    def install(self, monkeypatch):
        names = (
            "split", "concat", "pad", "topk", "typecast", "add", "untilize", "slice", "deallocate", "manual_seed",
            "sampling", "from_torch", "to_device", "copy_host_to_device_tensor", "mesh_partition",
        )  # fmt: skip
        for name in names:
            monkeypatch.setattr(ttnn, name, getattr(self, name))
        monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: ("replicate", mesh.n))
        monkeypatch.setattr(ttnn, "ShardTensorToMesh", lambda mesh, dim: ("shard", mesh.n, dim))

    def from_torch(self, t, dtype=None, layout=None, mesh_mapper=None, **kw):
        t = torch.as_tensor(t)
        if mesh_mapper[0] == "shard":
            return MT(torch.chunk(t, mesh_mapper[1], dim=mesh_mapper[2]), dtype, layout)
        return MT([t] * mesh_mapper[1], dtype, layout)

    def to_device(self, t, mesh, memory_config=None):
        return t

    def copy_host_to_device_tensor(self, host, dev):
        dev.parts = [h.reshape(d.shape).to(d.dtype).clone() for h, d in zip(host.parts, dev.parts)]

    def split(self, x, size, dim):
        self._live(x)
        per_device = [torch.split(p, size, dim=dim) for p in x.parts]
        return [MT([parts[i] for parts in per_device], x.dtype, x.layout) for i in range(len(per_device[0]))]

    def concat(self, ts, dim, memory_config=None):
        self._live(*ts)
        return MT([torch.cat(parts, dim=dim) for parts in zip(*(t.parts for t in ts))], ts[0].dtype, ts[0].layout)

    def pad(self, x, padding, value):
        self._live(x)
        flat = [n for before_after in reversed(padding) for n in before_after]
        return self._each(x, lambda p: torch.nn.functional.pad(p, flat, value=value))

    def topk(self, x, k, dim=-1, largest=True, sorted=True, stable=False):
        self._live(x)
        self.topk_calls.append(dict(input=[p.clone() for p in x.parts], largest=largest, sorted=sorted, stable=stable))
        values, indices = [], []
        for p in x.parts:
            rows = p.reshape(-1, p.shape[-1]) + 0.0  # -0.0 is +0.0, as the stable top-k returns it
            order = torch.sort(rows, dim=-1, descending=True, stable=True).indices[:, :k]
            values.append(rows.gather(1, order).reshape(*p.shape[:-1], k))
            indices.append(order.reshape(*p.shape[:-1], k))
        return MT(values, x.dtype, x.layout), MT(indices, "uint16", x.layout)

    def all_gather(self, x):
        self._live(x)
        self.gathers += 1
        return MT([torch.cat(x.parts, dim=3)] * len(x.parts), x.dtype, x.layout)

    def mesh_partition(self, x, dim, cluster_axis, memory_config=None):
        self._live(x)
        n = len(x.parts)
        return MT([torch.chunk(p, n, dim=dim)[d] for d, p in enumerate(x.parts)], x.dtype, x.layout)

    def typecast(self, x, dtype):
        self._live(x)
        return self._each(x, lambda p: p.clone(), dtype)

    def add(self, a, b, dtype=None):
        self._live(a, b)
        return MT([p + q for p, q in zip(a.parts, b.parts)], dtype, a.layout)

    def untilize(self, x, use_multicore=True):
        self._live(x)
        return self._each(x, lambda p: p.clone(), layout="row_major")

    def slice(self, x, starts, ends):
        self._live(x)
        return self._each(x, lambda p: p[tuple(slice(s, e) for s, e in zip(starts, ends))].clone())

    def deallocate(self, x):
        self._live(x)
        x.freed = True

    def manual_seed(self, seeds, user_ids):
        self._live(seeds, user_ids)
        self.seeds = seeds.parts[0].clone()

    def sampling(self, vals, idx, k, p, temp, output_tensor):
        self._live(vals, idx, k, output_tensor)
        self.sampled.append(
            (vals.parts[0].reshape(-1, vals.shape[-1]).clone(), idx.parts[0].reshape(-1, idx.shape[-1]))
        )
        for d, (v, i) in enumerate(zip(vals.parts, idx.parts)):
            v, i = v.reshape(-1, v.shape[-1]), i.reshape(-1, i.shape[-1])
            output_tensor.parts[d] = output_tensor.parts[d].clone()
            output_tensor.parts[d].reshape(-1)[: v.shape[0]] = _draws(v, i, self.seeds, k.parts[d]).to(torch.int32)
        return output_tensor


def _draws(values, ids, seeds, k):
    """Per row, the id at rank (seed mod k) of the candidates sorted by value, descending, equal values by position."""
    rows = values.shape[0]
    ranked = torch.sort(-values, dim=-1, stable=True).indices
    rank = seeds[:rows].to(torch.int64) % k.reshape(-1)[:rows].to(torch.int64).clamp(min=1)
    return ids.gather(1, ranked.gather(1, rank.reshape(-1, 1))).reshape(-1)


def _bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def _logits(kind, rows, vocab, shards, seed):
    """[rows, vocab] logits with ties: ``coarse`` integers in [-8, 8] (thousands of equal maxima per shard),
    ``plateau`` 20 equal maxima and a second group of 40 equal values that holds the 32nd place, both spread over the
    shards and on every shard boundary, ``bf16`` random logits rounded to bfloat16 (ties at bfloat16 spacing),
    ``zeros`` 30 +0.0 and 30 -0.0 as the maximum, ``flat`` one value everywhere."""
    g = torch.Generator().manual_seed(seed)
    if kind == "coarse":
        return torch.randint(-8, 9, (rows, vocab), generator=g).to(torch.float32)
    if kind == "bf16":
        return _bf16(torch.randn(rows, vocab, generator=g) * 3)
    if kind == "flat":
        return torch.full((rows, vocab), 1.5)
    x = torch.full((rows, vocab), -30.0)
    width = vocab // shards
    edges = torch.tensor([e for d in range(1, shards) for e in (d * width - 1, d * width)] + [0, vocab - 1])
    for r in range(rows):
        if kind == "plateau":
            pick = torch.randperm(vocab, generator=g)
            x[r, pick[:20]] = 5.0
            x[r, pick[20:60]] = 4.0
            x[r, edges] = 4.0 if r % 2 else 5.0
        else:
            hits = torch.randperm(vocab, generator=g)[:60]
            x[r, hits[:30]], x[r, hits[30:]] = 0.0, -0.0
            x[r, edges] = 0.0 if r % 2 else -0.0
    return x


def reference_order(logits):
    """(values, ids) of each row sorted by value, descending, equal values lowest id first, -0.0 taken as +0.0: a
    stable torch sort of the whole vocabulary row."""
    x = logits + 0.0
    order = torch.sort(x, dim=-1, descending=True, stable=True).indices
    return x.gather(1, order), order


def reference_candidates(logits, shards):
    """[rows, shards * chunks * 32]: each shard chunk's top-32 by the stable sort, in shard and chunk order."""
    width = logits.shape[1] // shards
    cw = width // shard_chunks(width)
    values, ids = [], []
    for start in range(0, logits.shape[1], cw):
        v, i = reference_order(logits[:, start : start + cw])
        values.append(v[:, :SAMPLER_TOP_K])
        ids.append(i[:, :SAMPLER_TOP_K] + start)
    return torch.cat(values, dim=1), torch.cat(ids, dim=1)


def _sampler(ops, B, T, vocab, shards):
    return VerifySampler(Mesh(shards), B, T, vocab, shards=shards, gather=ops.all_gather)


def _run(sampler, logits, plan, mesh, replicated=False):
    """Per device, the sampler's candidates (values, ids) and draws of ``logits`` [rows, vocab] under ``plan``."""
    rows = logits.shape[0]
    sampler.stage(plan)
    t = logits.reshape(1, 1, rows, -1)
    x = _replicated(t, mesh) if replicated else _sharded(t, mesh)
    vals, gidx = sampler.candidates(x)
    cand = [(v.reshape(sampler.rows, -1), i.reshape(sampler.rows, -1)) for v, i in zip(vals.parts, gidx.parts)]
    sampler.draw(vals, gidx)
    assert not x.freed, "the sampler frees nothing it did not make"
    return cand, [d.reshape(-1)[: sampler.rows].to(torch.int64) for d in sampler.draws.parts]


def _plan(B, T, step, live=None, k=SAMPLER_TOP_K):
    params = SimpleNamespace(temperature=[1.0] * B, top_k=[k] * B, top_p=[1.0] * B, seed=[1000 + u for u in range(B)])
    live = B if live is None else live
    return sampling_plan(params, [50 + 9 * u + step if u < live else -1 for u in range(B)], T)


KINDS = ("coarse", "plateau", "bf16", "zeros", "flat")
# (shards, vocab): TP = 8, 4, 2 and 1 at the model's vocabulary (1, 2, 4, 8 chunks per shard), and a small vocabulary
# whose 1024-column chunks pad to the stable top-k's narrowest multi-core width.
MESHES = [(8, VOCAB), (4, VOCAB), (2, VOCAB), (1, VOCAB), (4, 4096)]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shards, vocab", MESHES)
@pytest.mark.parametrize("B, T", [(8, 4), (1, 4), (2, 12)])
def test_every_device_holds_each_shards_top_32_lowest_id_first_and_draws_what_a_stable_sort_draws(
    monkeypatch, kind, shards, vocab, B, T
):
    """Candidates equal the reference shard chunks' top-32 position for position, the same on every device; sorted
    as ttnn.sampling sorts them they begin with the whole row's stable top-32; and the draws of 3 seeded plans at
    top-k 32 and 3 at top-k 7 are the stable sort's tokens at the draw's rank, on every device."""
    ops = Ops()
    ops.install(monkeypatch)
    rows = B * T
    sampler = _sampler(ops, B, T, vocab, shards)
    logits = _logits(kind, rows, vocab, shards, seed=rows + len(kind) + shards)
    want_v, want_i = reference_candidates(logits, shards)
    top_v, top_i = reference_order(logits)
    for step, k in enumerate((32, 32, 32, 7, 7, 7)):
        plan = _plan(B, T, step, k=k)
        cand, draws = _run(sampler, logits, plan, Mesh(shards))
        for d, (v, i) in enumerate(cand):
            assert torch.equal(v, want_v) and torch.equal(i, want_i), f"device {d}: candidates differ"
            order = torch.sort(-v, dim=-1, stable=True).indices[:, :SAMPLER_TOP_K]
            assert torch.equal(v.gather(1, order), top_v[:, :SAMPLER_TOP_K]), f"device {d}: top-32 values"
            assert torch.equal(i.gather(1, order), top_i[:, :SAMPLER_TOP_K]), f"device {d}: top-32 ids"
        seeds = torch.tensor(plan.seeds, dtype=torch.int64)
        want = top_i.gather(1, (seeds % k).reshape(-1, 1)).reshape(-1)
        for d, got in enumerate(draws):
            assert torch.equal(got, want), f"step {step}, device {d}: the draws differ from the stable sort's"
    assert all(c["stable"] and c["largest"] and c["sorted"] for c in ops.topk_calls)


@pytest.mark.parametrize("shards", [8, 2])
def test_one_stable_top_k_call_takes_each_shard_chunk_padded_as_a_row(monkeypatch, shards):
    """One ttnn.topk call per sampler, over [1, chunks, B*T, 32768] on each device: row (c, r) is chunk c of the
    device's shard for row r of the logits, then PAD_LOGIT; the rows past the logits (a bucket's) are PAD_LOGIT."""
    ops = Ops()
    ops.install(monkeypatch)
    B, T, rows = 8, 4, 8
    chunks = shard_chunks(VOCAB // shards)
    cw = VOCAB // shards // chunks
    logits = _logits("bf16", rows, VOCAB, shards, seed=3)
    _run(_sampler(ops, B, T, VOCAB, shards), logits, _plan(B, T, 0, live=2), Mesh(shards))
    (call,) = ops.topk_calls
    assert call["stable"]
    for d, x in enumerate(call["input"]):
        assert x.shape == (1, chunks, B * T, TOPK_MAX_WIDTH)
        for c in range(chunks):
            start = d * VOCAB // shards + c * cw
            assert torch.equal(x[0, c, :rows, :cw], logits[:, start : start + cw])
        assert bool((x[0, :, :rows, cw:] == PAD_LOGIT).all()) and bool((x[0, :, rows:] == PAD_LOGIT).all())


@pytest.mark.parametrize("kind", ["coarse", "plateau", "bf16"])
@pytest.mark.parametrize("shards", [8, 2])
def test_replicated_logits_give_the_candidates_of_their_shards(monkeypatch, kind, shards):
    """Logits that every device holds whole (a verify that gathers them itself) are cut into the devices' shards first:
    the same candidates and draws as from the shards."""
    ops = Ops()
    ops.install(monkeypatch)
    B, T = 8, 4
    sampler = _sampler(ops, B, T, VOCAB, shards)
    logits = _logits(kind, B * T, VOCAB, shards, seed=11)
    plan = _plan(B, T, 0)
    got = _run(sampler, logits, plan, Mesh(shards), replicated=True)
    want = _run(sampler, logits, plan, Mesh(shards))
    for (gc, gd), (wc, wd) in zip(zip(*got), zip(*want)):
        assert all(torch.equal(a, b) for a, b in zip(gc, wc)) and torch.equal(gd, wd)


@pytest.mark.parametrize("shards, replicated", [(8, False), (2, False), (8, True), (1, False)])
@pytest.mark.parametrize("rows", [32, 8])
def test_the_sampler_frees_every_intermediate(monkeypatch, shards, replicated, rows):
    """Every tensor made while drawing is freed, the logits and the sampler's own buffers are not."""
    made = []
    ops = Ops()
    ops.install(monkeypatch)
    sampler = _sampler(ops, 8, 4, VOCAB, shards)
    own = {id(t) for t in vars(sampler).values() if isinstance(t, MT)}
    real_init = MT.__init__

    def tracked(self, *a, **kw):
        real_init(self, *a, **kw)
        made.append(self)

    monkeypatch.setattr(MT, "__init__", tracked)
    t = _logits("bf16", rows, VOCAB, shards, seed=5).reshape(1, 1, rows, -1)
    logits = _replicated(t, Mesh(shards)) if replicated else _sharded(t, Mesh(shards))
    sampler.stage(_plan(8, 4, 0, live=rows // 4))
    made.clear()
    sampler.body(logits)
    left = [t.shape for t in made if not t.freed and id(t) not in own]
    assert left == [] and not logits.freed


@pytest.mark.parametrize("kind", ["coarse", "plateau", "bf16"])
@pytest.mark.parametrize("b", [1, 2, 4])
def test_a_bucket_sampler_draws_its_rows_as_the_full_width_sampler_does(monkeypatch, kind, b):
    """The shard of the first b*T rows (a verify row bucket's) gives those rows the full-width sampler's candidates
    and draws; ttnn.sampling still draws B*T rows, those past the bucket from PAD_LOGIT candidates and no seeded core.
    """
    B, T, shards = 8, 4, 8
    rows = b * T
    ops = Ops()
    ops.install(monkeypatch)
    sampler = _sampler(ops, B, T, VOCAB, shards)
    logits = _logits(kind, B * T, VOCAB, shards, seed=b)
    for step in range(3):
        plan = _plan(B, T, step, live=b)
        assert plan.user_ids[rows:] == [NO_CORE] * (B * T - rows)
        full_c, full_d = _run(sampler, logits, plan, Mesh(shards))
        got_c, got_d = _run(sampler, logits[:rows], plan, Mesh(shards))
        for d in range(shards):
            (fv, fi), (gv, gi) = full_c[d], got_c[d]
            assert gv.shape == fv.shape == (B * T, shards * SAMPLER_TOP_K)
            assert torch.equal(gv[:rows], fv[:rows]) and torch.equal(gi[:rows], fi[:rows])
            assert bool((gv[rows:] == PAD_LOGIT).all())
            assert torch.equal(got_d[d][:rows], full_d[d][:rows])


def test_the_sampler_refuses_what_it_cannot_take(monkeypatch, expect_error):
    ops = Ops()
    ops.install(monkeypatch)
    sampler = _sampler(ops, 8, 4, VOCAB, 8)
    mesh = Mesh(8)
    for rows in (0, 33):
        with expect_error(ValueError, "are not a prefix of them"):
            sampler.candidates(_sharded(torch.zeros(1, 1, rows, VOCAB), mesh))
    with expect_error(ValueError, "the sampler takes 31040 \\(one of 8 shards\\)"):
        sampler.candidates(_sharded(torch.zeros(1, 1, 32, VOCAB // 2), mesh))
    with expect_error(ValueError, "does not split into 3 shards"):
        _sampler(ops, 8, 4, VOCAB, 3)
    with expect_error(ValueError, "not a power-of-two count of tiles"):
        _sampler(ops, 8, 4, 3 * 31040, 3)
    with expect_error(ValueError, "exceed the sampler"):
        _sampler(ops, 9, 4, VOCAB, 8)


def test_a_shard_splits_into_the_fewest_tile_aligned_chunks_the_top_k_takes(expect_error):
    assert [shard_chunks(VOCAB // tp) for tp in (8, 4, 2, 1)] == [1, 2, 4, 8]
    assert shard_chunks(1024) == 1 and shard_chunks(TOPK_MAX_WIDTH) == 1 and shard_chunks(3 * 31040) == 3
    with expect_error(ValueError, "does not split into tile-aligned top-k chunks"):
        shard_chunks(VOCAB + 1)
