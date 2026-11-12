# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the exact drafter and verify-tail changes give the ids and rows of the paths they replace.

* test_drafter_argmax_matches_the_sharded_tail: fp32 shards, b in (1, 2, 4, 8) rows; draft_argmax (fused per-shard
  argmax, tt/spec_argmax_fused.py) against Qwen36MTP._argmax_sharded and torch.argmax.
* test_verify_argmax_matches_the_gathered_argmax: bf16, R in (4, 8, 16, 32) rows (the verify's B*T); the fused ids
  of the vocab shards against ttnn.argmax over the untilized full row (the verify tail's argmax) and torch.argmax.
* test_fused_argmax_in_a_trace: both captured in a trace, replayed on fresh inputs, == the replaced tails == torch;
  and the replay time of each tail.
* test_one_row_embedding_matches_the_tile_lookup: Qwen36MTP._embed_rows against the TILE lookup plus reshape, byte
  for byte, at the drafter's (b) and the reseed's (B*T) row counts.

Inputs carry the tie rows of test_drafter_exact_host.tie_rows (ties inside and across shards, all-equal, +0 / -0, -inf,
1-ulp pairs, quantized ties) plus random rows.

    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_drafter_exact.py
"""

import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_drafter_exact_host import VOCAB, tie_rows
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape

MESH = _resolve_mesh_shape(8)


def _parametrize(fn):
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _ids(t, n):
    """The first n uint32 ids of a replicated ROW_MAJOR ids tensor; every device holds the same ids."""
    per_dev = [ttnn.to_torch(d).reshape(-1)[:n].to(torch.int64) for d in ttnn.get_device_tensors(t)]
    for d in per_dev[1:]:
        assert torch.equal(d, per_dev[0]), f"devices disagree: {d.tolist()} vs {per_dev[0].tolist()}"
    return per_dev[0]


def _rows(n, dtype, seed):
    """n rows: the tie cases rotated by the seed, so every case lands in the first n rows over the seeds."""
    x = tie_rows(dtype, rows=12, seed=seed)
    return torch.roll(x, shifts=-(seed % 12), dims=0)[:n].contiguous()


def _shards(mesh, x, dtype):
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )


def _drafter(mesh, b):
    """A Qwen36MTP holding only what the drafter argmax tails read, for b rows."""
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    nd = mesh.get_num_devices()
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.device = m.mesh_device = mesh
    m.num_devices = nd
    m.tt_ccl = TT_CCL(mesh)
    m.args = SimpleNamespace(
        vocab_size=VOCAB, max_batch_size=b, ccl_topology=lambda: ModelArgs.ccl_topology(SimpleNamespace(num_devices=nd))
    )
    m._sharded_argmax, m._fused_argmax = False, None
    m._init_sharded_argmax(SimpleNamespace(_lmhead_vocab_sharded=True))
    assert m._sharded_argmax and m._fused_argmax is not None
    return m


@torch.no_grad()
@_parametrize
def test_drafter_argmax_matches_the_sharded_tail(mesh_device):
    from models.demos.blackhole.qwen36.tt.mtp import draft_argmax

    for b in (1, 2, 4, 8):
        m = _drafter(mesh_device, b)
        for seed in range(12):
            x = _rows(b, torch.float32, seed)
            s = _shards(mesh_device, x, ttnn.float32)
            old, new = m._argmax_sharded(s), draft_argmax(m, s)
            for a, c in ((old.shape, new.shape), (old.dtype, new.dtype), (old.layout, new.layout)):
                assert a == c, f"output contract differs: {a} vs {c}"
            got_old, got_new = _ids(old, b), _ids(new, b)
            want = torch.argmax(x, dim=-1)
            assert torch.equal(got_new, got_old), f"b={b} seed={seed}: fused {got_new.tolist()} != {got_old.tolist()}"
            assert torch.equal(got_new, want), f"b={b} seed={seed}: {got_new.tolist()} != torch {want.tolist()}"
            for t in (s, old, new):
                ttnn.deallocate(t)
        logger.info(f"drafter b={b}: fused == _argmax_sharded == torch over 12 seeds")


def _gathered_argmax(mesh, x):
    """The verify tail's argmax: ttnn.argmax over the untilized full row, replicated as after the all-gather."""
    full = ttnn.from_torch(
        x.reshape(1, 1, *x.shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    u = ttnn.untilize(full, use_multicore=True)
    ids = ttnn.argmax(u, dim=-1, keepdim=False)
    for t in (full, u):
        ttnn.deallocate(t)
    return ids


@torch.no_grad()
@_parametrize
def test_verify_argmax_matches_the_gathered_argmax(mesh_device):
    from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES, FusedShardArgmax

    fused = FusedShardArgmax(mesh_device, VOCAB, mesh_device.get_num_devices())
    for rows in (4, 8, 16, 32):
        for seed in range(12):
            x = _rows(min(rows, 12), torch.bfloat16, seed)
            x = torch.cat([x, tie_rows(torch.bfloat16, rows=12, seed=100 + seed)] * 3)[:rows].contiguous()
            s = _shards(mesh_device, x, ttnn.bfloat16)
            old, new = _gathered_argmax(mesh_device, x), fused(s, LANES)
            got_old, got_new = _ids(old, rows), _ids(new, 32)
            want = torch.argmax(x.float(), dim=-1)
            assert torch.equal(got_new[rows:], torch.zeros(32 - rows, dtype=torch.int64)), got_new.tolist()
            assert torch.equal(got_new[:rows], got_old), f"rows={rows}: fused {got_new[:rows].tolist()} != {got_old}"
            assert torch.equal(got_old, want), f"rows={rows}: {got_old.tolist()} != torch {want.tolist()}"
            for t in (s, old, new):
                ttnn.deallocate(t)
        logger.info(f"verify rows={rows}: fused == gathered ttnn.argmax == torch over 12 seeds")


def _replay_us(mesh, tid, n=200):
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / n * 1e6


@torch.no_grad()
@_parametrize
def test_fused_argmax_in_a_trace(mesh_device):
    from models.demos.blackhole.qwen36.tt.mtp import draft_argmax
    from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES, FusedShardArgmax

    drafter = _drafter(mesh_device, 8)
    verify = FusedShardArgmax(mesh_device, VOCAB, mesh_device.get_num_devices())

    def gathered(inp):
        g = ttnn.all_gather(inp, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        u = ttnn.untilize(g, use_multicore=True)
        ids = ttnn.argmax(u, dim=-1, keepdim=False)
        for t in (g, u):
            ttnn.deallocate(t)
        return ids

    cases = {  # name: (rows, torch dtype, ttnn dtype, replaced tail, fused tail)
        "drafter": (8, torch.float32, ttnn.float32, drafter._argmax_sharded, lambda s: draft_argmax(drafter, s)),
        "verify": (32, torch.bfloat16, ttnn.bfloat16, gathered, lambda s: verify(s, LANES)),
    }
    for name, (rows, tdt, ndt, replaced, fused) in cases.items():
        host = lambda x: ttnn.from_torch(
            x.reshape(1, 1, *x.shape),
            dtype=ndt,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )
        x0 = torch.cat([_rows(12, tdt, 7)] * 3)[:rows].contiguous()
        inp = ttnn.to_device(host(x0), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tids, outs = {}, {}
        for tag, fn in (("replaced", replaced), ("fused", fused)):
            ttnn.deallocate(fn(inp))  # compile outside the trace
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            outs[tag] = fn(inp)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            tids[tag] = tid
        for seed in (11, 12, 13):
            x = torch.cat([_rows(12, tdt, seed)] * 3)[:rows].contiguous()
            ttnn.copy_host_to_device_tensor(host(x), inp)
            got = {}
            for tag in ("replaced", "fused"):
                ttnn.execute_trace(mesh_device, tids[tag], cq_id=0, blocking=True)
                got[tag] = _ids(outs[tag], rows)
            want = torch.argmax(x.float(), dim=-1)
            assert torch.equal(got["fused"], got["replaced"]), f"{name}: {got['fused']} != {got['replaced']}"
            assert torch.equal(got["fused"], want), f"{name}: {got['fused']} != torch {want}"
        us = {tag: _replay_us(mesh_device, tids[tag]) for tag in tids}
        logger.info(
            f"[FUSED_ARGMAX] {name} rows={rows}: replaced tail {us['replaced']:.1f} us, fused {us['fused']:.1f} us"
        )
        for tag in tids:
            ttnn.release_trace(mesh_device, tids[tag])
        ttnn.deallocate(inp)


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("rows", [1, 8, 32])
def test_one_row_embedding_matches_the_tile_lookup(mesh_device, rows):
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP

    nd, vocab, dim = mesh_device.get_num_devices(), 4096, 5120
    g = torch.Generator().manual_seed(rows)
    table = torch.randn(1, 1, vocab, dim, generator=g).to(torch.bfloat16)
    weights = ttnn.from_torch(
        table,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    mtp = Qwen36MTP.__new__(Qwen36MTP)
    mtp.embd = SimpleNamespace(weights=weights)
    ids_host = torch.randint(0, vocab, (rows, 1), generator=g, dtype=torch.int32)
    ids_host[0, 0] = vocab - 1
    ids = ttnn.from_torch(
        ids_host, dtype=ttnn.uint32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    old = ttnn.embedding(ids, weights, layout=ttnn.TILE_LAYOUT)
    old = ttnn.reshape(old, (1, 1, old.shape[0] * old.shape[1], old.shape[-1]))
    new = mtp._embed_rows(ids)
    assert tuple(new.shape) == tuple(old.shape) == (1, 1, rows, dim // nd) and new.layout == ttnn.TILE_LAYOUT
    for d, (a, b) in enumerate(zip(ttnn.get_device_tensors(old), ttnn.get_device_tensors(new))):
        a, b = ttnn.to_torch(a), ttnn.to_torch(b)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), f"device {d}: rows differ"
        assert torch.equal(b[0, 0], table[0, 0, ids_host.reshape(-1).long(), d * dim // nd : (d + 1) * dim // nd])
    for t in (old, new, ids, weights):
        ttnn.deallocate(t)
