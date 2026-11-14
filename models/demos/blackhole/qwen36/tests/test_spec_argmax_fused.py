# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_SPEC_ARGMAX_FUSED (tt/spec_argmax_fused.py) against the shipped argmax tails, on device.

Every check is on token ids and must be EXACT: the fused path has to return, for every row, the id the flag-off tail
returns (and torch.argmax over the full, dtype-rounded vocab row: the lowest id among the maxima).

  * test_fused_argmax_verify   bf16, R in (4, 8, 16, 32) rows (the verify buckets B*T):
                               Qwen36Model.shard_argmax_max_rows + _dl_greedy_token_row_batched (flag off) vs
                               _vfy_argmax_fused (flag on), both built by the real _vfy_setup_sharded_argmax. Whole
                               [1,1,1,32] ids rows compared (lanes >= R = 0).
  * test_fused_argmax_drafter  fp32, b in (1, 2, 4, 8) rows: Qwen36MTP._argmax_sharded (flag off) vs draft_argmax with
                               the flag on (built by the real _init_sharded_argmax).
  * test_fused_argmax_trace    both tails captured in a trace and replayed on fresh inputs (the production form), and
                               the replay time of the flag-off tail vs the fused one.

Inputs carry hand-built rows that pin the tie-break and the ranking: equal maxima across shards and inside one shard, at
stage-1 core boundaries and at the shard edges, an all-equal row, all-negative rows, +0/-0 maxima, -inf entries, 1-ulp
pairs, plus quantized random rows (many exact ties at the max).

Run (8-chip ring, 1x8 mesh):
    MESH_DEVICE=P150x8 QWEN36_FABRIC_RING=1 pytest models/demos/blackhole/qwen36/tests/test_spec_argmax_fused.py -s
"""

import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn

from .test_factory import parametrize_mesh_tp

VOCAB = 248320


def _ids_per_device(t, n):
    """The first n uint32 ids of a replicated RM ids tensor; every device must hold the same ids."""
    per_dev = [ttnn.to_torch(d).reshape(-1)[:n].to(torch.int64) for d in ttnn.get_device_tensors(t)]
    for d in per_dev[1:]:
        assert torch.equal(d, per_dev[0]), f"devices disagree: {d.tolist()} vs {per_dev[0].tolist()}"
    return per_dev[0]


def _stage1_cols(fused, rows, fp32):
    """Columns (within a shard) where stage-1 core ranges meet, from the plan the fused op builds."""
    plan = fused._plan(rows, fused.shard, fused.shard, fp32)
    tpc = plan["tpc"]
    return [c * tpc * 32 for c in range(1, plan["n"])]


def _make_logits(rows, shard, nd, boundaries, dtype, seed):
    """[1,1,rows,VOCAB] float32 logits with the special rows of the module docstring (row r uses case r % 12)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, rows, VOCAB, generator=g, dtype=torch.float32)
    bnd = boundaries[len(boundaries) // 2] if boundaries else 288
    for r in range(rows):
        row = x[0, 0, r]
        case = r % 12
        if case == 0:  # equal max on shard 0 and shard 2 -> the shard-0 id
            row[100] = row[2 * shard + 500] = 20.0
        elif case == 1:  # equal max twice inside the last shard -> the lower id
            row[(nd - 1) * shard + 1000] = row[(nd - 1) * shard + 30000] = 21.0
        elif case == 2:  # all-equal row -> id 0
            row.fill_(1.0)
        elif case == 3:  # all negative, a tied max on two devices
            row.copy_(-row.abs() - 3.0)
            row[shard + 17] = row[5 * shard + 3] = -0.5
        elif case == 4:  # max exactly 0: -0.0 first (shard 1), +0.0 later (shard 1 and shard 4) -> the -0.0 id
            row.copy_(-row.abs() - 1.0)
            row[shard + 40] = -0.0
            row[shard + 41] = 0.0
            row[4 * shard + 9] = 0.0
        elif case == 5:  # max at the last vocab id, and -inf entries elsewhere
            row[VOCAB - 1] = 22.0
            row[7] = row[shard + 9] = float("-inf")
        elif case == 6:  # max at id 0, tied with a stage-1 core boundary column of shard 3
            row[0] = row[3 * shard + bnd] = 19.0
        elif case == 7:  # tie straddling a stage-1 core boundary inside one shard -> the column before it
            row[6 * shard + bnd - 1] = row[6 * shard + bnd] = 18.0
        elif case == 8:  # 1-ulp pair in the logits' own dtype across devices: the larger one wins
            hi = torch.tensor(30.0, dtype=dtype)
            lo = torch.nextafter(hi.float(), torch.tensor(0.0)).to(dtype) if dtype == torch.float32 else None
            if lo is None:  # bf16: the next bf16 below 30.0
                lo = (hi.view(torch.int16) - 1).view(torch.bfloat16)
            row[shard + 7] = float(lo)
            row[3 * shard + 9] = float(hi)
        elif case == 9:  # quantized row with many exact ties at the max (first one wins)
            row.copy_(torch.clamp(torch.round(row * 2) / 2, max=2.5))
        elif case == 10:  # shifted row: a large negative offset (the slow/negative ranking path for a while)
            row.add_(-50.0)
            row[4 * shard + 31039 % shard] = -40.0
        else:  # plain random row
            pass
    return x


def _torch_ids(x, dtype):
    return torch.argmax(x.to(dtype).float(), dim=-1).reshape(-1)


@torch.no_grad()
@parametrize_mesh_tp()
def test_fused_argmax_verify(mesh_device, monkeypatch, reset_seeds):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES
    from models.tt_transformers.tt.ccl import TT_CCL

    monkeypatch.setenv("QWEN36_SPEC_ARGMAX_FUSED", "1")
    nd = mesh_device.get_num_devices()
    shard = VOCAB // nd
    m = Qwen36Model.__new__(Qwen36Model)
    m.mesh_device = m.device = mesh_device
    m.num_devices = nd
    m.vocab_size = VOCAB
    m.tt_ccl = TT_CCL(mesh_device)
    m._lmhead_vocab_sharded = True
    m.rope = SimpleNamespace(rope_delta=0, max_seq_len=1 << 20)

    for rows in (4, 8, 16, 32):
        m._vfy_setup_sharded_argmax(rows)
        assert m._vfy_sharded and m._vfy_argmax_fused is not None
        bnd = _stage1_cols(m._vfy_argmax_fused, rows, False)
        for seed in (rows, 1000 + rows):
            x = _make_logits(rows, shard, nd, bnd, torch.bfloat16, seed)
            expected = _torch_ids(x, torch.bfloat16)
            s = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
            idx, val = m.shard_argmax_max_rows(s, rows)
            old = m._dl_greedy_token_row_batched(idx, val, m._vfy_dl)
            new = m._vfy_argmax_fused.ids(s, LANES)
            assert new is not None
            for a, b in ((old.shape, new.shape), (old.dtype, new.dtype), (old.layout, new.layout)):
                assert a == b, f"output contract differs: {a} vs {b}"
            assert old.memory_config() == new.memory_config(), f"{old.memory_config()} vs {new.memory_config()}"
            got_old, got_new = _ids_per_device(old, 32), _ids_per_device(new, 32)
            assert torch.equal(got_new[rows:], torch.zeros(32 - rows, dtype=torch.int64)), got_new.tolist()
            assert torch.equal(got_old, got_new), f"rows={rows}: fused {got_new.tolist()} != shipped {got_old.tolist()}"
            assert torch.equal(got_new[:rows], expected), f"rows={rows}: {got_new[:rows].tolist()} != torch {expected}"
            for t in (s, idx, val, old, new):
                ttnn.deallocate(t)
        logger.info(f"verify rows={rows}: fused == shipped == torch ({expected[:8].tolist()} ...)")
        m._vfy_dl.release()


@torch.no_grad()
@parametrize_mesh_tp()
def test_fused_argmax_drafter(mesh_device, monkeypatch, reset_seeds):
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP, draft_argmax
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4,8")
    monkeypatch.setenv("QWEN36_SPEC_ARGMAX_FUSED", "1")
    nd = mesh_device.get_num_devices()
    shard = VOCAB // nd
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.device = m.mesh_device = mesh_device
    m.num_devices = nd
    m.tt_ccl = TT_CCL(mesh_device)
    m.args = SimpleNamespace(
        vocab_size=VOCAB,
        max_batch_size=8,
        ccl_topology=lambda: ModelArgs.ccl_topology(SimpleNamespace(num_devices=nd)),
    )
    m._init_sharded_argmax(SimpleNamespace(_lmhead_vocab_sharded=True))
    assert m._sharded_argmax and m._fused_argmax is not None

    for b in (1, 2, 4, 8):
        bnd = _stage1_cols(m._fused_argmax, b, True)
        for seed in (b, 2000 + b, 4000 + b):
            x = _make_logits(8, shard, nd, bnd, torch.float32, seed)
            if seed >= 2000:  # rotate the special rows so b < 8 sees every case
                x = torch.roll(x, shifts=-(seed % 8), dims=2)
            x = x[:, :, :b].contiguous()
            expected = _torch_ids(x, torch.float32)
            s = ttnn.from_torch(
                x,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
            )
            old = m._argmax_sharded(s)
            new = draft_argmax(m, s)
            for a, c in ((old.shape, new.shape), (old.dtype, new.dtype), (old.layout, new.layout)):
                assert a == c, f"output contract differs: {a} vs {c}"
            assert old.memory_config() == new.memory_config(), f"{old.memory_config()} vs {new.memory_config()}"
            got_old, got_new = _ids_per_device(old, b), _ids_per_device(new, b)
            assert torch.equal(got_old, got_new), f"b={b}: fused {got_new.tolist()} != shipped {got_old.tolist()}"
            assert torch.equal(got_new, expected), f"b={b}: {got_new.tolist()} != torch {expected.tolist()}"
            for t in (s, old, new):
                ttnn.deallocate(t)
        logger.info(f"drafter b={b}: fused == shipped == torch ({expected.tolist()})")
    for sh_off, sel in m._argmax_consts.values():
        ttnn.deallocate(sh_off)
        ttnn.deallocate(sel)


def _replay_ms(mesh_device, tid, n):
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) * 1e3 / n


@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=32 * 1024 * 1024)
def test_fused_argmax_trace(mesh_device, monkeypatch, reset_seeds):
    """Both fused tails inside a captured trace, replayed on fresh inputs, == torch; and the replay time of each tail
    (flag off vs fused, same input, nothing else in the trace)."""
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP, draft_argmax
    from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4,8")
    monkeypatch.setenv("QWEN36_SPEC_ARGMAX_FUSED", "1")
    nd = mesh_device.get_num_devices()
    shard = VOCAB // nd
    tt_ccl = TT_CCL(mesh_device)

    vm = Qwen36Model.__new__(Qwen36Model)
    vm.mesh_device = vm.device = mesh_device
    vm.num_devices, vm.vocab_size, vm.tt_ccl, vm._lmhead_vocab_sharded = nd, VOCAB, tt_ccl, True
    vm.rope = SimpleNamespace(rope_delta=0, max_seq_len=1 << 20)
    vm._vfy_setup_sharded_argmax(32)

    dm = Qwen36MTP.__new__(Qwen36MTP)
    dm.device = dm.mesh_device = mesh_device
    dm.num_devices, dm.tt_ccl = nd, tt_ccl
    dm.args = SimpleNamespace(
        vocab_size=VOCAB, max_batch_size=8, ccl_topology=lambda: ModelArgs.ccl_topology(SimpleNamespace(num_devices=nd))
    )
    dm._init_sharded_argmax(SimpleNamespace(_lmhead_vocab_sharded=True))

    def host(x, dtype):
        return ttnn.from_torch(
            x, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        )

    cases = {  # name: (rows, torch dtype, ttnn dtype, shipped tail, fused tail, n ids)
        "verify": (
            32,
            torch.bfloat16,
            ttnn.bfloat16,
            lambda s: vm._dl_greedy_token_row_batched(*vm.shard_argmax_max_rows(s, 32), vm._vfy_dl),
            lambda s: vm._vfy_argmax_fused.ids(s, LANES),
            32,
        ),
        "drafter": (8, torch.float32, ttnn.float32, dm._argmax_sharded, lambda s: draft_argmax(dm, s), 8),
    }
    for name, (rows, tdt, ndt, shipped, fused, n) in cases.items():
        bnd = _stage1_cols(vm._vfy_argmax_fused if name == "verify" else dm._fused_argmax, rows, ndt == ttnn.float32)
        x0 = _make_logits(rows, shard, nd, bnd, tdt, 7)
        inp = ttnn.to_device(host(x0, ndt), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tids, outs = {}, {}
        for tag, fn in (("shipped", shipped), ("fused", fused)):
            ttnn.deallocate(fn(inp))  # compile outside the trace
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            outs[tag] = fn(inp)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            tids[tag] = tid
        for seed in (11, 12, 13):
            x = _make_logits(rows, shard, nd, bnd, tdt, seed)
            x = torch.roll(x, shifts=-(seed % 12), dims=2)
            ttnn.copy_host_to_device_tensor(host(x, ndt), inp)
            expected = _torch_ids(x, tdt)
            got = {}
            for tag in ("shipped", "fused"):
                ttnn.execute_trace(mesh_device, tids[tag], cq_id=0, blocking=True)
                got[tag] = _ids_per_device(outs[tag], n)
            assert torch.equal(got["fused"], got["shipped"]), f"{name} replay: {got['fused']} != {got['shipped']}"
            assert torch.equal(got["fused"][:rows], expected), f"{name} replay: {got['fused'][:rows]} != {expected}"
        ms = {tag: _replay_ms(mesh_device, tids[tag], 200) for tag in ("shipped", "fused")}
        logger.info(
            f"[ARGMAX_FUSED_TIMING] {name} rows={rows}: shipped tail {ms['shipped'] * 1e3:.1f} us/replay, "
            f"fused {ms['fused'] * 1e3:.1f} us/replay"
        )
        for tag in ("shipped", "fused"):
            ttnn.release_trace(mesh_device, tids[tag])
        ttnn.deallocate(inp)
    for sh_off, sel in dm._argmax_consts.values():
        ttnn.deallocate(sh_off)
        ttnn.deallocate(sel)
    vm._vfy_dl.release()
