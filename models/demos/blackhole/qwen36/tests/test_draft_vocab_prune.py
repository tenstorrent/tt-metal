# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_DRAFT_VOCAB_TOPK (tt/draft_vocab.py) on device.

  * test_pick_remap        the fused per-shard argmax with an id table (FusedShardArgmax(remap=...)) over a K-wide
                           pruned logit row sharded over TP: every row's id is table[torch.argmax(row)] EXACTLY (ties ->
                           lowest pruned index = lowest token id), fp32 and bf16, ROWS and LANES contracts, b = 1..32.
  * test_pruned_head_ids   the drafter path end to end on a synthetic head (real vocab, small dim): Qwen36MTP.
                           _init_vocab_prune gathers the K rows (Qwen36Model._build_lm_head_weight(rows=...)), the
                           drafter's fp32 head linear + draft_argmax return the planted token when it is kept and the
                           planted in-set runner-up when it is not; the full head (flag off) returns the planted token.
  * test_pruned_head_timing  replay time in a trace of the drafter's head matmul + argmax tail, full head (31040 ids per
                           device) vs pruned K (zero weights at the real [5120, K/TP] shapes: timing only).

Run (8-chip ring, 1x8 mesh):
    MESH_DEVICE=P150x8 QWEN36_FABRIC_RING=1 pytest models/demos/blackhole/qwen36/tests/test_draft_vocab_prune.py -s
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
    per_dev = [ttnn.to_torch(d).reshape(-1)[:n].to(torch.int64) for d in ttnn.get_device_tensors(t)]
    for d in per_dev[1:]:
        assert torch.equal(d, per_dev[0]), f"devices disagree: {d.tolist()} vs {per_dev[0].tolist()}"
    return per_dev[0]


def _table(mesh_device, ids):
    k = int(ids.shape[0])
    return ttnn.from_torch(
        ids.to(torch.int32).reshape(k // 16, 16).contiguous(),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _pruned_logits(rows, k, nd, seed):
    """[1,1,rows,k] fp32 with tie rows: equal maxima across devices / inside one device / all-equal / max at index 0
    and at k-1, plus random rows."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, rows, k, generator=g)
    sh = k // nd
    for r in range(rows):
        row, case = x[0, 0, r], r % 6
        if case == 0:
            row[sh + 5] = row[3 * sh + 7] = 20.0  # tie across devices 1 and 3 -> device 1's
        elif case == 1:
            row[(nd - 1) * sh + 3] = row[(nd - 1) * sh + sh - 1] = 21.0  # tie inside the last device -> lower column
        elif case == 2:
            row.fill_(1.0)  # all equal -> index 0
        elif case == 3:
            row[k - 1] = 30.0  # last index
        elif case == 4:
            row[0] = 30.0  # first index
    return x


@pytest.mark.timeout(900)
@torch.no_grad()
@parametrize_mesh_tp()
def test_pick_remap(mesh_device, reset_seeds):
    from models.demos.blackhole.qwen36.tt import draft_vocab as dv
    from models.demos.blackhole.qwen36.tt.spec_argmax_fused import LANES, ROWS, FusedShardArgmax

    nd = mesh_device.get_num_devices()
    for k in (32768, 98304):
        ids = dv.select_draft_vocab(k, VOCAB)
        table = _table(mesh_device, ids)
        fused = FusedShardArgmax(mesh_device, k, nd, remap=table)
        for dtype, tdt in ((ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)):
            for rows, mode in ((1, ROWS), (3, ROWS), (8, ROWS), (32, LANES), (16, LANES)):
                x = _pruned_logits(rows, k, nd, seed=rows * 7 + k)
                expected = ids[torch.argmax(x.to(tdt).float(), dim=-1).reshape(-1)]
                s = ttnn.from_torch(
                    x,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
                )
                out = fused.ids(s, mode)
                assert out is not None
                got = _ids_per_device(out, rows if mode == ROWS else 32)
                assert torch.equal(got[:rows], expected), f"k={k} {dtype} rows={rows}: {got[:rows]} != {expected}"
                if mode == LANES:
                    assert torch.equal(got[rows:], torch.zeros(32 - rows, dtype=torch.int64))
                ttnn.deallocate(s)
                ttnn.deallocate(out)
        logger.info(f"pick remap k={k}: ids == table[torch.argmax] (fp32 + bf16, ROWS + LANES)")
        ttnn.deallocate(table)


def _mtp_and_parent(mesh_device, vocab, max_batch=8):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model
    from models.demos.blackhole.qwen36.tt.mtp import Qwen36MTP
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    nd = mesh_device.get_num_devices()
    parent = Qwen36Model.__new__(Qwen36Model)
    parent.mesh_device = parent.device = mesh_device
    parent.num_devices = nd
    parent._lmhead_vocab_sharded = True
    m = Qwen36MTP.__new__(Qwen36MTP)
    m.device = m.mesh_device = mesh_device
    m.num_devices = nd
    m.tt_ccl = TT_CCL(mesh_device)
    m.args = SimpleNamespace(
        vocab_size=vocab,
        max_batch_size=max_batch,
        ccl_topology=lambda: ModelArgs.ccl_topology(SimpleNamespace(num_devices=nd)),
    )
    m._init_sharded_argmax(SimpleNamespace(_lmhead_vocab_sharded=True))
    return m, parent


@pytest.mark.timeout(900)
@torch.no_grad()
@parametrize_mesh_tp()
def test_pruned_head_ids(mesh_device, monkeypatch, reset_seeds):
    from models.demos.blackhole.qwen36.tt import draft_vocab as dv
    from models.demos.blackhole.qwen36.tt.mtp import draft_argmax

    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4,8")
    monkeypatch.setenv("QWEN36_SPEC_ARGMAX_FUSED", "1")
    dim, b, k = 512, 8, 65536
    m, parent = _mtp_and_parent(mesh_device, VOCAB)
    assert m._sharded_argmax and m._fused_argmax is not None
    kept = set(dv.select_draft_vocab(k, VOCAB).tolist())
    g = torch.Generator().manual_seed(5)
    w = (torch.randn(VOCAB, dim, generator=g) * 0.02).to(torch.bfloat16)
    x = torch.randn(b, dim, generator=g)
    x = x / x.norm(dim=-1, keepdim=True)
    # Row r: a planted winner t (logit ~0.6, the random ones are ~N(0, 0.02^2)); odd rows plant an id outside the kept
    # set and an in-set runner-up s (logit ~0.3), so the pruned head must draft s there and t on the even rows.
    out_ids = [i for i in range(VOCAB) if i not in kept]
    in_ids = sorted(kept)
    want_full, want_pruned = [], []
    for r in range(b):
        t = in_ids[(r * 7919) % len(in_ids)] if r % 2 == 0 else out_ids[(r * 104729) % len(out_ids)]
        w[t] = (0.6 * x[r]).to(torch.bfloat16)
        want_full.append(t)
        if r % 2:
            s = in_ids[(r * 15485863) % len(in_ids)]
            w[s] = (0.3 * x[r]).to(torch.bfloat16)
            want_pruned.append(s)
        else:
            want_pruned.append(t)
    planted = want_full + [s for r, s in enumerate(want_pruned) if r % 2]
    assert len(set(planted)) == len(planted), f"planted ids collide: {planted}"
    m.draft_lm_head_weight = parent._build_lm_head_weight(w, None, ttnn.bfloat8_b)  # the full head (flag off)
    m._init_vocab_prune(parent, {"output.weight": w}, None, k)
    vp = m._vocab_prune
    assert vp is not None and vp.k == k and vp.shard == k // m.num_devices
    xin = ttnn.from_torch(
        x.reshape(1, 1, b, dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    got = {}
    for name, weight in (("full", m.draft_lm_head_weight), ("pruned", vp.weight)):
        logits = ttnn.linear(xin, weight, dtype=ttnn.float32, compute_kernel_config=m._fp32_cfg)
        ids = draft_argmax(m, logits)  # the drafter's pick: routes the K/TP-wide shard through the id remap
        got[name] = _ids_per_device(ids, b).tolist()
        for t in (logits, ids):
            ttnn.deallocate(t)
    assert got["full"] == want_full, f"full head {got['full']} != planted {want_full}"
    assert got["pruned"] == want_pruned, f"pruned head {got['pruned']} != {want_pruned}"
    assert all(i in kept for i in got["pruned"])
    logger.info(f"pruned head k={k}: full {got['full']} / pruned {got['pruned']} as planted")
    vp.release()
    for t in (xin, m.draft_lm_head_weight, *(c for pair in m._argmax_consts.values() for c in pair)):
        ttnn.deallocate(t)


def _replay_us(mesh_device, tid, n):
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    return (time.perf_counter() - t0) * 1e6 / n


@pytest.mark.timeout(900)
@torch.no_grad()
@parametrize_mesh_tp(trace_region_size=32 * 1024 * 1024)
def test_pruned_head_timing(mesh_device, monkeypatch, reset_seeds):
    """The drafter step's head + argmax tail in a trace (b = 8 rows, dim 5120, fp32 logits, HiFi2 + fp32 acc as the
    drafter runs it), full head vs pruned K. Zero weights: matmul and argmax time do not depend on the values."""
    from models.demos.blackhole.qwen36.tt import draft_vocab as dv
    from models.demos.blackhole.qwen36.tt.mtp import draft_argmax

    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4,8")
    monkeypatch.setenv("QWEN36_SPEC_ARGMAX_FUSED", "1")
    dim, b = 5120, 8
    m, _ = _mtp_and_parent(mesh_device, VOCAB)
    nd = m.num_devices
    xin = ttnn.from_torch(
        torch.randn(1, 1, b, dim),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def zeros_w(n):
        return ttnn.from_torch(
            torch.zeros(dim, n, dtype=torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    results = {}
    for k in (0, 32768, 65536, 98304, 131072):
        if k == 0:
            w, m._vocab_prune = zeros_w(VOCAB // nd), None
        else:
            ids = dv.select_draft_vocab(k, VOCAB)
            w = zeros_w(k // nd)
            m._vocab_prune = dv.DraftVocabPrune(mesh_device, ids, w, nd)

        def step():
            lg = ttnn.linear(xin, w, dtype=ttnn.float32, compute_kernel_config=m._fp32_cfg)
            out = draft_argmax(m, lg)
            ttnn.deallocate(lg)
            return out

        def head():
            return ttnn.linear(xin, w, dtype=ttnn.float32, compute_kernel_config=m._fp32_cfg)

        us = {}
        for tag, fn in (("head+argmax", step), ("head", head)):
            ttnn.deallocate(fn())  # compile outside the trace
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            out = fn()
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            us[tag] = _replay_us(mesh_device, tid, 200)
            ttnn.release_trace(mesh_device, tid)
            ttnn.deallocate(out)
        results[k] = us
        logger.info(
            f"[DRAFT_VOCAB_TIMING] K={k or VOCAB} ({(k or VOCAB) // nd} per device): head {us['head']:.1f} us, "
            f"head+argmax {us['head+argmax']:.1f} us per replay"
        )
        if m._vocab_prune is not None:
            m._vocab_prune.release()
            m._vocab_prune = None
        else:
            ttnn.deallocate(w)
    ttnn.deallocate(xin)
    for t in (c for pair in m._argmax_consts.values() for c in pair):
        ttnn.deallocate(t)
    base = results[0]["head+argmax"]
    for k in (32768, 65536, 98304, 131072):
        logger.info(
            f"[DRAFT_VOCAB_TIMING] K={k}: saves {base - results[k]['head+argmax']:.1f} us per draft step "
            f"({3 * (base - results[k]['head+argmax']) / 1000:.3f} ms per K=3 iteration)"
        )
        assert results[k]["head+argmax"] < base
