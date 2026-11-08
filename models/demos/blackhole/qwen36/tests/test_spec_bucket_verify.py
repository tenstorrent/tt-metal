# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify row buckets (QWEN36_SPEC_BUCKETS) on the shared 8-wide state: the b-row verify trace (b = 1, 2, 4) of rows 0..b-1
against the 8-row verify trace of the same rows, on the same seeded state, through the real model (first N layers of the
checkpoint: GDN + full attention + the LM head tail).

Per b: build the model at B = 8, T = 4; prepare + capture the 8-row trace, then the b-row bucket (V8 first, then Vb: the
capture order of the plan); seed random GDN state; replay (1) the bucket, (2) the 8-row trace, (3) the bucket again, each from
the same state (``seed_spec_state()`` and the window parity reset between the runs; the ring is rewritten in place).
Compared bit for bit (rows 0..b-1): ids, full logits, top-k values / ids, the drafter-feed rows (the in-trace hand-off into
the shared 8-row buffers), the GDN ring blocks and window rows, and the attention KV the rows wrote. Rows b..7 of the shared
state are untouched by the bucket run; the hand-off leaves rows >= b*T of the shared rows buffer as they were.

Pinned (QWEN36_VERIFY_SDPA_MAX_CORES=13, QWEN36_MTP_SDPA_MAX_CORES=13): the SDPA core split no longer depends on the row count,
so the traces are bit-identical. Unpinned (production split): max |logit diff| and argmax agreement are logged; mismatches must be
near-ties only.

One node id per device run:
  pytest models/demos/blackhole/qwen36/tests/test_spec_bucket_verify.py::test_trace_rows_bit_identical[2] -sv
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.demo.text_demo import BLOCK_SIZE
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _marks
from models.demos.blackhole.qwen36.tests.test_spec_lossless import NEAR_TIE_GAP, _top2_gap
from models.demos.blackhole.qwen36.tests.test_spec_verify_specstep import _seed
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

B = 8
T = 4
NB = 4  # page-table width in blocks (256 tokens)
NUM_BLOCKS = 1 + B * NB
N_LAYERS = int(os.environ.get("QWEN36_BUCKET_TEST_N_LAYERS", "4"))  # layers 0-2 GDN, layer 3 full attention


def _bits(t):
    t = t.contiguous()
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    return t


def _same(a, b):
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _cat(mesh, t, dim=0):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=dim))


def _rep0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


_SHARED_OUTS = ("_vfy_rows_out", "_vfy_ids_out", "_vfy_topk_vals_out", "_vfy_topk_ids_out")


def _build(mesh, b):
    mesh.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=B, max_seq_len=NB * BLOCK_SIZE, n_layers=N_LAYERS)
    model.set_gdn_fused_decode(True)
    model.allocate_kv_caches(
        [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim], ttnn.bfloat16, batch_size=B
    )
    gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]
    assert gdn and any(layer.is_full_attention for layer in model.layers), "need GDN and full attention layers"
    # random durable GDN state per user (before the spec buffers exist: reset_state frees them)
    g0 = gdn[0]
    torch.manual_seed(2024)
    for g in gdn:
        rec0 = [0.1 * torch.randn(1, g.Nv, g.Dk, g.Dv) for _ in range(B)]
        hist0 = [torch.randn(1, 3, g.qkv_dim_tp, dtype=torch.bfloat16) for _ in range(B)]
        _seed(mesh, g, rec0, hist0)
    pt = torch.stack([torch.arange(1 + u * NB, 1 + (u + 1) * NB, dtype=torch.int32) for u in range(B)])
    model.prepare_verify_trace(pt, T, warm_positions=[0] * B)
    model.prepare_verify_buckets((b,))
    model.record_verify_trace()
    shared = {n: getattr(model, n).buffer_address() for n in _SHARED_OUTS}
    model.record_verify_buckets((b,))
    # the hand-off must write the shared outputs in place: an op that re-binds the tensor (slice_write into a TILE dst) leaves
    # the V8 trace writing the old buffer, which later captures then reuse
    moved = {
        n: (hex(a), hex(getattr(model, n).buffer_address()))
        for n, a in shared.items()
        if getattr(model, n).buffer_address() != a
    }
    assert not moved, f"the bucket capture moved shared B-row outputs: {moved}"
    assert model._vfy_bk[b].trace_id is not None
    logger.info(f"[bucket_verify] trace / DRAM memory after each capture: {getattr(model, '_vfy_mem_log', None)}")
    return model, gdn, pt, g0._spec_win_par


def _snapshot(mesh, model, gdn, pt):
    """Host copies of everything a verify writes: ring [nd,T,B,Nv,Dk,Dv] and window pair [nd,B,L,C] per GDN layer, KV blocks of
    every user [nd, B, NB, ...] per attention layer."""
    nd = mesh.get_num_devices()
    ring = [_cat(mesh, g._spec_ring).reshape(nd, T, B, g.Nv, g.Dk, g.Dv) for g in gdn]
    win = [[_cat(mesh, w).reshape(nd, B, w.shape[1], w.shape[2]) for w in g._verify_win_pair] for g in gdn]
    kv = []
    for k, v in model._paged_kv_caches:
        for c in (k, v):
            full = _cat(mesh, c)
            full = full.reshape(nd, NUM_BLOCKS, *full.shape[1:])
            kv.append(full[:, pt.reshape(-1).long()].reshape(nd, B, NB, *full.shape[2:]))
    return dict(ring=ring, win=win, kv=kv)


def _run(mesh, model, gdn, pt, par0, tokens, positions, bucket):
    """Replay one verify (``bucket`` None = the 8-row trace) from the freshly seeded state; return outputs + state."""
    for g in gdn:
        g.seed_spec_state()
        g._spec_win_par = par0
    ttnn.synchronize_device(mesh)
    pre = _snapshot(mesh, model, gdn, pt)
    h = model.verify_stage_and_enqueue(
        tokens, positions, [0] * B, read_logits=True, page_tables=pt, execute_topk=True, bucket=bucket
    )
    ids, feed, lt = model.verify_sync_and_read(h)
    out = dict(
        ids=torch.tensor(ids),
        logits=lt,
        topk_vals=_rep0(model._vfy_topk_vals_out).reshape(32, -1).clone(),
        topk_ids=_rep0(model._vfy_topk_ids_out).reshape(32, -1).clone(),
        rows=_cat(mesh, model._vfy_rows_out, dim=3).reshape(32, -1).clone(),
        pre=pre,
        post=_snapshot(mesh, model, gdn, pt),
    )
    assert all(g._spec_win_par == par0 ^ 1 for g in gdn), "the replay flips the window parity"
    return out


def _rows_diff(x, y, ndev=8):
    """Where two [R, D] feed-row blocks differ: differing rows, the column span per device shard, and how many values of each side
    are subnormal / non-finite (stale or foreign memory shows up as tiny non-zero values: a bucket hand-off bug does not).
    """
    if x.shape != y.shape:
        return f"shapes {tuple(x.shape)} vs {tuple(y.shape)}"
    ne = x.contiguous().view(torch.int16) != y.contiguous().view(torch.int16)
    rows = torch.nonzero(ne.any(dim=1)).flatten().tolist()
    shard = x.shape[1] // ndev
    per_dev = [int(ne[:, d * shard : (d + 1) * shard].sum()) for d in range(ndev)]

    def odd(t):
        f = t.float()
        return int(((f != 0) & (f.abs() < 1e-30)).sum() + (~torch.isfinite(f)).sum())

    return (
        f"rows {rows}, differing values per device shard {per_dev}, subnormal/non-finite values a={odd(x)} b={odd(y)}"
    )


def _compare_rows(tag, a, bk, b, exact_msg=True):
    """Rows 0..b-1 of every state element of run ``a`` equal run ``bk`` (bit for bit)."""
    R = b * T
    assert _same(a["ids"][:R], bk["ids"][:R]), f"{tag}: ids differ {a['ids'][:R].tolist()} vs {bk['ids'][:R].tolist()}"
    assert _same(a["logits"][:R], bk["logits"][:R]), f"{tag}: logits differ"
    assert _same(a["topk_vals"][:R], bk["topk_vals"][:R]), f"{tag}: topk values differ"
    assert _same(a["topk_ids"][:R], bk["topk_ids"][:R]), f"{tag}: topk ids differ"
    assert _same(
        a["rows"][:R], bk["rows"][:R]
    ), f"{tag}: drafter-feed rows differ: {_rows_diff(a['rows'][:R], bk['rows'][:R])}"
    for i, (x, y) in enumerate(zip(a["post"]["ring"], bk["post"]["ring"])):
        assert _same(x[:, :, :b], y[:, :, :b]), f"{tag}: GDN layer {i} ring blocks of users < {b} differ"
    for i, (x, y) in enumerate(zip(a["post"]["win"], bk["post"]["win"])):
        for half in (0, 1):
            assert _same(x[half][:, :b], y[half][:, :b]), f"{tag}: GDN layer {i} window half {half} users < {b} differ"
    for i, (x, y) in enumerate(zip(a["post"]["kv"], bk["post"]["kv"])):
        assert _same(x[:, :b], y[:, :b]), f"{tag}: attention KV (cache {i}) of users < {b} differs"


def _tokens_positions():
    g = torch.Generator().manual_seed(77)
    tokens = torch.randint(0, 5000, (B, T), generator=g).tolist()
    positions = [20 + 9 * u for u in range(B)]
    return tokens, positions


@_marks
@pytest.mark.parametrize("b", [1, 2, 4])
def test_trace_rows_bit_identical(mesh_device, monkeypatch, b):
    monkeypatch.setenv("QWEN36_VERIFY_SDPA_MAX_CORES", "13")
    monkeypatch.setenv("QWEN36_MTP_SDPA_MAX_CORES", "13")
    mesh = mesh_device
    model, gdn, pt, par0 = _build(mesh, b)
    tokens, positions = _tokens_positions()
    R = b * T
    r_b = _run(mesh, model, gdn, pt, par0, tokens, positions, b)
    r_8 = _run(mesh, model, gdn, pt, par0, tokens, positions, None)
    r_b2 = _run(mesh, model, gdn, pt, par0, tokens, positions, b)
    ids_b = r_b["ids"]
    assert torch.isfinite(r_8["logits"][:R].float()).all()
    # 1. the b-row trace == the 8-row trace on rows 0..b-1
    _compare_rows(f"b={b} vs 8", r_b, r_8, b)
    # 2. replaying the bucket after the 8-row trace (interchangeable traces, same state) gives the same rows again
    _compare_rows(f"b={b} again vs 8", r_b2, r_8, b)
    # 3. rows b..7 of the shared state are untouched by the b-row run
    for i, (x, y) in enumerate(zip(r_b["pre"]["ring"], r_b["post"]["ring"])):
        assert _same(x[:, :, b:], y[:, :, b:]), f"GDN layer {i}: ring blocks of users >= {b} changed"
    for i, (x, y) in enumerate(zip(r_b["pre"]["win"], r_b["post"]["win"])):
        for half in (0, 1):
            assert _same(x[half][:, b:], y[half][:, b:]), f"GDN layer {i}: window half {half} users >= {b} changed"
    for i, (x, y) in enumerate(zip(r_b["pre"]["kv"], r_b["post"]["kv"])):
        assert _same(x[:, b:], y[:, b:]), f"attention KV (cache {i}) of users >= {b} changed"
    # 4. the b-row run really wrote its rows (not a no-op)
    assert not _same(r_b["pre"]["ring"][0][:, 1:, :b], r_b["post"]["ring"][0][:, 1:, :b]), "ring not written"
    assert not _same(r_b["pre"]["kv"][0][:, :b], r_b["post"]["kv"][0][:, :b]), "KV not written"
    # 5. the hand-off: ids lanes >= b*T are the argmax pad (0); the topk pad rows are -1e30; the drafter-feed rows >= b*T keep
    # what the 8-row run left there
    assert bool((ids_b[R:] == 0).all()), f"ids lanes >= {R} are not zero: {ids_b[R:].tolist()}"
    assert bool((r_b["topk_vals"][R:].float() < -1e29).all()), "topk pad rows are not -1e30"
    assert _same(r_b2["rows"][R:], r_8["rows"][R:]), "the hand-off changed drafter-feed rows >= b*T"
    logger.info(f"[bucket_verify] b={b}: rows 0..{R - 1} bit-identical to the 8-row trace; memory {model._vfy_mem_log}")
    model.release_verify_trace()
    assert not model._vfy_bk
    model.free_kv_caches()


@_marks
@pytest.mark.parametrize("b", [1, 2, 4])
def test_trace_rows_unpinned(mesh_device, monkeypatch, b):
    """Production SDPA split (no pins): the core split differs per row count, so the online softmax order differs. Logs the max
    |logit diff| and the argmax agreement of rows 0..b-1; a mismatch must be a near-tie."""
    for k in ("QWEN36_VERIFY_SDPA_MAX_CORES", "QWEN36_MTP_SDPA_MAX_CORES"):
        monkeypatch.delenv(k, raising=False)
    mesh = mesh_device
    model, gdn, pt, par0 = _build(mesh, b)
    tokens, positions = _tokens_positions()
    R = b * T
    r_b = _run(mesh, model, gdn, pt, par0, tokens, positions, b)
    r_8 = _run(mesh, model, gdn, pt, par0, tokens, positions, None)
    lb, l8 = r_b["logits"][:R].float(), r_8["logits"][:R].float()
    assert torch.isfinite(lb).all() and torch.isfinite(l8).all()
    diff = (lb - l8).abs().max().item()
    bad = [i for i in range(R) if int(r_b["ids"][i]) != int(r_8["ids"][i])]
    gaps = {i: _top2_gap(l8[i]) for i in bad}
    same_ring = all(_same(x[:, :, :b], y[:, :, :b]) for x, y in zip(r_b["post"]["ring"], r_8["post"]["ring"]))
    logger.info(
        f"[bucket_verify] UNPINNED b={b}: max|logit diff|={diff:.5f} over {R} rows, argmax ids equal on {R - len(bad)}/{R} "
        f"rows (mismatch rows {bad}, top-2 gaps {gaps}), ring bit-identical: {same_ring}, logits bit-identical: "
        f"{_same(lb, l8)}"
    )
    assert all(g < NEAR_TIE_GAP for g in gaps.values()), f"a mismatch that is not a near-tie: {gaps}"
    model.release_verify_trace()
    model.free_kv_caches()


# ---------------------------------------------------------------------------------------------------------------------------
# Drafter row buckets: the b-row drafter body on rows 0..b-1 against the 8-row body on the same inputs.
# ---------------------------------------------------------------------------------------------------------------------------
def _drafter_engine(mesh, b):
    """Model + spec engine at B = 8, K = 3 with the drafter bucket ``b`` warmed (prepare only, no capture: the bodies run
    eagerly)."""
    from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
    from models.demos.blackhole.qwen36.tt.spec_engine import get_spec_engine

    mesh.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=B, max_seq_len=NB * BLOCK_SIZE, n_layers=N_LAYERS)
    assert model.mtp is not None, "MTP head not built"
    model.set_gdn_fused_decode(True)
    model.allocate_kv_caches(
        [NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim], ttnn.bfloat16, batch_size=B
    )
    pt = torch.stack([torch.arange(1 + u * NB, 1 + (u + 1) * NB, dtype=torch.int32) for u in range(B)])
    eng = get_spec_engine(model, B, tc.K)
    assert eng.T == T
    eng.prepare(pt)
    assert eng._dr_bks == (b,), f"the drafter bucket {b} was not warmed: {eng._dr_bks}"
    assert eng._hp_bufs and set(eng._hp_bufs) == {b}
    return model, eng, pt


def _zero_mtp_kv(model):
    for c in model._mtp_kv_cache:
        ttnn.copy(ttnn.zeros_like(c), c)
    ttnn.synchronize_device(model.device)


def _mtp_kv_users(mesh, model, pt):
    """Host copies of the MTP KV blocks of every user: per cache [nd, B, NB, ...]."""
    nd = mesh.get_num_devices()
    out = []
    for c in model._mtp_kv_cache:
        full = _cat(mesh, c)
        full = full.reshape(nd, c.shape[0], *full.shape[1:])
        out.append(full[:, pt.reshape(-1).long()].reshape(nd, B, NB, *full.shape[2:]))
    return out


def _drafter_inputs(eng, pt, feed_kind):
    """Stage the drafter's persistent inputs (pack, reseed table, MTP page table) with plausible values; return the feed."""
    from models.demos.blackhole.qwen36.tt.spec_engine import reseed_rows_vec

    K = eng.K
    g = torch.Generator().manual_seed(11)
    m = torch.tensor([3, 2, 4, 1, 0, 2, 3, 4], dtype=torch.int32)[:B]
    p = torch.tensor([60 + 7 * u for u in range(B)], dtype=torch.int32)
    ctok = torch.randint(1, 5000, (B, T), generator=g, dtype=torch.int32)
    cpos = (p[:, None] - m[:, None] + 1 + torch.arange(T)[None, :]).to(torch.int32)
    pending = torch.randint(1, 5000, (B,), generator=g, dtype=torch.int32)
    pack, _, _, real = eng._dr_pack_host(ctok, cpos, m.tolist(), pending.tolist(), p.tolist(), m.tolist())
    h = ttnn.from_torch(
        pack,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=None,
        mesh_mapper=ttnn.ReplicateTensorToMesh(eng.mesh),
    )
    ttnn.copy_host_to_device_tensor(h, eng._dr_pack)
    _, _, rpt, _ = reseed_rows_vec(ctok, cpos, m.tolist(), pt, T, eng.nb, eng._scratch_block)
    eng._stage(rpt, eng._dr_rs_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    eng._stage(pt, eng._mtp_pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    dim = eng._dim_frac
    x = (
        torch.randn(1, 1, B * T, dim, generator=g).to(torch.bfloat16)
        if feed_kind == "random"
        else torch.zeros(1, 1, B * T, dim)
    )
    z = eng._alloc_host_replicated(x.to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    feed = eng.model.spec_feed_rows(z)
    return feed


@_marks
@pytest.mark.parametrize("b", [1, 2, 4])
def test_drafter_rows(mesh_device, monkeypatch, b):
    """The b-row drafter body (anchor gather, batched reseed, K draft steps, in-place hand-off of the K ids) against the 8-row body
    on identical inputs (a random feed, then a zero feed), MTP SDPA pinned: the K draft ids and the MTP KV written for rows
    0..b-1 are equal; the ids of rows b..7 of the shared outputs keep their old value; the KV of users b..7 is not written.
    """
    monkeypatch.setenv("QWEN36_VERIFY_SDPA_MAX_CORES", "13")
    monkeypatch.setenv("QWEN36_MTP_SDPA_MAX_CORES", "13")
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", f"{b},8")
    mesh = mesh_device
    model, eng, pt = _drafter_engine(mesh, b)
    K = eng.K
    SENT = 777
    for kind in ("random", "zero"):
        feed = _drafter_inputs(eng, pt, kind)
        # b-row body: hands its ids over to the stand-ins (pre-filled with a sentinel)
        for t in eng._dr_standins:
            eng._stage(torch.full((1, 1, B, 1), SENT, dtype=torch.int32), t, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        addrs = [t.buffer_address() for t in eng._dr_standins]
        _zero_mtp_kv(model)
        outs_b = eng._drafter_body(feed, b, list(eng._dr_standins))
        ttnn.synchronize_device(mesh)
        assert [t.buffer_address() for t in outs_b] == addrs, "the hand-off moved the shared K ids outputs"
        ids_b = eng._ids_to_host(outs_b)  # [K][B]
        kv_b = _mtp_kv_users(mesh, model, pt)
        hp_b = _cat(mesh, eng._hp_bufs[b], dim=3)[..., :, :].clone()
        # 8-row body on the same inputs
        _zero_mtp_kv(model)
        outs_8 = eng._drafter_body(feed)
        ttnn.synchronize_device(mesh)
        ids_8 = eng._ids_to_host(outs_8)
        kv_8 = _mtp_kv_users(mesh, model, pt)
        hp_8 = _cat(mesh, eng._hp_buf, dim=3).clone()
        for t in outs_8:
            ttnn.deallocate(t)
        ttnn.deallocate(feed)
        for k in range(K):
            assert [int(x) for x in ids_b[k][:b]] == [
                int(x) for x in ids_8[k][:b]
            ], f"{kind}: draft step {k} ids of rows 0..{b - 1}: bucket {ids_b[k][:b]} vs 8-row {ids_8[k][:b]}"
            assert all(
                int(x) == SENT for x in ids_b[k][b:]
            ), f"{kind}: step {k} ids of rows >= {b} changed: {ids_b[k][b:]}"
        assert any(int(ids_8[k][u]) != 0 for k in range(K) for u in range(b)), f"{kind}: all drafts are 0 (degenerate)"
        for i, (x, y) in enumerate(zip(kv_b, kv_8)):
            assert _same(x[:, :b], y[:, :b]), f"{kind}: MTP KV (cache {i}) of users 0..{b - 1} differs"
            assert float(x[:, b:].abs().sum()) == 0.0, f"{kind}: MTP KV (cache {i}) of users >= {b} was written"
        assert float(kv_b[0][:, :b].abs().sum()) > 0.0 or kind == "zero", "no MTP KV written for rows < b"
        assert _same(hp_b[..., :b, :], hp_8[..., :b, :]), f"{kind}: the anchor hidden of rows < {b} differs"
        logger.info(
            f"[bucket_drafter] b={b} {kind} feed: ids {[list(map(int, s[:b])) for s in ids_b]} equal to the 8-row body, KV equal"
        )
    model.free_kv_caches()


@_marks
def test_engine_capture_buckets(mesh_device, monkeypatch):
    """QWEN36_SPEC_BUCKETS=1,2,4,8 on the full model (QWEN36_SPEC_TEST_N_LAYERS unset = 64 layers) with the fused accept and chain
    traces: prepare + capture succeed with program-cache misses disallowed (capture_decode), every shared buffer keeps its address
    across all captures (asserted by capture_decode and again after the replays below), every bucket has its verify (main, logits,
    topk) and drafter trace, replaying each of them compiles nothing, and the trace memory per capture is logged."""
    from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _env, _setup

    _env(monkeypatch)
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4,8")
    model, eng, _, _ = _setup(mesh_device)
    mesh = mesh_device
    assert eng._buckets == (1, 2, 4, 8) and eng._dr_bks == (4, 2, 1), (eng._buckets, eng._dr_bks)
    assert set(eng._dr_traces) == {1, 2, 4} and all(t is not None for t in eng._dr_traces.values())
    assert eng._dr_trace is not None and eng._dr_trace not in eng._dr_traces.values()
    assert set(model._vfy_bk) == {1, 2, 4}
    for b, bk in model._vfy_bk.items():
        assert bk.trace_id is not None and bk.logits_trace_id is not None, f"bucket {b}: verify traces missing"
        assert not bk.standins, f"bucket {b}: stand-ins were not freed"
    assert not eng._dr_standins and eng._addr_snap and any(n.startswith("dr_outs") for n in eng._addr_snap)
    mem = dict(model._vfy_mem_log)
    labels = list(mem)
    for lab in ("V8", "V4", "V2", "V1", "D8", "D4", "D2", "D1", "accept", "chain", "seed"):
        assert lab in labels, f"no memory log for {lab}: {labels}"
    prev = 0
    lines = []
    for lab in labels:
        tb, db = mem[lab]
        lines.append(f"{lab}: trace {tb / 2**20:.1f} MiB (+{(tb - prev) / 2**20:.1f}), DRAM {db / 2**20:.0f} MiB")
        prev = tb
    logger.info("[bucket_engine] trace memory (sum over banks, 64 layers): " + "; ".join(lines))
    n_prog = mesh.num_program_cache_entries()
    mesh.set_program_cache_misses_allowed(False)
    try:
        for b in (1, 2, 4):
            bk = model._vfy_bk[b]
            for tid in (bk.trace_id, bk.logits_trace_id, bk.topk_trace_id):
                if tid is not None:
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.execute_trace(mesh, eng._dr_traces[b], cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
        ttnn.execute_trace(mesh, eng._dr_trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
    finally:
        mesh.set_program_cache_misses_allowed(True)
    assert mesh.num_program_cache_entries() == n_prog, "replaying a bucket trace compiled a program"
    eng._check_shared_addrs(eng._addr_snap, "after replaying every bucket trace")
    eng.shutdown()
    assert not eng._dr_traces and not model._vfy_bk and eng._hp_bufs is None
    model.free_kv_caches()
