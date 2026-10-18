# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests for the sampled verify's device-side top-k (QWEN36_VERIFY_DEVICE_TOPK, note 26). One Galaxy slot
(TP mesh, MESH_DEVICE=P150x8 as the demo); NOT run on the host.

(a) ttnn.topk(k=64, stable=True) on the padded shard cell [1,1,32,32768] bf16 with crafted ties and -1e30 pad
    lanes vs a torch stable sort: ids equal, values bit-exact.
(b) trace parity: candidates from the topk trace vs the exact top-64 of the logits trace's readback, every row,
    over many replays of a real spec loop; and the lazy fetch returns the same logits the logits trace would.
    Also counts the plugin's overflow positions (the tie group at the request's k-th value reaches the 64th
    candidate, so the plugin would fetch the full logits) for a few request top_k values.
(c) determinism: two identical runs in topk mode give identical candidates; the flag off leaves the main-trace
    argmax ids and the logits-trace readback unchanged (greedy and sampled paths untouched).
(d) a prof05-style timing print of the T0 (argmax) / T6 logits / T6 topk reads at B=1 and B=8 (logged, not
    asserted): per step the decode_forward total, the engine's (device, readback) split (QWEN36_SPEC_TIMING) and,
    for topk, the lazy full-logits fetch timed outside the step.
Every capture logs the DRAM / TRACE / L1 bytes allocated per bank, so the third trace's footprint is the
difference between the flag-on and the flag-off runs.

Run (device, one test at a time; see work/notes/26 and the disk guard in the memory notes):
  QWEN36_VERIFY_DEVICE_TOPK=1 MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_spec_device_topk.py \
      -k topk_stable_matches -v -s
  QWEN36_VERIFY_DEVICE_TOPK=1 MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_spec_device_topk.py \
      -k trace_parity -v -s
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.spec_helpers import _build, _fresh_kv, _perm_table, _prompt_pool
from models.demos.blackhole.qwen36.tests.test_spec_lossless import _N_LAYERS
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_engine import exact_topk, get_spec_engine

NUM_BLOCKS = 72  # B=1: KV pool and page-table width (4608 tokens)
K = int(os.environ.get("QWEN36_SPEC_K", "3"))
STEPS = int(os.environ.get("QWEN36_TOPK_TEST_STEPS", "100"))
M = 64
# B=8 timing: one 16-block (1024-token) table per user over blocks 1.. (block 0 stays the null block), distinct
# prompt lengths so every user sits at a different, unaligned frontier (as test_spec_continuous).
B8_NB = 16
B8_PROMPT_LENS = (130, 257, 389, 512, 600, 211, 345, 460)


def _reference_topk(values, m):
    v = values.to(torch.float32)
    sv, si = torch.sort(v, dim=-1, descending=True, stable=True)
    return sv[..., :m], si[..., :m].to(torch.int32)


def _log_mem(mesh_device, tag):
    """DRAM / TRACE / L1 (allocated, free) bytes per bank after a capture; informational, never fails the test."""
    used = {}
    for name in ("DRAM", "TRACE", "L1"):
        try:
            view = ttnn.get_memory_view(mesh_device, getattr(ttnn.BufferType, name))
            used[name] = (int(view.total_bytes_allocated_per_bank), int(view.total_bytes_free_per_bank))
        except Exception as e:  # pragma: no cover - depends on the ttnn build
            used[name] = f"n/a: {e}"
    logger.info(f"[devtopk] memory after capture ({tag}): (allocated, free) bytes per bank {used}")


# --------------------------------------------------------------------------- #
# (a) the stable top-k op on our exact cell
# --------------------------------------------------------------------------- #
@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_topk_stable_matches_torch_on_the_padded_shard_cell(mesh_device):
    """[1,1,32,32768] bf16 (per-shard width 31040 real lanes + -1e30 pads): ids and bit-exact values."""
    per_shard, width = 31040, 32768
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(1, 1, 32, per_shard, generator=g) * 3).to(torch.bfloat16)
    # crafted ties: a group inside the top-64, a group straddling the 64th boundary, a group at the very top
    for r in range(32):
        top = torch.topk(x[0, 0, r].float(), 80).indices
        x[0, 0, r, top[2:6]] = x[0, 0, r, top[2]].clone()
        x[0, 0, r, top[60:70]] = x[0, 0, r, top[60]].clone()
        x[0, 0, r, [5, 17, 4000, 31039]] = (x[0, 0, r].float().max() + 1).to(torch.bfloat16)
    x[0, 0, 31] = -1e30  # a padding row: all equal, lowest ids first
    # Pad in bf16: the device pads a bf16 tensor, so its pad lanes hold bf16(-1e30) = -1.00026e30, the same value
    # as row 31's real lanes (an fp32 -1e30 pad would outrank them and put pad ids in that row's reference).
    padded = torch.nn.functional.pad(x, (0, width - per_shard), value=-1e30).float()
    want_v, want_i = _reference_topk(padded, M)

    t = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    p = ttnn.pad(t, [(0, 0), (0, 0), (0, 0), (0, width - per_shard)], value=-1e30)
    vals, idx = ttnn.topk(p, k=M, dim=-1, largest=True, sorted=True, stable=True)
    idx32 = ttnn.typecast(idx, ttnn.uint32)
    got_v = ttnn.to_torch(ttnn.get_device_tensors(vals)[0]).float().reshape(32, M)
    got_i = ttnn.to_torch(ttnn.get_device_tensors(idx32)[0]).reshape(32, M).to(torch.int64)
    want_v, want_i = want_v.reshape(32, M), want_i.reshape(32, M).to(torch.int64)
    bad = ((got_v != want_v) | (got_i != want_i)).any(-1).nonzero().view(-1).tolist()
    for r in bad[:4]:
        cols = ((got_v[r] != want_v[r]) | (got_i[r] != want_i[r])).nonzero().view(-1).tolist()
        logger.error(
            f"row {r}: {len(cols)} columns differ, first {cols[:8]}; got {[(float(got_v[r, c]), int(got_i[r, c])) for c in cols[:4]]} "
            f"want {[(float(want_v[r, c]), int(want_i[r, c])) for c in cols[:4]]}"
        )
    assert torch.equal(got_v, want_v), f"values must be the bf16 inputs, bit-exact (rows {bad})"
    assert torch.equal(got_i, want_i), f"stable=True: lowest id first among ties (rows {bad})"
    assert bool((got_i[:31] < per_shard).all()), "no pad lane in a real row's top-64"
    for tt in (t, p, vals, idx, idx32):
        ttnn.deallocate(tt)


# --------------------------------------------------------------------------- #
# (b) + (c) + (d): the engine's topk read vs its logits read, over a real spec loop
# --------------------------------------------------------------------------- #
def _setup(mesh_device, B):
    """Model (max_batch_size=B), tokenizer, B prompts, [B, nb] block tables and the KV shape for B users."""
    if B == 1:
        model, tokenizer, kv_shape = _build(mesh_device, NUM_BLOCKS)
        pool = _prompt_pool(tokenizer, NUM_BLOCKS * BLOCK_SIZE)
        prompts = [pool[:300]]
        pts = _perm_table(NUM_BLOCKS, seed=5)[:, :NUM_BLOCKS]
        return model, tokenizer, prompts, pts, kv_shape
    from transformers import AutoTokenizer

    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=B, max_seq_len=B8_NB * BLOCK_SIZE, n_layers=_N_LAYERS
    )
    assert model.mtp is not None, "MTP head not built"
    model.set_gdn_fused_decode(True)  # verify runs the fused GDN op; plain decode must too
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    pool = _prompt_pool(tokenizer, 4096)
    prompts = [pool[97 * i : 97 * i + n] for i, n in enumerate(B8_PROMPT_LENS[:B])]
    assert [len(p) for p in prompts] == list(B8_PROMPT_LENS[:B])
    pts = torch.stack([torch.arange(1 + u * B8_NB, 1 + (u + 1) * B8_NB, dtype=torch.int32) for u in range(B)])
    kv_shape = [1 + B * B8_NB, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    return model, tokenizer, prompts, pts, kv_shape


def _fresh(model, kv_shape, B):
    if B == 1:
        _fresh_kv(model, kv_shape)
    else:
        model.free_kv_caches()  # also shuts down any engine
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=B)


def _engine(model, pts, B):
    """A prepared and captured engine for B users (the warm table is the B=1 identity table, as before)."""
    eng = get_spec_engine(model, B, K)
    eng.prepare(pts if B > 1 else torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, -1))
    eng.capture()
    return eng


def _spec_steps(eng, prompts, steps, pts, read, keep=True):
    """Drive B requests as the plugin runner does: prefill each (its argmax is the first committed token, verified
    at position P as the seed step), then ``steps`` verifies with the drafter's K drafts per user, reading
    ``read`` ("topk" / "logits" / None) on every verify. ``pts``: [B, nb] block tables, one row per user.
    Returns (per-step readbacks when ``keep``: (topk_values, topk_ids, full logits) [B, T, .] in topk mode or
    the [B, T, V] logits in logits mode; per-step decode_forward seconds; committed tokens per user; per-step
    engine.last_verify_timing (device_s, readback_s) when QWEN36_SPEC_TIMING is on, else None; per-step lazy
    fetch seconds in topk mode)."""
    B = len(prompts)
    last, pos0 = [], []
    for u, prompt in enumerate(prompts):
        lt = eng.prefill(u, prompt, pts[u : u + 1])
        last.append(int(lt.argmax()))
        pos0.append(len(prompt))
    committed = [[t] for t in last]
    proposal = [torch.empty(0, dtype=torch.int32) for _ in range(B)]
    acc = [1] * B
    reads, times, splits, fetches = [], [], [], []
    mode = {"topk": "topk", "logits": "logits", None: "argmax_ids"}[read]
    for step in range(steps):
        tokens = torch.full((B, 1 + K), -1, dtype=torch.int32)
        start = torch.full((B, 1 + K), -1, dtype=torch.int32)
        ns = [int(len(p)) for p in proposal]
        for u in range(B):
            tokens[u, : 1 + ns[u]] = torch.tensor([last[u]] + proposal[u].tolist(), dtype=torch.int32)
            start[u, : 1 + ns[u]] = torch.arange(pos0[u], pos0[u] + 1 + ns[u], dtype=torch.int32)
        t0 = time.perf_counter()
        res = eng.decode_forward(
            tokens,
            start,
            num_valid_drafts=torch.tensor(ns, dtype=torch.int32),
            accepted_counts=torch.tensor(acc, dtype=torch.int32),
            page_table=pts,
            spec_mode=mode,
        )
        times.append(time.perf_counter() - t0)
        splits.append(eng.last_verify_timing)
        if read == "topk":
            t1 = time.perf_counter()
            full = res.fetch_logits()  # the lazy fallback: one logits-trace replay (a host copy on a seed step)
            fetches.append(time.perf_counter() - t1)
            if keep:  # plus this step's draft counts, so an overflow column can be told from an unused column
                reads.append((res.topk_values.clone(), res.topk_ids.clone(), full.clone(), list(ns)))
        elif read == "logits" and keep:
            reads.append(res.logits.clone())
        blk = torch.full((B, 1 + K), -1, dtype=torch.int32)
        cpos = torch.full((B, 1 + K), -1, dtype=torch.int32)
        for u in range(B):
            ids, drafts, n = res.argmax_ids[u].tolist(), proposal[u].tolist(), ns[u]
            m = 0
            while m < n and drafts[m] == ids[m]:
                m += 1
            new = drafts[:m] + [ids[m]]
            acc[u] = len(new)
            committed[u] += new
            last[u] = new[-1]
            blk[u, : acc[u]] = torch.tensor(new, dtype=torch.int32)
            cpos[u, : acc[u]] = torch.arange(pos0[u] + 1, pos0[u] + 1 + acc[u], dtype=torch.int32)
            pos0[u] += acc[u]
        d = eng.propose_draft_tokens(K, blk, cpos, torch.tensor(acc, dtype=torch.int32), pts)
        proposal = [d.draft_token_ids[u, : int(d.num_valid[u])].clone() for u in range(B)]
    for u in range(B):
        eng.release(u)
    return reads, times, committed, splits, fetches


def _ms(xs, skip=2, col=None):
    """(median, mean) ms of a per-step series without its first ``skip`` steps (the seed and the first drafts);
    ``col`` picks one element of a per-step tuple. None entries (a seed step sets no verify timing) are dropped."""
    xs = [x if col is None else (x[col] if x is not None else None) for x in xs[skip:]]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return round(1e3 * sorted(xs)[len(xs) // 2], 2), round(1e3 * sum(xs) / len(xs), 2)


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_trace_parity_topk_candidates_equal_the_logits_exact_top64(mesh_device, monkeypatch):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x8")
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "1")
    model, tokenizer, prompts, pt, kv_shape = _setup(mesh_device, 1)
    _fresh(model, kv_shape, 1)
    eng = _engine(model, pt, 1)
    assert model._vfy_topk and model._vfy_topk_trace_id is not None, "the third trace was not captured"
    _log_mem(mesh_device, "flag on, B=1")
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        reads, times, committed, _, fetches = _spec_steps(eng, prompts, STEPS, pt, read="topk")
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    per_shard = model.vocab_size // model.num_devices
    T = K + 1
    mism, positions, boundary_ties = 0, 0, 0
    overflow = {k: 0 for k in (1, 20, 32, 63)}
    detail = []  # (step, drafts n, overflowing columns at top_k=20, their 1st and 64th candidate values)
    for step, (vals, ids, full, ns) in enumerate(reads):
        vals, ids, full = vals[0], ids[0], full[0]  # user 0: [T, M], [T, M], [T, V]
        over = (vals[:, M - 1] >= vals[:, 19]).nonzero().view(-1).tolist()
        if over:
            detail.append((step, ns[0], over, [(float(vals[c, 0]), float(vals[c, M - 1])) for c in over]))
        want_v, want_i = exact_topk(full, M)
        if not (torch.equal(ids, want_i) and torch.equal(vals, want_v)):
            mism += 1
            bad = (ids != want_i).any(-1) | (vals != want_v).any(-1)
            logger.error(
                f"step {step}: columns {bad.nonzero().view(-1).tolist()} differ; "
                f"got ids {ids[bad][:, :8].tolist()} want {want_i[bad][:, :8].tolist()}"
            )
        assert bool((ids >= 0).all()) and bool((ids < model.vocab_size).all())
        positions += T
        # The plugin's overflow rule for a request with top_k = k: the tie group at the k-th value reaches the
        # M-th candidate (sorted desc, so >= means equal), and the plugin fetches the full logits for that row.
        for k in overflow:
            overflow[k] += int((vals[:, M - 1] >= vals[:, k - 1]).sum())
        boundary_ties += int((vals[:, M - 1] == vals[:, M - 2]).sum())
    assert mism == 0, f"{mism} of {len(reads)} steps: topk-trace candidates != exact top-64 of the logits"
    logger.info(
        f"[devtopk] parity OK over {len(reads)} steps x T={T} = {positions} positions; per_shard={per_shard}; "
        f"overflow positions by request top_k {overflow}; positions with a tie at the 64th candidate "
        f"{boundary_ties}; committed {len(committed[0])} tokens; step (median, mean) ms {_ms(times)} "
        f"(topk read, fetch outside); lazy fetch (median, mean) ms {_ms(fetches)}"
    )
    logger.info(
        f"[devtopk] overflow positions at top_k=20 (step, n drafts, columns, (1st, 64th) values): {detail[:24]}"
    )
    # (c) determinism: a second run from fresh state gives identical candidates
    _fresh(model, kv_shape, 1)
    eng = _engine(model, pt, 1)
    reads2, _, committed2, _, _ = _spec_steps(eng, prompts, STEPS, pt, read="topk")
    assert committed2 == committed
    assert all(torch.equal(a[1], b[1]) and torch.equal(a[0], b[0]) for a, b in zip(reads, reads2))
    model.free_kv_caches()


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_flag_off_keeps_the_old_traces_and_reads(mesh_device, monkeypatch, expect_error):
    """QWEN36_VERIFY_DEVICE_TOPK=0: two traces, no shard logits kept, read_topk refused, greedy loop unchanged."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x8")
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "0")
    monkeypatch.setenv("QWEN36_SPEC_TIMING", "1")
    model, tokenizer, prompts, pt, kv_shape = _setup(mesh_device, 1)
    _fresh(model, kv_shape, 1)
    eng = _engine(model, pt, 1)
    assert not model._vfy_topk and getattr(model, "_vfy_topk_trace_id", None) is None
    assert model._vfy_shard_logits_out is None
    _log_mem(mesh_device, "flag off, B=1")
    with expect_error(ValueError, "QWEN36_VERIFY_DEVICE_TOPK=1"):
        model.verify_traced([[0] * (K + 1)], [10], [0], read_topk=True)
    reads, times, committed, splits, _ = _spec_steps(eng, prompts, 20, pt, read="logits")
    assert len(committed[0]) >= 20 and all(tuple(r.shape) == (1, K + 1, model.vocab_size) for r in reads)
    logger.info(
        f"[devtopk] flag off: logits read step (median, mean) ms {_ms(times)}; "
        f"device {_ms(splits, col=0)}; readback {_ms(splits, col=1)}"
    )
    model.free_kv_caches()


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("B", [1, 8])
def test_step_times_argmax_vs_logits_vs_topk(mesh_device, monkeypatch, B):
    """(d) prof05-style: the three reads on the same loop at B users (logged; the A/B at the server is the
    benchmark). device = staging + replay + synchronize until the ids read; readback = ids + payload to_torch
    (the payload is 15.9 MB in logits mode, ~96 KB in topk mode); fetch = the lazy full-logits replay + read,
    outside the step (the plugin calls it only on an overflow)."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x8")
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "1")
    monkeypatch.setenv("QWEN36_SPEC_TIMING", "1")
    model, tokenizer, prompts, pts, kv_shape = _setup(mesh_device, B)
    rows = {}
    for read in (None, "logits", "topk"):
        _fresh(model, kv_shape, B)
        eng = _engine(model, pts, B)
        if read is None:
            _log_mem(mesh_device, f"flag on, B={B}")
        _, times, committed, splits, fetches = _spec_steps(eng, prompts, 60, pts, read=read, keep=False)
        rows[read or "argmax_ids"] = {
            "step": _ms(times),
            "device": _ms(splits, col=0),
            "readback": _ms(splits, col=1),
            "fetch": _ms(fetches),
            "committed": [len(c) for c in committed],
        }
    logger.info(f"[devtopk] B={B} K={K}: (median, mean) ms per decode_forward step by read: {rows}")
    model.free_kv_caches()
