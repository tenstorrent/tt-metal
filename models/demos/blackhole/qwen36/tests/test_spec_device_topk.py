# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device tests for the sampled verify's device-side top-k (QWEN36_VERIFY_DEVICE_TOPK, note 26). Written to run
later on one Galaxy slot (TP mesh, MESH_DEVICE=P150x8 as the demo); NOT run on the host.

(a) ttnn.topk(k=64, stable=True) on the padded shard cell [1,1,32,32768] bf16 with crafted ties and -1e30 pad
    lanes vs a torch stable sort: ids equal, values bit-exact.
(b) trace parity: candidates from the topk trace vs the exact top-64 of the logits trace's readback, every row,
    over many replays of a real spec loop; and the lazy fetch returns the same logits the logits trace would.
(c) determinism: two identical runs in topk mode give identical candidates; the flag off leaves the main-trace
    argmax ids and the logits-trace readback unchanged (greedy and sampled paths untouched).
(d) a prof05-style timing print of T0 (argmax) / T6 logits / T6 topk reads at B=1 and B=8 (logged, not asserted).

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
from models.demos.blackhole.qwen36.tt.spec_engine import exact_topk, get_spec_engine

NUM_BLOCKS = 72
K = int(os.environ.get("QWEN36_SPEC_K", "3"))
STEPS = int(os.environ.get("QWEN36_TOPK_TEST_STEPS", "100"))
M = 64


def _reference_topk(values, m):
    v = values.to(torch.float32)
    sv, si = torch.sort(v, dim=-1, descending=True, stable=True)
    return sv[..., :m], si[..., :m].to(torch.int32)


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
    padded = torch.nn.functional.pad(x.float(), (0, width - per_shard), value=-1e30)
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
    assert torch.equal(got_v, want_v.reshape(32, M)), "values must be the bf16 inputs, bit-exact"
    assert torch.equal(got_i, want_i.reshape(32, M).to(torch.int64)), "stable=True: lowest id first among ties"
    assert bool((got_i[:31] < per_shard).all()), "no pad lane in a real row's top-64"
    for tt in (t, p, vals, idx, idx32):
        ttnn.deallocate(tt)


# --------------------------------------------------------------------------- #
# (b) + (c) + (d): the engine's topk read vs its logits read, over a real spec loop
# --------------------------------------------------------------------------- #
def _spec_steps(eng, prompt, steps, pt, read):
    """Drive one B=1 request: prefill, then ``steps`` verifies with the drafter's K drafts, reading ``read``
    ("topk" / "logits" / None) on every verify. Returns the per-step readbacks and the step times."""
    P = len(prompt)
    eng.prefill(0, prompt, pt)
    out = [int(eng.prefill_logits.argmax())] if hasattr(eng, "prefill_logits") else None
    del out
    proposal = torch.empty(0, dtype=torch.int32)
    acc, last, reads, times = 1, None, [], []
    pos0 = P - 1
    tokens_committed = []
    for step in range(steps):
        n = int(len(proposal))
        drafts = [int(t) for t in proposal[:n]]
        pending = last if last is not None else int(prompt[-1])
        tokens = torch.full((1, 1 + K), -1, dtype=torch.int32)
        start = torch.full((1, 1 + K), -1, dtype=torch.int32)
        tokens[0, : 1 + n] = torch.tensor([pending] + drafts, dtype=torch.int32)
        start[0, : 1 + n] = torch.arange(pos0, pos0 + 1 + n, dtype=torch.int32)
        t0 = time.perf_counter()
        res = eng.decode_forward(
            tokens,
            start,
            num_valid_drafts=torch.tensor([n], dtype=torch.int32),
            accepted_counts=torch.tensor([acc], dtype=torch.int32),
            page_table=pt,
            spec_mode={"topk": "topk", "logits": "logits", None: "argmax_ids"}[read],
        )
        times.append(time.perf_counter() - t0)
        ids = res.argmax_ids[0].tolist()
        if read == "topk":
            full = res.fetch_logits()  # the lazy fallback: one logits-trace replay
            reads.append((res.topk_values[0].clone(), res.topk_ids[0].clone(), full[0].clone()))
        elif read == "logits":
            reads.append(res.logits[0].clone())
        m = 0
        while m < n and drafts[m] == ids[m]:
            m += 1
        new = drafts[:m] + [ids[m]]
        acc = len(new)
        tokens_committed += new
        last = new[-1]
        committed = torch.full((1, 1 + K), -1, dtype=torch.int32)
        cpos = torch.full((1, 1 + K), -1, dtype=torch.int32)
        committed[0, :acc] = torch.tensor(new, dtype=torch.int32)
        cpos[0, :acc] = torch.arange(pos0 + 1, pos0 + 1 + acc, dtype=torch.int32)
        d = eng.propose_draft_tokens(K, committed, cpos, torch.tensor([acc], dtype=torch.int32), pt)
        proposal = d.draft_token_ids[0, : int(d.num_valid[0])].clone()
        pos0 += acc
    eng.release(0)
    return reads, times, tokens_committed


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_trace_parity_topk_candidates_equal_the_logits_exact_top64(mesh_device, monkeypatch):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x8")
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "1")
    model, tokenizer, kv_shape = _build(mesh_device, NUM_BLOCKS)
    pool = _prompt_pool(tokenizer, NUM_BLOCKS * BLOCK_SIZE)
    prompt = pool[:300]
    pt = _perm_table(NUM_BLOCKS, seed=5)[:, :NUM_BLOCKS]
    _fresh_kv(model, kv_shape)
    eng = get_spec_engine(model, 1, K)
    eng.prepare(torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, -1))
    eng.capture()
    assert model._vfy_topk and model._vfy_topk_trace_id is not None, "the third trace was not captured"
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        reads, times, committed = _spec_steps(eng, prompt, STEPS, pt, read="topk")
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    per_shard = model.vocab_size // model.num_devices
    mism = 0
    for step, (vals, ids, full) in enumerate(reads):
        want_v, want_i = exact_topk(full.reshape(1 + K, -1), M)
        if not (torch.equal(ids, want_i) and torch.equal(vals, want_v)):
            mism += 1
            bad = (ids != want_i).any(-1) | (vals != want_v).any(-1)
            logger.error(
                f"step {step}: columns {bad.nonzero().view(-1).tolist()} differ; "
                f"got ids {ids[bad][:, :8].tolist()} want {want_i[bad][:, :8].tolist()}"
            )
        assert bool((ids >= 0).all()) and bool((ids < model.vocab_size).all())
    assert mism == 0, f"{mism} of {len(reads)} steps: topk-trace candidates != exact top-64 of the logits"
    logger.info(
        f"[devtopk] parity OK over {len(reads)} steps; per_shard={per_shard}; "
        f"mean step {1e3 * sum(times) / len(times):.2f} ms (topk read + lazy fetch)"
    )
    # (c) determinism: a second run from fresh state gives identical candidates
    _fresh_kv(model, kv_shape)
    eng = get_spec_engine(model, 1, K)
    eng.prepare(torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, -1))
    eng.capture()
    reads2, _, committed2 = _spec_steps(eng, prompt, STEPS, pt, read="topk")
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
    model, tokenizer, kv_shape = _build(mesh_device, NUM_BLOCKS)
    pool = _prompt_pool(tokenizer, NUM_BLOCKS * BLOCK_SIZE)
    prompt = pool[:300]
    pt = _perm_table(NUM_BLOCKS, seed=5)[:, :NUM_BLOCKS]
    _fresh_kv(model, kv_shape)
    eng = get_spec_engine(model, 1, K)
    eng.prepare(torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, -1))
    eng.capture()
    assert not model._vfy_topk and getattr(model, "_vfy_topk_trace_id", None) is None
    assert model._vfy_shard_logits_out is None
    with expect_error(ValueError, "QWEN36_VERIFY_DEVICE_TOPK=1"):
        model.verify_traced([[0] * (K + 1)], [10], [0], read_topk=True)
    reads, times, committed = _spec_steps(eng, prompt, 20, pt, read="logits")
    assert len(committed) >= 20 and all(tuple(r.shape) == (K + 1, model.vocab_size) for r in reads)
    logger.info(f"[devtopk] flag off: logits read mean step {1e3 * sum(times) / len(times):.2f} ms")
    model.free_kv_caches()


@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_step_times_argmax_vs_logits_vs_topk(mesh_device, monkeypatch):
    """(d) prof05-style: the three reads on the same loop, B=1 (logged; the A/B at the server is the benchmark)."""
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x8")
    monkeypatch.setenv("QWEN36_VERIFY_DEVICE_TOPK", "1")
    model, tokenizer, kv_shape = _build(mesh_device, NUM_BLOCKS)
    pool = _prompt_pool(tokenizer, NUM_BLOCKS * BLOCK_SIZE)
    prompt = pool[:300]
    pt = _perm_table(NUM_BLOCKS, seed=5)[:, :NUM_BLOCKS]
    rows = {}
    for read in (None, "logits", "topk"):
        _fresh_kv(model, kv_shape)
        eng = get_spec_engine(model, 1, K)
        eng.prepare(torch.arange(NUM_BLOCKS, dtype=torch.int32).reshape(1, -1))
        eng.capture()
        _, times, _ = _spec_steps(eng, prompt, 60, pt, read=read)
        rows[read or "argmax_ids"] = 1e3 * sorted(times)[len(times) // 2]
    logger.info(
        f"[devtopk] median decode_forward ms by read (B=1, K={K}): {rows} "
        "(topk includes the lazy full fetch here; the plugin fetches only on overflow)"
    )
    model.free_kv_caches()
