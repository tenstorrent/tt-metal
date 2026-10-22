# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The batched sparse-prefill selection (``TtDSA._sparse_rows``: index stems once per slab, scorer + bounded top-k
per score sub-block of ``GLM53_PREFILL_QBLOCK`` rows, id expansion + sparse attention per attention block of
``GLM53_PREFILL_ABLOCK`` rows) must select exactly the pool ids (sets AND order) the per-sub-block loop selects,
and attend to the same output, on a 16 K chunk at the end of a 64 K synthetic prefix (layer 3, the real
``R16385`` rows, ``tests/test_mesh_heads2users_prefill.py`` seeding pattern):

* ``test_sparse_block_plan_host``: host only (runs on CPU now): the planner and the knob parsing.
* ``test_select_batched_vs_per_subblock[65536]``: variants on the same slot (identical state each time):
  ``loop`` = ABLOCK == QBLOCK == 512 (the previous structure: one attention call per score sub-block),
  ``batched`` = QBLOCK 512, ABLOCK 2048 (4 sub-blocks selected, then one expansion + one attention per slab),
  ``wide`` = QBLOCK 2048 (one scorer call per slab, memory permitting). Selections must be identical across the
  variants, outputs within the row-PCC bar (bit-equality is logged: the absorbed-query matmul of a 2048-row block
  and of a 512-row block are the default dispatch, which may differ by an ulp), two passes of one variant bit-equal.

    TT_VISIBLE_DEVICES=11,10,15,14,7,6,3,2 TT_MESH_GRAPH_DESC_PATH=<p150_x8 descriptor> MESH_DEVICE=P150x8 \
        GLM53_MESH_SHAPE=2,4 pytest models/demos/glm53_flash/tests/test_prefill_select_batched.py -s --timeout 1500
    TT_VISIBLE_DEVICES=0 GLM53_MESH_SHAPE=1,1 pytest ... (the single-row ``prefill_forward`` form of the same loop)
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

ttnn = pytest.importorskip("ttnn")
pytest.importorskip("models.demos.glm53_flash.tt.dsa")
# isort: off
from models.demos.glm53_flash.tt.dsa import prefill_blocks, sparse_block_plan  # noqa: E402
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params  # noqa: E402

# isort: on

TILE = 32
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
LONG_N = [int(v) for v in os.environ.get("GLM53_SEL_LONG_N", "65536").split(",")]
ROW_PCC_BAR = float(os.environ.get("GLM53_SEL_PCC_BAR", "0.99999"))
SEED_CHUNK = 32768
SENTINEL = 0xFFFFFFFF
DENSE_ROWS = 2048
CHUNK = 16384
VARIANTS = {"loop": (512, 512), "batched": (512, 2048), "wide": (2048, 2048)}  # name -> (QBLOCK, ABLOCK)


# ------------------------------------------------------------------------------------------ host only
def test_sparse_block_plan_host(monkeypatch):
    monkeypatch.setenv("GLM53_PREFILL_QBLOCK", "512")
    monkeypatch.setenv("GLM53_PREFILL_ABLOCK", "2048")
    assert prefill_blocks() == (512, 2048)
    monkeypatch.setenv("GLM53_PREFILL_QBLOCK", "1000")  # rounded down to tiles
    assert prefill_blocks()[0] == 992
    monkeypatch.delenv("GLM53_PREFILL_QBLOCK")
    monkeypatch.delenv("GLM53_PREFILL_ABLOCK")
    assert prefill_blocks() == (2048, 2048)
    # a 2048-row slab: 4 score sub-blocks under one attention block, or one call each
    assert sparse_block_plan(0, 2048, 512, 2048) == [(0, 2048, [(0, 512), (512, 1024), (1024, 1536), (1536, 2048)])]
    assert sparse_block_plan(0, 2048, 512, 512) == [(a, a + 512, [(a, a + 512)]) for a in range(0, 2048, 512)]
    assert sparse_block_plan(0, 2048, 2048, 2048) == [(0, 2048, [(0, 2048)])]
    # the single-row chunk form: sparse rows [split, T) of a 16 K chunk with a ragged tail block
    plan = sparse_block_plan(1024, 16384, 2048, 4096)
    assert plan[0] == (1024, 5120, [(1024, 3072), (3072, 5120)]) and plan[-1] == (
        13312,
        16384,
        [(13312, 15360), (15360, 16384)],
    )
    assert sum(a1 - a0 for a0, a1, _ in plan) == 16384 - 1024
    # ABLOCK below QBLOCK = the per-sub-block form; ABLOCK not a multiple of QBLOCK rounds down
    assert sparse_block_plan(0, 1024, 512, 100) == sparse_block_plan(0, 1024, 512, 512)
    assert [a1 - a0 for a0, a1, _ in sparse_block_plan(0, 2048, 256, 1000)] == [768, 768, 512]


# ------------------------------------------------------------------------------------------ helpers
class OrderedRecorder:
    """``TtDSA.on_indices`` sink: the valid-id prefix of every sparse query row by absolute position, in the op's
    order (``ordered``) and sorted (``rows``); read on column 0 of every mesh row (the replicated-cache layout:
    every column holds the same rows). The slab path hands ``(row_starts, indices)``, the single-row path
    ``(query_start, indices)``; positions below the dense boundary (a mixed chunk's clamped rows) are skipped."""

    def __init__(self, mesh: GLMMesh, max_pos: int):
        self.mesh, self.max_pos = mesh, max_pos
        self.rows: dict[int, torch.Tensor] = {}
        self.ordered: dict[int, torch.Tensor] = {}
        self.calls = 0

    def __call__(self, query_start, indices: ttnn.Tensor):
        self.calls += 1
        devs = self.mesh.device_tensors(indices)
        starts = query_start if isinstance(query_start, tuple) else (query_start,)
        for r, s in enumerate(starts):
            t = ttnn.to_torch(devs[r * self.mesh.cols]).reshape(-1, indices.shape[-1]).to(torch.int64) & SENTINEL
            for i in range(t.shape[0]):
                pos = s + i
                if pos < DENSE_ROWS or pos >= self.max_pos:
                    continue
                row = t[i]
                sent = (row == SENTINEL).nonzero()
                first = int(sent[0]) if sent.numel() else row.numel()
                assert bool((row[first:] == SENTINEL).all()), f"row {pos}: a sentinel inside the valid prefix"
                self.ordered[pos] = row[:first].clone()
                self.rows[pos] = row[:first].sort().values


def _prefill(attn, cache, mesh, rows_t, user, n, start, qblock, ablock, env):
    """The chunk [start, start + n) of ``rows_t`` (absolute positions) through the slab-local path with the given
    block sizes; returns (output [n, 4096] fp32, OrderedRecorder, wall seconds)."""
    from models.demos.glm53_flash.tests.test_mesh_slab_prefill import _chunk_to_device, _rows_out

    env.setenv("GLM53_PREFILL_QBLOCK", str(qblock))
    env.setenv("GLM53_PREFILL_ABLOCK", str(ablock))
    rec = OrderedRecorder(mesh, start + n)
    attn.on_indices = rec
    x = _chunk_to_device(mesh, rows_t, 0, n, True)
    ttnn.synchronize_device(mesh.device)
    t0 = time.perf_counter()
    y = attn.prefill_forward(x, cache, user, n, start=start, slab=True)
    ttnn.synchronize_device(mesh.device)
    wall = time.perf_counter() - t0
    out = _rows_out(mesh, y, n, True)
    ttnn.deallocate(y)
    ttnn.deallocate(x)
    attn.on_indices = None
    return out, rec, wall


def _compare_ids(ref: OrderedRecorder, got: OrderedRecorder) -> tuple[int, int, int]:
    """(rows compared, rows with a different id SET, rows with the same set in a different ORDER)."""
    assert set(ref.rows) == set(got.rows), (len(ref.rows), len(got.rows))
    diff_set = diff_order = 0
    for pos, a in ref.rows.items():
        b = got.rows[pos]
        if a.numel() != b.numel() or not torch.equal(a, b):
            diff_set += 1
        elif not torch.equal(ref.ordered[pos], got.ordered[pos]):
            diff_order += 1
    return len(ref.rows), diff_set, diff_order


# ----------------------------------------------------------------------------------------- the chunk
@pytest.mark.timeout(1500)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("n_pos", LONG_N)
def test_select_batched_vs_per_subblock(mesh_device, monkeypatch, n_pos):
    from models.demos.glm53_flash.tests.test_cache_1m import _x_stats, synthetic_rows
    from models.demos.glm53_flash.tests.test_mesh_heads2users_prefill import R16385, _attention, _load
    from models.demos.glm53_flash.tests.test_prefill_chunked import row_pcc
    from models.demos.glm53_flash.tests.utils import attention_input, log_dram
    from models.demos.glm53_flash.tt.dsa import TtLatentCache

    raw, cfg = _load()
    x_in = torch.load(R16385 / "layer3_prefill.pt")["x_in"][:, :CHUNK]
    rows_t = attention_input(raw, cfg, x_in)  # [CHUNK, 4096] bf16
    start0 = n_pos - CHUNK
    attn, mesh = _attention(mesh_device, raw, cfg)
    cache = TtLatentCache(mesh_device, cfg, max_batch=1, max_seq_len=n_pos + 64 * mesh.rows, mesh=mesh)
    log_dram(mesh_device, "attention + 1-slot replicated cache")
    if start0:
        mu, sigma = _x_stats(raw, cfg)
        filled, ci = 0, 0
        t0 = time.perf_counter()
        while filled < start0:
            T = min(SEED_CHUNK, start0 - filled)
            x_tt = mesh.upload(synthetic_rows(mu, sigma, ci, T).reshape(1, 1, T, -1), ttnn.bfloat16)
            attn.prefill_state(x_tt, cache, 0, T, start=filled)
            ttnn.deallocate(x_tt)
            filled += T
            ci += 1
        ttnn.synchronize_device(mesh_device)
        logger.info(f"seeded [0, {start0}) synthetically ({ci} chunks, {time.perf_counter() - t0:.1f} s)")
    slab_rows = CHUNK // mesh.rows
    results = {}
    failures = []
    for name, (qb, ab) in VARIANTS.items():
        if qb > slab_rows:
            qb = slab_rows  # a 1-row mesh scores 16 K rows at once only with the memory for it: cap at the slab
        passes = []
        for _ in range(2):  # the compile pass and the warm pass of every variant must agree
            passes.append(_prefill(attn, cache, mesh, rows_t, 0, CHUNK, start0, qb, ab, monkeypatch))
        (y0, rec0, _), (y1, rec1, wall) = passes
        assert torch.equal(y0, y1), f"{name}: the two passes differ bitwise"
        n_rows, ds, do = _compare_ids(rec0, rec1)
        assert ds == 0 and do == 0, f"{name}: the two passes select differently ({ds} sets, {do} orders)"
        n_blocks = len(sparse_block_plan(0, slab_rows, qb, ab))
        assert rec1.calls == n_blocks, (name, rec1.calls, n_blocks)
        results[name] = (y1, rec1, wall)
        logger.info(
            f"N={n_pos} {name} (QBLOCK {qb}, ABLOCK {ab}): {n_blocks} attention block(s) per slab, warm wall "
            f"{wall * 1e3:.0f} ms ({CHUNK / wall:.0f} tok/s), {n_rows} sparse rows recorded"
        )
    y_ref, rec_ref, w_ref = results["loop"]
    for name, (y, rec, wall) in results.items():
        if name == "loop":
            continue
        n_rows, ds, do = _compare_ids(rec_ref, rec)
        pc = row_pcc(y_ref, y)
        bit = torch.equal(y_ref, y)
        logger.info(
            f"N={n_pos} {name} vs loop: {n_rows} rows, different sets {ds}, different order {do}; output bitwise "
            f"equal {bit}, row pcc min {float(pc.min()):.7f}; wall x{w_ref / wall:.2f}"
        )
        if ds or do:
            failures.append(f"{name}: {ds} rows select a different set, {do} a different order")
        below = int((pc < ROW_PCC_BAR).sum())
        if below:
            failures.append(f"{name}: {below} rows below the row-pcc bar {ROW_PCC_BAR} (min {float(pc.min()):.7f})")
    log_dram(mesh_device, "done")
    assert not failures, "\n".join(failures)
