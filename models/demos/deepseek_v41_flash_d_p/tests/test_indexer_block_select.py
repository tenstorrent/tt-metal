# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.BM / MB.6 (OWNER 2026-11-21): the indexer's top-k by block-max pre-selection (INDEXER_BLOCK_SELECT=on) selects
the SAME key sets as the plain top-k (off), up to ties at the 512th value, on real text at long context, on the
single-request path (MB.6: model_bringup has no packed path; the batching branch's packed half is left out).

One request of real text (BRINGUP_POS_TOKENS's text, agentic_cc by default; repeated past its end) is prefilled from 0
in whole chunks of the load's chunk (5120 on the 8x4 run) through layers 0 .. the last plain index source of the load
(DeepSeek-V4.1: 0-14; plain index sources 2, 8, 14), every indexer call on the block path (force_block_select(True):
whatever INDEXER_BLOCK_SELECT_MIN). At each point P (IDX_BSEL_POS, default 138240 and the last whole chunk the load's
tables reach: 1018880 at the gate's max_seq 1024000), the
request's chunk [P, P + chunk) is checked: for every block top-k of it (every plain indexer layer, every chip, every
query row the chip scored), beside it on the same scores: the plain top-k (topk_large_indices over the keys' valid
length, the off path) and the scores at both id lists (ttnn.bringup.block_gather, block 1). On the host, per row: the
two id sets (the 0xFFFFFFFF tail dropped); when they differ, their sorted score lists: equal = the sets differ only by
keys tied at the 512th value (a tie row, counted); else every id outside the other set is a mismatch.

Metrics (gated ``idx_block_select_set_mismatches*`` == 0): idx_block_select_set_mismatches_single_p<P> and their sum
idx_block_select_set_mismatches; idx_block_select_tie_rows_single_p<P> (rows differing by 512th-value ties only),
idx_block_select_rows_single_p<P> (rows compared), idx_block_select_calls_single_p<P>.

Load: the accuracy load (BATCH_SHARED, the gate's shared process); the request's state holds P_max + a chunk positions,
so the default top point is fitted to the load's tables (an explicit IDX_BSEL_POS must fit them).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_indexer_block_select.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
CHUNK = int(S.data["target"]["chunk"])
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("IDX_BSEL_POS",)
BATCH_MAX_SEQ = 1024000  # (the gate's shared load: the top point is fitted to the load's tables, _points)
SENT = 0xFFFFFFFF


def _points(max_seq: int) -> list[int]:
    """IDX_BSEL_POS, default 138240 and the last whole chunk the load's tables reach (1018880 at max_seq 1024000)."""
    v = os.environ.get("IDX_BSEL_POS", "").strip()
    top = (int(max_seq) - CHUNK) // CHUNK * CHUNK
    pts = [int(x) for x in v.split(",") if x.strip()] if v else [138240, top]
    assert all(p % CHUNK == 0 and p >= CHUNK for p in pts), pts
    return sorted(pts)


def _u32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.int64) & 0xFFFFFFFF


class Checker:
    """BSEL_CHECK: after every block top-k, the plain top-k of the same scores and both lists' scores (device), kept
    until ``collect`` reads them after the chunk."""

    def __init__(self):
        self.pending = []
        self.on = False
        self.iota = {}  # (rows, k) -> [1, 1, rows, k] uint32 0..k-1 (replicated): block_keys at block 1 copies ids

    def __call__(self, ix, logits, out, keys_vl):
        import ttnn
        from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid
        from models.demos.deepseek_v41_flash_d_p.tt.indexer import topk_large_indices

        if not self.on:
            return
        vl = dict(valid_length=int(keys_vl)) if isinstance(keys_vl, int) else dict(valid_length_tensor=keys_vl)
        plain = topk_large_indices(logits, k=ix.topk, **vl)
        mg = max_grid()
        # (a copy of the block path's ids on the matmul grid: inside a gather-hide region only the compute sub-device
        # runs programs, ttnn.clone takes the whole grid; block_keys at block 1 over 0..k-1 is ids[p] exactly)
        iota = self.iota[(int(out.shape[-2]), int(out.shape[-1]))]  # (made before the chunks: prep)
        mine = ttnn.bringup.block_keys(iota, out, block=1, width=int(logits.shape[-1]), max_grid=mg)
        vm = ttnn.bringup.block_gather(logits, mine, block=1, max_grid=mg)
        vp = ttnn.bringup.block_gather(logits, plain, block=1, max_grid=mg)
        self.pending.append((ix, mine, plain, vm, vp))

    def prep(self, mesh, rows: int, k: int) -> None:
        import ttnn

        self.iota[(rows, k)] = ttnn.from_torch(
            torch.arange(k, dtype=torch.int32).view(1, 1, 1, k).expand(1, 1, rows, k).contiguous(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def collect(self) -> dict:
        import ttnn

        out = dict(mismatches=0, tie_rows=0, rows=0, calls=len(self.pending))
        for ix, *ts in self.pending:
            per = [[ttnn.to_torch(d).reshape(-1, d.shape[-1]) for d in ttnn.get_device_tensors(t)] for t in ts]
            for mine, plain, vm, vp in zip(*per):
                for r in range(mine.shape[0]):
                    a, b = _u32(mine[r]), _u32(plain[r])
                    ka, kb = a != SENT, b != SENT
                    sa, sb = set(a[ka].tolist()), set(b[kb].tolist())
                    out["rows"] += 1
                    if sa == sb:
                        continue
                    va = torch.sort(vm[r][ka].float(), descending=True).values
                    vb = torch.sort(vp[r][kb].float(), descending=True).values
                    if va.shape == vb.shape and torch.equal(va, vb):
                        out["tie_rows"] += 1
                    else:
                        out["mismatches"] += len(sa ^ sb)
            for t in ts:
                ttnn.deallocate(t)
        self.pending = []
        return out


@mesh_parametrize(timeout_s=7200)
def test_indexer_block_select(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.common.bringup.reference import texts
    from models.demos.deepseek_v41_flash_d_p.tt import indexer as IX
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv(settings.env_name("INDEXER_BLOCK_SELECT"), "on")  # (call-time: the block path needs it on)
    if settings.snap is not None:
        monkeypatch.setitem(settings.snap, "INDEXER_BLOCK_SELECT", "on")
    assert settings.get("INDEXER_BLOCK_SELECT") == "on"
    model = S.hooks().device_model(mesh_device, S, list(S.layers()), lm_head=False)
    points = _points(model.max_seq)
    assert model.chunk == CHUNK, (model.chunk, CHUNK)
    cap = points[-1] + CHUNK
    assert cap <= model.max_seq, f"the load's tables reach {model.max_seq} < {cap} (BATCH_MAX_SEQ)"
    cfg = model.cfg
    idx_layers = [
        i for i in model.layers if cfg.is_index_source(i) and not cfg.is_cand_source(i) and not cfg.uses_candidates(i)
    ]
    assert idx_layers, model.layers
    layers = [i for i in model.layers if i <= max(idx_layers)]
    skip0 = model.skip_mode
    if skip0 != "off":
        model.use_decoder_skip("off")  # (layers 0 .. 14 compute in full; FIX.28: host lists only)
    ids, info = texts.tokens(S, cap, texts.explicit() or "agentic_cc")
    print(f"block select sets: chunk {CHUNK}, layers {layers} (index {idx_layers}), points {points}, text {info}")

    # (DS41-311 / DS41-329: the held model's indexers may be instances of a class of an earlier module object than
    # the import above, after the runner reloaded tt/indexer.py in place: their hooks are their own module's globals)
    import contextlib
    import sys

    gl = [type(model.blocks[idx_layers[0]].indexer).block_topk.__globals__, vars(sys.modules[IX.__name__])]
    gl = list({id(g): g for g in gl}.values())  # (the instances' module and the current one, when they differ)
    chk = Checker()
    ix0 = model.blocks[idx_layers[0]].indexer
    chk.prep(model.mesh, int(ix0.sq), int(ix0.topk))
    for g in gl:
        g["BSEL_CHECK"][0] = chk
    res = {}

    def single(st, c0, check=False):
        toks = ids[c0 : c0 + CHUNK].clone()
        chk.on = check
        model.begin_chunk(toks, c0, last=False)
        h = model.embed(toks, c0)
        for i in layers:
            h2 = model.layer(i, h, c0, st)
            model.free(h)
            h = h2
        model.free(h)
        chk.on = False

    try:
        with contextlib.ExitStack() as forced:
            for g in gl:
                forced.enter_context(g["force_block_select"](True))
            st = model.new_state(cap)
            pos, t0 = 0, time.time()
            for p in points:
                while pos < p:
                    single(st, pos)
                    pos += CHUNK
                single(st, p, check=True)
                model.sync()
                res[("single", p)] = chk.collect()
                pos = p + CHUNK
                print(f"  point {p}: {res[('single', p)]} ({time.time() - t0:.0f} s)")
            st.free()
    finally:
        for g in gl:
            g["BSEL_CHECK"][0] = None
        for t in chk.iota.values():
            ttnn.deallocate(t)
        if skip0 != "off":
            model.use_decoder_skip(skip0)

    total = 0
    for (kind, p), r in sorted(res.items()):
        metrics.record(f"idx_block_select_set_mismatches_{kind}_p{p}", r["mismatches"])
        metrics.record(f"idx_block_select_tie_rows_{kind}_p{p}", r["tie_rows"])
        metrics.record(f"idx_block_select_rows_{kind}_p{p}", r["rows"])
        metrics.record(f"idx_block_select_calls_{kind}_p{p}", r["calls"])
        total += r["mismatches"]
        assert r["calls"] > 0 and r["rows"] > 0, f"{kind} p{p}: no block top-k ran ({r})"
    metrics.record("idx_block_select_set_mismatches", total)
    print(f"block select sets: mismatches {total}; {res}")
    assert total == 0, f"block select: {total} ids outside the plain top-k beyond 512th-value ties"
