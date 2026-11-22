# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.BM (OWNER 2026-11-21): the indexer's top-k by block-max pre-selection (INDEXER_BLOCK_SELECT=on) selects the SAME
key sets as the plain top-k (off), up to ties at the 512th value, on real text at long context, single and packed.

One request of real text (BRINGUP_POS_TOKENS's text, agentic_cc by default; repeated past its end) is prefilled from 0
through layers 0 .. the last plain index source of the load (DeepSeek-V4.1: 0-14; index sources 2, 8, 14), chunk 8192,
today's single-request path, every indexer call on the block path (force_block_select(True): whatever
INDEXER_BLOCK_SELECT_MIN). At each point P (IDX_BSEL_POS, default 139264 and 1024000):

    packed   a chip-aligned 4 x 2k packed chunk (PACKED=on: the glued one-call path): the request's segment
             [P - 2048, P) + three fresh requests [0, 2048) of other text (their own slots)
    single   the request's chunk [P, P + 8192) (PACKED=off)

and for every block top-k of those chunks (every plain indexer layer, every chip, every query row, the rows each chip
scored), beside it on the same scores: the plain top-k (topk_large_indices over the keys' valid length, the off path)
and the scores at both id lists (ttnn.bringup.block_gather, block 1). On the host, per row: the two id sets (the
0xFFFFFFFF tail dropped); when they differ, their sorted score lists: equal = the sets differ only by keys tied at the
512th value (a tie row, counted); else every id outside the other set is a mismatch.

Metrics (gated ``idx_block_select_set_mismatches*`` == 0): idx_block_select_set_mismatches_<single|packed>_p<P> and
their sum idx_block_select_set_mismatches; idx_block_select_tie_rows_<kind>_p<P> (rows differing by 512th-value ties
only), idx_block_select_rows_<kind>_p<P> (rows compared), idx_block_select_calls_<kind>_p<P>.

Load: the accuracy load (BATCH_SHARED); the request's state holds P_max + 8192 positions, so the load's tables must
reach it (BATCH_MAX_SEQ, core/batch.hints; at most the deployment's 1,044,480).

    PYTHONPATH=$PWD BRINGUP_CHUNK=8192 DS41_PACKED=on ... scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_indexer_block_select.py
"""

from __future__ import annotations

import importlib
import os
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
CHUNK, PAD_ID, packed = _pc.CHUNK, _pc.PAD_ID, _pc.packed

S = spec()
pytestmark = device_timeout(S)
BATCH_SHARED = True
BATCH_ITEM_VARS = ("IDX_BSEL_POS",)
BATCH_MAX_SEQ = 1032192  # (the request's state: 1024000 + a chunk)
SENT = 0xFFFFFFFF
SEG = 2048


def _points() -> list[int]:
    v = os.environ.get("IDX_BSEL_POS", "").strip()
    pts = [int(x) for x in v.split(",") if x.strip()] if v else [139264, 1024000]
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
def test_indexer_block_select(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    import ttnn
    from models.demos.common.bringup.reference import texts
    from models.demos.deepseek_v41_flash_d_p.tt import indexer as IX
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import run_chunk

    points = _points()
    model = S.hooks().device_model(mesh_device, S, list(S.layers()), lm_head=False)
    assert model.chunk == CHUNK, model.chunk
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
    ids, info = texts.tokens(S, cap + 4 * SEG, texts.explicit() or "agentic_cc")
    print(f"block select sets: layers {layers} (index {idx_layers}), points {points}, text {info}")

    # (DS41-311: the held model's indexers may be instances of a class of an earlier module object than the import
    # above, after the runner reloaded tt/indexer.py in place: their hooks are their own module's globals)
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

    def single(st, c0, end, check=None):
        toks = torch.full((CHUNK,), PAD_ID, dtype=torch.long)
        toks[: end - c0] = ids[c0:end]
        chk.on = check is not None
        with packed("off"):
            run_chunk(model, ChunkDescriptor.single(CHUNK, 0, c0, end), toks, [st], layers)
        chk.on = False

    try:
        with contextlib.ExitStack() as forced:
            for g in gl:
                forced.enter_context(g["force_block_select"](True))
            st = model.new_state(cap)
            pos, t0 = 0, time.time()
            for p in points:
                while pos + CHUNK <= p - CHUNK:
                    single(st, pos, pos + CHUNK)
                    pos += CHUNK
                if pos < p - SEG:
                    single(st, pos, p - SEG)  # (a partial chunk: the packed segment ends at p)
                    pos = p - SEG
                model.sync()
                # packed: the request's segment [p - 2048, p) + three fresh requests of other text, 4 x 2k chip-aligned
                others = [model.new_state(cap) for _ in range(3)]  # (one slot capacity a packed chunk: DS41-328)
                segs = [Segment(0, pos, p)] + [Segment(k + 1, 0, SEG) for k in range(3)]
                lane_ids = [ids[pos:p]] + [ids[cap + k * SEG : cap + (k + 1) * SEG] for k in range(3)]
                desc = ChunkDescriptor.packed(CHUNK, segs)
                chk.on = True
                with packed("on"):
                    run_chunk(model, desc, desc.pack_ids(lane_ids), [st] + others, layers)
                chk.on = False
                model.sync()
                res[("packed", p)] = chk.collect()
                for o in others:
                    o.free()
                pos = p
                single(st, p, p + CHUNK, check=True)
                model.sync()
                res[("single", p)] = chk.collect()
                pos = p + CHUNK
                print(f"  point {p}: {res[('packed', p)]} / {res[('single', p)]} ({time.time() - t0:.0f} s)")
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
