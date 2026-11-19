# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The multi-row prefill selection emulated on one chip: the shared scorer with a mesh row's chunk-start offset
(``TtDSAIndexer.row_offset_sim`` = r: ``chunk_start + r*Sq``, exactly what ``seq_shard_axes=[0]`` gives row r of
a multi-row mesh) scores a superset of the causal pools; the per-row bounded top-k must select exactly what the
exact single-chip path selects, for every sparse query row of a 16 K-token chunk, and the chunk's attention
output must be identical. Also the custom scorer (the previous multi-row path) for the cost comparison.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_select_rows.py -s --timeout 1500

Knobs: GLM53_SELECT_N (prefix curve, default 131072; the cache is allocated for the largest + 64),
GLM53_SELECT_ROWS (emulated rows, default 1,7), GLM53_SELECT_CUSTOM=0 (skip the custom scorer variant).
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tests.test_cache_1m import _x_stats, synthetic_rows
from models.demos.glm53_flash.tests.test_prefill_chunked import (
    LAYER_DSA,
    R16385,
    STAGE_GROUPS,
    IndexRecorder,
    _attention,
    _load,
    _rows_to_device,
    row_pcc,
)
from models.demos.glm53_flash.tests.utils import attention_input, log_dram
from models.demos.glm53_flash.tt.dsa import StageTimer

TILE = 32
CURVE = [int(v) for v in os.environ.get("GLM53_SELECT_N", "131072").split(",")]
ROWS = [int(v) for v in os.environ.get("GLM53_SELECT_ROWS", "1,7").split(",")]
WITH_CUSTOM = os.environ.get("GLM53_SELECT_CUSTOM", "1") == "1"
CHUNK, QBLOCK, SEED_CHUNK = 16384, 2048, 32768


def _grouped(timer: StageTimer) -> dict[str, float]:
    return {g: sum(timer.times.get(k, 0.0) for k in ks) for g, ks in STAGE_GROUPS.items()}


def _compare_ids(ref: dict[int, torch.Tensor], got: dict[int, torch.Tensor]) -> tuple[int, int, int]:
    """(rows compared, rows with a different id SET, rows with the same set in a different ORDER)."""
    diff_set = diff_order = 0
    for pos, a in ref.items():
        b = got[pos]
        if a.numel() != b.numel() or not torch.equal(a, b):
            diff_set += 1
    return len(ref), diff_set, diff_order


class OrderedRecorder(IndexRecorder):
    """The recorder keeps the valid prefix in the op's order too (bitwise order check)."""

    def __init__(self, max_pos: int):
        super().__init__(max_pos)
        self.ordered: dict[int, torch.Tensor] = {}

    def __call__(self, query_start: int, indices: ttnn.Tensor):
        t = ttnn.to_torch(indices).reshape(indices.shape[-2], indices.shape[-1]).to(torch.int64) & 0xFFFFFFFF
        for i in range(t.shape[0]):
            pos = query_start + i
            if pos >= self.max_pos:
                break
            row = t[i]
            sent = (row == 0xFFFFFFFF).nonzero()
            first = int(sent[0]) if sent.numel() else row.numel()
            self.ordered[pos] = row[:first].clone()
            self.rows[pos] = row[:first].sort().values


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_select_rows_emulated(mesh_device, monkeypatch):
    monkeypatch.setenv("GLM53_PREFILL_QBLOCK", str(QBLOCK))
    n_max = CURVE[-1]
    raw, cfg = _load(LAYER_DSA, attn_only=True)
    x_in = torch.load(R16385 / f"layer{LAYER_DSA}_prefill.pt")["x_in"][:, :CHUNK]
    rows = attention_input(raw, cfg, x_in)
    mu, sigma = _x_stats(raw, cfg)
    # headroom for the emulated rows' offsets (a mesh row's chunk start + r*Sq must lie inside the pooled-key
    # allocation, else the sub-block falls back to the custom scorer: the 8x4's last chunks, not what is tested here)
    max_seq_len = n_max + 64 + max(ROWS) * QBLOCK
    attn, cache = _attention(mesh_device, raw, cfg, max_batch=1, max_seq_len=max_seq_len)
    log_dram(mesh_device, f"attention + 1-slot cache for {max_seq_len} positions")
    timer = StageTimer(mesh_device)
    filled, ci = 0, 0
    failures = []
    for N in CURVE:
        start0 = N - CHUNK
        while filled < start0:
            T = min(SEED_CHUNK, start0 - filled)
            x_tt = ttnn.from_torch(
                synthetic_rows(mu, sigma, ci, T).reshape(1, 1, T, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
            )
            attn.prefill_state(x_tt, cache, 0, T, start=filled)
            ttnn.deallocate(x_tt)
            filled += T
            ci += 1
        ttnn.synchronize_device(mesh_device)
        variants = [("exact", 0, False)] + [(f"row{r}", r, False) for r in ROWS]
        if WITH_CUSTOM:
            variants.append(("custom", 0, True))
        results = {}
        for name, sim, custom in variants:
            monkeypatch.setenv("GLM53_PREFILL_SCORER", "custom" if custom else "auto")
            attn.indexer.row_offset_sim = sim
            rec = OrderedRecorder(N)
            attn.on_indices = rec
            outs, walls = [], []
            for variant in ("compile", "timed"):
                attn.set_stage_timer(timer if variant == "timed" else None)
                if variant == "timed":
                    timer.reset()
                x_tt = _rows_to_device(mesh_device, rows, 0, CHUNK)
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                y = attn.prefill_forward(x_tt, cache, 0, CHUNK, start=start0)
                ttnn.synchronize_device(mesh_device)
                walls.append(time.perf_counter() - t0)
                outs.append(ttnn.to_torch(y).reshape(CHUNK, -1))
                ttnn.deallocate(y)
                ttnn.deallocate(x_tt)
            attn.set_stage_timer(None)
            attn.on_indices = None
            attn.indexer.row_offset_sim = 0
            # the two passes of one variant must agree (determinism) before variants are compared
            assert torch.equal(outs[0], outs[1]), f"{name}: the compile and timed passes differ"
            results[name] = {
                "y": outs[1],
                "ids": rec.rows,
                "ordered": rec.ordered,
                "wall": walls[1],
                "stages": _grouped(timer),
            }
            g = results[name]["stages"]
            logger.info(
                f"N={N} {name}: chunk [{start0}, {N}) {walls[1]:.3f} s | kv_copy {g['kv_copy'] * 1e3:.1f} ms  "
                f"scorer {g['scorer'] * 1e3:.1f} ms  topk {g['topk'] * 1e3:.1f} ms  sparse_sdpa {g['sparse_sdpa'] * 1e3:.1f} ms"
            )
            logger.info(f"    raw: {timer.report()}")
        ref = results["exact"]
        for name, r in results.items():
            if name == "exact":
                continue
            n_rows, diff_set, _ = _compare_ids(ref["ids"], r["ids"])
            diff_order = sum(
                1
                for pos, a in ref["ordered"].items()
                if a.numel() != r["ordered"][pos].numel() or not torch.equal(a, r["ordered"][pos])
            )
            pc = row_pcc(ref["y"], r["y"])
            bit = torch.equal(ref["y"], r["y"])
            logger.info(
                f"N={N} {name} vs exact: selections {n_rows} rows, different sets {diff_set}, different order {diff_order}; "
                f"output bitwise equal {bit}, row pcc min {float(pc.min()):.6f}"
            )
            if name.startswith("row") and (diff_set or not bit):
                failures.append(
                    f"N={N} {name}: different sets {diff_set}, bitwise {bit}, pcc min {float(pc.min()):.6f}"
                )
        filled = N
        log_dram(mesh_device, f"N={N} done")
    assert not failures, failures
