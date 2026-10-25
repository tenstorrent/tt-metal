# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.sparse_push_plan: the on-device sparse KV push plan must rewrite the indices to the compact slots
a host reference computes, bit-for-bit, and its table must hold the bitmap and per-owner counts the owners read. Its cost is
timed on traced replays (test_sparse_push_plan_perf)."""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

SENTINEL = 0xFFFFFFFF
MAGIC = 0x5350504C
# Real GLM-5.3 top-k for the last 5120-token chunk of a 1,044,480-token prompt.
TRACE_1M = Path("/mnt/weka/model-cache/scratch/zai-org/GLM-5.3-Cache/golden_traces/glm53-1020k-last5120")
OWNER_OFF, OWNER_PREFIX_OFF, BITMAP_OFF, PAGE_WORDS = 32, 64, 128, 1024


def _layout(max_rows, sc):
    """(bitmap words, blocks, table pages): the table holds the header and the bitmap."""
    w = max_rows // 32
    return w, max_rows // sc, -(-(BITMAP_OFF + w) // PAGE_WORDS)


def _compact_reference(indices, num_devices, stripe_chunk, pages_per_packet, banks):
    """Host reference of the plan's index rewrite. indices: [num_devices, rows, topk] natural positions (SENTINEL =
    masked), chip-major. A requester's rows from owner o form a run starting at base(o) = sum over o' < o of
    round_up(count(o'), P * banks), in the owner's local-row order; row k of the run is at offset k % P of packet
    k / P, whose rows sit `banks` pages apart in bank lane (k / P) % banks."""
    unit = pages_per_packet * banks
    compact = np.full(indices.shape, SENTINEL, dtype=np.int64)
    for dest in range(num_devices):
        idx = indices[dest]
        valid = idx != SENTINEL
        picked = idx[valid]
        rows = np.flatnonzero(np.bincount(picked))  # sorted distinct rows
        blk = rows // stripe_chunk
        owner, local = blk % num_devices, (blk // num_devices) * stripe_chunk + rows % stripe_chunk
        order = np.lexsort((local, owner))  # owner-major, local-row order within an owner
        counts = np.bincount(owner, minlength=num_devices)
        bases = np.concatenate(([0], np.cumsum(-(-counts // unit) * unit)))
        starts = np.concatenate(([0], np.cumsum(counts)))
        k = np.arange(len(rows)) - np.repeat(starts[:-1], counts)
        j, i = k // pages_per_packet, k % pages_per_packet
        slot_by_row = np.empty(len(rows), dtype=np.int64)
        slot_by_row[order] = np.repeat(bases[:-1], counts) + (j % banks) + (j // banks) * unit + i * banks
        lut = np.empty(rows[-1] + 1 if len(rows) else 0, dtype=np.int64)
        lut[rows] = slot_by_row
        compact[dest][valid] = lut[picked]
    return compact


def _clustered_indices(rng, n_dev, rows, topk, prefix):
    """Top-k-like picks: each chip's queries share a few hot regions, so neighbouring rows overlap."""
    centers = rng.integers(0, prefix, size=(n_dev, 1, 64, 1))
    idx = (centers + rng.integers(0, 64, size=(n_dev, rows, 64, topk // 64))) % prefix
    idx = idx.reshape(n_dev, rows, topk).astype(np.int64)
    idx[rng.random(idx.shape) < 0.05] = SENTINEL
    return idx


def _plan_inputs(mesh_device, prefix, rows, topk, chunk_start):
    """(host indices [n_dev, rows, topk], device indices, device chunk start): clustered picks within `prefix`, or a
    layer's real top-k ("L<n>")."""
    n_dev = mesh_device.get_num_devices()
    if isinstance(prefix, str):
        from safetensors.numpy import load_file

        f = TRACE_1M / "dsa" / f"dsa_topk_indices_layer_{prefix[1:]}" / "rows_00000000_00005120.safetensors"
        if not f.exists():
            pytest.skip(f"golden top-k not staged: {f}")
        idx = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(n_dev, rows, topk)
        idx[idx < 0] = SENTINEL
    else:
        idx = _clustered_indices(np.random.default_rng(0), n_dev, rows, topk, prefix)
    common = dict(
        dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_idx = ttnn.from_torch(
        torch.from_numpy(idx.reshape(1, 1, n_dev * rows, topk).astype(np.uint32).view(np.int32)),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
        **common,
    )
    tt_split = ttnn.from_torch(
        torch.tensor([[[[chunk_start]]]], dtype=torch.int32),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        **common,
    )
    return idx, tt_idx, tt_split


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "max_rows, prefix",
    [(10240, 10240), (1044480, 1044480), (1044480, 61440), (1044480, "L14"), (1044480, "L58"), (1044480, "L62")],
    ids=["10k", "1M", "1M_cap_60k_prefix", "1M_real_L14", "1M_real_L58", "1M_real_L62"],
)
# Slot layout: packets spread over the 8 DRAM banks, or on consecutive slots (the push's L1 front and its spill).
@pytest.mark.parametrize("banks", [8, 1], ids=["dram_banks", "consecutive"])
def test_sparse_push_plan_matches_reference(mesh_device, max_rows, prefix, banks):
    n_dev = mesh_device.get_num_devices()
    rows, topk, sc, p = 160, 2048, 160, 12
    # Treat the last 5120 rows as the current chunk: the plan also records each owner's rows before it.
    split = max_rows - 5120
    idx, tt_idx, tt_split = _plan_inputs(mesh_device, prefix, rows, topk, split)
    want_compact = _compact_reference(idx, n_dev, sc, p, banks)

    table, compact = ttnn.experimental.sparse_push_plan(
        tt_idx,
        num_devices=n_dev,
        stripe_chunk=sc,
        max_rows=max_rows,
        pages_per_packet=p,
        banks=banks,
        chunk_start=tt_split,
    )

    def host(t):
        t = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
        return (t.to(torch.int64).numpy() & 0xFFFFFFFF).astype(np.uint32)

    got_compact = host(compact).reshape(n_dev, rows, topk).astype(np.int64)
    tbl = host(table).reshape(n_dev, -1)
    w, nb, pages = _layout(max_rows, sc)
    assert tuple(table.shape) == (1, 1, pages, PAGE_WORDS), f"table shape {table.shape}"
    bad = []
    for d in range(n_dev):
        need = np.zeros(max_rows, dtype=bool)
        v = idx[d][idx[d] != SENTINEL]
        need[v] = True
        want_bits = np.packbits(need.reshape(-1, 32)[:, ::-1], axis=1, bitorder="big").view(">u4").reshape(-1)
        want_counts = need.reshape(nb, sc).sum(1)
        want_owner = np.bincount(np.arange(nb) % n_dev, weights=want_counts, minlength=n_dev).astype(np.int64)
        sb = split // sc
        want_before = np.bincount(np.arange(sb) % n_dev, weights=want_counts[:sb], minlength=n_dev).astype(np.int64)
        # Only blocks up to the largest referenced row are written; the rest of the table may be stale.
        ab = -(-(int(v.max()) + 1) // sc) if v.size else 0
        aw = ab * (sc // 32)
        checks = {
            "header": tbl[d, 0] == MAGIC and tbl[d, 1] == n_dev and tbl[d, 2] == sc and tbl[d, 3] == ab,
            "owner_counts": np.array_equal(tbl[d, OWNER_OFF : OWNER_OFF + n_dev], want_owner),
            "owner_prefix": tbl[d, 4] == sb
            and np.array_equal(tbl[d, OWNER_PREFIX_OFF : OWNER_PREFIX_OFF + n_dev], want_before),
            "bitmap": np.array_equal(tbl[d, BITMAP_OFF : BITMAP_OFF + aw], want_bits[:aw].astype(np.uint32)),
            "compact": np.array_equal(got_compact[d], want_compact[d]),
        }
        bad += [(d, k) for k, ok in checks.items() if not ok]
    assert not bad, f"device plan differs from the host reference: {bad[:8]}"


HOT_MAGIC = 0x484F5431


def _hot_reference(indices, num_devices, stripe_chunk, pages_per_packet, banks, threshold, caps, split):
    """Host reference of sparse_push_hot. A row is hot when at least `threshold` chips need it; each owner's hot rows
    of the prefix part (blocks before the split block) and of the current part form sub-runs in a hot region at the
    front of every chip's compact buffer, identical on every chip: owner-major, part 0 then part 1, each padded to
    whole bank-lane units and holding at most caps[part] rows (an owner's first ones, in local-row order). Behind the
    region every requester keeps the plan's owner runs (one slot per needed row), now holding its cold rows (needed,
    not in the region): those of the current part first, then those of the prefix part, each in local-row order.
    Returns (hot slots H, bases [num_devices, 2], hot row mask, final compact indices)."""
    unit = pages_per_packet * banks
    valid = indices != SENTINEL
    rows_needed = np.zeros(int(indices[valid].max()) + 1, dtype=np.int32)
    for d in range(num_devices):
        rows_needed[np.unique(indices[d][indices[d] != SENTINEL])] += 1
    nrows = rows_needed.size
    row = np.arange(nrows)
    blk = row // stripe_chunk
    owner, part = blk % num_devices, (blk >= split // stripe_chunk).astype(np.int64)
    hot0 = rows_needed >= threshold
    rank = np.zeros(nrows, dtype=np.int64)
    hot = np.zeros(nrows, dtype=bool)
    counts = np.zeros((num_devices, 2), dtype=np.int64)
    for o in range(num_devices):
        for q in range(2):
            sel = np.flatnonzero(hot0 & (owner == o) & (part == q))  # ascending natural row = local-row order
            rank[sel] = np.arange(sel.size)
            keep = sel[: caps[q]]
            hot[keep] = True
            counts[o, q] = keep.size
    bases = np.zeros((num_devices, 2), dtype=np.int64)
    acc = 0
    for o in range(num_devices):
        for q in range(2):
            bases[o, q] = acc
            acc += -(-counts[o, q] // unit) * unit
    H = acc
    lane = (
        lambda r: (r // pages_per_packet % banks)
        + (r // pages_per_packet // banks) * unit
        + (r % pages_per_packet) * banks
    )
    out = np.full(indices.shape, SENTINEL, dtype=np.int64)
    for d in range(num_devices):
        ok = indices[d] != SENTINEL
        v = indices[d][ok]
        need = np.unique(v)
        needed_counts = np.bincount(owner[need], minlength=num_devices)
        run_base = np.concatenate(([0], np.cumsum(-(-needed_counts // unit) * unit)))[:-1]
        slot = np.zeros(nrows, dtype=np.int64)
        hv = need[hot[need]]
        slot[hv] = bases[owner[hv], part[hv]] + lane(rank[hv])
        for o in range(num_devices):
            cold = need[(owner[need] == o) & ~hot[need]]
            ordered = np.concatenate((cold[part[cold] == 1], cold[part[cold] == 0]))
            slot[ordered] = H + run_base[o] + lane(np.arange(ordered.size))
        flat = out[d][ok]
        flat[:] = slot[v]
        out[d][ok] = flat
    return H, bases, hot, out


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "max_rows, prefix",
    [(10240, 10240), (1044480, 61440), (1044480, "L14"), (1044480, "L62")],
    ids=["10k", "1M_cap_60k_prefix", "1M_real_L14", "1M_real_L62"],
)
@pytest.mark.parametrize("banks", [8, 1], ids=["dram_banks", "consecutive"])
# A low threshold and small caps exercise the per-owner caps and every part's sub-run.
@pytest.mark.parametrize("threshold, prefix_cap", [(12, 2048), (4, 96)], ids=["t12", "t4_cap96"])
def test_sparse_push_hot_matches_reference(mesh_device, max_rows, prefix, banks, threshold, prefix_cap):
    n_dev = mesh_device.get_num_devices()
    rows, topk, sc, p = 160, 2048, 160, 8
    split = max_rows - 5120
    idx, tt_idx, tt_split = _plan_inputs(mesh_device, prefix, rows, topk, split)
    plan_kw = dict(num_devices=n_dev, stripe_chunk=sc, max_rows=max_rows, pages_per_packet=p, banks=banks)
    table, compact = ttnn.experimental.sparse_push_plan(tt_idx, **plan_kw, chunk_start=tt_split)
    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import _shard_over_mesh

    _shard_over_mesh(table, mesh_device)
    tables = ttnn.from_torch(
        torch.zeros(1, 1, table.shape[2] * n_dev, table.shape[3], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.experimental.high_bw_all_gather(table, dim=2, output_tensor=tables, cluster_axis=None, num_links=2)
    compact_cols = lambda t: (
        ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2)).to(torch.int64).numpy() & 0xFFFFFFFF
    ).reshape(n_dev, rows, topk)
    current_cap = 2 * sc
    hot_table = ttnn.experimental.sparse_push_hot(
        tt_idx, compact, table, tables, **plan_kw, threshold=threshold, prefix_cap=prefix_cap, chunk_start=tt_split
    )
    want_H, want_bases, want_hot, want_compact = _hot_reference(
        idx, n_dev, sc, p, banks, threshold, (prefix_cap, current_cap), split
    )
    got_compact = compact_cols(compact)
    hot_host = (
        ttnn.to_torch(hot_table, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2)).to(torch.int64).numpy()
        & 0xFFFFFFFF
    ).reshape(n_dev, -1)
    w = max_rows // 32
    mask = np.zeros(w * 32, dtype=bool)
    mask[: want_hot.size] = want_hot
    want_words = np.packbits(mask.reshape(-1, 32)[:, ::-1], axis=1, bitorder="big").view(">u4").reshape(-1)
    v = idx[idx != SENTINEL]
    aw = -(-(int(v.max()) + 1) // sc) * (sc // 32)
    bad = []
    for d in range(n_dev):
        checks = {
            "header": hot_host[d, 0] == HOT_MAGIC and hot_host[d, 1] == want_H and hot_host[d, 3] * (sc // 32) == aw,
            "bases": np.array_equal(hot_host[d, OWNER_OFF : OWNER_OFF + 2 * n_dev], want_bases.reshape(-1)),
            # Only words of the active blocks are written; later ones may be stale.
            "hot_mask": np.array_equal(hot_host[d, BITMAP_OFF : BITMAP_OFF + aw], want_words[:aw].astype(np.int64)),
            "compact": np.array_equal(got_compact[d], want_compact[d]),
        }
        bad += [(d, k) for k, ok in checks.items() if not ok]
    logger.info(f"hot slots {want_H}, hot rows {int(want_hot.sum())} (threshold {threshold}, prefix cap {prefix_cap})")
    assert not bad, f"device hot plan differs from the host reference: {bad[:8]}"


# Cache capacity (max_rows) and populated prefix of the perf cases: GLM-5.3's 1M cache filled to a depth (clustered
# picks), and the real 1M top-k of a light (L14) and a heavy (L62) layer.
_PERF_CASES = [
    (1044480, 10240),
    (1044480, 102400),
    (1044480, 512000),
    (1044480, 1044480),
    (1044480, "L14"),
    (1044480, "L62"),
    (102400, 102400),
    (10240, 10240),
]


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="local production-shape perf qualification")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "max_rows, prefix",
    _PERF_CASES,
    ids=["1M_cap_10k", "1M_cap_100k", "1M_cap_500k", "1M_cap_1M", "1M_real_L14", "1M_real_L62", "100k", "10k"],
)
def test_sparse_push_plan_perf(mesh_device, max_rows, prefix):
    """The plan op alone, as GLM-5.3 runs it (5k chunk over 32 chips, top-k 2048, FP8 704 B pages: 8 per packet, on
    consecutive slots for the L1 front). Mean time per call over back-to-back replays of a trace of `reps` calls, so
    the per-replay host cost is amortised as it is inside the model's layer trace."""
    n_dev = mesh_device.get_num_devices()
    rows, topk, sc, p, n, reps = 160, 2048, 160, 8, 20, 10
    # The current chunk is the populated prefix's last 5120 rows; the real top-k is of a 1M prompt.
    populated = max_rows if isinstance(prefix, str) else prefix
    _, tt_idx, tt_split = _plan_inputs(mesh_device, prefix, rows, topk, populated - 5120)
    plan = lambda: ttnn.experimental.sparse_push_plan(
        tt_idx, num_devices=n_dev, stripe_chunk=sc, max_rows=max_rows, pages_per_packet=p, banks=1, chunk_start=tt_split
    )
    plan()
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    outputs = [plan() for _ in range(reps)]  # held until the trace is released: replays write these buffers
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / (n * reps) * 1e6
    ttnn.release_trace(mesh_device, tid)
    del outputs
    logger.info(f"sparse_push_plan max_rows={max_rows} prefix={prefix}: {us:.1f} us per call (traced)")
