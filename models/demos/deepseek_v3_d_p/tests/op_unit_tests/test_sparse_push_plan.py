# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn.experimental.sparse_push_plan: the on-device sparse KV push plan must rewrite the indices to the compact slots
a host reference computes, bit-for-bit, and its table must hold the bitmap and per-owner counts the owners read."""

from pathlib import Path

import numpy as np
import pytest
import torch

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


@pytest.mark.parametrize("mesh_device", [(8, 4)], ids=["8x4"], indirect=True)
@pytest.mark.parametrize(
    "max_rows, prefix",
    [(10240, 10240), (1044480, 1044480), (1044480, 61440), (1044480, "L14"), (1044480, "L58"), (1044480, "L62")],
    ids=["10k", "1M", "1M_cap_60k_prefix", "1M_real_L14", "1M_real_L58", "1M_real_L62"],
)
def test_sparse_push_plan_matches_reference(mesh_device, max_rows, prefix):
    n_dev = mesh_device.get_num_devices()
    rows, topk, sc, p, banks = 160, 2048, 160, 12, 8
    rng = np.random.default_rng(0)
    if isinstance(prefix, str):
        from safetensors.numpy import load_file

        f = TRACE_1M / "dsa" / f"dsa_topk_indices_layer_{prefix[1:]}" / "rows_00000000_00005120.safetensors"
        if not f.exists():
            pytest.skip(f"golden top-k not staged: {f}")
        idx = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(n_dev, rows, topk)
        idx[idx < 0] = SENTINEL
    else:
        idx = _clustered_indices(rng, n_dev, rows, topk, prefix)

    want_compact = _compact_reference(idx, n_dev, sc, p, banks)
    tt_idx = ttnn.from_torch(
        torch.from_numpy(idx.reshape(1, 1, n_dev * rows, topk).astype(np.uint32).view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    # Treat the last 5120 rows as the current chunk: the plan also records each owner's rows before it.
    split = max_rows - 5120
    tt_split = ttnn.from_torch(
        torch.tensor([[[[split]]]], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

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
