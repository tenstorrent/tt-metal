# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PROTOTYPE ttnn.experimental.sparse_push_plan: the on-device sparse KV push plan must match the host planner
(sparse_kv_push.build_plan_host) bit-for-bit, and its table must hold the bitmap and counts the owners read."""

import statistics
import time

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import SENTINEL, build_plan_host

MAGIC = 0x5350504C
OWNER_OFF, BITMAP_OFF, PAGE_WORDS = 16, 64, 1024


def _layout(max_rows, sc):
    w = max_rows // 32
    nb = max_rows // sc
    prefix_off = -(-(BITMAP_OFF + w) // 16) * 16
    counts_off = -(-(prefix_off + (w + 3) // 4) // 16) * 16
    return w, nb, prefix_off, counts_off


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
    [(10240, 10240), (1044480, 1044480), (1044480, 61440)],
    ids=["10k", "1M", "1M_cap_60k_prefix"],
)
def test_sparse_push_plan_matches_host(mesh_device, max_rows, prefix):
    n_dev = mesh_device.get_num_devices()
    rows, topk, sc, p, banks = 160, 2048, 160, 12, 8
    rng = np.random.default_rng(0)
    idx = _clustered_indices(rng, n_dev, rows, topk, prefix)

    _, want_compact, _ = build_plan_host(idx, n_dev, sc, 2, p, 10**9, banks)
    tt_idx = ttnn.from_torch(
        torch.from_numpy(idx.reshape(1, 1, n_dev * rows, topk).astype(np.uint32).view(np.int32)),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )

    def run():
        return ttnn.experimental.sparse_push_plan(
            tt_idx, num_devices=n_dev, stripe_chunk=sc, max_rows=max_rows, pages_per_packet=p, banks=banks
        )

    table, compact = run()
    ttnn.synchronize_device(mesh_device)
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        run()
        ttnn.synchronize_device(mesh_device)
        samples.append(time.perf_counter() - t0)

    def host(t):
        t = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=2))
        return (t.to(torch.int64).numpy() & 0xFFFFFFFF).astype(np.uint32)

    got_compact = host(compact).reshape(n_dev, rows, topk).astype(np.int64)
    tbl = host(table).reshape(n_dev, -1)
    w, nb, prefix_off, counts_off = _layout(max_rows, sc)
    bad = []
    for d in range(n_dev):
        need = np.zeros(max_rows, dtype=bool)
        v = idx[d][idx[d] != SENTINEL]
        need[v] = True
        want_bits = np.packbits(need.reshape(-1, 32)[:, ::-1], axis=1, bitorder="big").view(">u4").reshape(-1)
        want_counts = need.reshape(nb, sc).sum(1)
        want_owner = np.bincount(np.arange(nb) % n_dev, weights=want_counts, minlength=n_dev).astype(np.int64)
        words_bits = need.reshape(nb, sc // 32, 32).sum(2)
        want_prefix = (np.cumsum(words_bits, 1) - words_bits).reshape(-1).astype(np.uint8)
        got_prefix = tbl[d, prefix_off : prefix_off + (w + 3) // 4].view(np.uint8)[:w]
        checks = {
            "header": tbl[d, 0] == MAGIC and tbl[d, 1] == n_dev and tbl[d, 2] == sc,
            "owner_counts": np.array_equal(tbl[d, OWNER_OFF : OWNER_OFF + n_dev], want_owner),
            "bitmap": np.array_equal(tbl[d, BITMAP_OFF : BITMAP_OFF + w], want_bits.astype(np.uint32)),
            "prefix": np.array_equal(got_prefix, want_prefix),
            "counts": np.array_equal(tbl[d, counts_off : counts_off + nb], want_counts),
            "compact": np.array_equal(got_compact[d], want_compact[d]),
        }
        bad += [(d, k) for k, ok in checks.items() if not ok]
    logger.info(
        f"SPARSE_PUSH_PLAN max_rows={max_rows} prefix={prefix}: {statistics.median(samples) * 1e6:.0f} us "
        f"(median of {len(samples)}, host-timed), mismatches={bad[:8]}"
    )
    assert not bad, f"device plan differs from host planner: {bad[:8]}"
