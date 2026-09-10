# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""vsa_sdpa timed on REAL selections: a VSA_DUMP_INDICES dump (models/.../attention_minimax_h3.py) gives one
device's raw top-k rows; q/k/v are random at the real 15 s / 768p shape. Knobs: VSA_REAL_DUMP (file),
VSA_REAL_DEV (device index, default 5), VSA_ORDERS (comma list of identity|canonical|zorder|stride|bstrideR.S), TT_VSA_RMAX/DEPTH.
Run: ./scripts/run_safe_pytest.sh <this file> -q -s"""

import os
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_pcc, skip_for_wormhole_b0
import math

GRID = (107, 24, 42)  # 15 s / 768p latent token grid (27 x 6 x 11 cubes = 1782 video tiles)


def _stream_order(tile_ids, n_prefix, grid, kind):
    """Same construction as MiniMaxH3VSAGeometry.stream_order, from the dump's metadata."""
    n = tile_ids.numel()
    if kind in ("identity", "identity_perm"):
        return torch.arange(n, dtype=torch.long)
    if kind == "reversed":
        return torch.arange(n, dtype=torch.long).flip(0)
    hh, wh = math.ceil(grid[1] / 4), math.ceil(grid[2] / 4)
    keys = []
    for slot in range(n):
        c = int(tile_ids[slot])
        if c < n_prefix:
            keys.append((0, slot))
            continue
        if kind == "stride":
            # de-clustering order: spatial neighbours land far apart in the stream, so a core's listed
            # blocks (spatially clustered per row) spread evenly over the arrivals (977 is coprime to n)
            keys.append((1, (slot * 977) % n))
            continue
        if kind.startswith("bstride"):
            # blocked stride "bstrideR.S": runs of R consecutive blocks keep multi-block visits, and the runs
            # of S equal spatial segments are interleaved so one window carries work for several workers
            r_len, n_seg = (int(v) for v in kind[len("bstride") :].split("."))
            seg_len = (n + n_seg - 1) // n_seg
            seg, off = slot // seg_len, slot % seg_len
            keys.append((1, (off // r_len) * n_seg * r_len + seg * r_len + (off % r_len)))
            continue
        v = c - n_prefix
        ct, ch, cw = v // (hh * wh), (v // wh) % hh, v % wh
        if kind == "canonical":
            keys.append((1, v))
        else:
            m = 0
            for i in range(10):
                m |= ((ct >> i) & 1) << (3 * i) | ((ch >> i) & 1) << (3 * i + 1) | ((cw >> i) & 1) << (3 * i + 2)
            keys.append((1, m))
    return torch.tensor(sorted(range(n), key=lambda i: keys[i]), dtype=torch.long)


def _bench(device, fn, iters=8):
    out = fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / iters * 1e3, out


@skip_for_wormhole_b0("vsa_sdpa is Blackhole-only")
def test_vsa_sdpa_real(device):
    dump = os.environ.get("VSA_REAL_DUMP")
    if not dump:
        pytest.skip("set VSA_REAL_DUMP to a vsa_indices_call*.pt dump")
    D = torch.load(dump)
    dev = int(os.environ.get("VSA_REAL_DEV", "5"))
    idx = D["indices"][dev]  # [1, H, rows, W] int32 (raw top-k, padded numbering)
    H, rows, W = idx.shape[1], idx.shape[2], idx.shape[3]
    k, shift, tps, sp = D["k"], D["coarse_slots_shift"], D["tiles_per_shard"], D["sp_factor"]
    tile_ids, n_prefix = D["tile_ids"], D["n_prefix_tiles"]
    n_tiles = tile_ids.numel()
    assert rows == tps
    dim, blk = 128, 64
    torch.manual_seed(0)
    q = torch.randn(1, H, rows * blk, dim, dtype=torch.bfloat16)
    kv_len = n_tiles * blk
    kt = torch.randn(1, H, kv_len, dim, dtype=torch.bfloat16)
    vt = torch.randn(1, H, kv_len, dim, dtype=torch.bfloat16)
    counts = torch.where(tile_ids >= 0, torch.tensor(blk), torch.tensor(0)).to(torch.int32).reshape(1, 1, 1, n_tiles)
    tt = lambda t, lay, dt: ttnn.from_torch(t, device=device, layout=lay, dtype=dt)
    tt_q, tt_k, tt_v = (tt(x, ttnn.TILE_LAYOUT, ttnn.bfloat16) for x in (q, kt, vt))
    tt_idx = tt(idx.to(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    tt_counts = tt(counts, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    # dense-row mask for THIS device: exempt (dense-list) query tiles of shard `dev % sp` (sp axis is the
    # 8-wide mesh axis 1 in the model: device d -> shard d % 8)
    shard = dev % sp
    row_exempt = D["is_exempt"].reshape(sp, tps)[shard]
    dense_rows = torch.nonzero(row_exempt).reshape(-1).tolist()
    words = max(8, (tps + 31) // 32)
    words = (words + 7) // 8 * 8
    mask = torch.zeros(words, dtype=torch.int64)
    for r in dense_rows:
        mask[r // 32] |= 1 << (r % 32)
    tt_mask = tt(mask.to(torch.int32).reshape(1, 1, 1, words), ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
    common = dict(
        list_len=k,
        exempt_ids=[int(b) for b in D["exempt_ids"]],
        dense_row_mask=tt_mask,
        coarse_slots_shift=shift,
        coarse_real_per_shard=tps,
        dense_row_hint=[int(r) for r in D["dense_rows"]],
    )
    print(
        f"\nREAL dump={os.path.basename(dump)} dev={dev} shard={shard} H={H} rows={rows} k={k} dense_rows={dense_rows}"
    )
    ref = None
    for order in os.environ.get("VSA_ORDERS", "identity,zorder").split(","):
        kw = dict(common)
        if order != "identity":
            perm = _stream_order(tile_ids, n_prefix, GRID, order).to(torch.int32).reshape(1, 1, 1, -1)
            kw["stream_order"] = tt(perm, ttnn.ROW_MAJOR_LAYOUT, ttnn.uint32)
        ms, out = _bench(device, lambda: ttnn.transformer.vsa_sdpa(tt_q, tt_k, tt_v, tt_idx, tt_counts, **kw))
        out_t = ttnn.to_torch(out).float()
        if ref is None:
            ref = out_t
            note = "(reference)"
        else:
            ok, pcc = comp_pcc(ref, out_t, 0.999)
            note = f"pcc_vs_identity={pcc:.6f}"
        print(f"REAL order={order:10s} {ms:8.3f} ms  {note}")
        ttnn.deallocate(out)
