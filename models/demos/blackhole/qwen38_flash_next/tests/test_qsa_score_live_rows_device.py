# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_SCORE_LIVE_ROWS on the TP=8 line: the verify score path at R live rows (gather of R rows, 32-row merge,
top-k bounded by kv_block_start) against today's path (32 rows gathered, full-width top-k), rows < R.
Run: QWEN38_FUSED_DEVICE_TEST=1 pytest <file> -s -p no:cacheprovider -o addopts=  (the TP=8 environment)."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.demo import text_demo as demo
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import device_params  # noqa: F401
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_rows

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="needs the TP=8 mesh")
BLOCKS = 8192
TOPK = 512
MASK = -3.3895313892515355e38
TP_AXIS = 1


def _rep(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


@pytest.mark.parametrize("mesh_device", [demo.MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [{**demo.DEVICE_PARAMS, "trace_region_size": 8 * 1024 * 1024}], indirect=True)
@pytest.mark.parametrize("path", ["pages", "slice"])  # program 4 on / off (the TP=8 default is the slice form)
@pytest.mark.parametrize("live", [5, 1])
@pytest.mark.parametrize("position", [300, 2000, 16000, 32700])
def test_score_live_rows_matches_today(mesh_device, live, position, path):
    mesh = mesh_device
    n_dev = mesh.get_num_devices()
    g = torch.Generator().manual_seed(7 + position)
    partial = (
        (torch.randn(n_dev, 32, BLOCKS + 32, generator=g).abs() * 4)
        .to(torch.bfloat16)
        .reshape(1, n_dev, 32, BLOCKS + 32)
    )
    local = ttnn.from_torch(
        partial,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
    )
    complete = [min((position + r + 1) // 4, BLOCKS) for r in range(32)]
    cols = torch.arange(BLOCKS).unsqueeze(0)
    mask_t = (
        torch.where(cols >= torch.tensor(complete).unsqueeze(1), MASK, 0.0).to(torch.bfloat16).reshape(1, 1, 32, BLOCKS)
    )
    mask = _rep(mesh, mask_t, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    kv_start = _rep(
        mesh, torch.full((1, 1, 1, 1), position & ~31, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    )

    def run(rows_kept):
        if path == "pages":
            return qsa_rows.score_blocks_rows_from_scores(
                local,
                mask,
                cluster_axis=TP_AXIS,
                pages=qsa_rows.score_pages,
                live_rows=None if rows_kept == 32 else rows_kept,
            )
        sliced = ttnn.slice(local, (0, 0, 0, 0), (1, 1, rows_kept, BLOCKS), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = qsa_rows.score_blocks_rows(sliced, mask, cluster_axis=TP_AXIS)
        ttnn.deallocate(sliced)
        return out

    old = run(32)
    old_ids = ttnn.experimental.topk_large_indices(old, k=TOPK)
    new = run(live)
    new_ids = ttnn.experimental.topk_large_indices(new, k=TOPK)  # the default path: full-width top-k on the new scores
    new_live_ids = ttnn.experimental.topk_large_indices(
        new, k=TOPK, valid_length_tensor=kv_start, valid_length_offset=32
    )  # QWEN38_QSA_SCORE_LIVE_TOPK=1 (opt-in)
    old_s, new_s = _dev0(old), _dev0(new)
    old_i, new_i = _dev0(old_ids).to(torch.int64), _dev0(new_ids).to(torch.int64)
    live_i = _dev0(new_live_ids).to(torch.int64)
    # every device must hold the same replicated result
    for d, (a, b) in enumerate(zip(ttnn.get_device_tensors(new), ttnn.get_device_tensors(old))):
        assert ttnn.to_torch(a)[:, :, :live].view(torch.int16).equal(ttnn.to_torch(b)[:, :, :live].view(torch.int16)), d

    assert new_s.shape == old_s.shape == (1, 1, 32, BLOCKS)
    scores_equal = torch.equal(
        old_s[:, :, :live].contiguous().view(torch.int16), new_s[:, :, :live].contiguous().view(torch.int16)
    )
    ids_equal, live_slots, live_topk = True, [], []
    for r in range(live):
        lo = min(complete[r], TOPK)
        live_slots.append(lo)
        same = torch.equal(old_i[0, 0, r, :lo], new_i[0, 0, r, :lo])
        ids_equal &= same
        # the bounded top-k: the same set of ids, the order among equal scores may differ (tie order follows the chunking)
        sc = old_s[0, 0, r].float()
        a_, b_ = old_i[0, 0, r, :lo], live_i[0, 0, r, :lo]
        set_same = sorted(a_.tolist()) == sorted(b_.tolist())
        order_same = torch.equal(a_, b_)
        score_order_same = torch.equal(sc[a_], sc[b_])
        live_topk.append((set_same, order_same, score_order_same))
    dead_finite = bool(torch.isfinite(new_s[:, :, live:].float()).all())
    print(
        f"LIVEROWS path={path} R={live} P={position} scores_equal={scores_equal} ids_equal={ids_equal} lo={live_slots} dead_finite={dead_finite} live_topk(set,order,score_order)={live_topk}",
        flush=True,
    )
    assert scores_equal
    assert ids_equal
    assert dead_finite
    assert all(t[0] and t[2] for t in live_topk)
