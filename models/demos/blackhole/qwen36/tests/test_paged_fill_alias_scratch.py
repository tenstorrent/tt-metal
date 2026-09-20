# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kernel-level documentation of the p1b 63/64-token nondeterminism, plus the host guard that prevents it.

ONE ttnn.experimental.paged_fill_cache call whose fixed-width page table names the SAME physical block twice ([b, b]:
the fill table the old trusted-tail rule built from the row vLLM hands a 1-block prompt whose stale tail aliases its
own block). Rows 0..63 (value 1.0, the "real" K rows) and rows 64..127 (value 2.0, the bucket's pad rows) are both
written to block b; which one lands last is a multi-core write race (measured 47-69% of the block holding the pad
rows, different every run). The kernel is not at fault -- a page table naming one block twice is a caller error --
so this test REPORTS the aliased fill and asserts (a) a distinct-block fill keeps the real rows and (b) the host-side
fill_pt_row never builds such a row from the stale vLLM row (default rule pads with the scratch block; the rollback
rule trips the alias guard). Run on one die:
  pytest -svq models/demos/blackhole/qwen36/tests/test_paged_fill_alias_scratch.py
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.masked_bucket_trace import fill_pt_row

NKV, BLOCK, HD, NBLK = 8, 64, 128, 16


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 1), id="1x1")], indirect=True)
def test_paged_fill_same_block_twice(mesh_device):
    reps = int(os.environ.get("P1B_REPEATS", "20"))
    dtype = ttnn.bfloat8_b if os.environ.get("QWEN_SDPA_BF8", "1") == "1" else ttnn.bfloat16
    cache = ttnn.as_tensor(
        torch.zeros(NBLK, NKV, BLOCK, HD, dtype=torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    fill = torch.ones(1, NKV, 128, HD, dtype=torch.bfloat16)
    fill[:, :, 64:, :] = 2.0
    fill_t = ttnn.from_torch(fill, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    b = 5
    results = {"alias": [], "distinct": []}
    for name, row in (("alias", [b, b]), ("distinct", [b, b + 1])):
        pt = ttnn.from_torch(
            torch.tensor([row], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
        )
        for r in range(reps):
            ttnn.experimental.paged_fill_cache(cache, fill_t, pt, batch_idx=0)
            blk = ttnn.to_torch(ttnn.slice(cache, (b, 0, 0, 0), (b + 1, NKV, BLOCK, HD))).float()
            frac2 = float((blk == 2.0).float().mean())  # fraction of block b holding the PAD rows' value
            results[name].append(round(frac2, 4))
        ttnn.deallocate(pt)
    logger.info(
        f"[p1b-fill] page table [b, b]  -> fraction of block b overwritten by pad rows per run: {results['alias']}"
    )
    logger.info(f"[p1b-fill] page table [b, b+1] -> {results['distinct']}")
    assert all(f == 0.0 for f in results["distinct"]), "distinct blocks must keep the real rows"
    # The fixed rule's MANY-pad-entries shape ([real, pad, pad, ..., pad]: bucket 512 = 8 blocks, 1 real): 7 pad
    # entries name the scratch block in ONE paged_fill_cache; the real block must stay untouched every run.
    pad = NBLK - 1
    fill8 = torch.ones(1, NKV, 8 * BLOCK, HD, dtype=torch.bfloat16)
    fill8[:, :, BLOCK:, :] = 2.0
    fill8_t = ttnn.from_torch(fill8, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    pt8 = ttnn.from_torch(
        torch.tensor([[b] + [pad] * 7], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
    )
    results["multipad"] = []
    for r in range(reps):
        ttnn.experimental.paged_fill_cache(cache, fill8_t, pt8, batch_idx=0)
        blk = ttnn.to_torch(ttnn.slice(cache, (b, 0, 0, 0), (b + 1, NKV, BLOCK, HD))).float()
        results["multipad"].append(round(float((blk == 2.0).float().mean()), 4))
    logger.info(
        f"[p1b-fill] page table [b, pad x7] -> fraction of block b holding pad rows per run: {results['multipad']}"
    )
    assert all(
        f == 0.0 for f in results["multipad"]
    ), "7 pad entries on the scratch block must not touch the real block"
    # The aliased fill is the caller's bug: report what the kernel does with it (a race: typically clobbers, varies).
    clobbered = any(f > 0.0 for f in results["alias"])
    varies = len(set(results["alias"])) > 1
    logger.info(f"[p1b-fill] aliased fill clobbers real rows: {clobbered}; varies across runs: {varies}")
    # The host guard: the stale vLLM row [b, b, 0, ...] for a 1-block prompt never becomes the fill table [b, b].
    stale_row = torch.zeros(1, NBLK, dtype=torch.int32)
    stale_row[0, :2] = b
    assert fill_pt_row(stale_row, 0, 64, 128, pad, BLOCK).tolist() == [[b, pad]]
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only guard
        fill_pt_row(stale_row, 0, 64, 128, pad, BLOCK, trust_tail=True)
