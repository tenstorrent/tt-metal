# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.BM (OWNER 2026-11-21): op-level probe of the indexer's block pre-selected top-k (INDEXER_BLOCK_SELECT), mesh only
(the resident runner's mesh: 0 opens, 0 model loads).

(1) ttnn.bringup.block_gather / block_keys byte-exact vs torch (the fork's unit check, check_block_select).
(2) At the model's per-chip shapes (BSEL_ROWS, default the spec's chunk / the mesh's chips: 5120 on 8x4 -> 160 query
    rows a chip, MB.6; 8192 -> 256, PB.BM; ratio-2 keys: T = h / 2), random bf16
    scores with a causal -inf tail and tied runs: the selected SETS of the plain top-k (topk_large_indices k 512 over
    the T keys) vs the block path (block max fp32 -> bf16, top-512 blocks, block_gather, top-512 of the K B gathered,
    block_keys) are equal up to ties at the 512th value (checked on the host from the scores) for block 8 and 32,
    (BSEL_BLOCKS, default 8,32) and the device time (host-timed, synced, median of BSEL_ITERS) of each step. The block max itself comes from the
    score op's writer (block_max_out) in the model; here it is made on the host (its cost is the ladder's).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=5120 \\
        scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_block_select_probe.py"
"""

from __future__ import annotations

import json
import os
import statistics
import time

import pytest
import torch

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"
BATCH_ITEM_VARS = ("BSEL_ITERS", "BSEL_T", "BSEL_OUT", "BSEL_ROWS", "BSEL_BLOCKS")
SENT = 0xFFFFFFFF


def _u32(t):
    return t.to(torch.int64) & 0xFFFFFFFF


def _sets_check(s: torch.Tensor, plain: torch.Tensor, block: torch.Tensor, k: int) -> tuple[int, int]:
    """(rows whose sets differ beyond 512th-value ties, rows that differ only by such ties) of one chip's rows."""
    bad = ties = 0
    sf = s.float()
    for r in range(s.shape[0]):
        a, b = _u32(plain[r]), _u32(block[r])
        a, b = a[a != SENT], b[b != SENT]
        if torch.equal(torch.sort(a).values, torch.sort(b).values):
            continue
        va = torch.sort(sf[r, a], descending=True).values
        vb = torch.sort(sf[r, b], descending=True).values
        if torch.equal(va, vb):
            ties += 1
        else:
            bad += 1
    return bad, ties


def _time(fn, iters: int, mesh, reps: int = 8) -> float:
    """ms a call: ``reps`` calls back to back between two syncs (the device's pipelined time when device-bound),
    the median over ``iters``."""
    import ttnn

    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(reps)]
        ttnn.synchronize_device(mesh)
        ts.append((time.perf_counter() - t0) * 1e3 / reps)
        for out in outs:
            for o in out if isinstance(out, (list, tuple)) else [out]:
                ttnn.deallocate(o)
    return statistics.median(ts)


@mesh_parametrize(timeout_s=1800)
def test_block_select_probe(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from ttnn.bringup.block_select_ttnn.tests.unit.test_block_select import check_block_select

    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import cores
    from models.demos.deepseek_v41_flash_d_p.tt.cores import mm_cores

    mesh = mesh_device
    print("check_block_select:", check_block_select(mesh))
    iters = int(os.environ.get("BSEL_ITERS", "5"))
    widths = [int(x) for x in os.environ.get("BSEL_T", "69632,512000").split(",")]
    from models.demos.common.bringup.testing.harness import spec

    chips = int(mesh.shape[0]) * int(mesh.shape[1])
    rows = int(os.environ.get("BSEL_ROWS", "0") or 0) or int(spec().data["target"]["chunk"]) // chips
    blocks = [int(x) for x in os.environ.get("BSEL_BLOCKS", "8,32").split(",")]
    k = 512
    print(f"block select probe: {rows} rows a chip, widths {widths}, blocks {blocks}")
    res = {}

    def up(t, dtype):
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def topk(x, **kw):
        return ttnn.bringup.topk_large_indices(x, k=k, sub_core_grids=mm_cores(x), **kw)

    g = torch.Generator().manual_seed(1)
    for T in widths:
        s = (torch.randn(1, 1, rows, T, generator=g) * 2.0).to(torch.bfloat16)
        # causal tails: row i sees T - (rows - i) * 16 keys (the chunk's newest queries see the most)
        vis = T - (rows - torch.arange(rows)) * 16
        s[0, 0][torch.arange(T).view(1, -1) >= vis.view(-1, 1)] = float("-inf")
        s[0, 0, :, ::5] = 1.0  # a run of ties well above the 512th value
        st = up(s, ttnn.bfloat16)
        nb8 = T // 8
        nbm8 = -(-nb8 // 32) * 32
        bm8 = torch.full((1, 1, rows, nbm8), float("-inf"))
        bm8[..., :nb8] = s.float().view(1, 1, rows, nb8, 8).amax(-1)
        bmt = up(bm8, ttnn.float32)  # (the score writer's block_max_out, block_keys 8)
        for B in blocks:
            nb = T // B
            F = B // 8

            def cast():
                return ttnn.bringup.block_max_bf16(bmt, factor=F, n=nb)

            plain = topk(st, valid_length=T)

            def chain(keep=False):
                bf = cast()
                ids = topk(bf, valid_length=nb)
                gg = ttnn.bringup.block_gather(st, ids, block=B)
                pos = topk(gg)
                keys = ttnn.bringup.block_keys(pos, ids, block=B, width=T)
                outs = [bf, ids, gg, pos]
                if keep:
                    for o in outs:
                        ttnn.deallocate(o)
                    return keys
                return outs + [keys]

            keys = chain(keep=True)
            pk = ttnn.to_torch(ttnn.get_device_tensors(plain)[0])[0, 0]
            bk = ttnn.to_torch(ttnn.get_device_tensors(keys)[0])[0, 0]
            bad, ties = _sets_check(s[0, 0], pk, bk, k)
            chips_eq = all(torch.equal(ttnn.to_torch(t)[0, 0], bk) for t in ttnn.get_device_tensors(keys)[1:4])
            ttnn.deallocate(plain)
            ttnn.deallocate(keys)
            t_plain = _time(lambda: topk(st, valid_length=T), iters, mesh)
            t_chain = _time(chain, iters, mesh)
            bf = cast()
            ids = topk(bf, valid_length=nb)
            gg = ttnn.bringup.block_gather(st, ids, block=B)
            pos = topk(gg)
            t_cast = _time(cast, iters, mesh)
            t_cast3 = _time(
                lambda: cores.typecast(bmt, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG), iters, mesh
            )
            t_tb = _time(lambda: topk(bf, valid_length=nb), iters, mesh)
            t_g = _time(lambda: ttnn.bringup.block_gather(st, ids, block=B), iters, mesh)
            t_t2 = _time(lambda: topk(gg), iters, mesh)
            t_k = _time(lambda: ttnn.bringup.block_keys(pos, ids, block=B, width=T), iters, mesh)
            for o in (bf, ids, gg, pos):
                ttnn.deallocate(o)
            r = dict(
                T=T,
                block=B,
                rows=rows,
                set_mismatch_rows=bad,
                tie_rows=ties,
                chips_equal=chips_eq,
                plain_ms=round(t_plain, 3),
                chain_ms=round(t_chain, 3),
                cast_ms=round(t_cast, 3),
                typecast3_ms=round(t_cast3, 3),
                topk_blocks_ms=round(t_tb, 3),
                gather_ms=round(t_g, 3),
                topk_gathered_ms=round(t_t2, 3),
                keys_ms=round(t_k, 3),
            )
            print("BSEL", json.dumps(r))
            res[f"T{T}_B{B}"] = r
            assert bad == 0, r
        ttnn.deallocate(st)
        ttnn.deallocate(bmt)
    out = os.environ.get("BSEL_OUT", "/tmp/pb_bm_probe.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"written {out}")
