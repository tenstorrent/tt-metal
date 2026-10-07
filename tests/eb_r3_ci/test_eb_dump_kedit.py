# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58818 review): exhaustive bit dumps of ops whose hand-off is an opt-in define in the kernel .cpp
(dump_kedit.sh runs this module twice: EB_DUMP_STAGE=save with the define lines removed, main's program, then
EB_DUMP_STAGE=cmp with the kernels as in the PR, each process with its own kernel cache). Outputs go to EB_DUMP_DIR.

test_bcast: ttnn.bcast MUL (bcast_h.cpp, bcast_h_sharded_optimised.cpp, bcast_w.cpp, bcast_hw_metal2.cpp), at the op's
HiFi4 and with the CI toggles EB_R3_BCAST_FIDELITY (HiFi2, LoFi) and EB_R3_BCAST_FP32 (fp32 DEST).
test_rotate_half: ttnn.experimental.rotate_half (bcast_hw.cpp, a scalar multiply by -1)."""
import os
import time

import numpy as np
import pytest
import torch
import ttnn

from eb_dump_lib import PATS, b16_set, b16_small, bf16_from_bits, f32_of_bf16, kernel_variants, out_bits, set_env, stage

STAGE = os.environ.get("EB_DUMP_STAGE", "save")


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    kernel_variants(f" {STAGE}")
    ttnn.close_device(dev)


def _hs(shape, y, x):
    return ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=y, x=x), strategy=ttnn.ShardStrategy.HEIGHT)


BCAST_VARIANTS = [("HiFi4", False), ("HiFi2", False), ("LoFi", False), ("HiFi4", True), ("LoFi", True)]


def _bcast_data(dim, small):
    """a (bits, shape), b (bits, shape), per-element b operand bits."""
    B = b16_small(64) if small else b16_set()
    nb = B.size
    if dim == "H":  # b row (1, 1, 1, W): W = nb * q columns, column c gets B[c % nb]
        q = 2 if not small else 16
        W = nb * q
        R = 65536 // q
        r = np.arange(R)[:, None]
        c = np.arange(W)[None, :]
        a = PATS[(r * q + c // nb) % 65536]
        bvec = B[np.arange(W) % nb]
        return a, (1, 1, R, W), bvec, (1, 1, 1, W), np.broadcast_to(bvec[None, :], (R, W)).reshape(-1)
    if dim == "W":  # b column (1, 1, R, 1): row r gets B[r % nb]
        W = 1024
        R = 64 * nb
        r = np.arange(R)
        a = PATS[((r // nb)[:, None] * W + np.arange(W)[None, :]) % 65536]
        bcol = B[r % nb]
        return a, (1, 1, R, W), bcol, (1, 1, R, 1), np.repeat(bcol, W)
    # HW: b scalar per batch (N, 1, 1, 1)
    a = np.tile(PATS, nb).reshape(nb, 64, 1024)
    return a, (nb, 1, 64, 1024), B, (nb, 1, 1, 1), np.repeat(B, 65536)


BCASTS = [("H", "dram"), ("H", "bs"), ("H", "bs4"), ("W", "dram"), ("HW", "dram"), ("HW", "hs")]


@pytest.mark.parametrize("dim, mem", BCASTS, ids=["-".join(c) for c in BCASTS])
def test_bcast(device, dim, mem):
    t0 = time.time()
    small = mem != "dram"
    a, ashape, b, bshape, bel = _bcast_data(dim, small)
    if mem == "hs":
        tile_rows = int(np.prod(ashape[:-1])) // 32
        cores = next(c for c in (64, 32, 16, 8) if tile_rows % c == 0)
        mca = _hs(ashape, cores // 8, 8)
    elif mem == "bs":  # the sharded H factory (bcast_h_sharded_optimised.cpp) takes block or width sharding
        mca = ttnn.create_sharded_memory_config(ashape, core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.BLOCK)
    elif mem == "bs4":
        # shards of 4 x 4 tiles: the kernel's DEST block (h_blk = min(Ht, 8) tiles) stays within the 4 tiles of fp32 DEST
        return _bcast_bs4(device, a, ashape, b, bshape, bel, t0)
    else:
        mca = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(bf16_from_bits(a.reshape(-1)).reshape(ashape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mca)
    tb = ttnn.from_torch(bf16_from_bits(np.asarray(b).reshape(-1)).reshape(bshape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    bdim = {"H": ttnn.BcastOpDim.H, "W": ttnn.BcastOpDim.W, "HW": ttnn.BcastOpDim.HW}[dim]
    av = f32_of_bf16(a.reshape(-1))
    bv = f32_of_bf16(bel)
    for fid, fp32 in BCAST_VARIANTS:
        env = {"EB_R3_BCAST_FIDELITY": fid}
        if fp32:
            env["EB_R3_BCAST_FP32"] = "1"
        set_env(device, env)
        out = out_bits(ttnn.bcast(ta, tb, ttnn.BcastOpMath.MUL, bdim, memory_config=mca))
        if fid == "HiFi4" and not fp32:
            base = out
        elif out.dtype == base.dtype:
            print(f"\nDUMP control bcast_{dim}_{mem} {fid}_{'d32' if fp32 else 'd16'} against HiFi4_d16 (same side): differ {int((out != base).sum())}", flush=True)
        stage(f"bcast_{dim}_{mem}_{fid}_{'d32' if fp32 else 'd16'}", out, av, bv, "mul", f"({time.time() - t0:.1f} s)")
    set_env(device, {})


def test_rotate_half(device):
    """x (1, 1, 128, 1024): each half of the last dim holds every bf16 pattern; the second half is multiplied by -1."""
    t0 = time.time()
    half = PATS.reshape(128, 512)
    x = np.concatenate([half, half], axis=1)
    tx = ttnn.from_torch(bf16_from_bits(x.reshape(-1)).reshape(1, 1, 128, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = out_bits(ttnn.experimental.rotate_half(tx))
    # out = cat(-x2, x1): the multiplied half is the first one
    av = np.concatenate([f32_of_bf16(half), f32_of_bf16(half)], axis=1).reshape(-1)
    bv = np.concatenate([np.full((128, 512), -1.0, np.float32), np.full((128, 512), np.nan, np.float32)], axis=1).reshape(-1)
    stage("rotate_half", out, av, bv, None, f"({time.time() - t0:.1f} s)")


def _bcast_bs4(device, a, ashape, b, bshape, bel, t0):
    R, W = ashape[2], ashape[3]
    rows = 1024
    tb = ttnn.from_torch(bf16_from_bits(np.asarray(b).reshape(-1)).reshape(bshape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    av = f32_of_bf16(a.reshape(-1))
    bv = f32_of_bf16(bel)
    for fid, fp32 in BCAST_VARIANTS:
        env = {"EB_R3_BCAST_FIDELITY": fid}
        if fp32:
            env["EB_R3_BCAST_FP32"] = "1"
        set_env(device, env)
        outs = []
        for r0 in range(0, R, rows):
            shp = (1, 1, rows, W)
            mc = ttnn.create_sharded_memory_config(shp, core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.BLOCK)
            ta = ttnn.from_torch(bf16_from_bits(a[r0 : r0 + rows].reshape(-1)).reshape(shp), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            outs.append(out_bits(ttnn.bcast(ta, tb, ttnn.BcastOpMath.MUL, ttnn.BcastOpDim.H, memory_config=mc)))
            ttnn.deallocate(ta)
        out = np.concatenate(outs)
        if fid == "HiFi4" and not fp32:
            base = out
        else:
            print(f"\nDUMP control bcast_H_bs4 {fid}_{'d32' if fp32 else 'd16'} against HiFi4_d16 (same side): differ {int((out != base).sum())}", flush=True)
        stage(f"bcast_H_bs4_{fid}_{'d32' if fp32 else 'd16'}", out, av, bv, "mul", f"({time.time() - t0:.1f} s)")
    set_env(device, {})
