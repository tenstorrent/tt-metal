# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, fourth pass: exhaustive bit dumps of two binary_ng changes, side A against side B in one process
(EB_DUMP_PAIRS, toggles of the factory).

test_mp_dump (#58725): sharded ops with an operand activation, the operand pass over two DEST sections before one binary
init (EB_R3_MULTI_PASS=2) against one pass and one init per section; a holds every bf16 pattern, b 16 values of the special
and normal set (each pattern against each), or the Python scalar 0.375.
test_nat_dump (#58726): a block or width sharded a with a column or scalar b in DRAM, routed native (EB_R3_NATIVE_BCAST)
against the current routing; a holds every bf16 pattern against 64 (column) or 8 (scalar) b values."""
import time

import numpy as np
import pytest
import torch
import ttnn

from eb_dump_lib import PATS, b16_set, b16_small, bf16_from_bits, f32_of_bf16, kernel_variants, make_diffs, out_bits, run_chunk, set_env

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}
U = ttnn.UnaryWithParam


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    kernel_variants()
    ttnn.close_device(dev)


def _hs(shape, y, x):
    return ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=y, x=x), strategy=ttnn.ShardStrategy.HEIGHT)


MP_OPS = ["rsub", "add_arelu", "mul_asilu", "div", "ldexp", "logical_and", "rsub_s", "add_arelu_s"]
# 1024x1024 on 8 cores: 128 tiles per core; on 32 cores: 32 per core; 2560x256 on 8 cores: 80 per core (10 sections)
MP_SH = {"hs8_t128": ((1, 1, 1024, 1024), 1, 8), "hs32_t32": ((1, 1, 1024, 1024), 4, 8), "hs8_t80": ((1, 1, 2560, 256), 1, 8)}
MP = [(op, sh, o) for op in MP_OPS for sh in MP_SH for o in ("bf16", "fp32")]


def _mp_call(op, ta, tb, out_dt, mc):
    kw = dict(dtype=DT[out_dt], memory_config=mc)
    if op == "rsub":
        return ttnn.rsub(ta, tb, **kw)
    if op == "add_arelu":
        return ttnn.add(ta, tb, input_tensor_a_activations=[U(ttnn.UnaryOpType.RELU)], **kw)
    if op == "mul_asilu":
        return ttnn.multiply(ta, tb, input_tensor_a_activations=[U(ttnn.UnaryOpType.SILU)], fast_and_approximate_mode=True, **kw)
    if op == "div":
        return ttnn.divide(ta, tb, **kw)
    if op == "ldexp":
        return ttnn.ldexp(ta, tb, **kw)
    if op == "logical_and":
        return ttnn.logical_and(ta, tb, **kw)
    if op == "rsub_s":
        return ttnn.rsub(ta, 0.375, **kw)
    return ttnn.add(ta, 0.375, input_tensor_a_activations=[U(ttnn.UnaryOpType.RELU)], **kw)


@pytest.mark.parametrize("op, shape_id, out_dt", MP, ids=["-".join(c) for c in MP])
def test_mp_dump(device, op, shape_id, out_dt):
    t0 = time.time()
    shape, gy, gx = MP_SH[shape_id]
    mc = _hs(shape, gy, gx)
    per = int(np.prod(shape))
    B = b16_small(16)
    if op == "ldexp":
        B = np.array([0x0000, 0x3F80, 0xBF80, 0x4000, 0xC000, 0x4100, 0xC100, 0x4300, 0xC300, 0x42FE, 0xC2FE, 0x7F80, 0xFF80, 0x7FC0, 0x3F00, 0x4040], dtype=np.uint16)
    nb = B.size
    i = np.tile(np.arange(65536), nb)
    j = np.repeat(np.arange(nb), 65536)
    a_all, b_all = PATS[i], B[(i + j) % nb]
    total = a_all.size
    ncalls = -(-total // per)
    diffs = make_diffs(f"mp_{op}_{shape_id}_{out_dt}", None)
    try:
        for c in range(ncalls):
            idx = (np.arange(per) + c * per) % total
            ta = ttnn.from_torch(bf16_from_bits(a_all[idx]).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            tb = None
            if not op.endswith("_s"):
                tb = ttnn.from_torch(bf16_from_bits(b_all[idx]).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            run_chunk(device, diffs, lambda: out_bits(_mp_call(op, ta, tb, out_dt, mc)), f32_of_bf16(a_all[idx]), None if tb is None else f32_of_bf16(b_all[idx]), min(per, total - c * per))
            ttnn.deallocate(ta)
            if tb is not None:
                ttnn.deallocate(tb)
    except Exception as e:  # noqa: BLE001
        print(f"\nDUMP mp_{op}_{shape_id}_{out_dt}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:200]})", flush=True)
        set_env(device, {})
        return
    for *_, d in diffs:
        d.report(f"({ncalls} calls, every pattern against {nb} b values, {time.time() - t0:.1f} s)")


# column b: a (1, 1, R, W) block sharded on 8x8 cores (W 1280: 5 tiles per core row) or width sharded on 16 cores (W 2048:
# 4 tiles per core); scalar b: a (N, 1, H, W) block sharded with one plane per core row, b (N, 1, 1, 1).
NAT_K = {
    "col_bs": ("col", "block", 1280, 64),
    "col_ws": ("col", "width", 2048, 16),
    "scalar_bs": ("scalar", "block", 1024, 8),
    "scalar_ws": ("scalar", "width", 2048, 1),
}
NAT = [(k, op, o) for k in NAT_K for op in ("add", "sub", "mul") for o in ("bf16", "bfp8", "fp32")]


def _nat_call(op, ta, tb, out_dt, mc):
    kw = dict(dtype=DT[out_dt], memory_config=mc)
    if op == "mul":
        return ttnn.multiply(ta, tb, fast_and_approximate_mode=True, **kw)
    return {"add": ttnn.add, "sub": ttnn.subtract}[op](ta, tb, fast_and_approximate_mode=True, **kw)


@pytest.mark.parametrize("kind, op, out_dt", NAT, ids=["-".join(c) for c in NAT])
def test_nat_dump(device, kind, op, out_dt):
    t0 = time.time()
    form, layout, W, nbc = NAT_K[kind]
    B = b16_set()
    nb = B.size
    diffs = make_diffs(f"nat_{kind}_{op}_{out_dt}", op)
    done = 0
    try:
        for c0 in range(0, nb, nbc):
            bsel = B[(np.arange(nbc) + c0) % nb]
            if form == "col":
                rows_per_block = -(-65536 // W)
                R = -(-(rows_per_block * nbc) // 256) * 256
                r = np.arange(R)
                q, jj = r // nbc, r % nbc
                a = PATS[(q[:, None] * W + np.arange(W)[None, :]) % 65536]
                bcol = bsel[jj]
                ashape, bshape = (1, 1, R, W), (1, 1, R, 1)
                b_el = np.repeat(bcol, W)
                if layout == "block":
                    mc = ttnn.create_sharded_memory_config(ashape, core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.BLOCK)
                else:
                    mc = ttnn.create_sharded_memory_config(ashape, core_grid=ttnn.CoreGrid(y=2, x=8), strategy=ttnn.ShardStrategy.WIDTH)
            else:
                H = 256 if layout == "block" else 32
                per_b = H * W
                a = PATS[(np.arange(nbc)[:, None] * per_b + np.arange(per_b)[None, :] + c0 * 4099) % 65536].reshape(-1)
                ashape, bshape = (nbc, 1, H, W), (nbc, 1, 1, 1)
                bcol = bsel
                b_el = np.repeat(bsel, per_b)
                if layout == "block":
                    mc = ttnn.create_sharded_memory_config(ashape, core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.BLOCK)
                else:
                    mc = ttnn.create_sharded_memory_config(ashape, core_grid=ttnn.CoreGrid(y=2, x=8), strategy=ttnn.ShardStrategy.WIDTH)
            ta = ttnn.from_torch(bf16_from_bits(a.reshape(-1)).reshape(ashape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            tb = ttnn.from_torch(bf16_from_bits(bcol).reshape(bshape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            run_chunk(device, diffs, lambda: out_bits(_nat_call(op, ta, tb, out_dt, mc)), f32_of_bf16(a.reshape(-1)), f32_of_bf16(b_el))
            ttnn.deallocate(ta)
            ttnn.deallocate(tb)
            done += 1
    except Exception as e:  # noqa: BLE001
        print(f"\nDUMP nat_{kind}_{op}_{out_dt}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:200]})", flush=True)
        set_env(device, {})
        return
    for *_, d in diffs:
        d.report(f"({done} calls, every pattern against {min(nb, done * nbc)} b values, {time.time() - t0:.1f} s)")
