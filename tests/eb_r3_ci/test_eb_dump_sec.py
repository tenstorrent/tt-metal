# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58818 review): exhaustive bit dumps of the binary_ng sections of the PR, PR program against
main's program in one process (pairs as in test_eb_dump_bng: EB_DUMP_PAIRS or EB_DUMP_A / EB_DUMP_B).

test_block: sharded no-broadcast a, b, c at 16 or more tiles per core (the block unpack / block pack section).
test_bcast_sec: sharded a and c with a column or scalar broadcast b (the DEST sections of the broadcast kernels).
test_post_act: add with one fused post activation (the binary init kept after the activation), every kernel.
a holds every bf16 pattern; b the special and normal sets."""
import os
import time

import numpy as np
import pytest
import torch
import ttnn

from eb_dump_lib import PATS, Diff, b16_set, b16_small, bf16_from_bits, bfp_table, env_label, f32_of_bf16, kernel_variants, make_diffs, out_bits, pairs, run_chunk, set_env, tensor_vals, unique_envs

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    kernel_variants()
    ttnn.close_device(dev)


def _hs(shape, cores_y, cores_x):
    return ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=cores_y, x=cores_x), strategy=ttnn.ShardStrategy.HEIGHT)


def _bs(shape, cores_y, cores_x):
    return ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=cores_y, x=cores_x), strategy=ttnn.ShardStrategy.BLOCK)


def _op(op, x, y, out_dt, mc, acts=None):
    kw = dict(dtype=DT[out_dt], memory_config=mc)
    if acts:
        kw["activations"] = acts
    if op == "mul":
        return ttnn.multiply(x, y, fast_and_approximate_mode=True, **kw)
    if op == "add":
        return ttnn.add(x, y, fast_and_approximate_mode=True, **kw)
    return ttnn.subtract(x, y, fast_and_approximate_mode=True, **kw)


def _pairs_flat(bsrc):
    """Flat a (bf16 bits) and b (bf16 bits, or float32 for block float) covering every pattern against every b value:
    bf16 b: block j, index i: a = pattern i, b = B[(i + j) % NB]; bfp b: the test_eb_hifi2_dump arrangement (each pattern
    repeated 256 times against 16 groups of the table)."""
    if bsrc == "b16":
        B = b16_set()
        nb = B.size
        i = np.tile(np.arange(65536), nb)
        j = np.repeat(np.arange(nb), 65536)
        return PATS[i], B[(i + j) % nb], nb
    table = bfp_table(bsrc)
    n_groups = table.size // 16
    a = np.repeat(PATS, 256)
    slots = np.arange(a.size // 16)
    g = (slots * 7 + slots // n_groups) % n_groups
    return a, table.reshape(n_groups, 16)[g].reshape(-1), 256


# (shape, memory config builder): 32 tiles per core height sharded, 21 tiles per core (a partial DEST section), block sharded
SHAPES = {
    "hs32": ((1, 1, 2048, 1024), lambda s: _hs(s, 8, 8)),
    "hs21": ((1, 1, 2048, 672), lambda s: _hs(s, 8, 8)),
    "bs32": ((1, 1, 2048, 1024), lambda s: _bs(s, 8, 8)),
}
BLOCK = [(op, b, o, sh) for op in ("add", "sub", "mul") for b in ("b16", "bfp8") for o in ("bf16", "bfp8", "fp32") for sh in SHAPES]


@pytest.mark.parametrize("op, bsrc, out_dt, shape_id", BLOCK, ids=["-".join(c) for c in BLOCK])
def test_block(device, op, bsrc, out_dt, shape_id):
    t0 = time.time()
    shape, mcf = SHAPES[shape_id]
    mc = mcf(shape)
    per = int(np.prod(shape))
    a_all, b_all, nb = _pairs_flat(bsrc)
    total = a_all.size
    ncalls = -(-total // per)
    maxc = int(os.environ.get("EB_DUMP_MAX_CHUNKS", "0")) or ncalls
    diffs = make_diffs(f"block_{op}_{bsrc}_{out_dt}_{shape_id}", op)
    done = 0
    for c in range(min(ncalls, maxc)):
        idx = (np.arange(per) + c * per) % total
        a = a_all[idx]
        ta = ttnn.from_torch(bf16_from_bits(a).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        if bsrc == "b16":
            tb = ttnn.from_torch(bf16_from_bits(b_all[idx]).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        else:
            tb = ttnn.from_torch(torch.from_numpy(b_all[idx]).reshape(shape), dtype=DT[bsrc], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        av = f32_of_bf16(a)
        bv = tensor_vals(tb)
        run_chunk(device, diffs, lambda: out_bits(_op(op, ta, tb, out_dt, mc)), av, bv)
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
        done += 1
    for *_, d in diffs:
        d.report(f"({done} calls of {per}, pairs {min(total, done * per)} of {total} = 65536 x {nb}, {time.time() - t0:.1f} s)")


# column broadcast: a (1, 1, R, W) height sharded, b (1, 1, R, 1) in DRAM; scalar broadcast: a (N, 1, H, W) height sharded
# (one batch per core), b (N, 1, 1, 1) in DRAM. W of 32 tiles (whole DEST sections) and of 21 / 33 tiles (a partial one).
BCK = {
    "col_w32": ("col", 1024, 64),
    "col_w21": ("col", 672, 49),
    "scalar_w32": ("scalar", 1024, 64),
    "scalar_w33": ("scalar", 1056, 64),
}
BCAST = [(k, op, o) for k in BCK for op in ("add", "sub", "mul") for o in ("bf16", "bfp8", "fp32")]


def _grid(n):
    for y in range(min(n, 10), 0, -1):
        if n % y == 0 and n // y <= 13:
            return y, n // y
    raise ValueError(n)


@pytest.mark.parametrize("kind, op, out_dt", BCAST, ids=["-".join(c) for c in BCAST])
def test_bcast_sec(device, kind, op, out_dt):
    t0 = time.time()
    form, W, ncores = BCK[kind]
    B = b16_set()
    NBc = 64 if form == "col" else ncores  # b values per call
    nb = B.size
    diffs = make_diffs(f"bsec_{kind}_{op}_{out_dt}", op)
    gy, gx = _grid(ncores)
    done = 0
    maxc = int(os.environ.get("EB_DUMP_MAX_CHUNKS", "0")) or 10**9
    for c0 in range(0, nb, NBc):
        if done >= maxc:
            break
        bsel = B[(np.arange(NBc) + c0) % nb]
        if form == "col":
            rows_per_block = -(-65536 // W)  # rows holding every pattern once
            R = rows_per_block * NBc
            r = np.arange(R)
            q, j = r // NBc, r % NBc
            pat = (q[:, None] * W + np.arange(W)[None, :]) % 65536
            a = PATS[pat]
            bcol = bsel[j]
            ashape, bshape = (1, 1, R, W), (1, 1, R, 1)
            b_el = np.repeat(bcol, W)
        else:
            H = 64
            per_b = H * W
            pat = np.arange(per_b) % 65536
            a = np.tile(PATS[pat], NBc)
            ashape, bshape = (NBc, 1, H, W), (NBc, 1, 1, 1)
            bcol = bsel
            b_el = np.repeat(bsel, per_b)
        mc = ttnn.create_sharded_memory_config(ashape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=ttnn.ShardStrategy.HEIGHT)
        ta = ttnn.from_torch(bf16_from_bits(a.reshape(-1)).reshape(ashape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tb = ttnn.from_torch(bf16_from_bits(bcol).reshape(bshape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        av = f32_of_bf16(a.reshape(-1))
        bv = f32_of_bf16(b_el)
        run_chunk(device, diffs, lambda: out_bits(_op(op, ta, tb, out_dt, mc)), av, bv)
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
        done += 1
    for *_, d in diffs:
        d.report(f"({done} calls, every pattern against {min(nb, done * NBc)} b values, {time.time() - t0:.1f} s)")


ACTS = [
    "relu", "relu6", "gelu", "gelu_approx", "gelu_tanh", "silu", "sigmoid", "sigmoid_approx", "hardsigmoid", "hardtanh",
    "sqrt", "rsqrt", "exp", "recip", "log", "log1p", "tanh", "log2", "log10", "sin", "cos", "cosh", "sinh", "abs", "sign",
    "square", "softplus", "xielu", "selu", "alt_complex_rotate90", "hardmish", "mish", "mish_approx",
]
KINDS = ["nob_dram", "nob_hs", "col_dram", "col_hs", "scalar_dram", "scalar_hs", "pyscalar_dram", "pyscalar_hs"]
POST = [(act, k) for act in ACTS for k in KINDS]


def _act(name):
    return [ttnn._ttnn.activation.string_to_unary_with_param(name)]


@pytest.mark.parametrize("act, kind", POST, ids=["-".join(c) for c in POST])
def test_post_act(device, act, kind):
    """add(a, b, activations=[act]): a every bf16 pattern, b 16 values of the special and normal sets (each pattern against
    each), or the Python scalar 0.375."""
    t0 = time.time()
    B = b16_small(16)
    NB = B.size
    form, mem = kind.split("_")
    if form == "nob":
        ashape = bshape = (1, 1, 1024, 1024)
        i = np.tile(np.arange(65536), NB)
        j = np.repeat(np.arange(NB), 65536)
        a, b_el = PATS[i], B[(i + j) % NB]
        bdata = b_el
    elif form == "col":
        R = 64 * NB
        r = np.arange(R)
        a = PATS[((r // NB)[:, None] * 1024 + np.arange(1024)[None, :]) % 65536].reshape(-1)
        bdata = B[r % NB]
        b_el = np.repeat(bdata, 1024)
        ashape, bshape = (1, 1, R, 1024), (1, 1, R, 1)
    elif form == "scalar":
        a = np.tile(PATS, NB)
        bdata = B
        b_el = np.repeat(B, 65536)
        ashape, bshape = (NB, 1, 64, 1024), (NB, 1, 1, 1)
    else:
        a = PATS
        b_el = np.full(65536, 0x3EC0, dtype=np.uint16)  # 0.375
        ashape, bshape = (1, 1, 256, 256), None
    if mem == "hs":
        n_tile_rows = int(np.prod(ashape[:-1])) // 32
        cores = next(c for c in (16, 8, 4, 2, 1) if n_tile_rows % c == 0)
        mca = ttnn.create_sharded_memory_config(ashape, core_grid=ttnn.CoreGrid(y=max(cores // 8, 1), x=min(cores, 8)), strategy=ttnn.ShardStrategy.HEIGHT)
        mcb = mca if form == "nob" else ttnn.DRAM_MEMORY_CONFIG
    else:
        mca = mcb = ttnn.DRAM_MEMORY_CONFIG
    ta = ttnn.from_torch(bf16_from_bits(a.reshape(-1)).reshape(ashape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mca)
    if bshape is not None:
        tb = ttnn.from_torch(bf16_from_bits(np.asarray(bdata).reshape(-1)).reshape(bshape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mcb)

        def fn():
            return out_bits(_op("add", ta, tb, "bf16", mca, _act(act)))

    else:

        def fn():
            return out_bits(ttnn.add(ta, 0.375, dtype=ttnn.bfloat16, memory_config=mca, activations=_act(act)))

    diffs = make_diffs(f"post_{act}_{kind}", None)
    try:
        run_chunk(device, diffs, fn, f32_of_bf16(a.reshape(-1)), f32_of_bf16(b_el))
    except Exception as e:  # noqa: BLE001
        print(f"\nDUMP post_{act}_{kind}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:200]})", flush=True)
        set_env(device, {})
        return
    for *_, d in diffs:
        d.report(f"({time.time() - t0:.1f} s)")
