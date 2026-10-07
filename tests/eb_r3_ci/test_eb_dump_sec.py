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
    if bsrc == "bfp8both":  # a every bfp8_b datum (the table), b the reduced special and normal set, both bfp8_b
        table = bfp_table("bfp8")
        B = b16_small(64)
        i = np.tile(np.arange(table.size), B.size)
        j = np.repeat(np.arange(B.size), table.size)
        return table[i], f32_of_bf16(B[(i + j) % B.size]), B.size
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
    "hs12": ((1, 1, 2048, 384), lambda s: _hs(s, 8, 8)),
    "hs8": ((1, 1, 2048, 256), lambda s: _hs(s, 8, 8)),
}
BLOCK = [
    (op, b, o, sh)
    for op in ("add", "sub", "mul")
    for b in ("b16", "bfp8", "bfp8both", "bfp4")
    for o in ("bf16", "bfp8", "bfp4", "fp32")
    for sh in SHAPES
]


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
        if bsrc == "bfp8both":
            ta = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)).reshape(shape), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            tb = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(b_all[idx])).reshape(shape), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            av = tensor_vals(ta)
        else:
            ta = ttnn.from_torch(bf16_from_bits(a).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            if bsrc == "b16":
                tb = ttnn.from_torch(bf16_from_bits(b_all[idx]).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            else:
                tb = ttnn.from_torch(torch.from_numpy(b_all[idx]).reshape(shape), dtype=DT[bsrc], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            av = f32_of_bf16(a)
        bv = tensor_vals(tb)
        run_chunk(device, diffs, lambda: out_bits(_op(op, ta, tb, out_dt, mc)), av, bv, min(per, total - c * per))
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
        done += 1
    for *_, d in diffs:
        d.report(f"({done} calls of {per}, pairs {min(total, done * per)} of {total} = {total // nb} x {nb}, {time.time() - t0:.1f} s)")


# column broadcast: a (1, 1, R, W) height sharded, b (1, 1, R, 1) in DRAM; scalar broadcast: a (N, 1, H, W) height sharded
# (one batch per core), b (N, 1, 1, 1) in DRAM. W of 32 tiles (whole DEST sections) and of 21 / 33 tiles (a partial one).
BCK = {
    "col_w32": ("col", 1024, 64),
    "col_w21": ("col", 672, 49),
    "scalar_w32": ("scalar", 1024, 64),
    "scalar_w33": ("scalar", 1056, 64),
    # bfp8_b a and b: the column broadcast kernel also runs with fp32 DEST (a section of 4 tiles), every bfp8_b datum as a
    "col8_w32": ("col8", 1024, 64),
    "col8_w21": ("col8", 672, 49),
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
    NBc = 64 if form in ("col", "col8") else ncores  # b values per call
    nb = B.size
    diffs = make_diffs(f"bsec_{kind}_{op}_{out_dt}", op)
    gy, gx = _grid(ncores)
    done = 0
    maxc = int(os.environ.get("EB_DUMP_MAX_CHUNKS", "0")) or 10**9
    for c0 in range(0, nb, NBc):
        if done >= maxc:
            break
        bsel = B[(np.arange(NBc) + c0) % nb]
        if form == "col8":
            R = -(-65536 // W) * NBc
            r = np.arange(R)
            a = np.resize(bfp_table("bfp8"), R * W).reshape(R, W)
            a = np.roll(a.reshape(-1), c0 * 4099).reshape(R, W)
            bcol = bsel[r % NBc]
            ashape, bshape = (1, 1, R, W), (1, 1, R, 1)
            b_el = np.repeat(bcol, W)
        elif form == "col":
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
        if form == "col8":
            ta = ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).reshape(ashape), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            tb = ttnn.from_torch(torch.from_numpy(f32_of_bf16(bcol)).reshape(bshape), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            av = tensor_vals(ta)
            bv = np.repeat(tensor_vals(tb), W)
        else:
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


# ---- ci5 head: the block pack with a post activation, the Python-scalar block, broadcast sections with activations ----
BPOST = [(op, act, o, sh) for op in ("add", "mul") for act in ("gelu", "silu", "relu") for o in ("bf16", "fp32") for sh in ("hs32", "hs21", "hs8")]


@pytest.mark.parametrize("op, act, out_dt, shape_id", BPOST, ids=["-".join(c) for c in BPOST])
def test_block_post(device, op, act, out_dt, shape_id):
    """Sharded no-broadcast op with a post activation: the block unpack and block pack from 16 tiles per core (c bf16 or
    fp32); every bf16 pattern against 64 values of the special and normal set."""
    t0 = time.time()
    shape, mcf = SHAPES[shape_id]
    mc = mcf(shape)
    per = int(np.prod(shape))
    B = b16_small(64)
    nb = B.size
    i = np.tile(np.arange(65536), nb)
    j = np.repeat(np.arange(nb), 65536)
    a_all, b_all = PATS[i], B[(i + j) % nb]
    total = a_all.size
    ncalls = -(-total // per)
    diffs = make_diffs(f"bpost_{op}_{act}_{out_dt}_{shape_id}", None)
    for c in range(ncalls):
        idx = (np.arange(per) + c * per) % total
        ta = ttnn.from_torch(bf16_from_bits(a_all[idx]).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tb = ttnn.from_torch(bf16_from_bits(b_all[idx]).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        run_chunk(device, diffs, lambda: out_bits(_op(op, ta, tb, out_dt, mc, _act(act))), f32_of_bf16(a_all[idx]), f32_of_bf16(b_all[idx]), min(per, total - c * per))
        ttnn.deallocate(ta)
        ttnn.deallocate(tb)
    for *_, d in diffs:
        d.report(f"({ncalls} calls, every pattern against {nb} b values, {time.time() - t0:.1f} s)")


# Python scalar, a and c height sharded: (1, 1, 256, 256) holds every pattern; 2, 4 or 8 cores (32, 16 or 8 tiles per core)
SCSH = {"hs32": 2, "hs16": 4, "hs8": 8}
SCB = [(op, side, o, sh, post) for op in ("add", "sub", "mul") for side in ("rhs", "lhs") for o in ("bf16", "fp32") for sh in SCSH for post in ("none", "relu")]


def _scalar_call(op, side, x, s, out_dt, mc, acts):
    kw = dict(dtype=DT[out_dt], memory_config=mc)
    if acts:
        kw["activations"] = acts
    fn = {"add": ttnn.add, "sub": ttnn.subtract, "mul": ttnn.multiply}[op]
    if side == "rhs":
        return fn(x, s, fast_and_approximate_mode=True, **kw)
    return fn(s, x, fast_and_approximate_mode=True, **kw)


@pytest.mark.parametrize("op, side, out_dt, shape_id, post", SCB, ids=["-".join(c) for c in SCB])
def test_scalar_block(device, op, side, out_dt, shape_id, post):
    """op(tensor, scalar) (side rhs) and op(scalar, tensor) (side lhs, SCALAR_IS_LHS): every bf16 pattern in the tensor,
    every value of the special and normal set as the scalar (one call each)."""
    t0 = time.time()
    ncores = SCSH[shape_id]
    shape = (1, 1, 256, 256)
    mc = _hs(shape, 1, ncores)
    tx = ttnn.from_torch(bf16_from_bits(PATS).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    av = f32_of_bf16(PATS)
    B = b16_set()
    acts = _act("relu") if post == "relu" else None
    diffs = make_diffs(f"scblk_{op}_{side}_{out_dt}_{shape_id}_{post}", op if post == "none" else None)
    try:
        for s_bits in B:
            s = float(f32_of_bf16(np.array([s_bits], dtype=np.uint16))[0])
            sv = np.full(av.size, s, dtype=np.float32)
            x, y = (av, sv) if side == "rhs" else (sv, av)
            run_chunk(device, diffs, lambda: out_bits(_scalar_call(op, side, tx, s, out_dt, mc, acts)), x, y)
    except Exception as e:  # noqa: BLE001
        print(f"\nDUMP scblk_{op}_{side}_{out_dt}_{shape_id}_{post}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:200]})", flush=True)
        set_env(device, {})
        return
    for *_, d in diffs:
        d.report(f"({B.size} scalars, {time.time() - t0:.1f} s)")


def _bcast_inputs(device, kind, c0, NBc, B):
    form, W, ncores = BCK[kind]
    gy, gx = _grid(ncores)
    nb = B.size
    bsel = B[(np.arange(NBc) + c0) % nb]
    if form == "col":
        rows_per_block = -(-65536 // W)
        R = rows_per_block * NBc
        r = np.arange(R)
        a = PATS[((r // NBc)[:, None] * W + np.arange(W)[None, :]) % 65536]
        bcol = bsel[r % NBc]
        ashape, bshape = (1, 1, R, W), (1, 1, R, 1)
        b_el = np.repeat(bcol, W)
    else:
        per_b = 64 * W
        a = np.tile(PATS[np.arange(per_b) % 65536], NBc)
        ashape, bshape = (NBc, 1, 64, W), (NBc, 1, 1, 1)
        bcol = bsel
        b_el = np.repeat(bsel, per_b)
    mc = ttnn.create_sharded_memory_config(ashape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=ttnn.ShardStrategy.HEIGHT)
    ta = ttnn.from_torch(bf16_from_bits(a.reshape(-1)).reshape(ashape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tb = ttnn.from_torch(bf16_from_bits(bcol).reshape(bshape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ta, tb, f32_of_bf16(a.reshape(-1)), f32_of_bf16(b_el), mc


BACT = [(k, a) for k in ("col_w32", "col_w21", "scalar_w32", "scalar_w33") for a in ("add_gelu", "add_silu", "add_relu", "mul_gelu", "rsub", "add_arelu", "sub_brelu")]


@pytest.mark.parametrize("kind, case", BACT, ids=["-".join(c) for c in BACT])
def test_bcast_sec_act(device, kind, case):
    """Broadcast sections with one activation (height-sharded a, c; b broadcast from DRAM): a post activation (add + gelu,
    silu, relu; mul + gelu) or an operand activation (rsub: NEG on an operand; add with relu on a; sub with relu on b)."""
    t0 = time.time()
    form, W, ncores = BCK[kind]
    B = b16_set()
    NBc = 64 if form == "col" else ncores
    diffs = make_diffs(f"bsact_{kind}_{case}", None)
    relu = _act("relu")
    try:
        for c0 in range(0, B.size, NBc):
            ta, tb, av, bv, mc = _bcast_inputs(device, kind, c0, NBc, B)
            if case.startswith("add_") and case != "add_arelu":
                fn = lambda: out_bits(ttnn.add(ta, tb, activations=_act(case[4:]), memory_config=mc))
            elif case == "mul_gelu":
                fn = lambda: out_bits(ttnn.multiply(ta, tb, activations=_act("gelu"), fast_and_approximate_mode=True, memory_config=mc))
            elif case == "rsub":
                fn = lambda: out_bits(ttnn.rsub(ta, tb, memory_config=mc))
            elif case == "add_arelu":
                fn = lambda: out_bits(ttnn.add(ta, tb, input_tensor_a_activations=relu, memory_config=mc))
            else:
                fn = lambda: out_bits(ttnn.subtract(ta, tb, input_tensor_b_activations=relu, memory_config=mc))
            run_chunk(device, diffs, fn, av, bv)
            ttnn.deallocate(ta)
            ttnn.deallocate(tb)
    except Exception as e:  # noqa: BLE001
        print(f"\nDUMP bsact_{kind}_{case}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:200]})", flush=True)
        set_env(device, {})
        return
    for *_, d in diffs:
        d.report(f"(every pattern against {B.size} b values, {time.time() - t0:.1f} s)")
