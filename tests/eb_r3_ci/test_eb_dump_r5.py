# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary, fifth pass: exhaustive bit dumps of the widened native routing (#58726) and of the operand pass
over up to four DEST sections and over one (#58725), side A against side B in one process (EB_DUMP_PAIRS).

test_nat5_dump: a block or width sharded a with a column or scalar b in DRAM, block-float operands (a and b bfp8 or bfp4 into
the same format; a bf16 with a bfp8 b into bf16, scalar b); a holds every bf16 pattern quantized to its format, b the special
and normal set.
test_act5_dump: bf16 with an activation (relu, gelu or silu after the op, relu or silu on a, rsub, logical_and) at 4 tiles per
core on 32 cores, column and scalar b; scalar relu and gelu on 64-core block and 16-core width shards; column relu on 64-core
block shards.
test_mp5_dump: sharded ops with an operand activation at 4, 8 and 24 tiles per core (one partial section, one full, three) in
bf16, bfp8 and bfp4."""
import time

import numpy as np
import pytest
import ttnn

from eb_dump_lib import PATS, b16_set, b16_small, bf16_from_bits, kernel_variants, make_diffs, out_bits, run_chunk, set_env, tensor_vals

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}
U = ttnn.UnaryWithParam
UT = {"relu": ttnn.UnaryOpType.RELU, "gelu": ttnn.UnaryOpType.GELU, "silu": ttnn.UnaryOpType.SILU}


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    kernel_variants()
    ttnn.close_device(dev)


def _mc(shape, gy, gx, layout):
    st = {"block": ttnn.ShardStrategy.BLOCK, "width": ttnn.ShardStrategy.WIDTH, "height": ttnn.ShardStrategy.HEIGHT}[layout]
    return ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=gy, x=gx), strategy=st)


def _call(op, ta, tb, out_dt, mc, post=None, lact=None):
    kw = dict(memory_config=mc)
    if out_dt is not None:
        kw["dtype"] = DT[out_dt]
    if post:
        kw["activations"] = [U(UT[post])]
    if lact:
        kw["input_tensor_a_activations"] = [U(UT[lact])]
    if op == "mul":
        return ttnn.multiply(ta, tb, fast_and_approximate_mode=True, **kw)
    if op == "add":
        return ttnn.add(ta, tb, **kw)
    if op == "sub":
        return ttnn.subtract(ta, tb, **kw)
    if op == "rsub":
        return ttnn.rsub(ta, tb, **kw)
    if op == "logical_and":
        return ttnn.logical_and(ta, tb, **kw)
    if op == "ldexp":
        return ttnn.ldexp(ta, tb, **kw)
    if op == "div":
        return ttnn.divide(ta, tb, **kw)
    raise ValueError(op)


def _bcast_geometry(form, layout, nbc, c0, B, W, gy, gx):
    """a (every pattern, row by row) and b for one call: column b, one value per row; scalar b, one value per plane."""
    bsel = B[(np.arange(nbc) + c0) % B.size]
    if form == "col":
        rows_per_block = -(-65536 // W)
        R = -(-(rows_per_block * nbc) // (32 * gy)) * 32 * gy
        r = np.arange(R)
        q, jj = r // nbc, r % nbc
        a = PATS[(q[:, None] * W + np.arange(W)[None, :]) % 65536]
        return a.reshape(-1), (1, 1, R, W), bsel[jj], (1, 1, R, 1), np.repeat(bsel[jj], W)
    H = 256 if layout == "block" else 32
    per_b = H * W
    a = PATS[(np.arange(nbc)[:, None] * per_b + np.arange(per_b)[None, :] + c0 * 4099) % 65536].reshape(-1)
    return a, (nbc, 1, H, W), bsel, (nbc, 1, 1, 1), np.repeat(bsel, per_b)


def _dump_bcast(device, tag, op, form, layout, gy, gx, W, nbc, B, da, db, do, post=None, lact=None, calls=None):
    t0 = time.time()
    diffs = make_diffs(tag, op if op in ("add", "sub", "mul") and not post and not lact else None)
    done = 0
    try:
        starts = list(range(0, B.size, nbc)) if calls is None else calls
        for c0 in starts:
            a, ashape, bcol, bshape, _ = _bcast_geometry(form, layout, nbc, c0, B, W, gy, gx)
            mc = _mc(ashape, gy, gx, layout)
            ta = ttnn.from_torch(bf16_from_bits(a).reshape(ashape), dtype=DT[da], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            tb = ttnn.from_torch(bf16_from_bits(bcol).reshape(bshape), dtype=DT[db], layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            av = tensor_vals(ta)
            bv = np.repeat(tensor_vals(tb), av.size // tensor_vals(tb).size)
            run_chunk(device, diffs, lambda: out_bits(_call(op, ta, tb, do, mc, post, lact)), av, bv)
            ttnn.deallocate(ta)
            ttnn.deallocate(tb)
            done += 1
    except Exception as e:  # noqa: BLE001
        print(f"\nDUMP {tag}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:200]})", flush=True)
        set_env(device, {})
        return
    for *_, d in diffs:
        d.report(f"({done} calls, {time.time() - t0:.1f} s)")


# (form, layout, grid y, grid x, W, b values per call): column on 8x8 block shards (5 tiles per core row, 13 tile rows) and
# 2x8 width shards (4 tiles per core, 8 rows); scalar on 8x8 block shards (32 tiles per core) and 2x8 width shards (4).
GEO = {
    "col_bs64": ("col", "block", 8, 8, 1280, 64),
    "col_ws16": ("col", "width", 2, 8, 2048, 8),
    "scalar_bs64": ("scalar", "block", 8, 8, 1024, 8),
    "scalar_ws16": ("scalar", "width", 2, 8, 2048, 1),
}
FMT = {"bfp8": ("bfp8", "bfp8", "bfp8"), "bfp4": ("bfp4", "bfp4", "bfp4"), "mixed": ("bf16", "bfp8", "bf16")}
NAT5 = [(g, f, op) for g in GEO for f in FMT for op in ("add", "sub", "mul") if not (f == "mixed" and g.startswith("col"))]


@pytest.mark.parametrize("geo, fmt, op", NAT5, ids=["-".join(c) for c in NAT5])
def test_nat5_dump(device, geo, fmt, op):
    form, layout, gy, gx, W, nbc = GEO[geo]
    da, db, do = FMT[fmt]
    B = b16_set() if form == "col" else b16_small(64)
    _dump_bcast(device, f"nat5_{geo}_{fmt}_{op}", op, form, layout, gy, gx, W, nbc, B, da, db, do)


# 4 tiles per core on 32 cores (4x8 width, 32x4096): column b one value per row (32 per call), every a pattern against each
# in 16 calls per set of 32 values; scalar b one value per call.
ACTS = {"add_relu": ("add", "relu", None), "add_gelu": ("add", "gelu", None), "add_silu": ("add", "silu", None),
        "mul_relu": ("mul", "relu", None), "add_arelu": ("add", None, "relu"), "mul_asilu": ("mul", None, "silu"),
        "rsub": ("rsub", None, None), "logical_and": ("logical_and", None, None)}
ACT5 = [(k, a) for k in ("col_t4", "scalar_t4") for a in ACTS]
ACT5 += [("scalar_bs64", a) for a in ("add_relu", "add_gelu", "mul_relu")] + [("scalar_ws16", a) for a in ("add_relu", "mul_relu")]
ACT5 += [("col_bs64", a) for a in ("add_relu", "mul_relu")]


@pytest.mark.parametrize("geo, act", ACT5, ids=["-".join(c) for c in ACT5])
def test_act5_dump(device, geo, act):
    op, post, lact = ACTS[act]
    t0 = time.time()
    if geo == "col_t4":
        B = b16_small(64)
        tag = f"act5_{geo}_{act}"
        diffs = make_diffs(tag, None)
        done = 0
        shape, mc = (1, 1, 32, 4096), _mc((1, 1, 32, 4096), 4, 8, "width")
        try:
            for s in range(0, B.size, 32):
                bsel = B[s:s + 32]
                for c in range(16):
                    a = PATS[(np.arange(32)[:, None] * 0 + np.arange(4096)[None, :] + c * 4096) % 65536]
                    ta = ttnn.from_torch(bf16_from_bits(a.reshape(-1)).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                    tb = ttnn.from_torch(bf16_from_bits(bsel).reshape(1, 1, 32, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    run_chunk(device, diffs, lambda: out_bits(_call(op, ta, tb, None, mc, post, lact)), tensor_vals(ta), np.repeat(tensor_vals(tb), 4096))
                    ttnn.deallocate(ta)
                    ttnn.deallocate(tb)
                    done += 1
        except Exception as e:  # noqa: BLE001
            print(f"\nDUMP {tag}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:200]})", flush=True)
            set_env(device, {})
            return
        for *_, d in diffs:
            d.report(f"({done} calls, every pattern against 64 b values, {time.time() - t0:.1f} s)")
        return
    if geo == "scalar_t4":
        _dump_bcast(device, f"act5_{geo}_{act}", op, "scalar", "width", 4, 8, 4096, 1, b16_small(32), "bf16", "bf16", None, post, lact)
        return
    form, layout, gy, gx, W, nbc = GEO[geo]
    B = b16_set() if form == "col" else b16_small(64)
    _dump_bcast(device, f"act5_{geo}_{act}", op, form, layout, gy, gx, W, nbc, B, "bf16", "bf16", None, post, lact)


# one partial section (4 tiles per core), one full (8) and three (24), height sharded on 8 cores
MP5_SH = {"hs8_t4": (1, 1, 256, 128), "hs8_t8": (1, 1, 512, 128), "hs8_t24": (1, 1, 768, 256)}
MP5_OPS = ["rsub", "add_arelu", "mul_asilu", "div", "ldexp", "logical_and", "rsub_s", "add_arelu_s"]
MP5 = [(op, sh, d) for op in MP5_OPS for sh in MP5_SH for d in ("bf16", "bfp8", "bfp4")]


@pytest.mark.parametrize("op, shape_id, d", MP5, ids=["-".join(c) for c in MP5])
def test_mp5_dump(device, op, shape_id, d):
    t0 = time.time()
    shape = MP5_SH[shape_id]
    mc = _mc(shape, 2, 4, "height")
    per = int(np.prod(shape))
    B = b16_small(16 if d == "bf16" else 8)
    if op == "ldexp":
        B = np.array([0x0000, 0x3F80, 0xBF80, 0x4000, 0xC000, 0x4100, 0xC100, 0x4300, 0xC300, 0x42FE, 0xC2FE, 0x7F80, 0xFF80, 0x7FC0, 0x3F00, 0x4040], dtype=np.uint16)[: B.size]
    nb = B.size
    i = np.tile(np.arange(65536), nb)
    j = np.repeat(np.arange(nb), 65536)
    a_all, b_all = PATS[i], B[(i + j) % nb]
    total = a_all.size
    ncalls = -(-total // per)
    tag = f"mp5_{op}_{shape_id}_{d}"
    diffs = make_diffs(tag, None)
    kw = dict(memory_config=mc)
    try:
        for c in range(ncalls):
            idx = (np.arange(per) + c * per) % total
            ta = ttnn.from_torch(bf16_from_bits(a_all[idx]).reshape(shape), dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            tb = None
            if not op.endswith("_s"):
                tb = ttnn.from_torch(bf16_from_bits(b_all[idx]).reshape(shape), dtype=DT[d], layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
            if op == "rsub_s":
                fn = lambda: out_bits(ttnn.rsub(ta, 0.375, **kw))
            elif op == "add_arelu_s":
                fn = lambda: out_bits(ttnn.add(ta, 0.375, input_tensor_a_activations=[U(ttnn.UnaryOpType.RELU)], **kw))
            elif op == "add_arelu":
                fn = lambda: out_bits(_call("add", ta, tb, None, mc, None, "relu"))
            elif op == "mul_asilu":
                fn = lambda: out_bits(_call("mul", ta, tb, None, mc, None, "silu"))
            else:
                fn = lambda: out_bits(_call(op, ta, tb, None, mc))
            valid = min(per, total - c * per)
            run_chunk(device, diffs, fn, tensor_vals(ta), None if tb is None else tensor_vals(tb), valid)
            ttnn.deallocate(ta)
            if tb is not None:
                ttnn.deallocate(tb)
    except Exception as e:  # noqa: BLE001
        print(f"\nDUMP {tag}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:200]})", flush=True)
        set_env(device, {})
        return
    for *_, d_ in diffs:
        d_.report(f"({ncalls} calls, every pattern against {nb} b values, {time.time() - t0:.1f} s)")


# sixth pass (#58726): the classes taken in this pass. bfp8 column on 2 width cores at 4 tile rows (W 2048, 4 b values per
# call); column add and sub with relu on 4x8 block cores (W 2048, 16 tile rows); scalar add and sub with gelu on 2x8 width
# cores at 32 tiles per core (256x2048, one b value per call).
NAT6D = [("col_ws2_bfp8", op) for op in ("add", "sub", "mul")] + [("col_bs32_relu", op) for op in ("add", "sub")]
NAT6D += [("scalar_ws16_gelu", op) for op in ("add", "sub")]


@pytest.mark.parametrize("geo, op", NAT6D, ids=["-".join(c) for c in NAT6D])
def test_nat6_dump(device, geo, op):
    if geo == "col_ws2_bfp8":
        _dump_bcast(device, f"nat6_{geo}_{op}", op, "col", "width", 1, 2, 2048, 4, b16_set(), "bfp8", "bfp8", "bfp8")
    elif geo == "col_bs32_relu":
        _dump_bcast(device, f"nat6_{geo}_{op}", op, "col", "block", 4, 8, 2048, 64, b16_set(), "bf16", "bf16", None, "relu")
    else:
        t0 = time.time()
        B = b16_small(64)
        tag = f"nat6_{geo}_{op}"
        diffs = make_diffs(tag, None)
        shape = (1, 1, 256, 2048)
        mc = _mc(shape, 2, 8, "width")
        done = 0
        try:
            for c in range(B.size):
                a = PATS[(np.arange(256 * 2048) + c * 4099) % 65536]
                ta = ttnn.from_torch(bf16_from_bits(a).reshape(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
                tb = ttnn.from_torch(bf16_from_bits(B[c:c + 1]).reshape(1, 1, 1, 1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                run_chunk(device, diffs, lambda: out_bits(_call(op, ta, tb, None, mc, "gelu")), tensor_vals(ta), np.repeat(tensor_vals(tb), a.size))
                ttnn.deallocate(ta)
                ttnn.deallocate(tb)
                done += 1
        except Exception as e:  # noqa: BLE001
            print(f"\nDUMP {tag}: NOT RUN ({type(e).__name__}: {str(e).splitlines()[0][:200]})", flush=True)
            set_env(device, {})
            return
        for *_, d in diffs:
            d.report(f"({done} calls, every pattern against 64 b values, {time.time() - t0:.1f} s)")
