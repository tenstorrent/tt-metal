# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58818 review): exhaustive bit dumps of binary_ng FPU ops, two programs per comparison in one
process (eb_dump_lib). The pairs come from EB_DUMP_PAIRS ("A|B;A|B", each side "K=V,K=V" of EB_R3_* toggles; side A is the
reference) or from EB_DUMP_A / EB_DUMP_B.

test_nob_partial, test_row_partial, test_scalar_lhs: the configuration set of test_eb_hifi2_dump.py (every bf16 SrcA pattern
against bfp8_b and bfp4_b SrcB data under every shared exponent; the scalar kernel takes the full cross product).
test_nob_cross: the full cross product of every bf16 pattern (SrcA) with a SrcB source, through the no-broadcast kernel,
in chunks: b16set (bf16 special and normal set), b16all (all 65536 bf16 patterns), bfp8 / bfp4 (every datum under every
shared exponent)."""
import os
import time

import numpy as np
import pytest
import torch
import ttnn

from eb_dump_lib import (
    PATS,
    make_diffs,
    pairs,
    run_chunk,
    unique_envs,
    Diff,
    b16_set,
    bf16_from_bits,
    bfp_table,
    digest,
    env_label,
    f32_of_bf16,
    kernel_variants,
    out_bits,
    parse_env,
    set_env,
    tensor_vals,
    to_dev,
)

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}
MAX_DETAIL_CALLS = int(os.environ.get("EB_DUMP_MAX_DETAIL", "512"))


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    kernel_variants()
    ttnn.close_device(dev)


def binop(op, x, y, out_dt):
    kw = dict(dtype=DT[out_dt], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if op == "mul":
        return ttnn.multiply(x, y, fast_and_approximate_mode=True, **kw)
    if op == "add":
        return ttnn.add(x, y, fast_and_approximate_mode=True, **kw)
    return ttnn.subtract(x, y, fast_and_approximate_mode=True, **kw)


BIG = [(a, b, o, order) for a in ("bf16", "bfp8", "bfp4") for b in ("bfp8", "bfp4") for o in ("bf16", "fp32") for order in ("ab", "ba")]


@pytest.mark.parametrize("a_dt, b_dt, out_dt, order", BIG, ids=["-".join(c) for c in BIG])
def test_nob_partial(device, a_dt, b_dt, out_dt, order):
    """16384 tiles: every A pattern against 256 B values (16 groups of the table, the table cycled over the patterns)."""
    t0 = time.time()
    table = bfp_table(b_dt)
    n_groups = table.size // 16
    a = np.repeat(PATS, 256)
    slots = np.arange(a.size // 16)
    g = (slots * 7 + slots // n_groups) % n_groups
    b = table.reshape(n_groups, 16)[g].reshape(-1)
    shape = (1, 1, 16384, 1024)
    ta_t = bf16_from_bits(a).reshape(shape)
    ta = to_dev(ta_t if a_dt == "bf16" else ta_t.float(), DT[a_dt], device)
    tb = to_dev(torch.from_numpy(b).reshape(shape), DT[b_dt], device)
    x, y = (ta, tb) if order == "ab" else (tb, ta)
    xv, yv = tensor_vals(x), tensor_vals(y)
    diffs = make_diffs(f"nob_{a_dt}_{b_dt}_{out_dt}_{order}", "mul")
    run_chunk(device, diffs, lambda: out_bits(binop("mul", x, y, out_dt)), xv, yv)
    for *_, d in diffs:
        d.report(f"({time.time() - t0:.1f} s)")


ROW = [(b, o, order) for b in ("bfp8", "bfp4") for o in ("bf16", "fp32") for order in ("ab", "ba")]


@pytest.mark.parametrize("b_dt, out_dt, order", ROW, ids=["-".join(c) for c in ROW])
def test_row_partial(device, b_dt, out_dt, order):
    """A (1, 1, 32, 524288) holds every pattern 256 times; B (1, 1, 1, 524288), the table cycled, broadcast down the rows."""
    t0 = time.time()
    table = bfp_table(b_dt)
    W = 524288
    a = np.tile(PATS, 256).reshape(32, W)
    b = np.resize(table, W)
    ta = to_dev(bf16_from_bits(a.reshape(-1)).reshape(1, 1, 32, W), ttnn.bfloat16, device)
    tb = to_dev(torch.from_numpy(b).reshape(1, 1, 1, W), DT[b_dt], device)
    x, y = (ta, tb) if order == "ab" else (tb, ta)
    av = f32_of_bf16(a.reshape(-1))
    bv = np.broadcast_to(tensor_vals(tb)[:W], (32, W)).reshape(-1)
    xv, yv = (av, bv) if order == "ab" else (bv, av)
    diffs = make_diffs(f"row_{b_dt}_{out_dt}_{order}", "mul")
    run_chunk(device, diffs, lambda: out_bits(binop("mul", x, y, out_dt)), xv, yv)
    for *_, d in diffs:
        d.report(f"({time.time() - t0:.1f} s)")


SCAL = [(b, o) for b in ("bfp8", "bfp4") for o in ("bf16", "fp32")]


@pytest.mark.parametrize("b_dt, out_dt", SCAL, ids=["-".join(c) for c in SCAL])
def test_scalar_lhs(device, b_dt, out_dt):
    """multiply(scalar, B): every bf16 pattern as the scalar (SrcA) against the whole B table (SrcB): the full cross product.
    One digest per call and side; the differing calls are recomputed on both sides (up to EB_DUMP_MAX_DETAIL) and classified."""
    t0 = time.time()
    table = bfp_table(b_dt)
    W = 1024
    rows = -(-table.size // W)
    rows = -(-rows // 32) * 32
    b = np.resize(table, rows * W)
    tb = to_dev(torch.from_numpy(b).reshape(1, 1, rows, W), DT[b_dt], device)
    bv = tensor_vals(tb)
    scalars = f32_of_bf16(PATS)

    def call(i):
        return out_bits(ttnn.multiply(float(scalars[i]), tb, dtype=DT[out_dt], memory_config=ttnn.DRAM_MEMORY_CONFIG))

    prs = pairs()
    envs = unique_envs(prs)
    dig = {}
    for lab, env in envs.items():
        set_env(device, env)
        dig[lab] = [digest(call(i)) for i in range(PATS.size)]
    set_env(device, {})
    for a, b_ in prs:
        la, lb = env_label(a), env_label(b_)
        bad = [i for i in range(PATS.size) if dig[la][i] != dig[lb][i]]
        tag = f"scal_{b_dt}_{out_dt} [{la} | {lb}]"
        print(f"\nDUMP {tag}: calls {PATS.size}, outputs {PATS.size * b.size}, calls that differ {len(bad)} ({time.time() - t0:.1f} s)", flush=True)
        if not bad:
            continue
        print(f"DUMP {tag} differing scalars (first 64): {[hex(int(PATS[i])) for i in bad[:64]]}")
        from eb_dump_lib import cls_bits, NAMES

        sc = cls_bits(PATS[np.array(bad)])
        u, n = np.unique(sc, return_counts=True)
        print(f"DUMP {tag} differing calls by scalar class: {dict((NAMES[int(k)], int(c)) for k, c in zip(u, n))}")
        det = bad[:MAX_DETAIL_CALLS]
        outs = {}
        for lab, env in ((la, a), (lb, b_)):
            set_env(device, env)
            outs[lab] = [call(i) for i in det]
        set_env(device, {})
        d = Diff(tag + " detail", "mul", ("A", "B"))
        for k, i in enumerate(det):
            d.add(outs[la][k], outs[lb][k], np.full(bv.size, scalars[i], dtype=np.float32), bv)
        d.report(f"(detail of {len(det)} of {len(bad)} differing calls)")


def _chunk_plan(nbv, target=18_000_000):
    """Patterns per chunk P: P * nbv a multiple of 32 * 1024, about target elements."""
    g = 32768 // np.gcd(nbv, 32768)
    p = max(g, (target // (nbv * g)) * g)
    p = min(p, -(-65536 // g) * g)
    return int(p), -(-65536 // p)


def _b_source(src):
    """(values as float32, bit patterns as bf16 or None, device dtype)."""
    if src == "b16set":
        u = b16_set()
        return f32_of_bf16(u), u, ttnn.bfloat16
    if src == "b16all":
        return f32_of_bf16(PATS), PATS, ttnn.bfloat16
    return bfp_table(src), None, DT[src]


CROSS = [
    (src, op, o, order)
    for src in ("b16set", "b16all", "bfp8", "bfp4")
    for op in ("mul", "add", "sub")
    for o in ("bf16", "fp32")
    for order in ("ab", "ba")
    if not (src == "b16all" and order == "ba")
]


@pytest.mark.parametrize("src, op, out_dt, order", CROSS, ids=["-".join(c) for c in CROSS])
def test_nob_cross(device, src, op, out_dt, order):
    """Every bf16 pattern against every value of the B source (full cross product), interleaved DRAM, no broadcast. A chunk
    holds P patterns, each repeated against the whole source; the B tensor (the source tiled P times) stays on the device."""
    t0 = time.time()
    bvals, bbits, bdt = _b_source(src)
    nbv = bvals.size
    P, nchunks = _chunk_plan(nbv)
    maxc = int(os.environ.get("EB_DUMP_MAX_CHUNKS", "0")) or nchunks
    W = 1024
    rows = P * nbv // W
    shape = (1, 1, rows, W)
    if bbits is not None:
        tb = to_dev(bf16_from_bits(np.tile(bbits, P)).reshape(shape), bdt, device)
    else:
        tb = to_dev(torch.from_numpy(np.tile(bvals, P)).reshape(shape), bdt, device)
    bv_dev = tensor_vals(tb)
    diffs = make_diffs(f"cross_{src}_{op}_{out_dt}_{order}", op)
    done = 0
    for c in range(min(nchunks, maxc)):
        idx = (np.arange(P) + c * P) % 65536
        a = np.repeat(PATS[idx], nbv)
        ta = to_dev(bf16_from_bits(a).reshape(shape), ttnn.bfloat16, device)
        av = f32_of_bf16(a)
        x, y = (ta, tb) if order == "ab" else (tb, ta)
        xv, yv = (av, bv_dev) if order == "ab" else (bv_dev, av)
        run_chunk(device, diffs, lambda: out_bits(binop(op, x, y, out_dt)), xv, yv)
        ttnn.deallocate(ta)
        done += 1
    cover = min(65536, done * P)
    for *_, d in diffs:
        d.report(f"(a patterns {cover} of 65536 x {nbv} b values, {done} chunks of {P}, {time.time() - t0:.1f} s)")
