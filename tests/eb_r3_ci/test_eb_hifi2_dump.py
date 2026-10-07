# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723 review): exhaustive bit dump of binary_ng's HiFi2 rule against HiFi4.

SrcA: all 65536 bf16 bit patterns (and block-float SrcA built from them). SrcB: every bfp8_b and bfp4_b datum that from_torch
produces, under every shared exponent (raw 1 to 255, 255 holding Inf and NaN). Outputs bf16 (16-bit DEST) and Float32 (fp32
DEST). Kernels: no broadcast (both operand orders), row broadcast (both orders), scalar on the left (every bf16 scalar against
the whole SrcB table: the full cross product).

EB_DUMP_STAGE=ref with EB_R3_NO_HIFI2=1 writes HiFi4 outputs (scalar-left: one digest per call) under EB_DUMP_DIR;
EB_DUMP_STAGE=cmp (rule on) compares bit for bit and keeps the HiFi2 outputs of differing scalar calls;
EB_DUMP_STAGE=detail with EB_R3_NO_HIFI2=1 recomputes those calls at HiFi4 and prints every differing element."""
import hashlib
import os

import numpy as np
import pytest
import torch
import ttnn

STAGE = os.environ.get("EB_DUMP_STAGE", "ref")
DIR = os.environ.get("EB_DUMP_DIR", "/tmp/eb_hifi2_dump")
MAX_DETAIL_CALLS = 1024


@pytest.fixture(scope="module")
def device():
    dev = ttnn.CreateDevice(device_id=0, l1_small_size=24576)
    yield dev
    ttnn.close_device(dev)


def _bits32(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _f32(bits):
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def _b_table(fmt):
    """Groups of 16 float32 values (one shared exponent each): 15 datum codes and an anchor with the group's exponent.
    Returns (values [n_groups*16] float32, expected codes [n_groups*16] as (E, sign, m))."""
    mbits = 7 if fmt == "bfp8" else 3
    lead = 1 << (mbits - 1)
    mmax = (1 << mbits) - 1
    codes = [(0, 0)] + [(s, m) for m in range(1, mmax + 1) for s in (0, 1)]
    vals, meta = [], []
    for E in range(1, 256):
        def value(s, m):
            if E == 255 and m >= lead:
                frac = (m - lead) << (23 - (mbits - 1))
                return _f32((s << 31) | (255 << 23) | frac)[()]
            return np.float32((-1.0) ** s * m * 2.0 ** (E - 127 - (mbits - 1)))

        anchor = (0, mmax) if E < 255 else (0, lead)
        for i in range(0, len(codes), 15):
            chunk = codes[i : i + 15]
            while len(chunk) < 15:
                chunk.append((0, 0))
            for s, m in chunk + [anchor]:
                vals.append(value(s, m))
                meta.append((E, s, m))
        if E == 255:
            # one more group anchored by a NaN
            for s, m in codes[:15] + [(0, lead + 1)]:
                vals.append(value(s, m))
                meta.append((E, s, m))
    return np.array(vals, dtype=np.float32), meta


def _a_patterns():
    return np.arange(65536, dtype=np.uint32).astype(np.uint16)


def _bf16_from_bits(u16):
    return torch.from_numpy(u16.view(np.int16).copy()).view(torch.bfloat16)


def _out_bits(t):
    o = ttnn.to_torch(t)
    if o.dtype == torch.bfloat16:
        return o.view(torch.int16).numpy().view(np.uint16).copy()
    return o.float().numpy().view(np.uint32).copy()


def _to_dev(x, dtype, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}
OUT = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}


def _cls_bits(bits, dt):
    """0 zero, 1 denormal, 2 normal, 3 inf, 4 nan."""
    if dt == np.uint16:
        e, m = (bits >> 7) & 0xFF, bits & 0x7F
    else:
        e, m = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    c = np.full(bits.shape, 2, dtype=np.int8)
    c[(e == 0) & (m == 0)] = 0
    c[(e == 0) & (m != 0)] = 1
    c[(e == 0xFF) & (m == 0)] = 3
    c[(e == 0xFF) & (m != 0)] = 4
    return c


NAMES = ["zero", "denorm", "normal", "inf", "nan"]


def _report_diff(tag, ref, got, a_bits, b_vals):
    d = np.nonzero(ref != got)[0]
    print(f"\nDUMP {tag}: elements {ref.size}, differ {d.size}", flush=True)
    if not d.size:
        return
    a = _f32(np.asarray(a_bits, dtype=np.uint32)[d] << 16) if np.asarray(a_bits).dtype == np.uint16 else np.asarray(a_bits)[d]
    b = np.asarray(b_vals, dtype=np.float32)[d]
    ca = _cls_bits(_bits32(a), np.uint32)
    cb = _cls_bits(_bits32(b), np.uint32)
    cr = _cls_bits(ref[d], ref.dtype)
    cg = _cls_bits(got[d], got.dtype)
    with np.errstate(all="ignore"):
        prod = a.astype(np.float64) * b.astype(np.float64)
    lim = 3.3895313892515355e38 if ref.dtype == np.uint16 else 3.4028234663852886e38
    pc = np.where(np.isnan(prod), 4, np.where(np.isinf(prod) | (np.abs(prod) > lim), 3, np.where(prod == 0, 0, np.where(np.abs(prod) < 1.1754943508222875e-38, 1, 2))))
    key = ca.astype(np.int64) * 625 + cb.astype(np.int64) * 125 + pc.astype(np.int64) * 25 + cr.astype(np.int64) * 5 + cg.astype(np.int64)
    u, n = np.unique(key, return_counts=True)
    print(f"DUMP {tag} classes (a, b, exact product, hifi4, hifi2): count")
    for k, c in sorted(zip(u, n), key=lambda t: -t[1]):
        k = int(k)
        parts = [NAMES[(k // 625) % 5], NAMES[(k // 125) % 5], NAMES[(k // 25) % 5], NAMES[(k // 5) % 5], NAMES[k % 5]]
        sel = d[key == k][:3]
        ex = "; ".join(f"a={float(a[np.searchsorted(d, j)])!r} b={float(np.asarray(b_vals)[j])!r} h4=0x{int(ref[j]):x} h2=0x{int(got[j]):x}" for j in sel)
        print(f"DUMP {tag}   {parts}: {int(c)}   e.g. {ex}")
    fin = (ca == 2) & (cb == 2) & (cr == 2) & (cg == 2)
    if fin.any():
        r = _val_arr(ref[d][fin], ref.dtype); g = _val_arr(got[d][fin], got.dtype)
        rel = np.abs(r.astype(np.float64) - g) / np.maximum(np.abs(r.astype(np.float64)), 1e-300)
        print(f"DUMP {tag} finite normal operands and outputs: {int(fin.sum())} differ, max rel {rel.max():.3e}, median rel {np.median(rel):.3e}")


def _val_arr(bits, dt):
    if dt == np.uint16:
        return _f32(bits.astype(np.uint32) << 16)
    return _f32(bits)


def _val(bits, dt):
    if dt == np.uint16:
        return np.array([bits << 16], dtype=np.uint32).view(np.float32)[0]
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def _stage_io(tag, out):
    path = os.path.join(DIR, tag + ".npy")
    os.makedirs(DIR, exist_ok=True)
    if STAGE == "ref":
        np.save(path, out)
        print(f"\nDUMP {tag}: reference saved ({out.size} elements)", flush=True)
        return None
    return np.load(path)


BIG = [
    (a, b, o, order)
    for a in ("bf16", "bfp8", "bfp4")
    for b in ("bfp8", "bfp4")
    for o in ("bf16", "fp32")
    for order in ("ab", "ba")
]


@pytest.mark.parametrize("a_dt, b_dt, out_dt, order", BIG, ids=["-".join(c) for c in BIG])
def test_no_bcast(device, a_dt, b_dt, out_dt, order):
    """16384 tiles: every A pattern against 256 B values (16 groups of the table, the table cycled over the patterns)."""
    if STAGE == "detail":
        pytest.skip("big cases compare in the cmp stage")
    table, _ = _b_table(b_dt)
    n_groups = table.size // 16
    a = np.repeat(_a_patterns(), 256)
    slots = np.arange(a.size // 16)
    g = (slots * 7 + slots // n_groups) % n_groups
    b = table.reshape(n_groups, 16)[g].reshape(-1)
    shape = (1, 1, 16384, 1024)
    ta_t = _bf16_from_bits(a).reshape(shape)
    ta = _to_dev(ta_t if a_dt == "bf16" else ta_t.float(), DT[a_dt], device)
    tb = _to_dev(torch.from_numpy(b).reshape(shape), DT[b_dt], device)
    x, y = (ta, tb) if order == "ab" else (tb, ta)
    kw = dict(dtype=OUT[out_dt], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if a_dt == "bf16":
        kw["fast_and_approximate_mode"] = True
    out = _out_bits(ttnn.multiply(x, y, **kw)).reshape(-1)
    tag = f"nob_{a_dt}_{b_dt}_{out_dt}_{order}"
    ref = _stage_io(tag, out)
    if ref is not None:
        av = ttnn.to_torch(ta).float().numpy().reshape(-1)
        bv = ttnn.to_torch(tb).float().numpy().reshape(-1)
        _report_diff(tag, ref, out, av, bv)


ROW = [(b, o, order) for b in ("bfp8", "bfp4") for o in ("bf16", "fp32") for order in ("ab", "ba")]


@pytest.mark.parametrize("b_dt, out_dt, order", ROW, ids=["-".join(c) for c in ROW])
def test_row_bcast(device, b_dt, out_dt, order):
    """A (1, 1, 32, 524288) holds every pattern 256 times; B (1, 1, 1, 524288), the table cycled, broadcast down the rows."""
    if STAGE == "detail":
        pytest.skip("big cases compare in the cmp stage")
    table, _ = _b_table(b_dt)
    W = 524288
    a = np.tile(_a_patterns(), 256).reshape(32, W)
    b = np.resize(table, W)
    ta = _to_dev(_bf16_from_bits(a.reshape(-1)).reshape(1, 1, 32, W), ttnn.bfloat16, device)
    tb = _to_dev(torch.from_numpy(b).reshape(1, 1, 1, W), DT[b_dt], device)
    x, y = (ta, tb) if order == "ab" else (tb, ta)
    out = _out_bits(
        ttnn.multiply(x, y, dtype=OUT[out_dt], memory_config=ttnn.DRAM_MEMORY_CONFIG, fast_and_approximate_mode=True)
    ).reshape(-1)
    tag = f"row_{b_dt}_{out_dt}_{order}"
    ref = _stage_io(tag, out)
    if ref is not None:
        bv = ttnn.to_torch(tb).float().numpy().reshape(-1)[:W]
        _report_diff(tag, ref, out, _f32(a.reshape(-1).astype(np.uint32) << 16), np.broadcast_to(bv, (32, W)).reshape(-1))


SCAL = [(b, o) for b in ("bfp8", "bfp4") for o in ("bf16", "fp32")]


@pytest.mark.parametrize("b_dt, out_dt", SCAL, ids=["-".join(c) for c in SCAL])
def test_scalar_lhs(device, b_dt, out_dt):
    """multiply(scalar, B): every bf16 pattern as the scalar (SrcA) against the whole B table (SrcB, the tensor)."""
    table, meta = _b_table(b_dt)
    W = 1024
    rows = -(-table.size // W)
    rows = -(-rows // 32) * 32
    b = np.resize(table, rows * W)
    tb = _to_dev(torch.from_numpy(b).reshape(1, 1, rows, W), DT[b_dt], device)
    back = ttnn.to_torch(tb).float().numpy().reshape(-1)[: table.size]
    same = (_bits32(back) == _bits32(table)) | (np.isnan(back) & np.isnan(table))
    print(f"\nDUMP scal_{b_dt} table: {table.size} values, {int(same.sum())} read back as written", flush=True)
    pats = _a_patterns()
    scalars = _bf16_from_bits(pats).float().numpy()
    tag = f"scal_{b_dt}_{out_dt}"
    os.makedirs(DIR, exist_ok=True)
    dig_path = os.path.join(DIR, tag + "_digests.npy")
    bad_path = os.path.join(DIR, tag + "_bad.npy")

    def call(i):
        return _out_bits(
            ttnn.multiply(float(scalars[i]), tb, dtype=OUT[out_dt], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ).reshape(-1)

    if STAGE == "ref":
        dig = np.zeros((pats.size, 20), dtype=np.uint8)
        for i in range(pats.size):
            dig[i] = np.frombuffer(hashlib.sha1(call(i).tobytes()).digest(), dtype=np.uint8)
        np.save(dig_path, dig)
        print(f"DUMP {tag}: {pats.size} reference digests saved", flush=True)
        return
    if STAGE == "cmp":
        ref = np.load(dig_path)
        bad = []
        for i in range(pats.size):
            o = call(i)
            if np.frombuffer(hashlib.sha1(o.tobytes()).digest(), dtype=np.uint8).tobytes() != ref[i].tobytes():
                bad.append(i)
                if len(bad) <= 256 or len(bad) % 16 == 0:
                    np.save(os.path.join(DIR, f"{tag}_h2_{i}.npy"), o)
        np.save(bad_path, np.array(bad, dtype=np.int64))
        print(f"DUMP {tag}: calls {pats.size}, elements {pats.size * b.size}, calls that differ {len(bad)}", flush=True)
        if bad:
            print(f"DUMP {tag} differing scalars (first 64): {[hex(int(pats[i])) for i in bad[:64]]}")
        return
    bad = np.load(bad_path) if os.path.exists(bad_path) else np.array([], dtype=np.int64)
    bv = ttnn.to_torch(tb).float().numpy().reshape(-1)
    H4, H2, AV, BV = [], [], [], []
    for i in bad:
        f2 = os.path.join(DIR, f"{tag}_h2_{int(i)}.npy")
        if not os.path.exists(f2) or len(H4) >= MAX_DETAIL_CALLS:
            continue
        h4 = call(int(i))
        h2 = np.load(f2)
        H4.append(h4); H2.append(h2); AV.append(np.full(h4.shape, scalars[i], dtype=np.float32)); BV.append(bv)
    print(f"DUMP {tag}: detail of {len(H4)} of {len(bad)} differing calls", flush=True)
    if H4:
        _report_diff(tag + "_detail", np.concatenate(H4), np.concatenate(H2), np.concatenate(AV), np.concatenate(BV))
    sc = _cls_bits(_bits32(scalars[bad]), np.uint32) if len(bad) else np.array([])
    u, n = np.unique(sc, return_counts=True)
    print(f"DUMP {tag} differing calls by scalar class: {dict((NAMES[int(k)], int(c)) for k, c in zip(u, n))}")
