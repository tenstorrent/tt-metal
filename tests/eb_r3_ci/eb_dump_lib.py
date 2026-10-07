# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58818 review): helpers for the exhaustive bit-identity dumps.

Two programs are compared in one process: the binary_ng factory reads its CI toggles (EB_R3_*) at program creation, so the
dump sets the toggles of side A, clears the program cache, runs, sets those of side B, clears the cache, runs again and
compares the output bits. Every toggle changes a kernel define or the compute config, so the two sides compile to two kernel
binaries. Differences are classified by operand classes, the class of the exact result and the classes of both outputs."""
import hashlib
import os

import numpy as np
import torch
import ttnn

TOGGLES = (
    "EB_R3_PER_FACE",
    "EB_R3_FIDELITY",
    "EB_R3_NO_BLOCK",
    "EB_R3_NO_BLOCK_PACK",
    "EB_R3_NO_BCAST_CHUNK",
    "EB_R3_NO_BCAST_ACT",
    "EB_R3_NO_HIFI3",
    "EB_R3_MAIN_REINIT",
    "EB_R3_BU_MIN",
    "EB_R3_BP_MIN",
    "EB_R3_BCAST_FIDELITY",
    "EB_R3_BCAST_FP32",
)

NAMES = ["zero", "denorm", "normal", "inf", "nan", "-"]
PATS = np.arange(65536, dtype=np.uint32).astype(np.uint16)


def parse_env(s):
    """'K=V,K2=V2' (or empty) into a dict."""
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if part and part != "default":
            k, _, v = part.partition("=")
            out[k] = v if v else "1"
    return out


def env_label(env):
    return ",".join(f"{k}={v}" for k, v in sorted(env.items())) or "default"


def set_env(device, env):
    for k in TOGGLES:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = str(v)
    if hasattr(device, "clear_program_cache"):
        device.clear_program_cache()
    else:
        device.disable_and_clear_program_cache()
        device.enable_program_cache()


def sides():
    """Side A (the reference, usually main's program) and side B from EB_DUMP_A / EB_DUMP_B."""
    return parse_env(os.environ.get("EB_DUMP_A", "")), parse_env(os.environ.get("EB_DUMP_B", ""))


def with_env(base, extra):
    d = dict(base)
    d.update(extra)
    return d


def bf16_from_bits(u16):
    return torch.from_numpy(np.ascontiguousarray(u16, dtype=np.uint16).view(np.int16)).view(torch.bfloat16)


def f32_of_bf16(u16):
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


def f32_bits(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def out_bits(t):
    o = ttnn.to_torch(t)
    if o.dtype == torch.bfloat16:
        return o.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1).copy()
    return o.contiguous().float().numpy().view(np.uint32).reshape(-1).copy()


def tensor_vals(t):
    """float32 values of a device tensor as read back (block-float quantization included)."""
    return ttnn.to_torch(t).float().numpy().reshape(-1).copy()


def cls_bits(bits):
    """0 zero, 1 denormal, 2 normal, 3 inf, 4 nan, by bit width of the array."""
    bits = np.asarray(bits)
    if bits.dtype == np.uint16:
        e, m = (bits >> 7) & 0xFF, bits & 0x7F
    else:
        bits = bits.astype(np.uint32)
        e, m = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    c = np.full(bits.shape, 2, dtype=np.int64)
    c[(e == 0) & (m == 0)] = 0
    c[(e == 0) & (m != 0)] = 1
    c[(e == 0xFF) & (m == 0)] = 3
    c[(e == 0xFF) & (m != 0)] = 4
    return c


def cls_f32(x):
    return cls_bits(f32_bits(x))


def _exact_cls(op, a, b, width16):
    if op is None or a is None or b is None:
        return np.full(np.shape(a) if a is not None else (0,), 5, dtype=np.int64)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    with np.errstate(all="ignore"):
        r = {"mul": a * b, "add": a + b, "sub": a - b}[op]
    lim = 3.3895313892515355e38 if width16 else 3.4028234663852886e38
    tiny = 1.1754943508222875e-38
    return np.where(
        np.isnan(r), 4, np.where(np.isinf(r) | (np.abs(r) > lim), 3, np.where(r == 0, 0, np.where(np.abs(r) < tiny, 1, 2)))
    ).astype(np.int64)


def _hexv(x):
    return f"0x{int(f32_bits(np.float32(x))):08x}"


class Diff:
    """Bit comparison of two output streams of one configuration, accumulated over chunks."""

    def __init__(self, tag, op=None, labels=("A", "B"), max_examples=3):
        self.tag, self.op, self.labels, self.maxex = tag, op, labels, max_examples
        self.n = 0
        self.nd = 0
        self.counts = {}
        self.examples = {}
        self.fin_n = 0
        self.fin_maxrel = 0.0
        self.ulp_hist = {}
        self.chunks = 0

    def add(self, ref, got, a=None, b=None):
        ref = np.asarray(ref).reshape(-1)
        got = np.asarray(got).reshape(-1)
        assert ref.shape == got.shape and ref.dtype == got.dtype, (ref.shape, got.shape, ref.dtype, got.dtype)
        self.chunks += 1
        self.n += ref.size
        d = np.flatnonzero(ref != got)
        if not d.size:
            return
        self.nd += d.size
        av = None if a is None else np.asarray(a, dtype=np.float32).reshape(-1)[d]
        bv = None if b is None else np.asarray(b, dtype=np.float32).reshape(-1)[d]
        ca = cls_f32(av) if av is not None else np.full(d.size, 5, dtype=np.int64)
        cb = cls_f32(bv) if bv is not None else np.full(d.size, 5, dtype=np.int64)
        cx = _exact_cls(self.op, av, bv, ref.dtype == np.uint16) if (av is not None and bv is not None) else np.full(d.size, 5, dtype=np.int64)
        rr, gg = ref[d], got[d]
        cr, cg = cls_bits(rr), cls_bits(gg)
        key = (((ca * 6 + cb) * 6 + cx) * 6 + cr) * 6 + cg
        u, idx, n = np.unique(key, return_index=True, return_counts=True)
        for k, i0, c in zip(u.tolist(), idx.tolist(), n.tolist()):
            self.counts[k] = self.counts.get(k, 0) + c
            ex = self.examples.setdefault(k, [])
            if len(ex) < self.maxex:
                sel = np.flatnonzero(key == k)[: self.maxex - len(ex)]
                for j in sel.tolist():
                    w = 4 if ref.dtype == np.uint16 else 8
                    ex.append(
                        (f"a={float(av[j])!r}({_hexv(av[j])}) " if av is not None else "")
                        + (f"b={float(bv[j])!r}({_hexv(bv[j])}) " if bv is not None else "")
                        + f"{self.labels[0]}=0x{int(rr[j]):0{w}x} {self.labels[1]}=0x{int(gg[j]):0{w}x}"
                    )
        fin = (cr == 2) & (cg == 2)
        if fin.any():
            if ref.dtype == np.uint16:
                r = f32_of_bf16(rr[fin]).astype(np.float64)
                g = f32_of_bf16(gg[fin]).astype(np.float64)
            else:
                r = rr[fin].view(np.float32).astype(np.float64) if rr.dtype == np.uint32 else rr[fin].astype(np.float64)
                g = gg[fin].view(np.float32).astype(np.float64) if gg.dtype == np.uint32 else gg[fin].astype(np.float64)
            rel = np.abs(r - g) / np.maximum(np.abs(r), 1e-300)
            self.fin_n += int(fin.sum())
            self.fin_maxrel = max(self.fin_maxrel, float(rel.max()))
            same_sign = ((rr[fin] ^ gg[fin]) >> (15 if ref.dtype == np.uint16 else 31)) == 0
            ulp = np.abs(rr[fin].astype(np.int64) - gg[fin].astype(np.int64))
            ulp = np.where(same_sign, ulp, -1)
            for v, c in zip(*np.unique(np.minimum(ulp, 4), return_counts=True)):
                self.ulp_hist[int(v)] = self.ulp_hist.get(int(v), 0) + int(c)

    def report(self, extra=""):
        print(f"\nDUMP {self.tag}: outputs {self.n}, differ {self.nd}{(' ' + extra) if extra else ''}", flush=True)
        if not self.nd:
            return
        print(f"DUMP {self.tag} classes (a, b, exact, {self.labels[0]}, {self.labels[1]}): count")
        for k, c in sorted(self.counts.items(), key=lambda t: -t[1]):
            parts = [NAMES[(k // 6**p) % 6] for p in (4, 3, 2, 1, 0)]
            print(f"DUMP {self.tag}   {parts}: {c}   e.g. " + "; ".join(self.examples.get(k, [])))
        if self.fin_n:
            h = ", ".join(f"{'opposite sign' if k < 0 else (str(k) + ('+' if k == 4 else ''))} ulp: {v}" for k, v in sorted(self.ulp_hist.items()))
            print(f"DUMP {self.tag} normal outputs on both sides: {self.fin_n} differ, max rel {self.fin_maxrel:.3e}; {h}")


def run_sides(device, envs, fn):
    """fn() under each env (program cache cleared at each switch); returns the list of results."""
    out = []
    for env in envs:
        set_env(device, env)
        out.append(fn())
    set_env(device, {})
    return out


def to_dev(x, dtype, device, memory_config=None, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        x, dtype=dtype, layout=layout, device=device, memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG
    )


def b16_set():
    """bf16 bit patterns: zeros, denormals, min/max normals, infinities, NaN payloads, +-1, and two normals of every exponent."""
    s = [0x0000, 0x8000]
    s += [0x0001, 0x8001, 0x0002, 0x0010, 0x8010, 0x0040, 0x8040, 0x0055, 0x802A, 0x007F, 0x807F]
    s += [0x0080, 0x8080, 0x0081, 0x8081, 0x00FF, 0x80FF]
    s += [0x7F7F, 0xFF7F, 0x7F7E, 0xFF7E, 0x7F00, 0xFF00]
    s += [0x7F80, 0xFF80]
    s += [0x7FC0, 0xFFC0, 0x7F81, 0xFF81, 0x7FFF, 0xFFFF, 0x7FA5, 0xFFD3, 0x7F80 | 0x40, 0x7F80 | 0x01]
    s += [0x3F80, 0xBF80, 0x4000, 0xC000, 0x3F00, 0x3F81, 0x3FFF, 0xBFFF, 0x4049, 0xC049]
    for e in range(1, 255):
        s.append(((e & 1) << 15) | (e << 7) | ((e * 37 + 11) & 0x7F))
        s.append((((e + 1) & 1) << 15) | (e << 7) | ((e * 91 + 17) & 0x7F))
    out, seen = [], set()
    for v in s:
        if v not in seen:
            seen.add(v)
            out.append(v)
    while len(out) % 16:
        out.append(0x3F80)
    return np.array(out, dtype=np.uint16)


def b16_small(n=64):
    """A reduced set (n values): the special values first, then normals spread over the exponents."""
    full = b16_set()
    special = full[:52]
    rest = full[52:]
    pick = rest[np.linspace(0, rest.size - 1, max(n - special.size, 0)).astype(int)]
    return np.concatenate([special, pick])[:n].astype(np.uint16)


def bfp_table(fmt):
    """Groups of 16 float32 values, one shared exponent each: 15 datum codes and an anchor holding the group's exponent;
    every (sign, mantissa) code under every raw shared exponent 1..255 (255: Inf and NaN codes), as in test_eb_hifi2_dump."""
    mbits = 7 if fmt == "bfp8" else 3
    lead = 1 << (mbits - 1)
    mmax = (1 << mbits) - 1
    codes = [(0, 0)] + [(s, m) for m in range(1, mmax + 1) for s in (0, 1)]
    vals = []
    for E in range(1, 256):

        def value(s, m):
            if E == 255 and m >= lead:
                frac = (m - lead) << (23 - (mbits - 1))
                return np.array([(s << 31) | (255 << 23) | frac], dtype=np.uint32).view(np.float32)[0]
            return np.float32((-1.0) ** s * m * 2.0 ** (E - 127 - (mbits - 1)))

        anchor = (0, mmax) if E < 255 else (0, lead)
        for i in range(0, len(codes), 15):
            chunk = codes[i : i + 15]
            while len(chunk) < 15:
                chunk.append((0, 0))
            for s, m in chunk + [anchor]:
                vals.append(value(s, m))
        if E == 255:
            for s, m in codes[:15] + [(0, lead + 1)]:
                vals.append(value(s, m))
    return np.array(vals, dtype=np.float32)


def digest(x):
    return hashlib.sha1(np.ascontiguousarray(x).tobytes()).digest()


def kernel_variants(tag=""):
    """Print the compiled compute kernel variants (name: number of hashes) in TT_METAL_CACHE, the evidence that the two
    sides of a comparison ran two binaries."""
    root = os.environ.get("TT_METAL_CACHE")
    if not root or not os.path.isdir(root):
        return
    names = {}
    for dp, dn, fn in os.walk(root):
        parts = dp.split(os.sep)
        if "kernels" in parts:
            i = len(parts) - 1 - parts[::-1].index("kernels")
            if len(parts) == i + 3:
                names.setdefault(parts[i + 1], set()).add(parts[i + 2])
                dn[:] = []
    print(f"DUMP variants{tag}: " + ", ".join(f"{k}:{len(v)}" for k, v in sorted(names.items())), flush=True)


def pairs():
    spec = os.environ.get("EB_DUMP_PAIRS")
    if spec:
        out = []
        for p in spec.split(";"):
            if p.strip():
                a, _, b = p.partition("|")
                out.append((parse_env(a), parse_env(b)))
        return out
    return [(parse_env(os.environ.get("EB_DUMP_A", "")), parse_env(os.environ.get("EB_DUMP_B", "")))]


def unique_envs(prs):
    envs = {}
    for a, b in prs:
        envs.setdefault(env_label(a), a)
        envs.setdefault(env_label(b), b)
    return envs


def make_diffs(tag, op):
    return [(env_label(a), env_label(b), Diff(f"{tag} [{env_label(a)} | {env_label(b)}]", op, ("A", "B"))) for a, b in pairs()]


def run_chunk(device, diffs, fn, a_vals, b_vals, valid=None):
    """valid: compare only the first `valid` outputs (a chunk padded by wrapping around)."""
    envs = unique_envs(pairs())
    outs = {}
    for lab, env in envs.items():
        set_env(device, env)
        outs[lab] = fn()
    set_env(device, {})
    v = slice(None) if valid is None else slice(0, valid)
    for la, lb, d in diffs:
        d.add(outs[la][v], outs[lb][v], None if a_vals is None else a_vals[v], None if b_vals is None else b_vals[v])


def stage(tag, out, a=None, b=None, op=None, extra=""):
    """save: keep out; cmp: compare with the saved main-side output and report."""
    STAGE = os.environ.get("EB_DUMP_STAGE", "save")
    DIR = os.environ.get("EB_DUMP_DIR", "/tmp/eb_kedit")
    os.makedirs(DIR, exist_ok=True)
    path = os.path.join(DIR, tag.replace(" ", "_").replace("/", "_") + ".npy")
    if STAGE == "save":
        np.save(path, out)
        print(f"\nDUMP {tag}: saved {out.size} outputs {extra}", flush=True)
        return
    if not os.path.exists(path):
        print(f"\nDUMP {tag}: NO MAIN-SIDE OUTPUT", flush=True)
        return
    ref = np.load(path)
    if ref.shape != out.shape or ref.dtype != out.dtype:
        print(f"\nDUMP {tag}: SHAPE/DTYPE MISMATCH main {ref.shape} {ref.dtype} pr {out.shape} {out.dtype}", flush=True)
        return
    d = Diff(f"{tag} [main | PR]", op, ("main", "PR"))
    d.add(ref, out, a, b)
    d.report(extra)
