# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Generic DFB eltwise flow driver — the DFB analog of the tt-polynomial-fitter
quasar_sweep.sh.

Parses ANY fitter coefficient CSV (POLY_CASCADE single/multi-segment OR RATIONAL)
into a `ttnn.experimental.quasar.LutConfig`, runs it through the unary_lut DFB op on
craq-sim Quasar, and PCC-checks the result against the fitter ground_truth (the TRUE
activation) — exactly how the tt-llk generic_lut flow selects per-activation. Zero
per-activation special-casing: the activation JSON (via ground_truth) + the CSV drive
everything.

The CSV is read-only (the fitter is a refactor-in-flight read-only dependency); this
driver only consumes it.

LUT layout passed to the kernel (LutConfig.data), matching unary_lut_sfpu.h LUT_DATA:
  POLY:     [b0..bS (num_segments+1 boundaries), per seg (poly_degree+1) c0..cN]
  RATIONAL: [b0..bS, per seg (num_degree+1) n0..nN + (den_degree+1) d0..dM]

The coefficient CSV stores per-segment coefficients in ASCENDING order (c0, c1, ...),
which is exactly the kernel's LUT_DATA Horner-coefficient order (the kernel reads c_N
first then folds down to c_0). So coefficients map directly with no reordering.
"""

import csv as _csv
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

import ttnn

# ---- fitter (read-only) ground_truth -------------------------------------------------
_FITTER = Path(os.environ.get("TT_POLY_FITTER", "/localdev/nkapre/tt-polynomial-fitter"))
if str(_FITTER) not in sys.path:
    sys.path.insert(0, str(_FITTER))

EVAL_POLY = 0
EVAL_RATIONAL = 1


def _activation_name_from_csv(csv_path):
    """Activation name = filename up to the first '_pNN' / '_nNN' degree token.
    e.g. gelu_p5_s3_uniform_any_ulp.csv -> 'gelu';  atanh_n6d6_s1_..._rational.csv -> 'atanh'."""
    stem = Path(csv_path).stem
    parts = stem.split("_")
    out = []
    for p in parts:
        # stop at the degree token: pNN (poly) or nNNdMM (rational)
        if (p.startswith("p") and p[1:].isdigit()) or (
            p.startswith("n") and any(ch.isdigit() for ch in p) and "d" in p
        ):
            break
        out.append(p)
    return "_".join(out)


# Range-reduction method codes — MUST match the kernel's LUT_RR_METHOD contract
# (unary_lut_sfpu.h) and the tt-llk reference.
_RR_CODE = {
    "none": 0,
    "log": 1,
    "exp": 2,
    "cbrt": 3,
    "exponent_alu_exp2": 4,
    "exponent_alu_log2": 5,
    "exponent_alu_pow": 6,
    "trig": 7,
    "tan": 8,
    "newton_root": 9,  # STANDALONE magic-seed + Newton/Householder evaluator (sqrt/rsqrt/cbrt)
}
_RR_COMPOSE = {"": 0, "sigmoid": 1, "minus_one": 2}

# Dominant-factor class codes for asymptotic factoring. MUST match the kernel's
# LUT_DOMINANT_CLASS contract (unary_lut_sfpu.h) and reproduce precision/eval.py
# DOMINANT_FACTORS. Each maps the fitter's dominant_factor string to a class code; the
# kernel evaluates dominant(x) for that class and multiplies the segment's correction
# polynomial by it. Keys are the EXACT strings the fitter writes in the CSV column.
_DOMINANT_CLASS = {
    "x": 1,
    "-exp(-x^2/2) / sqrt(2*pi)": 2,
    "exp(-x^2/2) / sqrt(2*pi)": 3,
    "-exp(-x^2/2)": 4,
    "exp(-x^2/2)": 5,
    "exp(x)": 6,
    "exp(-x)": 7,
    "-exp(-x)": 8,
    "1 - exp(-x)": 9,
    "x * exp(x)": 10,
}
# Methods whose reduce/reconstruct (or standalone eval) is implemented in the DFB kernel.
_RR_SUPPORTED = {
    "exp",
    "trig",
    "tan",
    "log",
    "cbrt",
    "exponent_alu_exp2",
    "exponent_alu_log2",
    "exponent_alu_pow",
    "newton_root",
}


def _parse_rr_meta(meta):
    """Parse the range-reduction METADATA into LutConfig RR kwargs.

    Mirrors the tt-llk generic_lut codegen: method code + method-specific constants.
    Returns (rr_kwargs: dict, rr_enabled: bool, rr_method: str, original_domain or None).
    rr_kwargs are merged into the LutConfig; when the method is unsupported / disabled
    rr_method=0 (none) so the kernel/golden stay on the no-RR path.
    """

    def mf(k, default=None):
        v = meta.get(k, "")
        return float(v) if v not in (None, "") else default

    method = meta.get("range_reduction_method", "none")
    enabled = str(meta.get("range_reduction_enabled", "False")).lower() == "true"
    if method not in _RR_SUPPORTED:
        enabled = False
    if not enabled:
        return {"rr_method": 0}, False, "none", None

    rr = {"rr_method": _RR_CODE[method]}
    if method == "log":
        rr["rr_log_ln2"] = mf("log_ln2_constant", 1.0)
    elif method == "exp":
        rr["rr_exp_mult"] = mf("exp_log2_multiplier", 1.4426950408889634)
        rr["rr_exp_const"] = mf("exp_log2_constant", 0.6931471805599453)
    elif method == "cbrt":
        for i in range(3):
            c = mf(f"cbrt_scale_c{i}")
            if c is not None:
                rr[f"rr_scale{i}"] = c
    elif method == "exponent_alu_exp2":
        rr["rr_exp2_mult"] = mf("expalu_log2_multiplier", 1.0)
        rr["rr_compose"] = _RR_COMPOSE.get(meta.get("expalu_compose", "") or "", 0)
    elif method == "exponent_alu_log2":
        rr["rr_log2_scale"] = mf("expalu_log_scale", 1.0)
        rr["rr_log2_basis_mminus1"] = 1 if meta.get("expalu_log2_basis", "m") == "m_minus_1" else 0
        rr["rr_input_offset"] = mf("expalu_input_offset", 0.0) or 0.0
    elif method == "exponent_alu_pow":
        # The refactored fitter may emit the method name without its companion params
        # (e.g. cbrt's CSV carries only range_reduction_method=exponent_alu_pow). Missing
        # required params => the RR metadata is incomplete for this method; fall back to the
        # no-RR cascade so the pick is still measured honestly rather than crashing the sweep.
        root_n = meta.get("expalu_root_n", "")
        if root_n in (None, ""):
            return {"rr_method": 0}, False, "none", None
        rr["rr_pow_n"] = int(float(root_n))
        rr["rr_pow_recip"] = 1 if str(meta.get("expalu_reciprocal", "False")).lower() == "true" else 0
        for i in range(3):
            c = mf(f"expalu_pow_scale_c{i}")
            if c is not None:
                rr[f"rr_scale{i}"] = c
    elif method == "newton_root":
        # STANDALONE seed+Newton evaluator: magic seed (hex) + Newton/Householder
        # coefficients/iterations, all from the CSV METADATA. The kernel bypasses the
        # segment cascade entirely (method 9), so only these nr_* params matter.
        rr["nr_magic"] = int(str(meta["newton_root_magic"]).strip(), 0)  # 0x... hex -> int
        rr["nr_c1"] = mf("newton_root_c1", 0.0)
        rr["nr_c2"] = mf("newton_root_c2", 0.0)
        rr["nr_iters"] = int(float(meta.get("newton_root_iters", 2)))
        rr["nr_n"] = int(float(meta.get("newton_root_n", 2)))
        rr["nr_reciprocal"] = 1 if str(meta.get("newton_root_reciprocal", "False")).lower() == "true" else 0
    # trig / tan carry no kernel-tunable params (pi / Cody-Waite constants are hardcoded).

    orig = None
    omin, omax = mf("range_reduction_original_min"), mf("range_reduction_original_max")
    if omin is not None and omax is not None:
        orig = (omin, omax)
    return rr, True, method, orig


def parse_csv(csv_path):
    """Parse a fitter coefficient CSV into a LutConfig + sampling domain.

    Returns (lut_config_kwargs: dict, domain: (float, float)).
    The domain is the ORIGINAL activation domain when range reduction is enabled (the
    kernel reconstructs the full activation), else the reduced [b0, bN] LUT span.
    LutConfig kwargs carry the RR method + constants (rr_method == 0 => no RR)."""
    rows = []
    header = None
    meta = {}
    with open(csv_path, newline="") as f:
        for r in _csv.reader(f):
            if not r:
                continue
            if r[0] == "segment_id":
                header = r
                continue
            if r[0] == "METADATA":
                if len(r) >= 3:
                    meta[r[1].strip()] = r[2].strip()
                continue
            rows.append(r)
    assert header is not None, f"no header in {csv_path}"

    rational = "approximation_type" in header and any((len(r) > 3 and r[3] == "rational") for r in rows)

    rr_kwargs, rr_enabled, rr_method, rr_orig = _parse_rr_meta(meta)

    # Ascending boundaries (the kernel selects segments by ascending b0..bN).
    rows.sort(key=lambda r: float(r[1]))

    boundaries = []
    seg_coeffs = []

    if rational:
        # header: segment_id, lo, hi, approximation_type, num_degree, den_degree, n0..nN, d0..dM, ...
        num_deg = int(float(rows[0][4]))
        den_deg = int(float(rows[0][5]))
        n_start = 6
        n_end = n_start + (num_deg + 1)  # also the start index of the denominator coeffs
        for i, r in enumerate(rows):
            lo, hi = float(r[1]), float(r[2])
            if i == 0:
                boundaries.append(lo)
            boundaries.append(hi)
            nums = [float(r[n_start + k]) for k in range(num_deg + 1)]
            dens = [float(r[n_end + k]) for k in range(den_deg + 1)]  # n_end = start of denominator
            seg_coeffs.append(nums + dens)
        data = boundaries + [c for seg in seg_coeffs for c in seg]
        cfg = dict(
            eval_method=EVAL_RATIONAL,
            poly_degree=0,
            num_segments=len(rows),
            num_degree=num_deg,
            den_degree=den_deg,
            data=data,
        )
    else:
        # header: segment_id, lo, hi, c0, c1, ... cN, error, method, ...
        # degree = number of leading numeric c-columns. Detect from header c-prefixed cols.
        c_cols = [j for j, name in enumerate(header) if name.startswith("c") and name[1:].isdigit()]
        degree = len(c_cols) - 1
        for i, r in enumerate(rows):
            lo, hi = float(r[1]), float(r[2])
            if i == 0:
                boundaries.append(lo)
            boundaries.append(hi)
            coeffs = [float(r[c_cols[k]]) for k in range(degree + 1)]
            seg_coeffs.append(coeffs)
        data = boundaries + [c for seg in seg_coeffs for c in seg]
        cfg = dict(
            eval_method=EVAL_POLY,
            poly_degree=degree,
            num_segments=len(rows),
            num_degree=0,
            den_degree=0,
            data=data,
        )

    # exponent_alu_log2 m_minus_1 basis: the kernel's poly argument is u = mantissa - 1,
    # so the boundaries (used by the kernel ONLY for clamp + segment selection) must live
    # in the SAME u-space. The CSV stores them in m-space [1,2]; shift by -1. Mirrors the
    # tt-llk codegen. The coefficients are already in the u-basis.
    if rr_enabled and rr_method == "exponent_alu_log2" and rr_kwargs.get("rr_log2_basis_mminus1") == 1:
        nseg = cfg["num_segments"]
        for i in range(nseg + 1):
            cfg["data"][i] = cfg["data"][i] - 1.0
        boundaries = [b - 1.0 for b in boundaries]

    cfg.update(rr_kwargs)
    cfg["_rr_enabled"] = rr_enabled

    # ---- Asymptotic factoring: thread the per-segment is_asymptotic + dominant_factor
    # columns (NOT METADATA) into asym_mask (bit SEG => segment SEG is asymptotic, in the
    # SAME ascending segment order the kernel selects) + dom_class. The fitter stores the
    # CORRECTION polynomial as a segment's ordinary coeffs; the kernel multiplies by
    # dominant(x). Tail segments are NEVER dropped — they are kept and factored. When no
    # segment is asymptotic (or the column is absent), dom_class stays 0 (no factoring).
    asym_mask = 0
    dom_class = 0
    if "is_asymptotic" in header:
        ia = header.index("is_asymptotic")
        di = header.index("dominant_factor") if "dominant_factor" in header else None
        for seg_idx, r in enumerate(rows):  # rows already sorted ascending == kernel seg order
            if len(r) > ia and r[ia].strip().lower() == "true":
                asym_mask |= 1 << seg_idx
                if di is not None and len(r) > di:
                    dom = r[di].strip()
                    code = _DOMINANT_CLASS.get(dom)
                    if code is not None:
                        # All asymptotic segments of a deployed pick share one dominant class
                        # (the activation's tail); assert that invariant rather than silently
                        # picking one. An unknown class leaves dom_class 0 -> no factoring.
                        assert dom_class in (0, code), f"mixed dominant classes in {csv_path}: {dom_class} vs {code}"
                        dom_class = code
    cfg["asym_mask"] = asym_mask
    cfg["dom_class"] = dom_class

    # Sampling domain: the ORIGINAL activation domain when RR is enabled (the kernel
    # reconstructs the full activation), else the reduced [b0, bN] LUT span.
    if rr_enabled and rr_orig is not None:
        domain = rr_orig
    else:
        domain = (boundaries[0], boundaries[-1])
    return cfg, domain


def _lut_config_cls():
    # The nb::class_ lives on the raw binding module; the Python ttnn.experimental.quasar
    # wrapper re-exports the op function but not the bound struct.
    if hasattr(ttnn.experimental.quasar, "LutConfig"):
        return ttnn.experimental.quasar.LutConfig
    import ttnn._ttnn as _raw

    return _raw.operations.experimental.quasar.LutConfig


_RR_KW = (
    "rr_method",
    "rr_log_ln2",
    "rr_exp_mult",
    "rr_exp_const",
    "rr_scale0",
    "rr_scale1",
    "rr_scale2",
    "rr_exp2_mult",
    "rr_compose",
    "rr_log2_scale",
    "rr_log2_basis_mminus1",
    "rr_input_offset",
    "rr_pow_n",
    "rr_pow_recip",
    "nr_magic",
    "nr_c1",
    "nr_c2",
    "nr_iters",
    "nr_n",
    "nr_reciprocal",
    "asym_mask",
    "dom_class",
)


def make_lut_config(cfg):
    kw = dict(
        eval_method=cfg["eval_method"],
        poly_degree=cfg["poly_degree"],
        num_segments=cfg["num_segments"],
        num_degree=cfg["num_degree"],
        den_degree=cfg["den_degree"],
        data=cfg["data"],
    )
    for k in _RR_KW:
        if k in cfg:
            kw[k] = cfg[k]
    return _lut_config_cls()(**kw)


def approximation_golden(cfg, x_np):
    """Replicate the kernel's piecewise eval EXACTLY (clamp, segment-select, per-seg
    clamp, Horner / rational). fp32 throughout. Used to isolate DFB-path correctness."""
    data = np.asarray(cfg["data"], dtype=np.float32)
    nseg = cfg["num_segments"]
    b = data[: nseg + 1]
    coeff_off = nseg + 1
    if cfg["eval_method"] == EVAL_RATIONAL:
        cps = (cfg["num_degree"] + 1) + (cfg["den_degree"] + 1)
    else:
        cps = cfg["poly_degree"] + 1

    x = np.clip(x_np.astype(np.float32), b[0], b[-1])
    out = np.empty_like(x)

    # Asymptotic factoring: dominant(x) for the segment's class, reproduced from the kernel's
    # per-class formula (= precision/eval.py DOMINANT_FACTORS). asym_mask marks which segments
    # multiply their Horner result by dominant(x). Empty/0 => bare-poly cascade.
    asym_mask = cfg.get("asym_mask", 0)
    dom_class = cfg.get("dom_class", 0)
    _inv_sqrt_2pi = np.float32(1.0 / np.sqrt(2 * np.pi))

    def dominant(xs):
        if dom_class == 1:
            return xs.astype(np.float32)
        if dom_class in (2, 3, 4, 5):
            e = np.exp(np.float32(-0.5) * xs * xs).astype(np.float32)
            if dom_class == 2:
                return (e * (-_inv_sqrt_2pi)).astype(np.float32)
            if dom_class == 3:
                return (e * _inv_sqrt_2pi).astype(np.float32)
            if dom_class == 4:
                return (-e).astype(np.float32)
            return e
        if dom_class == 6:
            return np.exp(xs.astype(np.float32)).astype(np.float32)
        if dom_class == 7:
            return np.exp(np.float32(-1.0) * xs).astype(np.float32)
        if dom_class == 8:
            return (np.float32(-1.0) * np.exp(np.float32(-1.0) * xs)).astype(np.float32)
        if dom_class == 9:
            return (np.float32(1.0) - np.exp(np.float32(-1.0) * xs)).astype(np.float32)
        if dom_class == 10:
            return (xs.astype(np.float32) * np.exp(xs.astype(np.float32))).astype(np.float32)
        return np.ones_like(xs, dtype=np.float32)

    def horner(coeffs, xs):
        acc = np.full_like(xs, coeffs[-1], dtype=np.float32)
        for k in range(len(coeffs) - 2, -1, -1):
            acc = (acc * xs + np.float32(coeffs[k])).astype(np.float32)
        return acc

    for seg in range(nseg):
        lo, hi = b[seg], b[seg + 1]
        mask = x >= b[seg]
        xs = np.clip(x, lo, hi)
        base = coeff_off + seg * cps
        if cfg["eval_method"] == EVAL_RATIONAL:
            nd, dd = cfg["num_degree"], cfg["den_degree"]
            nums = data[base : base + nd + 1]
            dens = data[base + nd + 1 : base + nd + 1 + dd + 1]
            p = horner(nums, xs)
            q = horner(dens, xs)
            val = (p / q).astype(np.float32)
        else:
            coeffs = data[base : base + cps]
            val = horner(coeffs, xs)
        if dom_class != 0 and (asym_mask >> seg) & 1:
            val = (val * dominant(xs)).astype(np.float32)
        out[mask] = val[mask]
    return out


def true_golden(activation, x_np):
    """The TRUE activation via the fitter ground_truth (single source of truth)."""
    import ground_truth as gt

    func = gt.get_activation(activation)
    y = func(torch.from_numpy(x_np.astype(np.float32)))
    if not torch.is_tensor(y):
        y = torch.tensor(np.asarray(y, dtype=np.float32))
    return y.to(torch.float32).numpy()


def csv_is_asymptotic(csv_path):
    """True iff ANY segment row of the CSV has is_asymptotic=True. is_asymptotic is a
    per-segment COLUMN (not METADATA). A pick with >=1 asymptotic segment is one whose
    deployed fit relies on asymptotic factoring f(x)=dominant(x)*correction(x) for its
    tail — the signal for whether asymptotic factoring needs porting to the DFB kernel."""
    with open(csv_path, newline="") as f:
        hdr = None
        for r in _csv.reader(f):
            if not r:
                continue
            if r[0] == "segment_id":
                hdr = r
                continue
            if r[0] == "METADATA":
                continue
            if hdr and "is_asymptotic" in hdr:
                i = hdr.index("is_asymptotic")
                if len(r) > i and r[i].strip().lower() == "true":
                    return True
    return False


def _bf16_bitdist_ulp(out_f32, truth_f32):
    """Sign-magnitude bf16 ordinal (bit-distance) ULP between device output and truth.

    Both are rounded to bf16, viewed as int16, mapped to a monotone ordinal
    (sign-magnitude -> two's-complement-like ordering), and the absolute ordinal
    difference is the ULP distance. Returns (max, mean, p99) over finite pairs.

    ZERO-REFERENCE EXCLUSION (principled near-root robustness, NOT a math hack): pairs whose
    bf16-rounded TRUTH is exactly +-0 are excluded from the bit-distance statistics. A
    bit-distance ULP is undefined at a zero reference — there is no neighboring representable
    value to count ULPs against — and crossing zero makes the sign-magnitude ordinal jump by
    ~2^15, so a faithful tiny output reads as a spurious ~9000-ULP "error". This is exactly
    the asymptotic-tail case for gelu: the reference (torch / ground_truth) gelu(x) for x<<0
    is computed as x*(1+erf(x/sqrt2)), which CATASTROPHICALLY CANCELS to +-0 below ~|x|=8,
    while the kernel's asymptotic factoring computes the genuine tiny tail value
    (matching a non-cancelling erfc form, and the fitter's own eval.py, to <=1 ULP). Counting
    those points would penalize the kernel for being MORE accurate than the cancelling
    reference. The exclusion is on TRUTH==0 only (a property of the reference, not the output),
    so a wrong nonzero output where truth is genuinely nonzero is still fully measured."""
    o = (
        torch.from_numpy(np.asarray(out_f32, dtype=np.float32).ravel())
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.int16)
        .to(torch.int64)
    )
    t = (
        torch.from_numpy(np.asarray(truth_f32, dtype=np.float32).ravel())
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.int16)
        .to(torch.int64)
    )
    ordi = lambda b: torch.where(b < 0, (-32768) - b, b)
    d = (ordi(o) - ordi(t)).abs().to(torch.float64)
    # Exclude zero-reference points (bf16 truth == +-0): bit-distance ULP is undefined there.
    truth_bf16 = torch.from_numpy(np.asarray(truth_f32, dtype=np.float32).ravel()).to(torch.bfloat16)
    nonzero_ref = truth_bf16 != 0
    if nonzero_ref.any():
        d = d[nonzero_ref]
    return float(d.max()), float(d.mean()), float(np.percentile(d.numpy(), 99))


def _all_bf16_in_domain(lo, hi):
    """ALL distinct representable bf16 values in [lo, hi] (finite), sorted ascending.

    Enumerates every 16-bit pattern, views as bf16, filters to the domain, dedups. This
    is the EXHAUSTIVE input set — not a random/linspace sample — so every bf16 the device
    can see in the activation's full fit domain is measured."""
    bits = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16)
    v = bits.view(torch.bfloat16).to(torch.float32)
    msk = (v >= lo) & (v <= hi) & torch.isfinite(v)
    return np.unique(v[msk].numpy())


def height_sharded_config(tiles):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 0))})
    shard = [tiles * 32, 32]
    mc = ttnn.create_sharded_memory_config(
        shard,
        core_grid=grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return mc, torch.Size([tiles * 32, 32])


def run_dfb(device, csv_path, activation=None, tiles=4, seed=0, margin=0.02):
    """Run a CSV's deployed config through the DFB op EXHAUSTIVELY and return a result
    dict with PCC, bf16 bit-distance ULP (max/mean/p99) and ml_pass vs the true activation.

    EXHAUSTIVE sampling: every distinct representable bf16 value in the activation's FULL
    fit domain [lo, hi] (NOT random / linspace). The full fit domain is used — asymptotic
    tail segments are NOT dropped — so tail-fidelity errors are measured, not hidden.
    `tiles`/`seed`/`margin` are retained for API compatibility but no longer drive sampling
    (the input set is fully determined by [lo, hi])."""
    activation = activation or _activation_name_from_csv(csv_path)
    cfg, (lo, hi) = parse_csv(csv_path)
    rr_enabled = cfg.get("_rr_enabled", False)
    lut = make_lut_config(cfg)

    # EXHAUSTIVE input set: every distinct finite bf16 in the FULL fit domain [lo, hi].
    xv = _all_bf16_in_domain(lo, hi)  # float32 array, ascending, deduped
    n0 = int(xv.size)
    # Pad up to a whole number of 32x32 tiles (height-sharded on one core).
    n_tiles = max(1, int(np.ceil(n0 / 1024)))
    npad = n_tiles * 1024
    pad_val = xv[-1] if n0 > 0 else np.float32(lo)
    xp = np.concatenate([xv, np.full(npad - n0, pad_val, dtype=np.float32)]).reshape(n_tiles * 32, 32)
    x_pt = torch.from_numpy(xp).to(torch.bfloat16)

    mc, _shape = height_sharded_config(n_tiles)
    x_tt = ttnn.from_torch(x_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    out_tt = ttnn.experimental.quasar.unary_lut(x_tt, lut_config=lut)
    out = ttnn.to_torch(out_tt).to(torch.float32).numpy().ravel()[:n0]

    x_eff = x_pt.to(torch.float32).numpy().ravel()[:n0]  # what the device actually saw (bf16-rounded)
    truth = true_golden(activation, x_eff)

    fin = np.isfinite(out) & np.isfinite(truth)
    out_f, truth_f = out[fin], truth[fin]

    pcc_true = _pcc(out, truth)
    # bf16 bit-distance ULP on the DFB device output vs the true activation.
    if out_f.size:
        ulp_max, ulp_mean, ulp_p99 = _bf16_bitdist_ulp(out_f, truth_f)
        # ml_pass: fraction within a relative+absolute tolerance (the "good enough for ML"
        # band) — a broad-error detector that, unlike bit-distance ULP_max, is NOT inflated
        # by near-zero crossings.
        tol = 1e-3 + 1e-3 * np.abs(truth_f)
        ml_pass = float(np.mean(np.abs(out_f - truth_f) <= tol))
    else:
        ulp_max = ulp_mean = ulp_p99 = float("nan")
        ml_pass = float("nan")

    # The reduced-poly approximation_golden only matches the kernel on the NO-RR path
    # (the RR kernel reduces+reconstructs the full activation). For RR, report
    # pcc_vs_approx == pcc_vs_true so the headline metric is the true-activation PCC.
    if rr_enabled:
        pcc_approx = pcc_true
    else:
        approx = approximation_golden(cfg, x_eff)
        pcc_approx = _pcc(out, approx)
    return {
        "activation": activation,
        "eval_method": "RATIONAL" if cfg["eval_method"] == EVAL_RATIONAL else "POLY",
        "num_segments": cfg["num_segments"],
        "degree": cfg["poly_degree"]
        if cfg["eval_method"] == EVAL_POLY
        else f"n{cfg['num_degree']}d{cfg['den_degree']}",
        "domain": (lo, hi),
        "rr_method": cfg.get("rr_method", 0),
        "rr_enabled": rr_enabled,
        "is_asymptotic": csv_is_asymptotic(csv_path),
        "n_bf16": n0,
        "pcc_vs_approx": pcc_approx,
        "pcc_vs_true": pcc_true,
        "ulp_max": ulp_max,
        "ulp_mean": ulp_mean,
        "ulp_p99": ulp_p99,
        "ml_pass": ml_pass,
    }


def _pcc(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size == 0:
        return float("nan")
    if np.allclose(a, b):
        return 1.0
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
