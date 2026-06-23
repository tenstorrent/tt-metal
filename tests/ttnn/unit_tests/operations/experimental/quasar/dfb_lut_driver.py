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


def parse_csv(csv_path):
    """Parse a fitter coefficient CSV into a LutConfig + (lo, hi) global domain.

    Returns (lut_config_kwargs: dict, domain: (float, float))."""
    rows = []
    header = None
    with open(csv_path, newline="") as f:
        for r in _csv.reader(f):
            if not r:
                continue
            if r[0] == "segment_id":
                header = r
                continue
            if r[0] == "METADATA":
                continue
            rows.append(r)
    assert header is not None, f"no header in {csv_path}"

    rational = "approximation_type" in header and any((len(r) > 3 and r[3] == "rational") for r in rows)

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

    domain = (boundaries[0], boundaries[-1])
    return cfg, domain


def _lut_config_cls():
    # The nb::class_ lives on the raw binding module; the Python ttnn.experimental.quasar
    # wrapper re-exports the op function but not the bound struct.
    if hasattr(ttnn.experimental.quasar, "LutConfig"):
        return ttnn.experimental.quasar.LutConfig
    import ttnn._ttnn as _raw

    return _raw.operations.experimental.quasar.LutConfig


def make_lut_config(cfg):
    return _lut_config_cls()(
        eval_method=cfg["eval_method"],
        poly_degree=cfg["poly_degree"],
        num_segments=cfg["num_segments"],
        num_degree=cfg["num_degree"],
        den_degree=cfg["den_degree"],
        data=cfg["data"],
    )


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
    """Run a CSV's deployed config through the DFB op and return a result dict with
    PCC vs the true activation AND vs the approximation."""
    activation = activation or _activation_name_from_csv(csv_path)
    cfg, (lo, hi) = parse_csv(csv_path)
    lut = make_lut_config(cfg)

    # Sample the deployed domain with a margin past both ends (exercises the clamp).
    mc, shape = height_sharded_config(tiles)
    n = int(np.prod(shape))
    torch.manual_seed(seed)
    span = hi - lo
    x_np = (lo - margin * span) + (span * (1 + 2 * margin)) * np.random.RandomState(seed).rand(n)
    x_np = x_np.astype(np.float32).reshape(tuple(shape))
    x_pt = torch.from_numpy(x_np).to(torch.bfloat16)

    x_tt = ttnn.from_torch(x_pt, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=mc)
    out_tt = ttnn.experimental.quasar.unary_lut(x_tt, lut_config=lut)
    out = ttnn.to_torch(out_tt).to(torch.float32).numpy()

    x_eff = x_pt.to(torch.float32).numpy()  # what the device actually saw (bf16-rounded)
    approx = approximation_golden(cfg, x_eff)
    truth = true_golden(activation, x_eff)

    pcc_approx = _pcc(out, approx)
    pcc_true = _pcc(out, truth)
    return {
        "activation": activation,
        "eval_method": "RATIONAL" if cfg["eval_method"] == EVAL_RATIONAL else "POLY",
        "num_segments": cfg["num_segments"],
        "degree": cfg["poly_degree"]
        if cfg["eval_method"] == EVAL_POLY
        else f"n{cfg['num_degree']}d{cfg['den_degree']}",
        "domain": (lo, hi),
        "pcc_vs_approx": pcc_approx,
        "pcc_vs_true": pcc_true,
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
