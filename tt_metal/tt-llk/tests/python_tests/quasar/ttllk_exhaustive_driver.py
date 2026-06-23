# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
EXHAUSTIVE bf16 driver for the tt-llk Quasar generic-LUT path.

The DFB analog (tests/.../quasar/dfb_lut_driver.py) drives the DFB `unary_lut`
op with EVERY representable bf16 in an activation's full fit domain and computes
the bf16 bit-distance ULP + Torch-tolerance `ml_pass` against the fitter
`ground_truth`. This module is the tt-llk counterpart: it drives the SFPI
generic-LUT kernel (polynomial / rational / newton_root) on craq-sim Quasar with
the SAME exhaustive bf16 input set and computes the SAME metrics, so the two paths
can be compared head-to-head.

Why a custom driver and not the existing quasar tests: the per-method tests
(test_generic_lut_*_quasar.py) drive the kernel with SAMPLED stimuli
(generate_stimuli) and report PCC/ULP on a 32x32 tile. We need EXHAUSTIVE inputs
+ the DFB metric definitions. We REUSE each test module's CSV parser + the
kernel-baking template parameter class + the kernel source path + the per-method
MathOperation; only the input (exhaustive instead of sampled) and the host-side
metric (DFB bit-distance ULP + ml_pass) differ.

Input order is preserved 1:1 by the harness: generate_stimuli's src_A is a flat
tensor whose element i maps to res_from_L1[i] (the quasar tests rely on exactly
this for their element-wise golden compare). We therefore replace src_A's VALUES
with the exhaustive bf16 set (padded to a whole number of tiles) and slice both
the device output and the truth back to the real count n0.

Method detection (mirrors dfb_lut_driver / quasar_sweep.sh resolve_deployed):
  * rational     iff the CSV has approximation_type=='rational' segment rows.
  * newton_root  iff METADATA range_reduction_method=='newton_root'.
  * polynomial   otherwise (incl. all other RR methods the poly kernel implements).
"""

import csv as _csv
import importlib
import math
import os
import sys

import numpy as np
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES

_FITTER = os.environ.get("TT_POLY_FITTER", "/localdev/nkapre/tt-polynomial-fitter")
if _FITTER not in sys.path:
    sys.path.insert(0, _FITTER)

# Per-method test modules (reused for parsing + kernel baking). Imported lazily by
# name so a parse failure in one method does not break the others.
_POLY_MOD = "quasar.test_generic_lut_activation_quasar"
_RAT_MOD = "quasar.test_generic_lut_rational_quasar"
_NR_MOD = "quasar.test_generic_lut_newton_root_quasar"


# ---- DFB metric (copied verbatim from dfb_lut_driver, same definitions) -------------
def _bf16_bitdist_ulp(out_f32, truth_f32):
    """Sign-magnitude bf16 ordinal (bit-distance) ULP. (max, mean, p99) over finite
    pairs, EXCLUDING zero-reference points (bf16 truth == +-0). Verbatim from the DFB
    driver so the two paths' numbers are directly comparable."""
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
    truth_bf16 = torch.from_numpy(np.asarray(truth_f32, dtype=np.float32).ravel()).to(
        torch.bfloat16
    )
    nonzero_ref = truth_bf16 != 0
    if nonzero_ref.any():
        d = d[nonzero_ref]
    return float(d.max()), float(d.mean()), float(np.percentile(d.numpy(), 99))


def _all_bf16_in_domain(lo, hi):
    """ALL distinct finite representable bf16 in [lo, hi], sorted. Verbatim from the
    DFB driver (and identical to the tt-llk _enumerate_representable)."""
    bits = torch.arange(0, 2**16, dtype=torch.int32).to(torch.int16)
    v = bits.view(torch.bfloat16).to(torch.float32)
    msk = (v >= lo) & (v <= hi) & torch.isfinite(v)
    return np.unique(v[msk].numpy())


def _log_expand_constant(activation):
    """Base-specific log-expansion constant logB(2) for the rr=log reconstruction.

    Verbatim policy from the DFB driver's _log_expand_constant: the deployed rr=log
    CSVs (log_n5d4, log10_n2d2, log2_*) do NOT emit `log_ln2_constant`, so source the
    per-base scale from the GROUND-TRUTH activation JSON (range_reduction.log_scale),
    falling back to ln(2) (the fitter's own default). This keeps the tt-llk path's RR
    reconstruction identical to the DFB path's for the log family."""
    import json

    cfg = os.path.join(_FITTER, "activations", f"{activation}.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            s = json.load(f).get("range_reduction", {}).get("log_scale")
        if s is not None:
            return float(s)
    return math.log(2.0)


def _backfill_rr_params(lut, activation):
    """Backfill RR params the deployed CSV omits, using the SAME defaults the DFB
    driver (_parse_rr_meta) uses — so the tt-llk RR reconstruction is identical to
    the DFB path for the log / exp families. Mutates lut in place.

    * rr=log: ln2 (logB(2)) from the activation JSON log_scale, else ln(2).
      (DFB _log_expand_constant.)
    * rr=exp: Cody-Waite mult = log2(e) = 1.4426950408889634,
              const = ln(2) = 0.6931471805599453.
      (DFB _parse_rr_meta exp branch defaults.)
    """
    p = getattr(lut, "rr_params", None)
    if not (getattr(lut, "rr_enabled", False) and isinstance(p, dict)):
        return lut
    if lut.rr_method == "log" and p.get("ln2") is None:
        p["ln2"] = _log_expand_constant(activation)
    elif lut.rr_method == "exp":
        if p.get("mult") is None:
            p["mult"] = 1.4426950408889634
        if p.get("const") is None:
            p["const"] = 0.6931471805599453
    return lut


def true_golden(activation, x_np):
    """TRUE activation via the fitter ground_truth (single source of truth)."""
    from ground_truth import compute_ground_truth

    y = compute_ground_truth(activation, x_np.astype(np.float64))
    return np.asarray(y, dtype=np.float32)


# ---- method detection ---------------------------------------------------------------
def detect_method(csv_path):
    """Return 'rational' | 'newton_root' | 'polynomial' from a fitter CSV.

    Mirrors dfb_lut_driver: rational iff any segment row is approximation_type=='rational';
    newton_root iff METADATA range_reduction_method=='newton_root'; else polynomial.
    """
    rr_method = "none"
    header = None
    rows = []
    with open(csv_path, newline="") as f:
        for r in _csv.reader(f):
            if not r:
                continue
            if r[0] == "segment_id":
                header = r
                continue
            if r[0] == "METADATA":
                if len(r) >= 3 and r[1].strip() == "range_reduction_method":
                    rr_method = r[2].strip()
                continue
            rows.append(r)
    rational = (
        header is not None
        and "approximation_type" in header
        and any((len(r) > 3 and r[3] == "rational") for r in rows)
    )
    if rr_method == "newton_root":
        return "newton_root"
    if rational:
        return "rational"
    return "polynomial"


# ---- per-method wiring (kernel source, template class, math op, parser) -------------
def _method_wiring(method):
    """Return (module, kernel_src, math_op, build_template_fn) for `method`.

    build_template_fn(lut) -> the TemplateParameter that bakes `lut` into the kernel
    (exactly the class the per-method test uses)."""
    if method == "polynomial":
        m = importlib.import_module(_POLY_MOD)

        def parse(csv_path, act):
            return m._parse_fitter_csv(csv_path, act)

        def tmpl(lut):
            return m.GENERIC_LUT_DATA(lut=lut, emit=lut.use_ground_truth)

        return (
            m,
            "sources/quasar/generic_lut_activation_quasar_test.cpp",
            MathOperation.Sigmoid,
            parse,
            tmpl,
            False,  # newton golden uses model, others use rr_enabled flag on lut
        )
    if method == "rational":
        m = importlib.import_module(_RAT_MOD)

        def parse(csv_path, act):
            return m.parse_rational_csv(csv_path, act)

        def tmpl(lut):
            return m.RATIONAL_LUT(lut=lut)

        return (
            m,
            "sources/quasar/generic_lut_rational_quasar_test.cpp",
            MathOperation.Sigmoid,
            parse,
            tmpl,
            False,
        )
    if method == "newton_root":
        m = importlib.import_module(_NR_MOD)

        def parse(csv_path, act):
            return m.parse_nr_csv(csv_path, act)

        def tmpl(lut):
            return m.NR_DEFINES(lut=lut)

        return (
            m,
            "sources/quasar/generic_lut_newton_root_quasar_test.cpp",
            MathOperation.Sqrt,
            parse,
            tmpl,
            True,
        )
    raise ValueError(f"unknown method {method}")


def _domain_for(method, lut):
    """The full evaluation domain [lo, hi] the kernel covers (original domain under
    range reduction, else the reduced LUT span). Mirrors each test's dom_lo/dom_hi."""
    if method == "newton_root":
        return float(lut.orig_min), float(lut.orig_max)
    if getattr(lut, "rr_enabled", False):
        return float(lut.rr_original_min), float(lut.rr_original_max)
    if method == "rational":
        return float(lut.boundaries[0]), float(lut.boundaries[-1])
    return float(lut.boundaries[0]), float(lut.boundaries[lut.num_segments])


# DestSync.Half holds 8 tiles of 16-bit Dest. The kernel packs all TILE_CNT tiles
# from Dest in one section, so a single op invocation is capped at 8 tiles; larger
# exhaustive sets are processed in <=8-tile BATCHES and concatenated. (Validated
# empirically: TILE_CNT>8 with DestSync.Half overruns Dest -> qsr PACR src_tile_idx
# out of range.)
_MAX_TILES_PER_RUN = 8


# bf16 output format pick (the deployed bf16 row).
_BF16 = DataFormat.Float16_b


def run_exhaustive(csv_path, activation, method=None):
    """Drive the tt-llk kernel for `csv_path` with EXHAUSTIVE bf16 inputs over the full
    fit domain and return the DFB metrics dict.

    Returns dict with: eval_method, n_bf16, ulp_max, ulp_mean, ulp_p99, ml_pass, domain.
    Raises on a parse/build/run failure (the caller marks the pick out-of-scope).
    """
    method = method or detect_method(csv_path)
    mod, kernel_src, math_op, parse, tmpl, _ = _method_wiring(method)

    lut = parse(csv_path, activation)
    _backfill_rr_params(lut, activation)  # log/exp picks omit RR params (DFB parity)
    lo, hi = _domain_for(method, lut)

    # EXHAUSTIVE input set: every distinct finite bf16 in [lo, hi].
    xv = _all_bf16_in_domain(lo, hi)
    n0 = int(xv.size)
    if n0 == 0:
        raise ValueError(f"no bf16 in domain [{lo}, {hi}]")

    formats = input_output_formats([_BF16], same=True)[0]
    num_faces = MAX_NUM_FACES
    dest_acc = DestAccumulation.No  # bf16 input -> 16-bit Dest

    # The combine pass trips ttsim's SFPADDI stub for the rational/newton reciprocal
    # NR step; the per-method tests disable it. Match them (identical numerics).
    if method in ("rational", "newton_root"):
        TestConfig.ARCH_SPECIFIC_OPTIONS = "-mno-tt-tensix-optimize-combine"
    else:
        TestConfig.ARCH_SPECIFIC_OPTIONS = ""

    def _run_one(values_1d):
        """Run the kernel once on a tile-aligned flat input (<=_MAX_TILES_PER_RUN
        tiles). Returns the device output as a flat float32 array of equal length."""
        n_tiles = len(values_1d) // 1024
        src_A = torch.from_numpy(values_1d).to(format_dict[formats.input_format])
        src_B = torch.zeros_like(src_A)
        configuration = TestConfig(
            kernel_src,
            formats,
            templates=[
                MATH_OP(mathop=math_op),
                IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
                DATA_COPY_TYPE(DataCopyType.A2D),
                UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
                DEST_SYNC(DestSync.Half),
                tmpl(lut),
            ],
            runtimes=[
                TILE_COUNT(n_tiles),
                NUM_FACES(num_faces),
                TEST_FACE_DIMS(),
                DEST_INDEX(0),
            ],
            variant_stimuli=StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=n_tiles,
                tile_count_B=n_tiles,
                tile_count_res=n_tiles,
                num_faces=num_faces,
            ),
            unpack_to_dest=False,
            dest_acc=dest_acc,
        )
        res = configuration.run().result
        return torch.tensor(res, dtype=torch.float32).numpy().ravel()

    # Pad to a whole number of tiles, then split into <=_MAX_TILES_PER_RUN-tile
    # batches (DestSync.Half Dest cap). Run each batch, concatenate, slice to n0.
    total_tiles = max(1, math.ceil(n0 / 1024))
    npad = total_tiles * 1024
    pad_val = xv[-1]
    xp = np.concatenate([xv, np.full(npad - n0, pad_val, dtype=np.float32)])

    out_chunks = []
    elems_per_batch = _MAX_TILES_PER_RUN * 1024
    for start in range(0, npad, elems_per_batch):
        chunk = xp[start : start + elems_per_batch]
        out_chunks.append(_run_one(chunk))
    out = np.concatenate(out_chunks)[:n0]

    # What the device actually saw (bf16-rounded), and the TRUE activation on it.
    # xv is already exact bf16 values (from _all_bf16_in_domain), so x_eff == xv[:n0].
    x_eff = (
        torch.from_numpy(xv)
        .to(format_dict[formats.input_format])
        .to(torch.float32)
        .numpy()
        .ravel()[:n0]
    )
    truth = true_golden(activation, x_eff)

    fin = np.isfinite(out) & np.isfinite(truth)
    out_f, truth_f = out[fin], truth[fin]

    if out_f.size:
        ulp_max, ulp_mean, ulp_p99 = _bf16_bitdist_ulp(out_f, truth_f)
        tol = 1e-3 + 1e-3 * np.abs(truth_f)
        ml_pass = float(np.mean(np.abs(out_f - truth_f) <= tol))
    else:
        ulp_max = ulp_mean = ulp_p99 = float("nan")
        ml_pass = float("nan")

    return {
        "eval_method": method,
        "n_bf16": n0,
        "domain": (lo, hi),
        "ulp_max": ulp_max,
        "ulp_mean": ulp_mean,
        "ulp_p99": ulp_p99,
        "ml_pass": ml_pass,
    }
