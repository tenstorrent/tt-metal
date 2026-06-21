# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generic piecewise-polynomial LUT activation on Quasar (sim-qsr / ttsim).

UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest -> SFPU(embedded LUT) -> PACK.
The SFPU evaluates an embedded piecewise-polynomial LUT using the sfpi DSL on
the Quasar 2-rows-per-iteration model. The LUT is generic in degree N
(POLY_DEGREE) and segment count S (NUM_SEGMENTS).

Two modes (selected by env):
  * DEFAULT (no env): the proven hardcoded deg-2 / 4-seg sigmoid LUT. The golden
    replicates the EXACT baked-in coefficients so the PCC isolates kernel
    correctness (~0.9999999).
  * REAL FITTER COEFFS (QUASAR_LUT_CSV + QUASAR_ACT set): parse the
    tt-polynomial-fitter coefficient CSV -> per-segment breakpoints (lo/hi) +
    Horner coeffs -> infer degree N and segment count S -> bake them into the
    kernel via the GENERIC_LUT_DATA template parameter. The golden is the TRUE
    activation from the fitter's compute_ground_truth(QUASAR_ACT, x), so PCC/ULP
    measure the end-to-end (fit + kernel) accuracy of the real LUT on sim-qsr.

Shared CONTRACT (stable; the Quasar-backend driver relies on it):
  * Env QUASAR_LUT_CSV  : path to a fitter polynomial coefficient CSV.
  * Env QUASAR_ACT      : activation name understood by compute_ground_truth
                          (sigmoid / tanh / gelu / exp / ...).
  * Env QUASAR_FITTER_REPO (optional): path to the tt-polynomial-fitter repo
                          (for compute_ground_truth). Defaults to
                          /localdev/nkapre/tt-polynomial-fitter.
  * Prints, per format : "[generic_lut_activation_quasar] PCC = <number>"
                         "[generic_lut_activation_quasar] ULP = <number>"
                         "[generic_lut_activation_quasar] config = <act> ..."

Reproduce (from tt_metal/tt-llk/tests/python_tests):
  TT_METAL_HOME=/localdev/nkapre/tt-metal-nkapreTT \
  TT_METAL_SIMULATOR=/home/nkapre/sim-qsr/libttsim.so CHIP_ARCH=quasar \
  QUASAR_ACT=sigmoid \
  QUASAR_LUT_CSV=/localdev/nkapre/tt-polynomial-fitter/data/coefficients/sigmoid_p5_s16_uniform_any_ulp.csv \
  ../.venv/bin/python -m pytest --run-simulator \
  quasar/test_generic_lut_activation_quasar.py -x -s -q
"""

import csv
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest
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
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
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
    TemplateParameter,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import calculate_pcc, passed_test

# ---------------------------------------------------------------------------
# Default (hardcoded) LUT — the proven deg-2 / 4-seg sigmoid approximation.
# MUST match the C++ kernel defaults (sources/quasar/...test.cpp) exactly.
# ---------------------------------------------------------------------------
DEFAULT_NUM_SEGMENTS = 4
DEFAULT_POLY_DEGREE = 2
DEFAULT_BOUNDARIES = [-4.0, -2.0, 0.0, 2.0, 4.0]
DEFAULT_COEFFS = [
    [0.38296354, 0.17515847, 0.02109685],  # seg0: c0, c1, c2
    [0.50329190, 0.27505103, 0.04113654],  # seg1
    [0.49670810, 0.27505103, -0.04113654],  # seg2
    [0.61703646, 0.17515847, -0.02109685],  # seg3
]


# ---------------------------------------------------------------------------
# Parsed LUT container.
# ---------------------------------------------------------------------------
@dataclass
class LutConfig:
    activation: str
    degree: int
    num_segments: int
    boundaries: list  # length num_segments+1, ascending
    coeffs: list  # num_segments lists of (degree+1) Horner coeffs (c0..cN)
    use_ground_truth: bool  # True -> golden is the true activation; else algorithmic
    # Range reduction (parsed from CSV METADATA; method='none' -> disabled).
    rr_method: str = "none"
    rr_enabled: bool = False
    rr_original_min: float = None
    rr_original_max: float = None
    rr_reduced_min: float = None
    rr_reduced_max: float = None
    rr_params: dict = None  # method-specific (see _parse_fitter_csv)


def _default_lut() -> LutConfig:
    return LutConfig(
        activation="sigmoid",
        degree=DEFAULT_POLY_DEGREE,
        num_segments=DEFAULT_NUM_SEGMENTS,
        boundaries=list(DEFAULT_BOUNDARIES),
        coeffs=[list(c) for c in DEFAULT_COEFFS],
        use_ground_truth=False,
    )


def _longest_nonasymptotic_run(seg_rows):
    """Return the longest contiguous run of NON-asymptotic segments.

    The fitter marks tail segments (where the function decays to an asymptote,
    e.g. sigmoid/gelu on [-10,10]) with is_asymptotic=True. Those segments are
    NOT plain Horner polynomials — the stored coeffs are a correction term that
    multiplies a dominant factor (exp(x), 1-exp(-x), ...). A pure-polynomial LUT
    kernel cannot reproduce them, so we restrict to the largest contiguous span
    of genuine polynomial segments (the function's core). For configs that are
    fully polynomial (tanh, exp), this keeps every segment.
    """
    best, cur = [], []
    for r in seg_rows:
        if not r[3]:  # not asymptotic
            cur.append(r)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return best


def _parse_fitter_csv(path: str, activation: str) -> LutConfig:
    """Parse a tt-polynomial-fitter polynomial coefficient CSV.

    Columns: segment_id,lo,hi,c0,c1,...,cN,error,method,...,is_asymptotic,...
    (N = degree). Returns per-segment boundaries (lo/hi) and Horner coeffs c0..cN
    for the largest contiguous run of non-asymptotic (pure polynomial) segments.
    Rows whose segment_id is non-numeric (e.g. METADATA) are skipped.
    """
    seg_rows = []
    meta = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Coeff columns are the contiguous c0,c1,... columns after lo,hi.
        coeff_idx = [i for i, h in enumerate(header) if h.strip().lower().startswith("c") and h.strip()[1:].isdigit()]
        coeff_idx.sort(key=lambda i: int(header[i].strip()[1:]))
        lo_idx = header.index("lo")
        hi_idx = header.index("hi")
        asy_idx = header.index("is_asymptotic") if "is_asymptotic" in header else None
        for row in reader:
            if not row:
                continue
            if row[0] == "METADATA":
                meta[row[1].strip()] = row[2].strip()
                continue
            try:
                int(row[0])
            except ValueError:
                continue  # blank / non-segment rows
            lo = float(row[lo_idx])
            hi = float(row[hi_idx])
            coeffs = [float(row[i]) for i in coeff_idx]
            asy = asy_idx is not None and str(row[asy_idx]).strip().lower() == "true"
            seg_rows.append((lo, hi, coeffs, asy))

    if not seg_rows:
        raise ValueError(f"No segment rows parsed from {path}")

    # ---- Parse range-reduction metadata -------------------------------------
    def _mf(k):  # metadata float (None if absent/empty)
        return float(meta[k]) if k in meta and meta[k] != "" else None

    rr_method = meta.get("range_reduction_method", "none")
    rr_enabled = meta.get("range_reduction_enabled", "False").lower() == "true"
    # Only the exponent family is implemented in the kernel; everything else
    # (none/trig/tan) falls back to the legacy [b0,bN] path.
    _SUPPORTED_RR = {"log", "exp", "cbrt", "exponent_alu_exp2", "exponent_alu_log2", "exponent_alu_pow", "trig", "tan"}
    if rr_method not in _SUPPORTED_RR:
        rr_enabled = False
    params = {}
    if rr_enabled:
        if rr_method == "log":
            params["ln2"] = _mf("log_ln2_constant")
        elif rr_method == "exp":
            params["mult"] = _mf("exp_log2_multiplier")
            params["const"] = _mf("exp_log2_constant")
        elif rr_method == "cbrt":
            params["scale"] = [_mf("cbrt_scale_c0"), _mf("cbrt_scale_c1"), _mf("cbrt_scale_c2")]
        elif rr_method == "exponent_alu_exp2":
            params["mult"] = _mf("expalu_log2_multiplier")
            params["compose"] = meta.get("expalu_compose", "") or ""
        elif rr_method == "exponent_alu_log2":
            params["scale"] = _mf("expalu_log_scale")
            params["basis"] = meta.get("expalu_log2_basis", "m")
            params["offset"] = _mf("expalu_input_offset") or 0.0
        elif rr_method == "exponent_alu_pow":
            params["n"] = int(float(meta["expalu_root_n"]))
            params["recip"] = meta.get("expalu_reciprocal", "False").lower() == "true"
            params["scale"] = [_mf("expalu_pow_scale_c0"), _mf("expalu_pow_scale_c1"), _mf("expalu_pow_scale_c2")]
        elif rr_method in ("trig", "tan"):
            # No kernel-tunable params: the kernel hardcodes pi / Cody-Waite
            # constants (matching range_reduction.py). trig_symmetry is metadata
            # only (the golden activation name already selects sin/cos/tan); the
            # reduced-domain bounds drive the clamp via b0/bN, parsed below.
            pass

    # Sort by lo to guarantee ascending boundaries (kernel relies on this).
    seg_rows.sort(key=lambda r: r[0])
    # RR (exponent-family) configs are fully-polynomial over the reduced domain
    # and carry no asymptotic tails, so the asymptotic-run filter does not apply.
    if rr_enabled:
        poly_rows = seg_rows
    else:
        poly_rows = _longest_nonasymptotic_run(seg_rows)
    if not poly_rows:
        raise ValueError(
            f"{path}: no non-asymptotic (pure polynomial) segments — this config "
            f"is entirely asymptotic and not representable by the pure-polynomial kernel."
        )
    n_dropped = len(seg_rows) - len(poly_rows)
    if n_dropped:
        print(
            f"[generic_lut_activation_quasar] note: dropped {n_dropped} asymptotic "
            f"segment(s); testing the {len(poly_rows)}-segment polynomial core "
            f"[{poly_rows[0][0]}, {poly_rows[-1][1]}]"
        )
    seg_rows = poly_rows
    degree = len(seg_rows[0][2]) - 1
    num_segments = len(seg_rows)

    # Boundaries: b0..bS. Use each segment's lo, plus the last segment's hi.
    boundaries = [r[0] for r in seg_rows] + [seg_rows[-1][1]]
    coeffs = [r[2] for r in seg_rows]

    return LutConfig(
        activation=activation,
        degree=degree,
        num_segments=num_segments,
        boundaries=boundaries,
        coeffs=coeffs,
        use_ground_truth=True,
        rr_method=rr_method,
        rr_enabled=rr_enabled,
        rr_original_min=_mf("range_reduction_original_min"),
        rr_original_max=_mf("range_reduction_original_max"),
        rr_reduced_min=_mf("range_reduction_reduced_min"),
        rr_reduced_max=_mf("range_reduction_reduced_max"),
        rr_params=params,
    )


def _load_lut_config() -> LutConfig:
    csv_path = os.environ.get("QUASAR_LUT_CSV")
    if not csv_path:
        return _default_lut()
    act = os.environ.get("QUASAR_ACT", "sigmoid")
    return _parse_fitter_csv(csv_path, act)


# ---------------------------------------------------------------------------
# Template parameter that bakes the LUT into the kernel via the build header
# (build.h), which the kernel includes at file scope. Emits LUT_POLY_DEGREE /
# LUT_NUM_SEGMENTS / LUT_DATA_INIT; when absent the kernel uses its hardcoded
# sigmoid defaults, keeping the env-free run identical to the proven baseline.
# ---------------------------------------------------------------------------
@dataclass
class GENERIC_LUT_DATA(TemplateParameter):
    lut: LutConfig = None
    emit: bool = True  # default-LUT runs leave the kernel defaults untouched

    def convert_to_cpp(self) -> str:
        if not self.emit or self.lut is None:
            return "// GENERIC_LUT_DATA: using kernel-default (hardcoded) LUT"

        def f(v):
            # Coeffs below ~FLT_MIN (1.18e-38) underflow fp32: a literal like
            # 2.2e-91f truncates to a subnormal/zero and trips -Werror=overflow
            # ("floating constant truncated to zero"). They are below fp32
            # representability and contribute nothing, so flush to 0.0
            # (numerically safe). Matches the rational test's _fmt_one.
            v = 0.0 if abs(float(v)) < 1.18e-38 else float(v)
            return f"{v:.10e}f"

        # For the exponent_alu_log2 m_minus_1 basis, the kernel's polynomial
        # argument is u = mantissa - 1 (it does r_arg = rr_mantissa(x) - 1). The
        # boundaries are used by the kernel ONLY for the clamp + segment selection,
        # so they must live in the SAME u-space as r_arg. The CSV stores them in
        # m-space [1,2]; shift by -1 so the clamp/selection match r_arg and the
        # u-basis coeffs. Without this a single [1,2] segment clamps every u in
        # [0,1] up to 1.0, collapsing the poly to a constant -> the result becomes
        # a pure exponent step function (PCC ~0.975, the observed log/log2/log10 fail).
        emit_boundaries = list(self.lut.boundaries)
        if (
            self.lut.rr_enabled
            and self.lut.rr_method == "exponent_alu_log2"
            and (self.lut.rr_params or {}).get("basis") == "m_minus_1"
        ):
            emit_boundaries = [b - 1.0 for b in emit_boundaries]

        # Only /* */ block comments here: a // line comment inside a
        # backslash-spliced macro would swallow the following line.
        parts = ["/* boundaries b0..bS */ " + ", ".join(f(b) for b in emit_boundaries) + ","]
        for seg in range(self.lut.num_segments):
            parts.append(f"/* seg{seg} */ " + ", ".join(f(c) for c in self.lut.coeffs[seg]) + ",")
        # Trailing comma on the last line is fine inside { ... }.
        body = " \\\n    ".join(parts)
        lines = [
            f"#define LUT_POLY_DEGREE {self.lut.degree}",
            f"#define LUT_NUM_SEGMENTS {self.lut.num_segments}",
            f"#define LUT_DATA_INIT {{ \\\n    {body} }}",
        ]

        # ---- Range-reduction macros (only when enabled). Numeric method codes
        # match the kernel's LUT_RR_METHOD contract; non-RR CSVs and the default
        # sigmoid build emit none of these -> byte-identical to the legacy path.
        _CODE = {
            "none": 0,
            "log": 1,
            "exp": 2,
            "cbrt": 3,
            "exponent_alu_exp2": 4,
            "exponent_alu_log2": 5,
            "exponent_alu_pow": 6,
            "trig": 7,
            "tan": 8,
        }
        _COMP = {"": 0, "sigmoid": 1, "minus_one": 2}
        rr = self.lut
        if rr.rr_enabled:
            p = rr.rr_params
            lines.append(f"#define LUT_RR_METHOD {_CODE[rr.rr_method]}")
            if rr.rr_method == "log":
                lines.append(f"#define LUT_RR_LOG_LN2 {p['ln2']:.10e}f")
            elif rr.rr_method == "exp":
                lines.append(f"#define LUT_RR_EXP_MULT {p['mult']:.10e}f")
                lines.append(f"#define LUT_RR_EXP_CONST {p['const']:.10e}f")
            elif rr.rr_method == "cbrt":
                for i, c in enumerate(p["scale"]):
                    if c is not None:
                        lines.append(f"#define LUT_RR_SCALE{i} {c:.10e}f")
            elif rr.rr_method == "exponent_alu_exp2":
                lines.append(f"#define LUT_RR_EXP2_MULT {p['mult']:.10e}f")
                lines.append(f"#define LUT_RR_COMPOSE {_COMP[p['compose']]}")
            elif rr.rr_method == "exponent_alu_log2":
                lines.append(f"#define LUT_RR_LOG2_SCALE {p['scale']:.10e}f")
                lines.append(f"#define LUT_RR_LOG2_BASIS_MMINUS1 {1 if p['basis'] == 'm_minus_1' else 0}")
                lines.append(f"#define LUT_RR_INPUT_OFFSET {p['offset']:.10e}f")
            elif rr.rr_method == "exponent_alu_pow":
                lines.append(f"#define LUT_RR_POW_N {p['n']}")
                lines.append(f"#define LUT_RR_POW_RECIP {1 if p['recip'] else 0}")
                for i, c in enumerate(p["scale"]):
                    if c is not None:
                        lines.append(f"#define LUT_RR_SCALE{i} {c:.10e}f")
            elif rr.rr_method in ("trig", "tan"):
                # Method code already emitted via LUT_RR_METHOD above; the kernel
                # hardcodes pi / Cody-Waite constants, so no further #defines.
                pass
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Goldens.
# ---------------------------------------------------------------------------
def _eval_poly_horner(coeffs, x):
    # coeffs[k] is coefficient of x^k; Horner from the top.
    acc = torch.full_like(x, float(coeffs[-1]))
    for k in range(len(coeffs) - 2, -1, -1):
        acc = acc * x + float(coeffs[k])
    return acc


def _algorithmic_golden(lut: LutConfig, x: torch.Tensor) -> torch.Tensor:
    """Replicate the EXACT kernel algorithm: clamp x to [b0, bS], select segment
    by ascending boundaries, Horner-eval that segment."""
    x = x.to(torch.float64)
    b0 = lut.boundaries[0]
    bN = lut.boundaries[lut.num_segments]
    x_clamped = torch.clamp(x, b0, bN)

    result = _eval_poly_horner(lut.coeffs[0], x_clamped)
    for seg in range(1, lut.num_segments):
        mask = x_clamped >= lut.boundaries[seg]
        result = torch.where(mask, _eval_poly_horner(lut.coeffs[seg], x_clamped), result)
    return result


def _ground_truth_golden(lut: LutConfig, x: torch.Tensor) -> torch.Tensor:
    """Golden = TRUE activation (fitter's compute_ground_truth).

    When range reduction is enabled, the kernel reproduces the full activation
    over the ORIGINAL domain via reduce+reconstruct, so the golden is the exact
    activation over the full domain (NO clamp to [b0,bN]). Otherwise the golden
    is the true activation over the clamped reduced domain [b0,bN]."""
    repo = os.environ.get("QUASAR_FITTER_REPO", "/localdev/nkapre/tt-polynomial-fitter")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from ground_truth import compute_ground_truth  # noqa: E402

    if lut.rr_enabled:
        x_eval = x.to(torch.float64)  # full original domain, no clamp
    else:
        b0 = lut.boundaries[0]
        bN = lut.boundaries[lut.num_segments]
        x_eval = torch.clamp(x.to(torch.float64), b0, bN)
    y = compute_ground_truth(lut.activation, x_eval.numpy())
    return torch.as_tensor(np.asarray(y, dtype=np.float64))


def _golden(lut: LutConfig, x: torch.Tensor) -> torch.Tensor:
    if lut.use_ground_truth:
        return _ground_truth_golden(lut, x)
    return _algorithmic_golden(lut, x)


# ---------------------------------------------------------------------------
# ULP: max units-in-the-last-place between kernel result and golden, sized by
# the OUTPUT format's mantissa width (bf16: 7 bits, fp32: 23 bits). The ULP step
# is taken from the golden's binade. Note: for an fp32 golden compared against
# the TRUE activation, tiny-magnitude outputs (e.g. sigmoid/gelu near 0) make
# the polynomial-fit error span many fp32 ULPs even at PCC ~1.0 — that is the
# fit's intrinsic error in fp32 terms, not a kernel defect. The bf16 ULP (= 1
# here) is the operationally meaningful figure for the low-precision path.
# ---------------------------------------------------------------------------
def _max_ulp(golden: torch.Tensor, result: torch.Tensor, out_format: DataFormat) -> float:
    g = golden.to(torch.float32).flatten().numpy()
    r = result.to(torch.float32).flatten().numpy()
    finite = np.isfinite(g) & np.isfinite(r)
    g = g[finite]
    r = r[finite]
    if g.size == 0:
        return 0.0
    if out_format == DataFormat.Float16_b:
        # bf16: 8-bit exponent, 7-bit mantissa; ULP = 2^(exp - 7).
        ge = np.frexp(np.where(g == 0.0, 1.0, g))[1] - 1  # unbiased exponent
        ulp_size = np.ldexp(1.0, ge - 7)
    else:
        # fp32: 23-bit mantissa.
        ge = np.frexp(np.where(g == 0.0, 1.0, g))[1] - 1
        ulp_size = np.ldexp(1.0, ge - 23)
    ulp = np.abs(r - g) / ulp_size
    return float(np.max(ulp))


# Matched in/out pairs only (same=True). fp32 isolates kernel correctness;
# Float16_b proves the low-precision path. The dest_acc / Dest bit-width is
# derived from the input format below, so mixed-width pairs are out of scope.
LUT_FORMATS = input_output_formats(
    [DataFormat.Float32, DataFormat.Float16_b], same=True
)


@pytest.mark.quasar
@parametrize(
    formats=LUT_FORMATS,
    dest_sync=[DestSync.Half],
    implied_math_format=[ImpliedMathFormat.Yes],
    input_dimensions=[[32, 32]],
)
def test_generic_lut_activation_quasar(
    formats,
    dest_sync,
    implied_math_format,
    input_dimensions,
):
    torch.manual_seed(0)

    lut = _load_lut_config()
    b0 = lut.boundaries[0]
    bN = lut.boundaries[lut.num_segments]

    # 32-bit inputs require a 32-bit Dest (dest_acc=Yes); 16-bit use Dest=No.
    dest_acc = (
        DestAccumulation.Yes
        if formats.input_format.is_32_bit()
        else DestAccumulation.No
    )

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    # Scale inputs into the evaluation domain. With range reduction the kernel
    # covers the FULL original domain; otherwise only the reduced LUT span.
    if lut.rr_enabled:
        dom_lo, dom_hi = lut.rr_original_min, lut.rr_original_max
    else:
        dom_lo, dom_hi = b0, bN
    src_A_f = src_A.to(torch.float32)
    lo, hi = src_A_f.min(), src_A_f.max()
    if hi > lo:
        src_A_f = (src_A_f - lo) / (hi - lo) * (dom_hi - dom_lo) + dom_lo
    else:
        src_A_f = torch.full_like(src_A_f, (dom_lo + dom_hi) / 2.0)
    src_A = src_A_f.to(format_dict[formats.input_format])

    num_faces = MAX_NUM_FACES

    golden_values = _golden(lut, src_A.to(torch.float32))
    golden_tensor = golden_values.to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/quasar/generic_lut_activation_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Sigmoid),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
            DEST_SYNC(dest_sync),
            GENERIC_LUT_DATA(lut=lut, emit=lut.use_ground_truth),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
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
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    pcc = calculate_pcc(golden_tensor, res_tensor)
    ulp = _max_ulp(golden_tensor, res_tensor, formats.output_format)
    print(f"\n[generic_lut_activation_quasar] PCC = {pcc}")
    print(f"[generic_lut_activation_quasar] ULP = {ulp}")
    print(
        f"[generic_lut_activation_quasar] config = act={lut.activation} "
        f"degree={lut.degree} segments={lut.num_segments} "
        f"format={formats.output_format.name} golden={'ground_truth' if lut.use_ground_truth else 'algorithmic'}"
    )

    if lut.use_ground_truth:
        # Real fitter coeffs validated against the TRUE activation: require a
        # high correlation (the driver gates on PCC >= 0.99).
        assert pcc >= 0.99, f"PCC {pcc} below 0.99 for {lut.activation}"
    else:
        # Default sigmoid LUT: golden replicates the kernel exactly -> ~1.0.
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            print_pcc=True,
        )
