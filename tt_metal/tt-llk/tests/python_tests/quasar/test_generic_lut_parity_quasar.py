# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generic piecewise-polynomial LUT activation on Quasar (sim-qsr / ttsim) with
PARITY x^2-HORNER and ADAPTIVE PER-SEGMENT DEGREE (eval-method parity_adaptive).

Sibling of test_generic_lut_activation_quasar.py. Same UNPACK -> SrcA -> FPU
datacopy -> Dest -> SFPU(embedded LUT) -> PACK recipe, but the kernel
(sources/quasar/generic_lut_parity_quasar_test.cpp) ports two PURE-EVALUATOR
features from the Blackhole/embedded kernels:

  * PARITY x^2-HORNER (POLY_PARITY_ODD / POLY_PARITY_EVEN): polynomials whose
    coefficients have a parity (odd: c0=c2=...=0; even: c1=c3=...=0) are
    evaluated in the x^2 basis with stride-2 access, halving the FMA count.
  * ADAPTIVE PER-SEGMENT DEGREE (SEGMENT_DEGREES[]): each segment unrolls Horner
    to its own effective degree (trailing zero high-order coeffs are skipped).

Both are exact reformulations of the SAME Horner polynomial, so the golden is
the plain natural-basis algorithmic Horner over the (clamped) coefficients — it
does NOT need to know about parity / adaptive degree. The kernel result must
therefore match the algorithmic golden to ~1.0 PCC. With real fitter coeffs and
QUASAR_ACT set, the golden is the TRUE activation (the fit's accuracy is then
measured end-to-end).

Modes (selected by env):
  * DEFAULT (no env): a built-in ODD-parity tanh-like deg-5 / 4-seg LUT (odd
    function -> all even coeffs are exactly 0), exercising the parity x^2-Horner
    path with adaptive per-segment degree. The golden replicates the EXACT baked
    coeffs (algorithmic), so PCC isolates kernel correctness (~1.0).
  * REAL FITTER COEFFS (QUASAR_LUT_CSV + QUASAR_ACT): parse the fitter CSV,
    auto-DETECT parity from the actual coefficients, derive per-segment degrees
    from trailing-zero high coeffs, bake them in. Golden = true activation.

Env CONTRACT:
  * QUASAR_LUT_CSV : path to a fitter polynomial coefficient CSV.
  * QUASAR_ACT     : activation name for compute_ground_truth (sin/tanh/...).
  * QUASAR_FITTER_REPO (optional): tt-polynomial-fitter repo path.
  * QUASAR_FORCE_PARITY (optional): "odd" / "even" / "none" to override detection.
  * Prints: "[generic_lut_parity_quasar] PCC = <n>", "... ULP = <n>",
            "... config = <act> ... parity=<odd|even|none> degrees=[...]".

Reproduce (from tt_metal/tt-llk/tests/python_tests):
  TT_METAL_HOME=/localdev/nkapre/tt-metal \
  TT_METAL_SIMULATOR=/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 CHIP_ARCH=quasar \
  python -m pytest --run-simulator quasar/test_generic_lut_parity_quasar.py -x -s -q
"""

import csv
import os
import sys
from dataclasses import dataclass, field

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
# Default (hardcoded) LUT — an ODD-parity deg-5 / 4-seg approximation of an odd
# function. All even-index coefficients are exactly 0.0, so the odd-parity
# x^2-Horner path is exercised. Per-segment effective degrees vary (adaptive).
# Coeffs are c0..c5 (low-order first). seg0 truncated to deg-3 (c5=0) to also
# exercise the adaptive-degree path.
# ---------------------------------------------------------------------------
DEFAULT_NUM_SEGMENTS = 4
DEFAULT_POLY_DEGREE = 5
DEFAULT_BOUNDARIES = [-2.0, -1.0, 0.0, 1.0, 2.0]
# tanh-ish odd fit; values are illustrative but self-consistent (the algorithmic
# golden replays these EXACT coeffs, so correctness is isolated regardless).
DEFAULT_COEFFS = [
    [0.0, 0.80, 0.0, -0.08, 0.0, 0.0],  # seg0: odd, effective deg 3 (c5=0)
    [0.0, 0.95, 0.0, -0.28, 0.0, 0.045],  # seg1: odd, deg 5
    [0.0, 0.95, 0.0, -0.28, 0.0, 0.045],  # seg2: odd, deg 5
    [0.0, 0.80, 0.0, -0.08, 0.0, 0.0],  # seg3: odd, deg 3
]


@dataclass
class LutConfig:
    activation: str
    degree: int
    num_segments: int
    boundaries: list
    coeffs: list
    use_ground_truth: bool
    parity: str = "none"  # "odd" | "even" | "none"
    seg_degrees: list = field(default_factory=list)  # effective degree per segment


def _default_lut() -> LutConfig:
    return LutConfig(
        activation="tanh",
        degree=DEFAULT_POLY_DEGREE,
        num_segments=DEFAULT_NUM_SEGMENTS,
        boundaries=list(DEFAULT_BOUNDARIES),
        coeffs=[list(c) for c in DEFAULT_COEFFS],
        use_ground_truth=False,
    )


# ---------------------------------------------------------------------------
# Parity + adaptive-degree detection on the ACTUAL coefficients.
# ---------------------------------------------------------------------------
_PAR_EPS = 1e-12


def _detect_parity(coeffs_per_seg, degree):
    """Detect a coefficient parity shared by ALL segments.

    odd  : every even-index coefficient (c0,c2,c4,...) is ~0 across all segments
           AND at least one odd-index coeff is nonzero.
    even : every odd-index coefficient (c1,c3,c5,...) is ~0 across all segments
           AND at least one even-index coeff is nonzero.
    Returns "odd" / "even" / "none". An explicit override wins.
    """
    forced = os.environ.get("QUASAR_FORCE_PARITY", "").strip().lower()
    if forced in ("odd", "even", "none"):
        return forced

    def all_zero(idxs):
        return all(
            abs(float(seg[i])) < _PAR_EPS
            for seg in coeffs_per_seg
            for i in idxs
            if i <= degree
        )

    def any_nonzero(idxs):
        return any(
            abs(float(seg[i])) >= _PAR_EPS
            for seg in coeffs_per_seg
            for i in idxs
            if i <= degree
        )

    even_idx = list(range(0, degree + 1, 2))
    odd_idx = list(range(1, degree + 1, 2))
    if all_zero(even_idx) and any_nonzero(odd_idx):
        return "odd"
    if all_zero(odd_idx) and any_nonzero(even_idx):
        return "even"
    return "none"


def _seg_effective_degree(seg_coeffs, parity):
    """Effective degree = highest index with a nonzero coeff respecting parity.

    For odd parity the highest LIVE index must be odd; for even, even. We return
    the highest index whose coeff is nonzero (the kernel walks down in stride-2
    from the parity-appropriate TOP <= this value), with a floor that keeps at
    least one live coefficient for the parity.
    """
    hi = 0
    for i, c in enumerate(seg_coeffs):
        if abs(float(c)) >= _PAR_EPS:
            hi = i
    if parity == "odd":
        # need an odd TOP; ensure >= 1
        if hi % 2 == 0:
            hi = max(hi - 1, 1)
        hi = max(hi, 1)
    elif parity == "even":
        if hi % 2 == 1:
            hi = max(hi - 1, 0)
    return hi


# ---------------------------------------------------------------------------
# CSV parsing (shared with the poly test; RR metadata is intentionally ignored —
# parity_adaptive is orthogonal to range reduction).
# ---------------------------------------------------------------------------
def _longest_nonasymptotic_run(seg_rows):
    best, cur = [], []
    for r in seg_rows:
        if not r[3]:
            cur.append(r)
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    return best


def _parse_fitter_csv(path: str, activation: str) -> LutConfig:
    seg_rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        coeff_idx = [
            i
            for i, h in enumerate(header)
            if h.strip().lower().startswith("c") and h.strip()[1:].isdigit()
        ]
        coeff_idx.sort(key=lambda i: int(header[i].strip()[1:]))
        lo_idx = header.index("lo")
        hi_idx = header.index("hi")
        asy_idx = header.index("is_asymptotic") if "is_asymptotic" in header else None
        for row in reader:
            if not row or row[0] == "METADATA":
                continue
            try:
                int(row[0])
            except ValueError:
                continue
            lo = float(row[lo_idx])
            hi = float(row[hi_idx])
            coeffs = [float(row[i]) for i in coeff_idx]
            asy = asy_idx is not None and str(row[asy_idx]).strip().lower() == "true"
            seg_rows.append((lo, hi, coeffs, asy))

    if not seg_rows:
        raise ValueError(f"No segment rows parsed from {path}")

    seg_rows.sort(key=lambda r: r[0])
    poly_rows = _longest_nonasymptotic_run(seg_rows)
    if not poly_rows:
        raise ValueError(f"{path}: no non-asymptotic segments to test.")
    seg_rows = poly_rows

    degree = len(seg_rows[0][2]) - 1
    num_segments = len(seg_rows)
    boundaries = [r[0] for r in seg_rows] + [seg_rows[-1][1]]
    coeffs = [r[2] for r in seg_rows]

    parity = _detect_parity(coeffs, degree)
    seg_degrees = [_seg_effective_degree(c, parity) for c in coeffs]

    return LutConfig(
        activation=activation,
        degree=degree,
        num_segments=num_segments,
        boundaries=boundaries,
        coeffs=coeffs,
        use_ground_truth=True,
        parity=parity,
        seg_degrees=seg_degrees,
    )


def _load_lut_config() -> LutConfig:
    csv_path = os.environ.get("QUASAR_LUT_CSV")
    if not csv_path:
        lut = _default_lut()
        lut.parity = _detect_parity(lut.coeffs, lut.degree)
        lut.seg_degrees = [_seg_effective_degree(c, lut.parity) for c in lut.coeffs]
        return lut
    act = os.environ.get("QUASAR_ACT", "tanh")
    return _parse_fitter_csv(csv_path, act)


# ---------------------------------------------------------------------------
# Build-header template parameter: bakes the LUT + parity/adaptive macros.
# ---------------------------------------------------------------------------
@dataclass
class GENERIC_LUT_DATA(TemplateParameter):
    lut: LutConfig = None
    emit: bool = True

    def convert_to_cpp(self) -> str:
        if not self.emit or self.lut is None:
            return "// GENERIC_LUT_DATA: using kernel-default (hardcoded) LUT"

        def f(v):
            v = 0.0 if abs(float(v)) < 1.18e-38 else float(v)
            return f"{v:.10e}f"

        parts = [
            "/* boundaries b0..bS */ "
            + ", ".join(f(b) for b in self.lut.boundaries)
            + ","
        ]
        for seg in range(self.lut.num_segments):
            parts.append(
                f"/* seg{seg} */ " + ", ".join(f(c) for c in self.lut.coeffs[seg]) + ","
            )
        body = " \\\n    ".join(parts)
        lines = [
            f"#define LUT_POLY_DEGREE {self.lut.degree}",
            f"#define LUT_NUM_SEGMENTS {self.lut.num_segments}",
            f"#define LUT_DATA_INIT {{ \\\n    {body} }}",
        ]

        # Parity macro (mutually exclusive odd/even; none -> natural-basis Horner).
        if self.lut.parity == "odd":
            lines.append("#define POLY_PARITY_ODD")
        elif self.lut.parity == "even":
            lines.append("#define POLY_PARITY_EVEN")

        # Adaptive per-segment degree. Only emit when it actually varies / is
        # below the nominal degree (otherwise the kernel's POLY_DEGREE fallback
        # is identical and we keep the build minimal).
        if self.lut.seg_degrees and any(
            d != self.lut.degree for d in self.lut.seg_degrees
        ):
            arr = ", ".join(str(int(d)) for d in self.lut.seg_degrees)
            lines.append(f"#define SEGMENT_DEGREES_INIT {{ {arr} }}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Goldens — natural-basis Horner over the EXACT coeffs (parity/adaptive degree
# are exact reformulations, so the golden is parity-agnostic).
# ---------------------------------------------------------------------------
def _eval_poly_horner(coeffs, x):
    acc = torch.full_like(x, float(coeffs[-1]))
    for k in range(len(coeffs) - 2, -1, -1):
        acc = acc * x + float(coeffs[k])
    return acc


def _algorithmic_golden(lut: LutConfig, x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float64)
    b0 = lut.boundaries[0]
    bN = lut.boundaries[lut.num_segments]
    x_clamped = torch.clamp(x, b0, bN)
    result = _eval_poly_horner(lut.coeffs[0], x_clamped)
    for seg in range(1, lut.num_segments):
        mask = x_clamped >= lut.boundaries[seg]
        result = torch.where(
            mask, _eval_poly_horner(lut.coeffs[seg], x_clamped), result
        )
    return result


def _ground_truth_golden(lut: LutConfig, x: torch.Tensor) -> torch.Tensor:
    repo = os.environ.get("QUASAR_FITTER_REPO", "/localdev/nkapre/tt-polynomial-fitter")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from ground_truth import compute_ground_truth  # noqa: E402

    b0 = lut.boundaries[0]
    bN = lut.boundaries[lut.num_segments]
    x_eval = torch.clamp(x.to(torch.float64), b0, bN)
    y = compute_ground_truth(lut.activation, x_eval.numpy())
    return torch.as_tensor(np.asarray(y, dtype=np.float64))


def _golden(lut: LutConfig, x: torch.Tensor) -> torch.Tensor:
    if lut.use_ground_truth:
        return _ground_truth_golden(lut, x)
    return _algorithmic_golden(lut, x)


def _max_ulp(golden, result, out_format):
    g = golden.to(torch.float32).flatten().numpy()
    r = result.to(torch.float32).flatten().numpy()
    finite = np.isfinite(g) & np.isfinite(r)
    g, r = g[finite], r[finite]
    if g.size == 0:
        return 0.0
    mant = 7 if out_format == DataFormat.Float16_b else 23
    ge = np.frexp(np.where(g == 0.0, 1.0, g))[1] - 1
    ulp_size = np.ldexp(1.0, ge - mant)
    return float(np.max(np.abs(r - g) / ulp_size))


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
def test_generic_lut_parity_quasar(
    formats,
    dest_sync,
    implied_math_format,
    input_dimensions,
):
    torch.manual_seed(0)

    lut = _load_lut_config()
    b0 = lut.boundaries[0]
    bN = lut.boundaries[lut.num_segments]

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

    # Scale inputs into the LUT span [b0, bN].
    src_A_f = src_A.to(torch.float32)
    lo, hi = src_A_f.min(), src_A_f.max()
    if hi > lo:
        src_A_f = (src_A_f - lo) / (hi - lo) * (bN - b0) + b0
    else:
        src_A_f = torch.full_like(src_A_f, (b0 + bN) / 2.0)
    src_A = src_A_f.to(format_dict[formats.input_format])

    num_faces = MAX_NUM_FACES

    golden_values = _golden(lut, src_A.to(torch.float32))
    golden_tensor = golden_values.to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/quasar/generic_lut_parity_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Sigmoid),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
            DEST_SYNC(dest_sync),
            GENERIC_LUT_DATA(lut=lut, emit=True),
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
    print(f"\n[generic_lut_parity_quasar] PCC = {pcc}")
    print(f"[generic_lut_parity_quasar] ULP = {ulp}")
    print(
        f"[generic_lut_parity_quasar] config = act={lut.activation} "
        f"degree={lut.degree} segments={lut.num_segments} "
        f"parity={lut.parity} degrees={lut.seg_degrees} "
        f"format={formats.output_format.name} "
        f"golden={'ground_truth' if lut.use_ground_truth else 'algorithmic'}"
    )

    if lut.use_ground_truth:
        assert pcc >= 0.99, f"PCC {pcc} below 0.99 for {lut.activation}"
    else:
        assert passed_test(
            golden_tensor,
            res_tensor,
            formats.output_format,
            print_pcc=True,
        )
