# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generic piecewise-RATIONAL LUT activation on Quasar (sim-qsr / ttsim).

Sibling of test_generic_lut_activation_quasar.py (piecewise-polynomial). Each
segment evaluates a rational approximation P(x)/Q(x) (numerator degree
num_degree, denominator degree den_degree) emitted by the tt-polynomial-fitter.
The SFPU computes the division as P(x) * (1 / Q(x)) using the iterative sfpi
reciprocal (_sfpu_reciprocal_), the silicon-representative reciprocal path.

UNPACK -> SrcA -> FPU datacopy (MOVA2D) -> Dest -> SFPU(embedded rational LUT)
-> PACK.

Follows the SAME env contract as the poly test so a single sweep driver can
call both uniformly:
  * QUASAR_LUT_CSV : path to a fitter rational coefficients CSV
                     (data/coefficients/<act>_n<num>d<den>_s<seg>_..._rational_*.csv)
  * QUASAR_ACT     : activation name (used for logging / optional ground-truth
                     reference PCC)
Prints, per format:
  [generic_lut_rational_quasar] PCC = <number>
  [generic_lut_rational_quasar] ULP = <number>

The Python golden replicates the EXACT same algorithm the kernel runs (clamp ->
segment select -> P(x)/Q(x)) on the SAME injected coefficients, so the PCC
isolates kernel correctness on the fitter's rational coefficients.

Reproduce (from tt_metal/tt-llk/tests/python_tests):
  TT_METAL_HOME=/localdev/nkapre/tt-metal-nkapreTT \
  TT_METAL_SIMULATOR=/home/nkapre/sim-qsr/libttsim.so CHIP_ARCH=quasar \
  QUASAR_ACT=sigmoid \
  QUASAR_LUT_CSV=/localdev/nkapre/tt-polynomial-fitter/data/coefficients/sigmoid_n4d4_s2_uniform_rational_ulp.csv \
  ../.venv/bin/python -m pytest --run-simulator \
  quasar/test_generic_lut_rational_quasar.py -x -s -q
"""

import csv
import os
from dataclasses import dataclass

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

# Default config: sigmoid 2-segment n4d4 rational (fp32 mae 6e-6, bf16 max_ulp ~1).
DEFAULT_CSV = (
    "/localdev/nkapre/tt-polynomial-fitter/data/coefficients/"
    "sigmoid_n4d4_s2_uniform_rational_ulp.csv"
)
DEFAULT_ACT = "sigmoid"


# ---------------------------------------------------------------------------
# Rational LUT parsed from the fitter CSV.
# ---------------------------------------------------------------------------
@dataclass
class RationalLUT:
    num_segments: int
    num_degree: int
    den_degree: int
    boundaries: list  # length num_segments + 1
    num_coeffs: list  # num_segments lists, each length num_degree + 1 (c0..cn)
    den_coeffs: list  # num_segments lists, each length den_degree + 1 (c0..cm)


def parse_rational_csv(path: str) -> RationalLUT:
    seg_rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["segment_id"] == "METADATA":
                continue
            seg_rows.append(row)

    seg_rows.sort(key=lambda r: int(r["segment_id"]))
    if not seg_rows:
        raise ValueError(f"No segment rows in {path}")

    num_degree = int(seg_rows[0]["num_degree"])
    den_degree = int(seg_rows[0]["den_degree"])

    # The flat C++ array layout requires a uniform per-segment stride. The
    # fitter emits uniform degrees within one config (filename n<num>d<den>).
    for r in seg_rows:
        if int(r["num_degree"]) != num_degree or int(r["den_degree"]) != den_degree:
            raise ValueError("Non-uniform per-segment degrees are unsupported")

    boundaries = [float(seg_rows[0]["lo"])] + [float(r["hi"]) for r in seg_rows]
    num_coeffs = [[float(r[f"n{i}"]) for i in range(num_degree + 1)] for r in seg_rows]
    den_coeffs = [[float(r[f"d{i}"]) for i in range(den_degree + 1)] for r in seg_rows]

    return RationalLUT(
        num_segments=len(seg_rows),
        num_degree=num_degree,
        den_degree=den_degree,
        boundaries=boundaries,
        num_coeffs=num_coeffs,
        den_coeffs=den_coeffs,
    )


# ---------------------------------------------------------------------------
# Template parameter: inject the rational LUT into build.h (-> params.h ->
# kernel). Self-contained here so the shared test_variant_parameters.py is not
# touched.
# ---------------------------------------------------------------------------
@dataclass
class RATIONAL_LUT(TemplateParameter):
    lut: RationalLUT = None

    @staticmethod
    def _fmt_one(v) -> str:
        # Coeffs below ~FLT_MIN (1.18e-38) underflow fp32 -> a float literal
        # like 1e-56f truncates to a subnormal/zero and trips
        # -Werror=overflow ("floating constant truncated to zero"). These are
        # below fp32 representability and contribute nothing, so flush to 0.0f
        # (numerically correct). Otherwise emit with enough precision + 'f'.
        if abs(v) < 1.0e-37:
            return "0.0f"
        return f"{v:.9e}f"

    @staticmethod
    def _fmt_floats(values) -> str:
        return ", ".join(RATIONAL_LUT._fmt_one(v) for v in values)

    def convert_to_cpp(self) -> str:
        lut = self.lut
        nseg = lut.num_segments
        nd = lut.num_degree
        dd = lut.den_degree

        flat_num = [c for seg in lut.num_coeffs for c in seg]
        flat_den = [c for seg in lut.den_coeffs for c in seg]

        lines = [
            "// === Injected piecewise-rational LUT (from QUASAR_LUT_CSV) ===",
            f"constexpr std::uint32_t RAT_NUM_SEGMENTS = {nseg};",
            f"constexpr std::uint32_t RAT_NUM_DEGREE = {nd};",
            f"constexpr std::uint32_t RAT_DEN_DEGREE = {dd};",
            f"constexpr std::array<float, {nseg + 1}> RAT_BOUNDARIES = {{ {self._fmt_floats(lut.boundaries)} }};",
            f"constexpr std::array<float, {nseg * (nd + 1)}> RAT_NUM_COEFFS = {{ {self._fmt_floats(flat_num)} }};",
            f"constexpr std::array<float, {nseg * (dd + 1)}> RAT_DEN_COEFFS = {{ {self._fmt_floats(flat_den)} }};",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Golden: EXACT replica of the kernel algorithm on the same coefficients.
# clamp x to [b0, bN] -> select segment by boundaries -> P(x)/Q(x).
# ---------------------------------------------------------------------------
def _eval_poly(coeffs, x):
    # Horner, low-order-first coeffs.
    acc = torch.full_like(x, float(coeffs[-1]))
    for c in reversed(coeffs[:-1]):
        acc = acc * x + float(c)
    return acc


def piecewise_rational_golden(x: torch.Tensor, lut: RationalLUT) -> torch.Tensor:
    x = x.to(torch.float64)
    b = lut.boundaries
    x_clamped = torch.clamp(x, b[0], b[lut.num_segments])

    p = _eval_poly(lut.num_coeffs[0], x_clamped)
    q = _eval_poly(lut.den_coeffs[0], x_clamped)
    for seg in range(1, lut.num_segments):
        mask = x_clamped >= b[seg]
        p = torch.where(mask, _eval_poly(lut.num_coeffs[seg], x_clamped), p)
        q = torch.where(mask, _eval_poly(lut.den_coeffs[seg], x_clamped), q)
    return p / q


def _ulp_error(golden: torch.Tensor, result: torch.Tensor, out_format) -> float:
    """Mean ULP distance in the output format's representation."""
    is_bf16 = out_format == DataFormat.Float16_b
    g = golden.to(torch.float32).flatten()
    r = result.to(torch.float32).flatten()
    if is_bf16:
        # bf16 = top 16 bits of fp32; ULP step = 1 in those 16 bits.
        gi = (g.view(torch.int32) >> 16).to(torch.int64)
        ri = (r.view(torch.int32) >> 16).to(torch.int64)
    else:
        gi = g.view(torch.int32).to(torch.int64)
        ri = r.view(torch.int32).to(torch.int64)
    return float((gi - ri).abs().to(torch.float64).mean())


# fp32 isolates kernel correctness; Float16_b proves the low-precision path.
LUT_FORMATS = input_output_formats([DataFormat.Float32, DataFormat.Float16_b], same=True)


@pytest.mark.quasar
@parametrize(
    formats=LUT_FORMATS,
    dest_sync=[DestSync.Half],
    implied_math_format=[ImpliedMathFormat.Yes],
    input_dimensions=[[32, 32]],
)
def test_generic_lut_rational_quasar(
    formats,
    dest_sync,
    implied_math_format,
    input_dimensions,
):
    torch.manual_seed(0)

    csv_path = os.environ.get("QUASAR_LUT_CSV", DEFAULT_CSV)
    act = os.environ.get("QUASAR_ACT", DEFAULT_ACT)
    lut = parse_rational_csv(csv_path)
    print(
        f"\n[generic_lut_rational_quasar] act={act} csv={os.path.basename(csv_path)} "
        f"segs={lut.num_segments} num_deg={lut.num_degree} den_deg={lut.den_degree} "
        f"domain=[{lut.boundaries[0]}, {lut.boundaries[-1]}]"
    )

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

    # Scale inputs into the LUT domain [b0, bN].
    lo_dom, hi_dom = lut.boundaries[0], lut.boundaries[-1]
    src_A_f = src_A.to(torch.float32)
    lo, hi = src_A_f.min(), src_A_f.max()
    if hi > lo:
        span = hi_dom - lo_dom
        src_A_f = (src_A_f - lo) / (hi - lo) * span + lo_dom
    else:
        src_A_f = torch.zeros_like(src_A_f)
    src_A = src_A_f.to(format_dict[formats.input_format])

    num_faces = MAX_NUM_FACES

    # Disable the Tensix instruction-combine pass for this test's kernels. The
    # SFPU reciprocal's Newton-Raphson step (x*y - vConstFloatPrgm0, with
    # vConstFloatPrgm0 == 2.0) is otherwise folded into an SFPADDI immediate at
    # -O3. The Quasar ttsim build implements SFPADDI only as a throwing stub
    # (MissingSpecification: tensix_execute_sfpaddi), so high-degree /
    # single-segment configs (e.g. silu/swish n10d10_s1) abort. With the combine
    # pass off, the subtract stays a register-operand SFPADD instead — identical
    # numerics, ttsim-supported, and it does not change any other config's PCC.
    TestConfig.ARCH_SPECIFIC_OPTIONS = "-mno-tt-tensix-optimize-combine"

    golden_values = piecewise_rational_golden(src_A.to(torch.float32), lut)
    golden_tensor = golden_values.to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/quasar/generic_lut_rational_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Sigmoid),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
            DEST_SYNC(dest_sync),
            RATIONAL_LUT(lut=lut),
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
    ulp = _ulp_error(golden_tensor, res_tensor, formats.output_format)
    print(f"[generic_lut_rational_quasar] PCC = {pcc}")
    print(f"[generic_lut_rational_quasar] ULP = {ulp}")

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        print_pcc=True,
    )
