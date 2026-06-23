# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
STANDALONE exponent-ALU eval method on Quasar (sim-qsr / ttsim).

Drives sources/quasar/generic_lut_expalu_quasar_test.cpp, the DISTINCT eval
method EVAL_METHOD_EXPONENT_ALU: a single reduced-domain Horner that BYPASSES
the segment cascade (no clamp / boundaries / segment-select). The whole
activation comes from the hardware exponent ALU bit-decompose
(exexp / exman(ImplicitOne) / setexp / shft) plus one Horner over the fitter's
reduced-domain coefficients.

This is the Quasar sibling of the Blackhole craq-sim drivers
test_{exp2,expalu,log2alu,pow}_craqsim_resolution.py (which drive the BH
exp_hw_eval / log_hw_eval / pow_hw_eval). It follows the SAME env contract as the
poly / rational quasar tests so the quasar sweep driver can call it uniformly:
  * QUASAR_LUT_CSV : path to a fitter polynomial coefficients CSV with
                     exponent-ALU range reduction in METADATA.
  * QUASAR_ACT     : activation name (for the ground_truth golden).

The kernel is selected automatically from the CSV's range_reduction_method:
  exponent_alu_exp2 -> EXPALU_MODE 1 (exp_hw_eval, optional sigmoid / minus_one)
  exponent_alu_log2 -> EXPALU_MODE 2 (log_hw_eval, optional offset / m_minus_1)
  exponent_alu_pow  -> EXPALU_MODE 3 (pow_hw_eval, optional reciprocal)

Golden = the TRUE activation over the FULL original domain (the fitter's
ground_truth), since the standalone exponent-ALU kernel reproduces the whole
activation. PCC is gated >= 0.99 (same bar as the rational ground-truth path).

Reproduce (from tt_metal/tt-llk/tests/python_tests):
  TT_METAL_HOME=/localdev/nkapre/tt-metal \
  TT_METAL_SIMULATOR=/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 CHIP_ARCH=quasar \
  QUASAR_ACT=sigmoid \
  QUASAR_LUT_CSV=<...exponent_alu...csv> \
  python -m pytest --run-simulator \
  quasar/test_generic_lut_expalu_quasar.py -x -s -q
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
from helpers.utils import calculate_pcc

DEFAULT_CSV = os.environ.get("QUASAR_LUT_CSV", "")
DEFAULT_ACT = os.environ.get("QUASAR_ACT", "sigmoid")

_MODE_CODE = {
    "exponent_alu_exp2": 1,
    "exponent_alu_log2": 2,
    "exponent_alu_pow": 3,
}
_COMPOSE_CODE = {"": 0, "sigmoid": 1, "minus_one": 2}


# ---------------------------------------------------------------------------
# LUT parsed from a single-segment polynomial CSV with exponent-ALU reduction.
# ---------------------------------------------------------------------------
@dataclass
class ExpaluLUT:
    mode: int  # 1 exp, 2 log, 3 pow
    method: str
    degree: int
    coeffs: list  # c0..cDEG, low-order first
    domain: tuple  # (lo, hi) ORIGINAL full domain
    activation: str
    params: dict


def parse_expalu_csv(path: str, act: str) -> ExpaluLUT:
    seg_rows, meta = [], {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["segment_id"] == "METADATA":
                meta[(row["lo"] or "").strip()] = (row["hi"] or "").strip()
                continue
            seg_rows.append(row)
    seg_rows.sort(key=lambda r: int(r["segment_id"]))
    if not seg_rows:
        raise ValueError(f"No segment rows in {path}")

    method = meta.get("range_reduction_method", "none")
    if method not in _MODE_CODE:
        raise ValueError(
            f"CSV range_reduction_method='{method}' is not an exponent-ALU method "
            f"(expected one of {sorted(_MODE_CODE)})"
        )
    mode = _MODE_CODE[method]

    # Degree = number of c{i} columns present in the (single) segment row.
    deg = 0
    r0 = seg_rows[0]
    while f"c{deg + 1}" in r0 and r0[f"c{deg + 1}"] not in (None, ""):
        deg += 1
    coeffs = [float(r0[f"c{i}"]) for i in range(deg + 1)]

    def _mf(k):
        return float(meta[k]) if k in meta and meta[k] != "" else None

    dom_lo = _mf("range_reduction_original_min")
    dom_hi = _mf("range_reduction_original_max")

    params = {}
    if mode == 1:
        params["mult"] = _mf("expalu_log2_multiplier")
        params["compose"] = meta.get("expalu_compose", "").strip()
    elif mode == 2:
        params["scale"] = _mf("expalu_log_scale")
        params["basis"] = meta.get("expalu_log2_basis", "m")
        params["offset"] = _mf("expalu_input_offset") or 0.0
    elif mode == 3:
        params["n"] = int(float(meta["expalu_root_n"]))
        params["recip"] = meta.get("expalu_reciprocal", "False").lower() == "true"
        params["scale"] = [
            _mf("expalu_pow_scale_c0"),
            _mf("expalu_pow_scale_c1"),
            _mf("expalu_pow_scale_c2"),
        ]

    return ExpaluLUT(
        mode=mode,
        method=method,
        degree=deg,
        coeffs=coeffs,
        domain=(dom_lo, dom_hi),
        activation=act,
        params=params,
    )


# ---------------------------------------------------------------------------
# Template parameter: inject the LUT into build.h (-> params.h -> kernel).
# ---------------------------------------------------------------------------
@dataclass
class EXPALU_LUT(TemplateParameter):
    lut: ExpaluLUT = None

    @staticmethod
    def _fmt_one(v) -> str:
        if abs(v) < 1.0e-37:
            return "0.0f"
        return f"{v:.9e}f"

    @staticmethod
    def _fmt_floats(values) -> str:
        return ", ".join(EXPALU_LUT._fmt_one(v) for v in values)

    def convert_to_cpp(self) -> str:
        lut = self.lut
        p = lut.params
        lines = [
            "// === Injected standalone exponent-ALU LUT (from QUASAR_LUT_CSV) ===",
            f"#define EXPALU_MODE {lut.mode}",
            f"#define EXPALU_DEG {lut.degree}",
            f"#define EXPALU_COEFFS_INIT {self._fmt_floats(lut.coeffs)}",
        ]
        if lut.mode == 1:
            lines.append(f"#define EXPALU_EXP_MULT {p['mult']:.10e}f")
            lines.append(f"#define EXPALU_COMPOSE {_COMPOSE_CODE[p['compose']]}")
        elif lut.mode == 2:
            lines.append(f"#define EXPALU_LOG_SCALE {p['scale']:.10e}f")
            lines.append(
                f"#define EXPALU_LOG_BASIS_MMINUS1 {1 if p['basis'] == 'm_minus_1' else 0}"
            )
            if abs(p["offset"]) > 0.0:
                lines.append(f"#define EXPALU_LOG_OFFSET {p['offset']:.10e}f")
        elif lut.mode == 3:
            lines.append(f"#define EXPALU_POW_N {p['n']}")
            scale = p["scale"]
            for i, key in enumerate(
                ("EXPALU_POW_C0", "EXPALU_POW_C1", "EXPALU_POW_C2")
            ):
                c = scale[i] if i < len(scale) else None
                if c is not None:
                    lines.append(f"#define {key} {c:.10e}f")
            if p["recip"]:
                lines.append("#define EXPALU_POW_RECIP")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Golden: TRUE activation over the full original domain (fitter ground_truth).
# ---------------------------------------------------------------------------
def _ground_truth(lut: ExpaluLUT, x: torch.Tensor) -> torch.Tensor:
    repo = os.environ.get("QUASAR_FITTER_REPO", "/localdev/nkapre/tt-polynomial-fitter")
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from ground_truth import compute_ground_truth  # noqa: E402

    y = compute_ground_truth(lut.activation, x.to(torch.float64).numpy())
    return torch.as_tensor(np.asarray(y, dtype=np.float64))


def _ulp_error(golden: torch.Tensor, result: torch.Tensor, out_format) -> float:
    is_bf16 = out_format == DataFormat.Float16_b
    g = golden.to(torch.float32).flatten()
    r = result.to(torch.float32).flatten()
    if is_bf16:
        gi = (g.view(torch.int32) >> 16).to(torch.int64)
        ri = (r.view(torch.int32) >> 16).to(torch.int64)
    else:
        gi = g.view(torch.int32).to(torch.int64)
        ri = r.view(torch.int32).to(torch.int64)
    return float((gi - ri).abs().to(torch.float64).mean())


# fp32 isolates kernel correctness; Float16_b proves the low-precision path.
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
def test_generic_lut_expalu_quasar(
    formats,
    dest_sync,
    implied_math_format,
    input_dimensions,
):
    torch.manual_seed(0)

    csv_path = os.environ.get("QUASAR_LUT_CSV", DEFAULT_CSV)
    act = os.environ.get("QUASAR_ACT", DEFAULT_ACT)
    if not csv_path:
        pytest.skip("QUASAR_LUT_CSV not set")
    lut = parse_expalu_csv(csv_path, act)
    print(
        f"\n[generic_lut_expalu_quasar] act={act} csv={os.path.basename(csv_path)} "
        f"method={lut.method} mode={lut.mode} deg={lut.degree} "
        f"domain=[{lut.domain[0]}, {lut.domain[1]}]"
    )

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

    # Scale inputs into the FULL original domain (the standalone exponent-ALU
    # kernel covers the whole activation domain).
    dom_lo, dom_hi = lut.domain
    src_A_f = src_A.to(torch.float32)
    lo, hi = src_A_f.min(), src_A_f.max()
    if hi > lo:
        src_A_f = (src_A_f - lo) / (hi - lo) * (dom_hi - dom_lo) + dom_lo
    else:
        src_A_f = torch.full_like(src_A_f, (dom_lo + dom_hi) / 2.0)
    src_A = src_A_f.to(format_dict[formats.input_format])

    num_faces = MAX_NUM_FACES

    # The sigmoid / rsqrt composes use the iterative reciprocal whose
    # Newton-Raphson subtract would otherwise fold into an SFPADDI immediate at
    # -O3 (ttsim aborts on SFPADDI). Disable the instruction-combine pass — same
    # numerics, ttsim-supported (matches the rational quasar test).
    TestConfig.ARCH_SPECIFIC_OPTIONS = "-mno-tt-tensix-optimize-combine"

    golden_values = _ground_truth(lut, src_A.to(torch.float32))
    golden_tensor = golden_values.to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/quasar/generic_lut_expalu_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Sigmoid),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
            DEST_SYNC(dest_sync),
            EXPALU_LUT(lut=lut),
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
    print(f"[generic_lut_expalu_quasar] PCC = {pcc}")
    print(f"[generic_lut_expalu_quasar] ULP = {ulp}")

    assert pcc >= 0.99, f"PCC {pcc} below 0.99 for {lut.activation} ({lut.method})"
