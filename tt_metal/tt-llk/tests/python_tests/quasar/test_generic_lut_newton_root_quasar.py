# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
NEWTON_ROOT (magic-seed + Newton/Householder) LUT activation on Quasar
(sim-qsr / ttsim).

Quasar sibling of the Blackhole scratch harness
sources/generic_lut_newton_root_bh_test.cpp + test_newton_root_craqsim_resolution.py.
Drives sources/quasar/generic_lut_newton_root_quasar_test.cpp, the BH->Quasar
port of the production EVAL_METHOD_NEWTON_ROOT path. The only BH->Quasar SFPU
substitution is addexp(x,-1) -> x*0.5f (SFPADDEXP missing on the Quasar ttsim
build); everything else is the same shift / int-subtract / FMA / exponent-field
arithmetic.

Env contract (mirrors the rational/poly quasar tests + the BH newton driver):
  * QUASAR_NR_CSV : path to a newton_root fitter CSV (METADATA carries
                    newton_root_magic / c1 / c2 / n / reciprocal / iters and
                    range_reduction_original_{min,max}).
  * QUASAR_ACT    : activation name (sqrt / rsqrt / cbrt), for ground_truth.

The Python golden is the ttpoly MODEL rangered._eval_newton_root (the exact bh
SFPU-FMA arithmetic the kernel runs), so the PCC / ULP isolates the BH->Quasar
translation. We also report bit-distance vs the TRUE activation for context.

Reproduce (from tt_metal/tt-llk/tests/python_tests):
  TT_METAL_HOME=/localdev/nkapre/tt-metal \
  TT_METAL_SIMULATOR=/localdev/nkapre/craq-sim-quasar/src/_out/release_qsr/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 CHIP_ARCH=quasar \
  QUASAR_ACT=sqrt \
  QUASAR_NR_CSV=/localdev/nkapre/tt-polynomial-fitter/data/coefficients/sqrt_p0_s1_uniform_fpminimax_ulp.csv \
  ../.venv/bin/python -m pytest --run-simulator \
  quasar/test_generic_lut_newton_root_quasar.py -x -s -q
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

FITTER = os.environ.get("TT_POLY_FIT_DIR", "/localdev/nkapre/tt-polynomial-fitter")
if FITTER not in sys.path:
    sys.path.insert(0, FITTER)

DEFAULT_CSV = (
    "/localdev/nkapre/tt-polynomial-fitter/data/coefficients/"
    "sqrt_p0_s1_uniform_fpminimax_ulp.csv"
)
DEFAULT_ACT = "sqrt"


# ---------------------------------------------------------------------------
# Newton-root LUT parsed from the fitter CSV METADATA.
# ---------------------------------------------------------------------------
@dataclass
class NewtonRootLUT:
    magic: int
    iters: int
    c1: float
    c2: float
    root_n: int
    recip: bool
    orig_min: float
    orig_max: float
    activation: str = DEFAULT_ACT


def parse_nr_csv(path: str, act: str = DEFAULT_ACT) -> NewtonRootLUT:
    meta = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["segment_id"] == "METADATA":
                meta[(row["lo"] or "").strip()] = (row["hi"] or "").strip()

    # DATA-DRIVEN, NO per-activation hardcoding. The newton_root params (magic
    # seed, root order n, reciprocal flag, Newton constants, iters) are genuine
    # method parameters sourced from the CSV METADATA — NOT from the activation
    # name. `newton_root_reciprocal` is what distinguishes sqrt (double-Newton)
    # from rsqrt (inverse-sqrt Newton), and the magic seed follows it. If the CSV
    # omits these, we CANNOT evaluate without hardcoding the activation, so the
    # config is OUT OF SCOPE — skip rather than guess a seed (mirrors the DFB
    # dfb_lut_driver _parse_rr_meta newton_root branch nr_keys check).
    nr_keys = ("newton_root_reciprocal", "newton_root_n", "newton_root_magic")
    if not any(k in meta for k in nr_keys):
        pytest.skip(
            f"{os.path.basename(path)}: newton_root_* METADATA missing "
            "(newton_root_reciprocal/_n/_magic) — out of scope (would require "
            "hardcoding the activation; the fitter must re-emit newton_root_* in the CSV)"
        )

    recip = str(meta.get("newton_root_reciprocal", "False")).strip().lower() in (
        "true",
        "1",
    )
    mm = str(meta.get("newton_root_magic", "")).strip()
    # The magic seed follows the reciprocal flag (a per-variant method constant,
    # not an activation key): rsqrt -> 0x5f3759df, sqrt -> 0x5f1110a0.
    magic = int(mm, 0) if mm else (0x5F3759DF if recip else 0x5F1110A0)

    return NewtonRootLUT(
        magic=magic,
        iters=int(float(meta.get("newton_root_iters", "2") or "2")),
        c1=float(meta.get("newton_root_c1", "0.0") or "0.0"),
        c2=float(meta.get("newton_root_c2", "0.0") or "0.0"),
        root_n=int(float(meta.get("newton_root_n", "2") or "2")),
        recip=recip,
        orig_min=float(meta.get("range_reduction_original_min", "1.0e-3")),
        orig_max=float(meta.get("range_reduction_original_max", "1.0e2")),
        activation=act,
    )


def _fmt(v) -> str:
    if abs(v) < 1.0e-37:
        return "0.0f"
    return f"{v:.9e}f"


# ---------------------------------------------------------------------------
# Template parameter: inject the newton-root LUT as -D defines (matches the BH
# driver's NR_DEFINES.convert_to_cpp; the kernel reads NEWTON_ROOT_*).
# ---------------------------------------------------------------------------
@dataclass
class NR_DEFINES(TemplateParameter):
    lut: NewtonRootLUT = None

    def convert_to_cpp(self) -> str:
        lut = self.lut
        # build.h is included (via params.h) AFTER the kernel's #ifndef fallback
        # defaults, so #undef each macro first to let the injected fitter values
        # win without tripping -Werror=redefined.
        defs = [
            ("NEWTON_ROOT_N", str(lut.root_n)),
            ("NEWTON_ROOT_ITERS", str(lut.iters)),
            ("NEWTON_ROOT_MAGIC", f"0x{lut.magic:08x}"),
            ("NEWTON_ROOT_C1", _fmt(lut.c1)),
            ("NEWTON_ROOT_C2", _fmt(lut.c2)),
        ]
        lines = []
        for name, val in defs:
            lines.append(f"#undef {name}")
            lines.append(f"#define {name} {val}")
        if lut.recip:
            lines.append("#define NEWTON_ROOT_RECIPROCAL")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Golden: ttpoly MODEL rangered._eval_newton_root (bh SFPU-FMA arithmetic) — the
# EXACT arithmetic the kernel runs, so PCC/ULP isolates the BH->Quasar port.
# ---------------------------------------------------------------------------
def model_newton_root(x_np: np.ndarray, lut: NewtonRootLUT) -> np.ndarray:
    from ttpoly.precision import rangered

    params = {
        "newton_root_magic": f"0x{lut.magic:08x}",
        "newton_root_iters": lut.iters,
        "newton_root_c1": lut.c1,
        "newton_root_c2": lut.c2,
        "expalu_root_n": lut.root_n,
        "expalu_reciprocal": lut.recip,
    }
    return rangered._eval_newton_root(
        params, x_np.astype(np.float32), "fp32", "bh"
    ).astype(np.float32)


def true_activation(act: str, x: torch.Tensor) -> torch.Tensor:
    from ground_truth import compute_ground_truth

    y = compute_ground_truth(act, x.to(torch.float64).numpy())
    return torch.as_tensor(np.asarray(y, dtype=np.float64))


def _bitdist_ulp(
    golden: torch.Tensor, result: torch.Tensor, out_format
) -> torch.Tensor:
    """Monotone-ordered bit-distance ULP (sign-correct), bf16 or fp32."""
    g = golden.to(torch.float32).flatten()
    r = result.to(torch.float32).flatten()
    if out_format == DataFormat.Float16_b:
        a = (g.view(torch.int32) >> 16).to(torch.int64)
        b = (r.view(torch.int32) >> 16).to(torch.int64)
        mask = 0xFFFF
        sign = 0x8000
    else:
        a = g.view(torch.int32).to(torch.int64)
        b = r.view(torch.int32).to(torch.int64)
        mask = 0xFFFFFFFF
        sign = 0x80000000

    def mono(v):
        v &= mask
        neg = (v & sign) != 0
        return torch.where(neg, (~v) & mask, v | sign).to(torch.int64)

    return (mono(a) - mono(b)).abs()


# fp32 isolates the translation; Float16_b proves the low-precision path.
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
def test_generic_lut_newton_root_quasar(
    formats,
    dest_sync,
    implied_math_format,
    input_dimensions,
):
    torch.manual_seed(0)

    csv_path = os.environ.get("QUASAR_NR_CSV", DEFAULT_CSV)
    act = os.environ.get("QUASAR_ACT", DEFAULT_ACT)
    lut = parse_nr_csv(csv_path, act)
    print(
        f"\n[generic_lut_newton_root_quasar] act={act} csv={os.path.basename(csv_path)} "
        f"root_n={lut.root_n} recip={lut.recip} iters={lut.iters} "
        f"magic=0x{lut.magic:08x} domain=[{lut.orig_min}, {lut.orig_max}]"
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

    # Scale inputs into the activation's original domain [orig_min, orig_max].
    dom_lo, dom_hi = lut.orig_min, lut.orig_max
    src_A_f = src_A.to(torch.float32)
    lo, hi = src_A_f.min(), src_A_f.max()
    if hi > lo:
        src_A_f = (src_A_f - lo) / (hi - lo) * (dom_hi - dom_lo) + dom_lo
    else:
        src_A_f = torch.full_like(src_A_f, (dom_lo + dom_hi) / 2.0)
    src_A = src_A_f.to(format_dict[formats.input_format])

    num_faces = MAX_NUM_FACES

    # The newton_root body has no SFPADDI-prone reciprocal NR subtract, but keep
    # the combine pass disabled for parity with the rational/poly quasar tests
    # (identical numerics; ttsim-safe).
    TestConfig.ARCH_SPECIFIC_OPTIONS = "-mno-tt-tensix-optimize-combine"

    # Golden = ttpoly model on the SAME (quantized) inputs the kernel sees.
    x_for_golden = src_A.to(torch.float32)
    golden_model = torch.from_numpy(
        model_newton_root(x_for_golden.numpy(), lut).astype(np.float32)
    )
    golden_tensor = golden_model.to(format_dict[formats.output_format])

    configuration = TestConfig(
        "sources/quasar/generic_lut_newton_root_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Sqrt),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(UnpackerEngine.UnpA),
            DEST_SYNC(dest_sync),
            NR_DEFINES(lut=lut),
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

    # Congruence (sim vs ttpoly model): the load-bearing metric for the port.
    pcc = calculate_pcc(golden_tensor, res_tensor)
    smv = _bitdist_ulp(golden_tensor, res_tensor, formats.output_format)
    finite = torch.isfinite(golden_tensor.to(torch.float32).flatten()) & torch.isfinite(
        res_tensor.to(torch.float32).flatten()
    )
    smv_f = smv[finite]

    # Context (sim vs true activation).
    x_true = src_A.to(torch.float32)
    y_true = true_activation(act, x_true).to(torch.float32)
    y_true_t = y_true.to(format_dict[formats.output_format])
    svt = _bitdist_ulp(y_true_t, res_tensor, formats.output_format)
    svt_f = svt[
        torch.isfinite(y_true.flatten())
        & torch.isfinite(res_tensor.to(torch.float32).flatten())
    ]

    print(f"[generic_lut_newton_root_quasar] PCC(sim_vs_model) = {pcc}")
    print(
        f"[generic_lut_newton_root_quasar] sim_vs_model bitdist max={int(smv_f.max()) if smv_f.numel() else -1} "
        f"mean={float(smv_f.to(torch.float64).mean()) if smv_f.numel() else -1:.4f} "
        f"frac_bit_exact={float((smv_f == 0).to(torch.float64).mean()) if smv_f.numel() else 0:.4f}"
    )
    print(
        f"[generic_lut_newton_root_quasar] sim_vs_true  bitdist max={int(svt_f.max()) if svt_f.numel() else -1} "
        f"mean={float(svt_f.to(torch.float64).mean()) if svt_f.numel() else -1:.4f}"
    )

    # The model is the exact arithmetic; the BH->Quasar port differs only in the
    # addexp->*0.5 substitution (bit-identical for finite normals), so the sim
    # must agree with the model to high PCC.
    assert pcc >= 0.99, f"PCC(sim_vs_model) {pcc} below 0.99 for {act}"
