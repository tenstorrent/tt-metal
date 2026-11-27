# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 SFPU activations (llk2::vec_*) through the Compute API.

One kernel (llk2_vec_compute_api_test.cpp) picks the activation from a compile-time parameter; only the golden
changes per activation. Stimuli domains, goldens and tolerances are the ones test_eltwise_unary_sfpu.py uses.

test_llk2_vec_exp_recip covers vec_exp and vec_recip in their forms (Exact, Fast, Polynomial) and regions (All,
LeftFaces, LeftFacesEvenColumns)
with torch goldens and per-path tolerances in the style of test_sfpu_sdpa.py, and checks which datums each one writes.
"""

import struct
from dataclasses import dataclass

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.sfpu_accuracy_budget import accuracy_contract
from helpers.sfpu_domains import exclude_undefined, for_op_pipeline
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, TemplateParameter
from helpers.utils import passed_test
from llk2_paths import cpp_source, include_dirs

# The llk2::VecFunction enumerator (resolver/vec_function.h) for each activation, and the harness op whose golden,
# domain and tolerance it uses.
VEC_FUNCTIONS = {
    "silu": ("Silu", MathOperation.Silu),
    "tanh": ("Tanh", MathOperation.Tanh),
    "gelu": ("Gelu", MathOperation.Gelu),
    "gelu_tanh": ("GeluTanh", MathOperation.GeluTanh),
    "relu_max": ("ReluMax", MathOperation.ReluMax),
    "sigmoid": ("Sigmoid", MathOperation.Sigmoid),
    "hardsigmoid": ("Hardsigmoid", MathOperation.Hardsigmoid),
    "hardtanh": ("Hardtanh", MathOperation.Hardtanh),
    "selu": ("Selu", MathOperation.Selu),
    "softplus": ("Softplus", MathOperation.Softplus),
}

# Activations the target kernel also runs in their fast (approximate) form.
HAS_FAST_FORM = {"tanh", "gelu", "gelu_tanh", "sigmoid"}

# Where the fast form is a different kernel with its own harness op (domain and tolerance), that op.
FAST_FORM_OP = {"sigmoid": MathOperation.SigmoidAppx}


# The llk2::Thread enumerator (tensor/tensor.h) for the thread that drives the SFPU.
VEC_THREADS = {"math": "Math", "pack": "Pack"}


@dataclass
class LLK2_VEC(TemplateParameter):
    function: str
    form: str  # llk2::VecForm: Exact, Fast or Polynomial
    thread: str
    rows: int
    cols: int
    scale: int = 0x3F800000  # exp's input scale, an FP32 bit pattern
    region: str = "All"  # llk2::VecRegion: All, LeftFaces or LeftFacesEvenColumns

    def convert_to_cpp(self) -> str:
        return (
            '#include "resolver/vec_function.h"\n'
            '#include "tensor/tensor.h"\n'
            f"constexpr auto VEC_FUNCTION = llk2::VecFunction::{self.function};\n"
            f"constexpr auto VEC_FORM = llk2::VecForm::{self.form};\n"
            f"constexpr auto VEC_REGION = llk2::VecRegion::{self.region};\n"
            f"constexpr auto VEC_THREAD = llk2::Thread::{self.thread};\n"
            f"constexpr std::uint32_t ROWS = {self.rows};\n"
            f"constexpr std::uint32_t COLS = {self.cols};\n"
            f"constexpr std::uint32_t VEC_SCALE = {self.scale:#010x}u;"
        )


def _variants():
    variants = []
    for name in VEC_FUNCTIONS:
        for fast in (False, True) if name in HAS_FAST_FORM else (False,):
            variants.append((name, fast))
    return variants


# bf16 in both Dest modes; fp32 in the 32-bit Dest mode, unpacked straight to Dest.
FORMATS_AND_DEST = [
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
    ),
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.Yes,
    ),
    (InputOutputFormat(DataFormat.Float32, DataFormat.Float32), DestAccumulation.Yes),
]


@parametrize(
    variant=_variants(),
    format_and_dest=FORMATS_AND_DEST,
    thread=list(VEC_THREADS),
    input_dimensions=[[32, 32], [64, 64]],
)
def test_llk2_vec(variant, format_and_dest, thread, input_dimensions):
    name, fast = variant
    formats, dest_acc = format_and_dest
    function, mathop = VEC_FUNCTIONS[name]
    if fast:
        mathop = FAST_FORM_OP.get(name, mathop)
    approx_mode = ApproximationMode.Yes if fast else ApproximationMode.No
    torch.manual_seed(0)

    spec_A = exclude_undefined(
        mathop,
        for_op_pipeline(
            mathop,
            formats.input_format,
            formats.output_format,
            approx_mode=approx_mode,
        ).spec_A,
    )
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
    )

    golden = get_golden_generator(UnarySFPUGolden)(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    configuration = TestConfig(
        cpp_source("llk2_vec_compute_api_test.cpp"),
        formats,
        templates=[
            DEST_SYNC(),
            LLK2_VEC(
                function,
                "Fast" if fast else "Exact",
                VEC_THREADS[thread],
                input_dimensions[0],
                input_dimensions[1],
            ),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit()
        and dest_acc == DestAccumulation.Yes,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    contract = accuracy_contract(
        mathop,
        output_format=formats.output_format,
        input_format=formats.input_format,
        approx_mode=approx_mode,
        dest_acc=dest_acc,
        arch=TestConfig.CHIP_ARCH,
    )
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden, res_tensor, formats.output_format, **contract.tolerance_kwargs()
    ), "Assert against golden failed"


# ---- The SDPA functions: exp (score and statistics forms) and reciprocal ----

TILE_DATUMS = 1024
FACE_DATUMS = 256
FACE_DIM = 16

FP32_ONE = 0x3F800000
SDPA_SCALE = 0x3DB504F3  # 1 / sqrt(128) = 0.08838834764831845, Wan 2.2's head dim; bf16 0x3DB5 = 0.08837890625


def _fp32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bf16(bits: int) -> float:
    """The FP32 pattern truncated to bf16, as 1.0's exp takes its scale (scale_fp32 >> 16)."""
    return _fp32(bits & 0xFFFF0000)


def _tile_masks():
    """Datums of one tile, in the tile order L1 holds (four 16 x 16 faces, row-major each), that a walk writes."""
    i = torch.arange(TILE_DATUMS)
    face = i // FACE_DATUMS
    col = i % FACE_DIM
    left = face % 2 == 0  # faces 0 and 2: tile columns 0..15
    return {
        "all": torch.ones(TILE_DATUMS, dtype=torch.bool),
        "left": left,  # 1.0's VectorMode::C
        "left_even": left & (col % 2 == 0),  # VecRegion::LeftFacesEvenColumns
    }


# name: (llk2::VecFunction, llk2::VecRegion, the datums that region writes, the forms run on it)
SDPA_FUNCTIONS = {
    # The fast exp is unclamped Schraudolph, right for scale x in [-88, 0.72]: scores after max subtraction.
    "exp": ("Exp", "All", "all", ("Exact", "Fast")),
    "exp_left_faces": ("Exp", "LeftFaces", "left", ("Exact",)),
    "exp_even_columns": (
        "Exp",
        "LeftFacesEvenColumns",
        "left_even",
        ("Polynomial", "Exact"),
    ),
    "recip": ("Reciprocal", "All", "all", ("Exact", "Fast")),
    "recip_left_faces": ("Reciprocal", "LeftFaces", "left", ("Exact", "Fast")),
}

# (atol, rtol), bf16 out. The fast exp is Schraudolph: up to 3.1% off by construction (modelled), plus the bf16 store
# truncating its low mantissa bit (0.8%). The others: twice the bf16 step, as test_sfpu_sdpa.py's bf16 rows, with an
# atol far below the smallest output (exp(-20) ~ 2e-9, 1 / 1e4) so it never decides a comparison.
SDPA_TOLERANCES = {
    ("exp", "Fast"): (1e-12, 5e-2),
    ("exp", "Exact"): (1e-12, 1.6e-2),
    ("exp_left_faces", "Exact"): (1e-12, 1.6e-2),
    ("exp_even_columns", "Exact"): (1e-12, 1.6e-2),
    ("exp_even_columns", "Polynomial"): (1e-12, 1.6e-2),
    ("recip", "Fast"): (1e-12, 1.6e-2),
    ("recip", "Exact"): (1e-12, 1.6e-2),
    ("recip_left_faces", "Fast"): (1e-12, 1.6e-2),
    ("recip_left_faces", "Exact"): (1e-12, 1.6e-2),
}


# rtol with Float32 data and results in a 32-bit Dest: no bf16 store, so what each function computes there. Measured
# on the board: the fast exp 3.12e-2 (Schraudolph's own error), the exact exp and reciprocal 1.2e-7 (one FP32 ulp), the
# polynomial exp 3.6e-6, the fast reciprocal 5.6e-3.
SDPA_FP32_RTOL = {
    ("exp", "Fast"): 3.2e-2,
    ("exp", "Exact"): 5e-7,
    ("exp_left_faces", "Exact"): 5e-7,
    ("exp_even_columns", "Exact"): 5e-7,
    ("exp_even_columns", "Polynomial"): 1e-5,
    ("recip", "Fast"): 8e-3,
    ("recip", "Exact"): 5e-7,
    ("recip_left_faces", "Fast"): 8e-3,
    ("recip_left_faces", "Exact"): 5e-7,
}


def _sdpa_variants():
    variants = []
    for name, (_, _, _, forms) in SDPA_FUNCTIONS.items():
        scales = (FP32_ONE, SDPA_SCALE) if name.startswith("exp") else (FP32_ONE,)
        for form in forms:
            for scale in scales:
                variants.append((name, form, scale))
    return variants


def _sdpa_stimulus(name, form, scale, tiles):
    """bf16 inputs. exp: scale * x uniform over its range (so the scale moves every output); recip: log-uniform."""
    n = tiles * TILE_DATUMS
    if name.startswith("recip"):
        x = torch.exp(torch.empty(n).uniform_(0.0, torch.log(torch.tensor(1e4)).item()))
    else:
        lo, hi = (-16.0, 0.5) if form == "Fast" else (-20.0, 5.0)
        x = torch.empty(n).uniform_(lo, hi) / _fp32(scale)
    return x.to(torch.bfloat16)


def _sdpa_golden(name, form, scale, x):
    xf = x.to(torch.float32)
    if name.startswith("recip"):
        return 1.0 / xf
    # The fast exp folds the FP32 scale into its constants; the others multiply by it rounded to bf16.
    s = _fp32(scale) if form == "Fast" else _bf16(scale)
    return torch.exp(xf * s)


@parametrize(
    variant=_sdpa_variants(),
    dest_acc=[
        DestAccumulation.No,
        DestAccumulation.Yes,
    ],  # SDPA's is 16-bit; the 32-bit Dest takes other SFPI forms
    thread=list(VEC_THREADS),
    input_dimensions=[[32, 64]],
)
def test_llk2_vec_exp_recip(variant, dest_acc, thread, input_dimensions):
    run_vec_exp_recip(variant, dest_acc, thread, input_dimensions)


# exp's half-face bodies on one tile, two tiles in a row and three in a column ([96, 32]: three tiles in one call).
@parametrize(
    variant=[
        ("exp_even_columns", form, scale)
        for form in ("Polynomial", "Exact")
        for scale in (FP32_ONE, SDPA_SCALE)
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    thread=list(VEC_THREADS),
    input_dimensions=[[32, 32], [32, 64], [96, 32]],
)
def test_llk2_vec_exp_even_columns_grids(variant, dest_acc, thread, input_dimensions):
    run_vec_exp_recip(variant, dest_acc, thread, input_dimensions)


# The fast exp's block walk (llk_lib/vec.h, vec_block_walk): every tile of a tensor in one call, the Dest row counter
# running on from tile to tile and the SFPU pipeline drained once. [1, 2] and [1, 4] tiles fit either Dest mode's
# section; [1, 8] and [2, 4] are a whole 16-bit section (SDPA's exp block), the longest walk there is: 512 rows.
# Every datum of every tile is compared, so a tile boundary that lost or repeated a row shows. The pack behind the
# exp takes the tensor from the SFPU on the SFPU's own thread and across the math -> pack hand-off.
@parametrize(
    variant=[("exp", "Fast", scale) for scale in (FP32_ONE, SDPA_SCALE)],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    thread=list(VEC_THREADS),
    input_dimensions=[[32, 64], [32, 128], [32, 256], [64, 128]],
)
def test_llk2_vec_exp_block_walk(variant, dest_acc, thread, input_dimensions):
    tiles = input_dimensions[0] * input_dimensions[1] // TILE_DATUMS
    if dest_acc == DestAccumulation.Yes and tiles > 4:
        pytest.skip("a 32-bit Dest section holds 4 tiles")
    run_vec_exp_recip(variant, dest_acc, thread, input_dimensions)


# The same functions on Float32 data in a 32-bit Dest (the input goes straight to Dest, 32 bits; the result is packed
# as Float32): what the function itself computes there, without the bf16 store's rounding. One tile row of two tiles,
# and the fast exp's block walk over a whole 32-bit section.
@parametrize(
    variant=_sdpa_variants(),
    thread=list(VEC_THREADS),
    input_dimensions=[[32, 64]],
)
def test_llk2_vec_exp_recip_fp32(variant, thread, input_dimensions):
    run_vec_exp_recip(
        variant, DestAccumulation.Yes, thread, input_dimensions, DataFormat.Float32
    )


@parametrize(
    variant=[("exp", "Fast", scale) for scale in (FP32_ONE, SDPA_SCALE)],
    thread=list(VEC_THREADS),
    input_dimensions=[[32, 128], [64, 64]],
)
def test_llk2_vec_exp_block_walk_fp32(variant, thread, input_dimensions):
    run_vec_exp_recip(
        variant, DestAccumulation.Yes, thread, input_dimensions, DataFormat.Float32
    )


def run_vec_exp_recip(
    variant, dest_acc, thread, input_dimensions, data_format=DataFormat.Float16_b
):
    name, form, scale = variant
    function, region, footprint, _ = SDPA_FUNCTIONS[name]
    formats = InputOutputFormat(data_format, data_format)
    tiles = input_dimensions[0] * input_dimensions[1] // TILE_DATUMS
    torch.manual_seed(0)

    # Tiles in L1 order: the stimulus is generated in that order, so the footprint masks index it directly.
    src_A = _sdpa_stimulus(name, form, scale, tiles).to(format_dict[data_format])
    src_B = torch.zeros_like(src_A)
    written = _tile_masks()[footprint].repeat(tiles)
    golden = torch.where(
        written, _sdpa_golden(name, form, scale, src_A), src_A.to(torch.float32)
    )

    configuration = TestConfig(
        cpp_source("llk2_vec_compute_api_test.cpp"),
        formats,
        templates=[
            DEST_SYNC(),
            LLK2_VEC(
                function,
                form,
                VEC_THREADS[thread],
                input_dimensions[0],
                input_dimensions[1],
                scale,
                region,
            ),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tiles,
            tile_count_B=tiles,
            tile_count_res=tiles,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=data_format.is_32_bit() and dest_acc == DestAccumulation.Yes,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])

    # What the function must not write keeps its input, bit for bit.
    untouched = ~written
    assert torch.equal(
        res_tensor[untouched].to(torch.float32), src_A[untouched].to(torch.float32)
    ), f"{name} wrote outside its footprint ({footprint})"

    atol, rtol = SDPA_TOLERANCES[(name, form)]
    if data_format == DataFormat.Float32:
        rtol = SDPA_FP32_RTOL[(name, form)]
        rel = (res_tensor.float() - golden).abs() / golden.abs().clamp(min=atol)
        assert (
            rel.max().item() <= rtol
        ), f"{name}: off by {rel.max().item():.3g} relative at datum {rel.argmax().item()} (rtol={rtol:g})"
        return
    assert passed_test(
        golden, res_tensor, formats.output_format, custom_atol=atol, custom_rtol=rtol
    ), f"Assert against golden failed (atol={atol:g}, rtol={rtol:g})"
