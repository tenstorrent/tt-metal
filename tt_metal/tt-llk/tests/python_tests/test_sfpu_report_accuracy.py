# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Raw device results for the LLK SFPU report (``tt-llk/sfpu_report``).

Not a test in the usual sense: it asserts nothing about accuracy. It runs each
requested op over two stimuli per variant and saves ``(src, golden, result)``, so
the report can compare the base side's results with the head side's lane by lane:

* ``sweep``    -- every finite value of the input format (float32: a stride of
  them), the same stimuli as the nightly ULP sweep in ``test_unary_sfpu_ulp.py``;
* ``specials`` -- the inputs a sweep never feeds: NaNs, infinities, signed zeros
  and subnormals.

It only runs when the report asks for it, through the environment::

    SFPU_REPORT_DUMP=<dir> SFPU_REPORT_OPS=Tanh,Exp pytest test_sfpu_report_accuracy.py
"""

import os
from pathlib import Path

import pytest
import torch
from helpers.data_format_inference import is_format_combination_outlier
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    TILE_DIMENSIONS,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    DestSync,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    generate_input_dim,
)
from helpers.ulp_sweep import stimuli_format_for, sweep_spec

pytestmark = pytest.mark.accuracy

DUMP_DIR = os.environ.get("SFPU_REPORT_DUMP")
OPS = [s for s in os.environ.get("SFPU_REPORT_OPS", "").split(",") if s]

#: (input, output) pairs the report measures: one per format a reviewer reads.
FORMAT_PAIRS = [
    (DataFormat.Float16_b, DataFormat.Float16_b),
    (DataFormat.Float16, DataFormat.Float16),
    (DataFormat.Float32, DataFormat.Float32),
]

#: 64 tiles hold every finite bfloat16/float16 value.
SWEEP_DIMENSIONS = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * 64]
SPECIALS_DIMENSIONS = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1]]


def _cells():
    return [
        (in_fmt, out_fmt, approx, dest)
        for in_fmt, out_fmt in FORMAT_PAIRS
        for approx in ApproximationMode
        for dest in DestAccumulation
        if not is_format_combination_outlier(in_fmt, out_fmt, dest)
        # A float32 input through a 16-bit Dest is truncated before the SFPU sees
        # it: that measures the Dest, not the kernel.
        and not (in_fmt == DataFormat.Float32 and dest == DestAccumulation.No)
    ]


def special_values(fmt):
    """One tile of the inputs a finite sweep skips, grouped by class.

    Returns ``(values, classes)``: ``classes[i]`` names the class of ``values[i]``.
    """
    dtype = format_dict[fmt]
    info = torch.finfo(dtype)
    tiny = info.tiny  # smallest normal
    groups = {
        "nan": [float("nan"), -float("nan")],
        "inf": [float("inf"), -float("inf")],
        "zero": [0.0, -0.0],
        "subnormal": [tiny / 2, -tiny / 2, tiny / 8, -tiny / 8],
        "extreme": [info.max, -info.max, tiny, -tiny],
    }
    values, classes = [], []
    for name, vals in groups.items():
        values += vals
        classes += [name] * len(vals)
    n = TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]
    reps = -(-n // len(values))
    src = torch.tensor(values * reps, dtype=torch.float64)[:n].to(dtype)
    return src, (classes * reps)[:n]


def _run(mathop, formats, approx_mode, dest_acc, src_A, tile_cnt, dimensions):
    src_B = src_A.clone()
    golden = get_golden_generator(UnarySFPUGolden)(
        mathop, src_A, formats.output_format, dest_acc, formats.input_format, dimensions
    )
    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        dimensions,
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )
    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(dimensions, dimensions),
            APPROX_MODE(approx_mode),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
    )
    result = configuration.run().result
    return golden, torch.tensor(result, dtype=format_dict[formats.output_format])


@pytest.mark.skipif(not DUMP_DIR or not OPS, reason="run by the LLK SFPU report only")
@pytest.mark.parametrize("kind", ["sweep", "specials"])
@pytest.mark.parametrize(
    "in_fmt, out_fmt, approx_mode, dest_acc",
    [
        pytest.param(
            *c, id=f"{c[0].name}-{c[1].name}-approx:{c[2].name}-dest_acc:{c[3].name}"
        )
        for c in _cells()
    ],
)
@pytest.mark.parametrize("op_name", OPS or ["none"])
def test_sfpu_report_accuracy(op_name, in_fmt, out_fmt, approx_mode, dest_acc, kind):
    mathop = MathOperation[op_name]
    formats = InputOutputFormat(in_fmt, out_fmt)
    classes = None
    if kind == "sweep":
        torch.manual_seed(0)
        stimuli_format = stimuli_format_for(in_fmt)
        src, tile_cnt, _, _ = generate_stimuli(
            stimuli_format_A=stimuli_format,
            input_dimensions_A=SWEEP_DIMENSIONS,
            stimuli_format_B=stimuli_format,
            input_dimensions_B=SWEEP_DIMENSIONS,
            spec_A=sweep_spec(in_fmt),
        )
        dims = SWEEP_DIMENSIONS
    else:
        src, classes = special_values(in_fmt)
        tile_cnt, dims = 1, SPECIALS_DIMENSIONS
    try:
        golden, result = _run(
            mathop, formats, approx_mode, dest_acc, src, tile_cnt, dims
        )
    except OverflowError as exc:
        pytest.skip(f"golden cannot be computed: {exc}")
    name = f"{mathop.name}__{in_fmt.name}-{out_fmt.name}__{approx_mode.name}__{dest_acc.name}__{kind}.pt"
    out = Path(DUMP_DIR)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "op": mathop.name,
            "in": in_fmt.name,
            "out": out_fmt.name,
            "approx": approx_mode.name,
            "dest_acc": dest_acc.name,
            "kind": kind,
            "src": src,
            "golden": golden,
            "result": result,
            "classes": classes,
        },
        out / name,
    )


# =============================================================================
# Binary SFPU ops
#
# Runs through sfpu_binary() of test_eltwise_binary_sfpu.py, the functional binary
# driver, so the operand layout (tile 2k = in0, 2k+1 = in1), the per-op stimuli
# domains and the golden are exactly the functional test's. Its final assertion is
# swapped for a capture: the report compares, it does not judge.
# =============================================================================

BINARY_OPS = [s for s in os.environ.get("SFPU_REPORT_BINARY_OPS", "").split(",") if s]

F16B, F16, F32 = DataFormat.Float16_b, DataFormat.Float16, DataFormat.Float32
I32, U32 = DataFormat.Int32, DataFormat.UInt32

#: op -> formats it is measured in (as test_eltwise_binary_sfpu tests it, in == out).
#: Ops whose result is not a rounded real number (comparisons, integers) are compared
#: exactly: a lane is right or wrong.
_FLOAT_FULL = [F16B, F16, F32]
BINARY_FORMATS = {
    **{
        op: _FLOAT_FULL
        for op in (
            "SfpuElwmul",
            "SfpuElwrsub",
            "SfpuElwpow",
            "SfpuXlogy",
            "SfpuLogaddexp",
            "SfpuLogaddexp2",
            "SfpuBinaryMax",
            "SfpuBinaryMin",
            "SfpuBinaryFmod",
            "SfpuBinaryRemainder",
            "SfpuElwEq",
            "SfpuElwNe",
        )
    },
    "SfpuElwadd": _FLOAT_FULL + [I32],
    "SfpuElwsub": _FLOAT_FULL + [I32],
    "SfpuElwdiv": [F16B, F32],
    "SfpuAtan2": [F16B, F32],
    "SfpuIsclose": [F16B, F32],
    "SfpuLogsigmoid": [F16B, F32],
    **{
        op: _FLOAT_FULL + [I32]
        for op in ("SfpuElwLt", "SfpuElwGt", "SfpuElwLe", "SfpuElwGe")
    },
    **{
        op: [I32]
        for op in (
            "SfpuElwLeftShift",
            "SfpuElwRightShift",
            "SfpuElwLogicalRightShift",
            "SfpuBitwiseAnd",
            "SfpuBitwiseOr",
            "SfpuBitwiseXor",
            "SfpuEqInt",
            "SfpuNeInt",
            "SfpuRsubInt32",
            "SfpuDivInt32",
            "SfpuDivInt32Floor",
            "SfpuGcd",
            "SfpuLcm",
            "SfpuMulInt32",
            "SfpuMaxInt32",
            "SfpuMinInt32",
            "SfpuRemainderInt32",
            "SfpuFmodInt32",
        )
    },
    **{op: [U32] for op in ("SfpuMaxUint32", "SfpuMinUint32", "SfpuRemainderUint32")},
}

#: Results that are not a rounded real number: compared lane by lane, right or wrong.
EXACT_BINARY_OPS = {
    "SfpuElwLt",
    "SfpuElwGt",
    "SfpuElwLe",
    "SfpuElwGe",
    "SfpuElwEq",
    "SfpuElwNe",
    "SfpuIsclose",
}

#: What the report says about an op's coverage when it is narrower than the format.
BINARY_COVERAGE_NOTES = {
    "SfpuLogsigmoid": "x in [-8, 3.9] only: the harness cannot supply the device-computed exp(-x) the x > 4 branch reads",
}


def _binary_cells():
    cells = []
    for op in BINARY_OPS:
        for fmt in BINARY_FORMATS.get(op, []):
            for dest in (
                (DestAccumulation.No, DestAccumulation.Yes)
                if not fmt.is_integer()
                else (DestAccumulation.Yes,)
            ):
                if fmt.is_32_bit() and dest == DestAccumulation.No:
                    continue
                cells.append((op, fmt, dest))
    return cells or [("none", F16B, DestAccumulation.No)]


def binary_special_pairs(fmt):
    """One tile pair: the cross product of the values a random draw never lands on."""
    if fmt.is_integer():
        lo = -(2**31) if fmt == I32 else 0
        hi = 2**31 - 1 if fmt == I32 else 2**32 - 1
        values = sorted(
            {lo, lo + 1, -2, -1, 0, 1, 2, 31, 32, 2**16, hi - 1, hi}
            & set(range(lo, hi + 1))
        )
        dtype = torch.int64
    else:
        info = torch.finfo(format_dict[fmt])
        values = [
            float("nan"),
            float("inf"),
            -float("inf"),
            0.0,
            -0.0,
            info.tiny / 2,
            -info.tiny / 2,
            info.tiny,
            info.max,
            -info.max,
            1.0,
            -1.0,
            2.0,
            0.5,
        ]
        dtype = torch.float64
    pairs = [(a, b) for a in values for b in values]
    n = TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]
    pairs = (pairs * (-(-n // len(pairs))))[:n]
    a = torch.tensor([p[0] for p in pairs], dtype=dtype)
    b = torch.tensor([p[1] for p in pairs], dtype=dtype)
    return a, b


def _binary_run(op_name, fmt, dest_acc, kind, monkeypatch):
    import test_eltwise_binary_sfpu as fb
    from helpers.stimuli_generator import DistributionKind, StimuliSpec

    mathop = MathOperation[op_name]
    formats = InputOutputFormat(fmt, fmt)
    captured = {}

    def capture_assert(_mathop, _formats, _dest_acc, golden, result, **_kwargs):
        captured["golden"], captured["result"] = golden, result

    real_generate = fb.generate_stimuli

    def capture_generate(*args, **kwargs):
        out = real_generate(*args, **kwargs)
        captured["src"] = out[0]
        return out

    monkeypatch.setattr(fb, "_assert_against_contract", capture_assert)
    monkeypatch.setattr(fb, "generate_stimuli", capture_generate)

    kwargs = {}
    if kind == "specials":
        a, b = binary_special_pairs(fmt)
        override = torch.cat([a, b])
        kwargs["src_A_override"] = override
    elif op_name == "SfpuAtan2":
        kwargs["spec_A"] = StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=-5.0, high=5.0
        )
    elif op_name == "SfpuLogsigmoid":
        kwargs["spec_A"] = fb._logsigmoid_stimuli_spec()
    elif mathop in fb._INT_BINARY_STIMULI:
        low, high = fb._INT_BINARY_STIMULI[mathop]
        kwargs["spec_A"] = StimuliSpec(
            distribution=DistributionKind.UNIFORM, low=low, high=high
        )
    if op_name == "SfpuRsubInt32":
        kwargs["twos_complement"] = True

    fb.sfpu_binary(formats, dest_acc, mathop, **kwargs)
    src = captured["src"].flatten()
    if kind == "specials":
        src = kwargs["src_A_override"].to(src.dtype).flatten()
        src = src.repeat(captured["result"].numel() // src.numel())
    n = TILE_DIMENSIONS[0] * TILE_DIMENSIONS[1]
    # Tile 2k holds in0 and the result, tile 2k+1 holds in1.
    pairs = lambda t: t.flatten()[: (t.numel() // (2 * n)) * 2 * n].reshape(
        -1, 2, n
    )  # noqa: E731
    s, g, r = (
        pairs(src),
        pairs(torch.as_tensor(captured["golden"])),
        pairs(captured["result"]),
    )
    return s[:, 0].flatten(), s[:, 1].flatten(), g[:, 0].flatten(), r[:, 0].flatten()


@pytest.mark.skipif(
    not DUMP_DIR or not BINARY_OPS, reason="run by the LLK SFPU report only"
)
@pytest.mark.parametrize("kind", ["random", "specials"])
@pytest.mark.parametrize(
    "op_name, fmt, dest_acc",
    [
        pytest.param(*c, id=f"{c[0]}-{c[1].name}-dest_acc:{c[2].name}")
        for c in _binary_cells()
    ],
)
def test_sfpu_report_accuracy_binary(op_name, fmt, dest_acc, kind, monkeypatch):
    if op_name == "none":
        pytest.skip("no binary op requested")
    from helpers.chip_architecture import ChipArchitecture

    if TestConfig.CHIP_ARCH == ChipArchitecture.BLACKHOLE:
        if op_name == "SfpuLcm":
            pytest.skip("SfpuLcm dest_acc=Yes hangs on Blackhole; see tt-metal#52997")
        if fmt == F16 and dest_acc == DestAccumulation.No:
            pytest.skip("Blackhole runs Float16 SFPU input through a 32-bit Dest only")
    a, b, golden, result = _binary_run(op_name, fmt, dest_acc, kind, monkeypatch)
    name = f"{op_name}__{fmt.name}-{fmt.name}__No__{dest_acc.name}__{kind}.pt"
    out = Path(DUMP_DIR)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "op": op_name,
            "in": fmt.name,
            "out": fmt.name,
            "approx": "No",
            "dest_acc": dest_acc.name,
            "kind": kind,
            "binary": True,
            "exact": fmt.is_integer() or op_name in EXACT_BINARY_OPS,
            "coverage": BINARY_COVERAGE_NOTES.get(op_name),
            "src": a,
            "src_b": b,
            "golden": golden,
            "result": result,
            "classes": None,
        },
        out / name,
    )
