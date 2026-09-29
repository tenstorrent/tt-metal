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
