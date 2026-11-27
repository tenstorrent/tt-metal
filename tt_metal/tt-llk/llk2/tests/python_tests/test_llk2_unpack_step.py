# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 unpack with counter-stepped streams (llk_lib/unpack.h): a streamed operand moves to its next tile with the
unpacker's channel 0 address counters (UNPACR post-increments) instead of a rewritten base address. The kernels are
the add / sub / mul and matmul Compute API tests; a wrong step reads the wrong tile, so the goldens catch it.

What each case walks:
  [1, 4], [1, 16]  one run of several tiles: both streams step every tile (None), or in1 stays (Col, Scalar)
  [2, 4]           two runs: the counters go back to 0 behind a run, in1 moves per run (Col) or not at all (Row)
  [3, 1], [4, 1]   a column of per-row statistics (SDPA's row max, alpha, 1 / sum): one tile per run
  matmul           a held operand and a streamed one, K steps: Q K^T's shape, and each reuse direction (with in1 held
                   in0 streams down its column, a step of K tiles, which keeps the base-address rewrite)
"""

from helpers.format_config import DataFormat
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
)
from helpers.param_config import input_output_formats, parametrize
from test_llk2_add import run_eltwise_binary
from test_llk2_matmul import TRANSPOSE_IN1_SHAPES, run_matmul_transpose_in1

# ((rt, ct), Dest ring sections)
STEP_GRIDS = [((1, 4), 2), ((2, 4), 2), ((3, 1), 2), ((4, 1), 2), ((1, 16), 1)]


@parametrize(
    math_op=[MathOperation.Elwadd, MathOperation.Elwsub, MathOperation.Elwmul],
    broadcast=[
        BroadcastType.None_,
        BroadcastType.Row,
        BroadcastType.Column,
        BroadcastType.Scalar,
    ],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    grid=STEP_GRIDS,
)
def test_llk2_unpack_step_eltwise(math_op, broadcast, formats, grid):
    (rt, ct), sections = grid
    run_eltwise_binary(
        math_op,
        broadcast,
        formats,
        DestAccumulation.No,
        MathFidelity.HiFi2,
        rt,
        ct,
        sections,
    )


def _matmul_cases():
    fp16b = input_output_formats([DataFormat.Float16_b], same=True)[0]
    return [
        (fp16b, DestAccumulation.No, MathFidelity.HiFi2, shape)
        for shape in TRANSPOSE_IN1_SHAPES
    ]


@parametrize(case=_matmul_cases())
def test_llk2_unpack_step_matmul(case):
    run_matmul_transpose_in1(case)
