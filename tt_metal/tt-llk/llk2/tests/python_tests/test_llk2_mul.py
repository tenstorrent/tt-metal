# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 mul through the Compute API, plain and with each broadcast, in the kernel's fidelity, against the goldens
of test_eltwise_binary.py (run_eltwise_binary: test_llk2_add.py).

mul runs ELWMUL once per fidelity phase, the phases summing in Dest, after zeroing the output tiles (with .accumulate
it does not zero them, and adds its product to what Dest holds: test_llk2_mul_accumulate). Col is SDPA's
rescale by the correction factor and final normalize by 1/sum (mul_tiles_bcast_cols); [1, 16] is one SDPA output
row block with one Col tile, in a one-section ring (its tiles 8..15 are in Dest's upper half).
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

ALL_BROADCASTS = [
    BroadcastType.None_,
    BroadcastType.Row,
    BroadcastType.Column,
    BroadcastType.Scalar,
]

# (dest_acc, (rt, ct), sections)
MUL_CASES = [
    (DestAccumulation.No, (1, 1), 2),
    (DestAccumulation.No, (1, 4), 2),
    (DestAccumulation.No, (2, 4), 2),
    (DestAccumulation.No, (1, 16), 1),
    (DestAccumulation.Yes, (2, 2), 2),
]


@parametrize(
    kernel_fidelity=[MathFidelity.HiFi2],
    broadcast=ALL_BROADCASTS,
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    case=MUL_CASES,
)
def test_llk2_mul(kernel_fidelity, broadcast, formats, case):
    dest_acc, (rt, ct), sections = case
    run_eltwise_binary(
        MathOperation.Elwmul,
        broadcast,
        formats,
        dest_acc,
        kernel_fidelity,
        rt,
        ct,
        sections,
    )


@parametrize(
    kernel_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi3, MathFidelity.HiFi4],
    broadcast=[BroadcastType.None_, BroadcastType.Column],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    grid=[(2, 2)],
)
def test_llk2_mul_fidelity(kernel_fidelity, broadcast, formats, grid):
    rt, ct = grid
    run_eltwise_binary(
        MathOperation.Elwmul,
        broadcast,
        formats,
        DestAccumulation.No,
        kernel_fidelity,
        rt,
        ct,
    )


# .accumulate: y = (a + b) + a * b, the product added into the add's result in Dest. Col is SDPA's U rescale added
# into P V; [2, 4] is the SDPA block of two output rows (a whole 8-tile section).
@parametrize(
    kernel_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2],
    broadcast=[BroadcastType.None_, BroadcastType.Column],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    case=[
        (DestAccumulation.No, (1, 4), 2),
        (DestAccumulation.No, (2, 4), 2),
        (DestAccumulation.Yes, (2, 2), 2),
    ],
)
def test_llk2_mul_accumulate(kernel_fidelity, broadcast, formats, case):
    dest_acc, (rt, ct), sections = case
    run_eltwise_binary(
        MathOperation.Elwmul,
        broadcast,
        formats,
        dest_acc,
        kernel_fidelity,
        rt,
        ct,
        sections,
        accumulate=True,
    )
