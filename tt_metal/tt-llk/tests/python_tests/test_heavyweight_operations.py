# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the heavyweight golden's operation chains.

No kernel, no device. The chains decide which src format each unpack and Dest-feedback
step lands in when the caller does not name one, and that choice has to match what the
harness programs or the golden computes on values the kernel never saw.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generator.heavyweight.operations.quasar_operations.quasar_eltwise import (
    QuasarEltwiseBinaryGolden,
)
from helpers.llk_params import MathFidelity, MathOperation

TILE = 1024


def _elwmul(in_format, out_format, a, b, **kwargs):
    golden = QuasarEltwiseBinaryGolden(MathOperation.Elwmul, MathFidelity.HiFi4)
    return golden.run(
        [torch.full((TILE,), a), torch.full((TILE,), b)],
        in_format,
        out_format,
        **kwargs,
    ).float()


def test_float32_input_under_a_float16_dest_unpacks_into_float16():
    """The harness lands Float32 in a Float16 src for a Float16 output, and that src
    flushes 2^-16 before the multiply. A Tf32 src would keep it and give 2^-14."""
    out = _elwmul(
        DataFormat.Float32,
        DataFormat.Float16,
        2.0**-16,
        4.0,
        dest_format=DataFormat.Float16,
    )
    assert out.tolist() == [0.0] * TILE


@pytest.mark.parametrize(
    "out_format, kwargs",
    [
        (DataFormat.Float16_b, dict(dest_format=DataFormat.Float16_b)),
        (DataFormat.Float32, dict(dest_acc=True)),
    ],
    ids=["float16_b_dest", "dest_acc"],
)
def test_float32_input_keeps_the_8_bit_exponent_range_otherwise(out_format, kwargs):
    out = _elwmul(DataFormat.Float32, out_format, 2.0**-16, 4.0, **kwargs)
    assert out.tolist() == [2.0**-14] * TILE
