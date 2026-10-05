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
from helpers.golden_generator.heavyweight.operations.quasar_operations.quasar_reuse_dest import (
    QuasarEltwiseBinaryReuseDestGolden,
)
from helpers.llk_params import EltwiseBinaryReuseDestType, MathFidelity, MathOperation

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


# ---------------------------------------------------------------------------
# reuse_dest block geometry


def _reuse_dest(tiles_a, tiles_b=None, **kwargs):
    golden = QuasarEltwiseBinaryReuseDestGolden(
        MathOperation.Elwadd, MathFidelity.LoFi, EltwiseBinaryReuseDestType.DEST_TO_SRCA
    )
    a = torch.cat([torch.full((TILE,), float(v)) for v in tiles_a])
    b = torch.cat([torch.full((TILE,), float(v)) for v in (tiles_b or tiles_a)])
    return golden.run([a, b], DataFormat.Float16_b, DataFormat.Float16_b, **kwargs)


@pytest.mark.parametrize(
    "tiles, kwargs",
    [
        ([1] * 3, dict(inner_dim=2)),  # tile 2 would go unused
        ([1] * 6, dict(inner_dim=2, output_tiles_in_block=2)),  # would read tile 6
        ([1] * 2, dict(inner_dim=0)),
        ([1] * 2, dict(inner_dim=2, output_tiles_in_block=-1)),
    ],
    ids=["leftover_tile", "past_the_end", "zero_inner_dim", "negative_block"],
)
def test_an_incomplete_reuse_dest_block_is_refused(tiles, kwargs):
    with pytest.raises(ValueError):
        _reuse_dest(tiles, **kwargs)


def test_reuse_dest_operands_must_match_in_size():
    with pytest.raises(ValueError, match="differ in size"):
        _reuse_dest([1] * 4, [1] * 2, inner_dim=2)


def test_a_complete_reuse_dest_block_folds_every_tile():
    """inner_dim=3, output_tiles_in_block=2: output j folds inputs j, j+2, j+4.

    DEST_TO_SRCA with Elwadd: Dest = seed(A[j]) + B[j], then Dest += B[j+2], then
    Dest += B[j+4]. Values are small integers, so every step is exact in bf16."""
    a = [1, 2, 10, 20, 100, 200]
    out = (
        _reuse_dest(a, a, inner_dim=3, output_tiles_in_block=2).float().reshape(2, TILE)
    )
    assert out[0].unique().tolist() == [1 + 1 + 10 + 100]
    assert out[1].unique().tolist() == [2 + 2 + 20 + 200]
