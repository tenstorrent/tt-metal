# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathOperation
from helpers.param_config import parametrize
from test_eltwise_binary_sfpu import sfpu_binary

ELEMENTS_PER_TILE = 1024
INF = float("inf")


def _edge_pairs(with_nan):
    pairs = []
    for divisor in (3.0, 5.0, 7.0, 11.0, 13.0):
        for k in range(1, 20):
            for sa, sb in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
                pairs.append((sa * k * divisor, sb * divisor))
    pairs += [(5.625, -1.875), (-5.625, 1.875), (5.625, 1.875), (-5.625, -1.875)]
    pairs += [
        (x, d)
        for x in (0.0, -0.0, 0.5, -0.5, 5.0, -5.0, 100.0, -100.0)
        for d in (INF, -INF)
    ]
    if with_nan:
        nan = float("nan")
        pairs += [
            (INF, INF),
            (-INF, -INF),
            (INF, -INF),
            (-INF, INF),
            (5.0, nan),
            (-5.0, nan),
            (nan, 5.0),
            (nan, -INF),
            (5.0, -nan),
            (-5.0, -nan),
            (0.5, -nan),
            (nan, nan),
            (-0.0, -0.0),
        ]
    return pairs


def _paired_tiles(pairs):
    a = [pairs[i % len(pairs)][0] for i in range(ELEMENTS_PER_TILE)]
    b = [pairs[i % len(pairs)][1] for i in range(ELEMENTS_PER_TILE)]
    return torch.tensor(a + b, dtype=torch.float32)


@parametrize(
    formats=[
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=[MathOperation.SfpuBinaryFmod],
)
def test_sfpu_binary_remainder_fmod_edges(formats, dest_acc, mathop):
    if formats.input_format == DataFormat.Float32 and dest_acc == DestAccumulation.No:
        pytest.skip("Float32 input needs DestAccumulation.Yes")
    with_nan = formats.input_format == DataFormat.Float32
    src = _paired_tiles(_edge_pairs(with_nan))
    res, golden = sfpu_binary(
        formats,
        dest_acc,
        mathop,
        src_A_override=src,
        input_dimensions=[64, 32],
        check_contract=False,
    )
    torch.testing.assert_close(
        res[:ELEMENTS_PER_TILE].float(),
        golden[:ELEMENTS_PER_TILE].float(),
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    if with_nan:
        a, b = src[:ELEMENTS_PER_TILE], src[ELEMENTS_PER_TILE:]
        inf_divisor = torch.isinf(b) & torch.isfinite(a)
        assert torch.equal(
            torch.signbit(res[:ELEMENTS_PER_TILE].float()[inf_divisor]),
            torch.signbit(golden[:ELEMENTS_PER_TILE].float()[inf_divisor]),
        )
