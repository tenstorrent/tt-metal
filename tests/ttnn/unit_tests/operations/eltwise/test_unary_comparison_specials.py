# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The unary comparison activations against infinities and NaNs.

A float comparison on the SFPU is a subtract and a test on the result, which is
not an IEEE comparison: two equal infinities subtract to NaN so they never
compare equal, and a NaN operand subtracts to a NaN that reads as positive so
`NaN > x` answers true. The binary comparisons do not have either hole, so each
case here is checked against the binary op as well as against torch.
"""


import pytest
import torch
import ttnn

SHAPE = (1, 1, 32, 32)
INF = float("inf")
NAN = float("nan")

# activation, binary op, and the python answer
OPS = [
    ("UNARY_EQ", ttnn.UnaryOpType.UNARY_EQ, ttnn.eq, lambda a, b: a == b),
    ("UNARY_NE", ttnn.UnaryOpType.UNARY_NE, ttnn.ne, lambda a, b: a != b),
    ("UNARY_GT", ttnn.UnaryOpType.UNARY_GT, ttnn.gt, lambda a, b: a > b),
    ("UNARY_LT", ttnn.UnaryOpType.UNARY_LT, ttnn.lt, lambda a, b: a < b),
    ("UNARY_GE", ttnn.UnaryOpType.UNARY_GE, ttnn.ge, lambda a, b: a >= b),
    ("UNARY_LE", ttnn.UnaryOpType.UNARY_LE, ttnn.le, lambda a, b: a <= b),
]

OPERANDS = [INF, -INF, 1.0, 0.0, -1.0, 2.0, NAN]


def _to_device(values, dtype, torch_dtype, device):
    t = torch.zeros(SHAPE, dtype=torch_dtype)
    for i, v in enumerate(values):
        t[0, 0, 0, i] = v
    return t, ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("parameter", [INF, -INF, NAN, 1.0], ids=["inf", "neg_inf", "nan", "one"])
@pytest.mark.parametrize("name, op, binary, expected", OPS, ids=[o[0] for o in OPS])
def test_unary_comparison_specials(device, dtype, parameter, name, op, binary, expected):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    n = len(OPERANDS)

    host, x = _to_device(OPERANDS, dtype, torch_dtype, device)
    _, zeros = _to_device([0.0] * n, dtype, torch_dtype, device)

    got = ttnn.to_torch(ttnn.add(x, zeros, input_tensor_a_activations=[ttnn.UnaryWithParam(op, parameter)])).flatten()[
        :n
    ]
    from_binary = ttnn.to_torch(binary(x, parameter)).flatten()[:n]

    # the reference is taken from what the device actually holds, so that a
    # bfloat16 operand that rounded on the way in is not counted as a difference
    held = host.flatten()[:n].to(torch.float64)
    want = expected(held, torch.tensor(parameter, dtype=torch.float64)).to(torch.float64)

    for i, operand in enumerate(OPERANDS):
        label = f"{name}({operand}, {parameter}) in {dtype}"
        assert got[i].item() == want[i].item(), f"{label}: got {got[i].item()}, expected {want[i].item()}"
        assert (
            got[i].item() == from_binary[i].item()
        ), f"{label}: activation says {got[i].item()}, the binary op says {from_binary[i].item()}"


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_unary_comparison_exhaustive_bfloat16_patterns(device, dtype):
    """Every bfloat16 bit pattern against an infinite parameter, both directions.

    The infinities are what the subtract cannot express, so this is where a
    regression would land; 254 of the patterns are NaN and they are the other.
    """
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    patterns = (
        torch.arange(0, 65536, dtype=torch.int64)
        .to(torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(1, 1, 256, 256)
    )
    x = ttnn.from_torch(patterns.to(torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    zeros = ttnn.from_torch(
        torch.zeros((1, 1, 256, 256), dtype=torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    for name, op, binary, expected in OPS:
        for parameter in (INF, -INF):
            got = ttnn.to_torch(
                ttnn.add(x, zeros, input_tensor_a_activations=[ttnn.UnaryWithParam(op, parameter)])
            ).flatten()
            want = expected(
                patterns.to(torch_dtype).flatten().to(torch.float64), torch.tensor(parameter, dtype=torch.float64)
            ).to(torch.float64)
            wrong = got.to(torch.float64) != want
            assert not wrong.any(), (
                f"{name} against {parameter} in {dtype}: {int(wrong.sum())} of 65536 patterns differ, "
                f"first at 0x{int(patterns.flatten().view(torch.int16)[wrong.nonzero()[0, 0]].item()) & 0xffff:04x}"
            )
