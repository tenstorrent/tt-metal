# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_reference(expression, operands):
    # Same grammar as the op: single-letter operands, + - * with the usual precedence, parentheses.
    return eval(expression, {"__builtins__": {}}, operands)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "expression",
    ["a+b", "a-b", "a*b", "a*b+c", "a+b*c", "(a-b)*c", "a*b+c-d", "(a+b)*(c-d)"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),  # one tile total: a single padded leftover window on one core
        (4, 512, 512),  # ~8 tiles per core: full windows
        (3, 160, 96),  # 45 tiles: 1 per core, leftover windows everywhere
    ],
)
def test_graph_kernel_binary_expression(device, expression, dtype, shape):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    letters = sorted({c for c in expression if c.isalpha()})
    torch_inputs = {letter: torch.randn(shape, dtype=torch_dtype) for letter in letters}
    inputs = [
        ttnn.from_torch(torch_inputs[letter], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) for letter in letters
    ]

    output = ttnn.graph_kernel(inputs, expression)

    expected = torch_reference(expression, torch_inputs)
    actual = ttnn.to_torch(output)
    assert_with_pcc(expected, actual, 0.9999)
    # Neither dtype is bit-exact against torch: bf16 rounds every intermediate and the odd element
    # lands one or two ulps off after three nodes, and float32 operands reach the FPU through
    # SrcA/SrcB, which truncates their mantissas. The absolute bound scales with the output range so
    # a one-ulp miss on a large product passes, and the PCC check above guards the bulk.
    atol = 2e-2 * max(1.0, expected.abs().max().item())
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=atol)


def run_and_check(device, expression, dtype, shape, pcc=0.9999, rtol=3e-2):
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    letters = sorted({c for c in expression if c.isalpha()})
    torch_inputs = {letter: torch.randn(shape, dtype=torch_dtype) for letter in letters}
    inputs = [
        ttnn.from_torch(torch_inputs[letter], dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) for letter in letters
    ]

    output = ttnn.graph_kernel(inputs, expression)

    expected = torch_reference(expression, torch_inputs)
    actual = ttnn.to_torch(output)
    assert_with_pcc(expected, actual, pcc)
    atol = rtol * max(1.0, expected.abs().max().item())
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return len(inputs)


# Stress: a long per-core stream. (8, 2048, 2048) tiled is 32768 tiles, about 252 per core on a
# 130-core grid. With 5 buffers the factory picks a window of ~130 pages, so every core runs one full
# window plus a padded leftover window through the reader, compute and writer.
@pytest.mark.timeout(600)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_graph_kernel_long_stream(device, dtype):
    run_and_check(device, "a*b+c", dtype, (8, 2048, 2048))


# Stress: L1. The factory sizes the window to fill the allocatable L1 (~1.43 MB on Blackhole) with
# one buffer per input, intermediate and output. 26 inputs summed pairwise is 26 + 24 + 1 = 51
# buffers, so the window shrinks to the smallest legal 8 pages for bf16 (816 KB); the fp32 variant
# with 16 inputs (31 buffers, 4 KB tiles) lands at 8 pages too (992 KB). Both stream several windows
# per input.
@pytest.mark.timeout(600)
@pytest.mark.parametrize("dtype, num_inputs", [(ttnn.bfloat16, 26), (ttnn.float32, 16)])
def test_graph_kernel_l1_stress(device, dtype, num_inputs):
    letters = [chr(ord("a") + i) for i in range(num_inputs)]
    # Alternate + and - so the running sum stays O(sqrt(n)) instead of drifting.
    expression = letters[0] + "".join(("+" if i % 2 else "-") + letter for i, letter in enumerate(letters[1:]))

    # (1, 1024, 4096) tiled is 4096 tiles: ~32 per core, four windows per input.
    run_and_check(device, expression, dtype, (1, 1024, 4096), pcc=0.999, rtol=5e-2)


def test_graph_kernel_rejects_l1_overflow(device, expect_error):
    # 26 float32 inputs: 51 buffers x 8 pages x 4 KB = 1.6 MB, more than the whole L1.
    letters = [chr(ord("a") + i) for i in range(26)]
    expression = "+".join(letters)
    inputs = [
        ttnn.from_torch(torch.randn(32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        for _ in letters
    ]
    with expect_error(RuntimeError, "does not fit"):
        ttnn.graph_kernel(inputs, expression)


def test_graph_kernel_rejects_unused_input(device, expect_error):
    x = ttnn.from_torch(torch.randn(32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "must be used exactly once"):
        ttnn.graph_kernel([x, x, x], "a+b")
