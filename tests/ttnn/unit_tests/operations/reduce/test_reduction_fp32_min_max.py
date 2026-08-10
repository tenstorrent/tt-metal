# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Accuracy of FLOAT32 ttnn.min / ttnn.max (issues #32274, #51889).

The FPU reduce path truncates fp32 operands to tf32 before comparing, so the surviving value
has its low 13 mantissa bits zeroed and is no longer an element of the input. FLOAT32 min/max
therefore default to the SFPU compare path (fast_and_approximate_mode=False).

These tests deliberately build tensors from torch.float32. The pre-existing min/max tests
parametrize dtype over float32 but source their data from torch.bfloat16 — and truncating a
bfloat16 value to tf32 is lossless (bf16 has 7 explicit mantissa bits, tf32 has 10), so those
cases cannot observe this defect no matter how many shapes or seeds they cover.
"""

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn

SHAPES = [
    (32, 32),  # single tile, exercises the HW reduce
    (1, 1, 64, 64),  # the shape reported in #51889
    (2, 3, 64, 96),  # rank 4, multi-tile, multi-core
    (4, 32, 32),  # rank 3, all-dim reduce goes through the transpose loop
    (1, 1, 37, 53),  # not tile-aligned, exercises the ±inf implicit padding
]

OPS = [("max", ttnn.max, torch.max, torch.amax), ("min", ttnn.min, torch.min, torch.amin)]


def _to_device(torch_tensor, device, dtype=ttnn.float32):
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name, ttnn_op, torch_global, torch_dim", OPS)
def test_fp32_global_result_is_an_input_element(device, shape, name, ttnn_op, torch_global, torch_dim):
    """#51889: min/max select, so the result must be a value that exists in the input."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)

    output = ttnn.to_torch(ttnn_op(_to_device(torch_input, device))).flatten()[0]

    assert (
        torch_input.flatten() == output
    ).any(), f"ttnn.{name} returned {output.item()!r}, which is not an element of the input"


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("name, ttnn_op, torch_global, torch_dim", OPS)
def test_fp32_global_matches_torch_exactly(device, shape, name, ttnn_op, torch_global, torch_dim):
    """#32274: a selection over exact fp32 compares must be bit-identical to torch."""
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)

    output = ttnn.to_torch(ttnn_op(_to_device(torch_input, device))).flatten()[0]

    assert output == torch_global(
        torch_input
    ), f"ttnn.{name}: {output.item()!r} != {torch_global(torch_input).item()!r}"


@pytest.mark.parametrize("dim", [-1, -2, 0, 1])
@pytest.mark.parametrize("name, ttnn_op, torch_global, torch_dim", OPS)
def test_fp32_per_axis_matches_torch_exactly(device, dim, name, ttnn_op, torch_global, torch_dim):
    """The defect is not limited to the global reduce; every axis must be exact too."""
    torch.manual_seed(0)
    torch_input = torch.randn((2, 3, 64, 96), dtype=torch.float32)

    output = ttnn.to_torch(ttnn_op(_to_device(torch_input, device), dim, True)).flatten()
    expected = torch_dim(torch_input, dim=dim, keepdim=True).flatten()

    assert (output == expected).all(), f"ttnn.{name} dim={dim}: {int((output != expected).sum())} mismatched elements"


@pytest.mark.parametrize("name, ttnn_op, torch_global, torch_dim", OPS)
def test_fp32_fast_and_approximate_mode_selects_the_fpu(device, name, ttnn_op, torch_global, torch_dim):
    """The opt-out must actually reach the FPU, where the tf32 truncation is observable.

    The distinguishing element differs from the rest only at mantissa bit 20, which tf32 (10
    explicit mantissa bits) cannot represent, so the two paths are guaranteed to disagree.
    """
    torch_input = torch.full((32, 32), 1.0, dtype=torch.float32)
    torch_input[5, 7] = 1.0 + 2**-20 if name == "max" else 1.0 - 2**-20
    tt_input = _to_device(torch_input, device)

    accurate = ttnn.to_torch(ttnn_op(tt_input)).flatten()[0]
    approximate = ttnn.to_torch(ttnn_op(tt_input, fast_and_approximate_mode=True)).flatten()[0]

    assert accurate == torch_global(torch_input)
    assert approximate != accurate, "fast_and_approximate_mode=True did not change the reduce path"


@pytest.mark.parametrize("scalar", [2.5, -2.5])
@pytest.mark.parametrize("name, ttnn_op, torch_global, torch_dim", OPS)
def test_fp32_scalar_post_multiply(device, scalar, name, ttnn_op, torch_global, torch_dim):
    """The SFPU ignores the scaler CB, so the scalar is applied as a post-multiply.

    A negative scalar also flips max<->min, since the scaling happens after the reduction.
    """
    torch.manual_seed(0)
    torch_input = torch.randn((1, 1, 64, 64), dtype=torch.float32)

    output = ttnn.to_torch(ttnn_op(_to_device(torch_input, device), scalar=scalar)).flatten()[0]

    assert torch.allclose(output, torch_global(scalar * torch_input), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name, ttnn_op, torch_global, torch_dim", OPS)
def test_non_fp32_dtypes_are_unaffected(device, name, ttnn_op, torch_global, torch_dim):
    """bfloat16 and int32 keep their existing paths (FPU and Int32 SFPU respectively)."""
    torch.manual_seed(0)

    bf16_input = torch.randn((1, 1, 64, 64), dtype=torch.bfloat16)
    output = ttnn.to_torch(ttnn_op(_to_device(bf16_input, device, ttnn.bfloat16))).flatten()[0]
    assert output == torch_global(bf16_input)

    int32_input = torch.randint(-10000, 10000, (1, 1, 64, 64), dtype=torch.int32)
    output = ttnn.to_torch(ttnn_op(_to_device(int32_input, device, ttnn.int32))).flatten()[0]
    assert output == torch_global(int32_input)
