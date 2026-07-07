# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY smoke tests for the repeat native/codegen routing added on ebanerjee/finalizing_poc.

Not meant to survive the eventual tt-metal merge (the manifest says the translation stage
wouldn't normally even compile, let alone need its own test suite) — this is a golden-standard
sanity pass to run once on real hardware before writing up the porting-guide/review-checklist
docs. Exercises the `implementation` kwarg ("auto" / "native" / "codegen") end to end:

  * codegen-eligible cases produce bit-exact-identical output between "native" and "codegen"
    (the call_parity requirement in agentic_port/manifests/repeat.yaml), and "auto" picks codegen.
  * codegen-ineligible cases (RM layout, W-dim repeat, non-tile-aligned H) still succeed under
    "auto" and "native", but forcing "codegen" TT_FATALs immediately instead of silently
    falling back.
  * an unrecognized `implementation` string TT_FATALs.
"""

from functools import reduce

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal

# (shape, repeat_dims, repeat_dim) — all unsharded/4D/TILE/bfloat16/H+W tile-aligned, so every
# one is expected to be codegen-eligible per repeat_codegen_supported.hpp / repeat.yaml.
ELIGIBLE_CASES = [
    ((1, 1, 32, 32), (1, 1, 2, 1), 2),  # H repeat — manifest in-scope case
    ((2, 3, 64, 128), (1, 1, 4, 1), 2),  # H repeat, larger shape — manifest in-scope case
    ((2, 1, 32, 32), (1, 3, 1, 1), 1),  # channel-dim repeat
    ((1, 2, 32, 32), (3, 1, 1, 1), 0),  # batch-dim repeat
]

# (shape, repeat_dims, layout, reason) — each fails supported_by_codegen for a distinct reason.
INELIGIBLE_CASES = [
    (
        (1, 1, 32, 32),
        (1, 1, 1, 3),
        ttnn.TILE_LAYOUT,
        "real-kernel-limit: W (dim 3) repeat is out of codegen scope",
    ),
    (
        (1, 1, 16, 32),
        (1, 1, 2, 1),
        ttnn.TILE_LAYOUT,
        "real-kernel-limit: H (16) is not TILE_HEIGHT-aligned",
    ),
    (
        (1, 1, 32, 32),
        (1, 1, 2, 1),
        ttnn.ROW_MAJOR_LAYOUT,
        "scope: codegen is TILE-only",
    ),
]


def _torch_reference(shape, repeat_dims):
    numel = reduce(lambda a, b: a * b, shape, 1)
    torch_input = torch.arange(0, numel, dtype=torch.bfloat16).reshape(shape)
    return torch_input, torch_input.repeat(repeat_dims)


@pytest.mark.parametrize("shape,repeat_dims,repeat_dim", ELIGIBLE_CASES)
@pytest.mark.parametrize("implementation", ["auto", "native", "codegen"])
def test_repeat_eligible_matches_torch(device, shape, repeat_dims, repeat_dim, implementation):
    torch_input, torch_result = _torch_reference(shape, repeat_dims)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_dims), implementation=implementation)
    output = ttnn.to_torch(output)

    assert output.shape == torch_result.shape
    assert_equal(torch_result, output)


@pytest.mark.parametrize("shape,repeat_dims,repeat_dim", ELIGIBLE_CASES)
def test_repeat_eligible_native_codegen_call_parity(device, shape, repeat_dims, repeat_dim):
    """Same (shape, kwargs) must drive native and codegen to a bit-exact-identical result."""
    torch_input, _ = _torch_reference(shape, repeat_dims)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    native_out = ttnn.to_torch(ttnn.repeat(input_tensor, ttnn.Shape(repeat_dims), implementation="native"))
    codegen_out = ttnn.to_torch(ttnn.repeat(input_tensor, ttnn.Shape(repeat_dims), implementation="codegen"))

    assert_equal(native_out, codegen_out)


@pytest.mark.parametrize("shape,repeat_dims,layout,reason", INELIGIBLE_CASES)
@pytest.mark.parametrize("implementation", ["auto", "native"])
def test_repeat_ineligible_still_succeeds_without_forcing_codegen(
    device, shape, repeat_dims, layout, reason, implementation
):
    torch_input, torch_result = _torch_reference(shape, repeat_dims)
    input_tensor = ttnn.from_torch(torch_input, layout=layout, device=device, dtype=ttnn.bfloat16)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_dims), implementation=implementation)
    output = ttnn.to_torch(output)

    assert output.shape == torch_result.shape
    assert_equal(torch_result, output)


@pytest.mark.parametrize("shape,repeat_dims,layout,reason", INELIGIBLE_CASES)
def test_repeat_forced_codegen_on_ineligible_input_raises(device, shape, repeat_dims, layout, reason):
    torch_input, _ = _torch_reference(shape, repeat_dims)
    input_tensor = ttnn.from_torch(torch_input, layout=layout, device=device, dtype=ttnn.bfloat16)

    with pytest.raises(RuntimeError, match="codegen-eligible"):
        ttnn.repeat(input_tensor, ttnn.Shape(repeat_dims), implementation="codegen")


def test_repeat_invalid_implementation_string_raises(device):
    torch_input, _ = _torch_reference((1, 1, 32, 32), (1, 1, 2, 1))
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    with pytest.raises(RuntimeError, match="invalid implementation"):
        ttnn.repeat(input_tensor, ttnn.Shape((1, 1, 2, 1)), implementation="not-a-real-mode")
