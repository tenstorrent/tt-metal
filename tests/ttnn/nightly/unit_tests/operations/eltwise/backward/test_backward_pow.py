# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "exponent",
    [
        -0.01,
        -1.0,
    ],
)
def test_negative_exponent(input_shapes, exponent, device, expect_error):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device, seed=1)

    with expect_error(RuntimeError, r"negative exponents are not supported"):
        ttnn.pow_bw(grad_tensor, input_tensor, exponent)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize(
    "exponent",
    [
        0,
    ],
)
def test_fw_exponent(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -90, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -20, 20, device, seed=1)

    golden_tensor = [
        torch.pow(grad_data, exponent),
    ]
    tt_output_tensor_on_device = ttnn.pow(grad_tensor, exponent)
    status = compare_pcc([tt_output_tensor_on_device], golden_tensor)
    assert status

    # assert "exponent >= 0.0" in str(_e)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent",
    [
        (0.0),
        (1.0),
        (2.0),
        (5.0),
        (0.5),
        (1.5),
        (2.5),
    ],
)
def test_bw_unary_pow(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 0, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device, seed=1)

    tt_output_tensor_on_device = ttnn.pow_bw(grad_tensor, input_tensor, exponent)

    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, pcc=0.99)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_unary_pow_test_inf(input_shapes, device):
    exponent = 2
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.74e38, 1.8e38, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, 1, 9, device, seed=1)

    tt_output_tensor_on_device = ttnn.pow_bw(grad_tensor, input_tensor, exponent)
    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
def test_bw_unary_pow_test_neg_inf(input_shapes, device):
    exponent = 2
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.74e38, 1.8e38, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, -1, device, seed=1)

    tt_output_tensor_on_device = ttnn.pow_bw(grad_tensor, input_tensor, exponent)

    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent",
    [
        (0.0),
        (1.0),
        (2.0),
        (5.0),
        (0.5),
        (1.5),
        (2.5),
    ],
)
def test_bw_unary_pow_output(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 0, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device, seed=1)
    input_grad = None

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = ttnn.pow_bw(
        grad_tensor,
        input_tensor,
        exponent=exponent,
        input_grad=input_grad,
        queue_id=cq_id,
    )

    in_data.retain_grad()

    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, pcc=0.99)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent",
    [
        (0.0),
        (1.0),
        (2.0),
        (5.0),
        (0.5),
        (1.5),
        (2.5),
    ],
)
def test_bw_unary_pow_negative_inputs(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True, seed=0)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device, seed=1)
    input_grad = None

    _, input_grad = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    tt_output_tensor_on_device = ttnn.pow_bw(
        grad_tensor,
        input_tensor,
        exponent=exponent,
        input_grad=input_grad,
        queue_id=cq_id,
    )

    in_data.retain_grad()

    golden_function = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_function(grad_data, in_data, exponent)

    status = compare_pcc(tt_output_tensor_on_device, golden_tensor, pcc=0.99)
    assert status


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "exponent",
    [(5.5), (8.5), (10.0), (11.6), (12.0), (13.2), (15.8), (16.45), (18.5), (20.0)],
)
@pytest.mark.parametrize(
    ("low1", "high1", "low2", "high2"),
    [
        (0, 30, -20, 20),
    ],
)
def test_bw_unary_pow_edge_case_exponents(device, input_shapes, exponent, high1, low1, high2, low2):
    in_data = (
        torch.rand(input_shapes, requires_grad=True).bfloat16() * (high1 - low1) + low1
    )  # Using only positive inputs as fractional exponents with negative bases yield NaN
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_data = torch.rand(input_shapes, requires_grad=True).bfloat16() * (high2 - low2) + low2
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    golden_fn = ttnn.get_golden_function(ttnn.pow_bw)
    golden_tensor = golden_fn(grad_data, in_data, exponent=exponent)

    in_data.retain_grad()

    output_tensor = ttnn.pow_bw(grad_tensor, input_tensor, exponent)

    status = compare_pcc(output_tensor, golden_tensor, pcc=0.99)
    assert status


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize("exponent", [0.0, 2.0])
def test_bw_pow_writes_through_preallocated_input_grad(input_shapes, exponent, device):
    """The caller's preallocated input_grad must be written, not replaced.

    Regression: for exponent 0, `input_grad = ttnn::zeros_like(input)` rebound the local
    optional instead of filling the supplied tensor, so a caller reading its own tensor
    got stale data. Comparing only the returned tensor does not catch that.
    """
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, 0.1, 10, device)

    sentinel = 7.0
    preallocated = ttnn.full(input_shapes, sentinel, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)

    ttnn.pow_bw(grad_tensor, input_tensor, exponent, input_grad=preallocated)

    written = ttnn.to_torch(preallocated)
    assert not torch.equal(
        written, torch.full(input_shapes, sentinel, dtype=torch.bfloat16)
    ), "preallocated input_grad was never written"
    if exponent == 0.0:
        assert torch.all(written == 0.0), "exponent 0 gradient must be zero"


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize("exponent", [0.0, 2.0])
def test_bw_pow_honours_requested_memory_config(input_shapes, exponent, device):
    """With no preallocated output, the gradient must land in the requested memory config.

    Regression: `empty_like`/`zeros_like` were called without `output_mem_config`, so the
    result inherited the input's config instead. Existing tests use the same config for
    inputs and output, so the two never diverge there.
    """
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, 0.1, 10, device)

    # inputs land in DRAM by default, so request L1 to force a divergence
    result = ttnn.pow_bw(grad_tensor, input_tensor, exponent, memory_config=ttnn.L1_MEMORY_CONFIG)

    assert (
        result[0].memory_config().buffer_type == ttnn.BufferType.L1
    ), f"requested L1 but gradient landed in {result[0].memory_config().buffer_type}"


@pytest.mark.parametrize("input_shapes", ((torch.Size([1, 1, 32, 32])),))
@pytest.mark.parametrize("exponent", [0.0, 2.0])
def test_bw_pow_preallocated_output_wins_over_memory_config(input_shapes, exponent, device):
    """A caller-supplied input_grad takes precedence over memory_config.

    This is the invariant whose behaviour changed: the gradient is written into the
    supplied tensor, so it stays in that tensor's memory space even when a different
    config is requested.

    The two parametrizations reach it by different routes, and neither goes through
    `full_impl`:

    - exponent 0: `zeros_like` -> `full_like_impl` takes the fast path for a device
      tensor in TILE layout whose dtype already matches, short-circuiting to
      `ttnn::fill(tensor, 0.0f, memory_config, optional_output_tensor)`. `fill` is a
      DEFINE_UNARY_OP_SCALAR_VARIANT, so precedence is decided by `unary_impl`
      (eltwise/unary/unary.cpp:38-40), which prefers
      `optional_output_tensor.value().memory_config()` over the requested config.
    - exponent 2: `full_like_impl` is not involved. The write-through is the
      `where(lez(input), inf, final_result, output_mem_config, input_grad)` at the end
      of `pow_bw`, via `where`'s output-tensor handling.

    Changing either of those would break this silently.
    """
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -10, 10, device)
    in_data, input_tensor = data_gen_with_range(input_shapes, 0.1, 10, device)

    preallocated = ttnn.full(input_shapes, 7.0, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)

    result = ttnn.pow_bw(
        grad_tensor, input_tensor, exponent, memory_config=ttnn.DRAM_MEMORY_CONFIG, input_grad=preallocated
    )

    assert result[0].memory_config().buffer_type == ttnn.BufferType.L1, (
        "preallocated output should win over memory_config, but the result moved to "
        f"{result[0].memory_config().buffer_type}"
    )
    assert preallocated.memory_config().buffer_type == ttnn.BufferType.L1
