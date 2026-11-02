# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import (
    assert_equal,
)

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "base_bits,pad_bits",
    [
        (0x7F800000, 0x7FC00000),
        (-0x800000, -0x400000),
    ],
    ids=["positive_nan_payloads", "negative_nan_payloads"],
)
def test_tanh_fp32_all_nan_payloads_propagate(device, base_bits, pad_bits):
    mantissas = torch.arange(1, 0x800000, dtype=torch.int32)
    input_bits = base_bits + mantissas
    input_bits = torch.cat([input_bits, torch.tensor([pad_bits], dtype=torch.int32)])
    torch_input_tensor = input_bits.view(torch.float32).reshape(8192, 1024)
    assert torch.isnan(torch_input_tensor).all()

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        preserve_nan_values=True,
    )

    output_tensor = ttnn.to_torch(ttnn.tanh(input_tensor))

    assert torch.isnan(output_tensor).all()


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_sign_signed_zero(device, dtype):
    # sign(-0.0) is 0, not -1. SFPSETCC is unspecified for -0.0 (VectorUnit.md), so the
    # kernel's (v < 0.0f) captured it; the zero test has to mask the sign bit first.
    x_torch = torch.tensor([[-0.0, 0.0, -1.0, 1.0, -5.5, 5.5, -1e-30, 1e-30]], dtype=torch.float32)
    y_torch = torch.sign(x_torch)

    x_tt = ttnn.from_torch(x_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    assert_equal(y_torch, ttnn.to_torch(ttnn.sign(x_tt)))


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_heaviside_signed_zero(device, dtype):
    # heaviside(-0.0, s) is s, not 0, because -0.0 == 0. Same SFPSETCC cause as sign above,
    # and the two are asserted to move together in the tt-llk edge table.
    value = 0.5
    x_torch = torch.tensor([[-0.0, 0.0, -1.0, 1.0, -5.5, 5.5]], dtype=torch.float32)
    y_torch = torch.heaviside(x_torch, torch.tensor(value, dtype=torch.float32))

    x_tt = ttnn.from_torch(x_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    assert_equal(y_torch, ttnn.to_torch(ttnn.heaviside(x_tt, value)))


@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.log,
        ttnn.log2,
        ttnn.log10,
    ],
    ids=["log", "log2", "log10"],
)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("fast_and_approximate_mode", [False, True])
def test_log_signed_zero(device, ttnn_function, dtype, fast_and_approximate_mode):
    # log(±0.0) is -inf (torch parity). The kernel normalises -0.0 to +0.0 before
    # reading the exponent; issue #57538 was the exponent being read off the raw -0.0,
    # which delivered +inf on the bf16 path. The fast path used to skip the
    # normalisation altogether, so its sign test excluded the -0.0 lane and the
    # lane kept the NaN the result is initialised with. The bfloat16 sweeps cannot
    # catch either: generate_bfloat16_bits_in_range rewrites bf16 -0.0 (0x8000) to
    # +0.0 before the tensor is sent to the device, so -0.0 is built directly here.
    x_torch = torch.tensor([[-0.0, 0.0, 1.0]], dtype=torch.float32)

    x_tt = ttnn.from_torch(x_torch, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn_function(x_tt, fast_and_approximate_mode=fast_and_approximate_mode))

    assert torch.isinf(result[0, 0]) and torch.signbit(result[0, 0]), (
        f"{ttnn_function.__name__}(-0.0) returned {result[0, 0].item()!r}, expected -inf; the "
        "signed-zero normalisation in calculate_log_body must run before the exponent read."
    )
    assert torch.isinf(result[0, 1]) and torch.signbit(
        result[0, 1]
    ), f"{ttnn_function.__name__}(+0.0) returned {result[0, 1].item()!r}, expected -inf."
    assert result[0, 2].item() == 0.0, (
        f"{ttnn_function.__name__}(1.0) returned {result[0, 2].item()!r}, expected 0.0; an "
        "over-wide signed-zero arm is swallowing ordinary inputs."
    )
