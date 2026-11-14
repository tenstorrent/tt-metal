# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("op_name", ["add", "multiply", "subtract"])
@pytest.mark.parametrize("scalar_rhs", [False, True])
@pytest.mark.parametrize("preallocated", [False, True])
def test_binary_ng_int8_output(device, input_dtype, op_name, scalar_rhs, preallocated, expect_error):
    # Integer values cover both signs and byte wrapping without rounding ambiguity.
    x = (torch.arange(1024).reshape(32, 32) % 127 - 63).float()
    rhs = torch.full_like(x, 2) if scalar_rhs else (torch.arange(1024).reshape(32, 32) % 3 + 1).float()
    a = ttnn.from_torch(x, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = 2.0 if scalar_rhs else ttnn.from_torch(rhs, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    kwargs = {"dtype": ttnn.int8}
    if preallocated:
        kwargs = {
            "output_tensor": ttnn.from_torch(
                torch.full((32, 32), 7, dtype=torch.int8),
                dtype=ttnn.int8,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        }
    if device.arch() == ttnn.device.Arch.QUASAR:
        with expect_error(RuntimeError, "INT8 output typecast is not supported on Quasar"):
            getattr(ttnn, op_name)(a, b, **kwargs)
        return
    expected = getattr(torch, op_name)(x, rhs)
    result = getattr(ttnn, op_name)(a, b, **kwargs)
    assert torch.equal(ttnn.to_torch(result), expected.to(torch.int8))
    if preallocated:
        assert result.buffer_address() == kwargs["output_tensor"].buffer_address()


def test_binary_ng_int8_quantization_control(device):
    if device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip("INT8 typecast is not available on Quasar")
    x = (torch.arange(1024).reshape(32, 32) % 255 - 127).float()
    a = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    quantized = ttnn.quantize(a, 1.0, 0, dtype=ttnn.int8)
    requantized = ttnn.requantize(quantized, 1.0, 0, 1.0, 0, dtype=ttnn.int8)
    assert torch.equal(ttnn.to_torch(quantized), x.to(torch.int8))
    assert torch.equal(ttnn.to_torch(requantized), x.to(torch.int8))
