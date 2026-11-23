# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize("predicate_dtype", [ttnn.uint16, ttnn.float32])
@pytest.mark.parametrize(
    "value_dtypes",
    [
        (ttnn.float32, ttnn.float32),
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("output_dtype", [ttnn.int32, ttnn.uint16])
@pytest.mark.parametrize("broadcast", ["none", "row", "column"])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_where_float_values_integer_output(
    device, predicate_dtype, value_dtypes, output_dtype, broadcast, memory_config
):
    if device.arch() == ttnn.device.Arch.QUASAR and (predicate_dtype == ttnn.uint16 or output_dtype == ttnn.uint16):
        pytest.skip("UINT16 typecast is not supported on Quasar")
    z = torch.arange(1024).reshape(32, 32)
    predicate = (z % 3 != 0).to(torch.int32)
    a = (z % 11 + 1).float()
    b = (z % 7 + 15).float() + 0.999
    a = a.to(torch.bfloat16).float() if value_dtypes[0] == ttnn.bfloat16 else a
    b = b.to(torch.bfloat16).float() if value_dtypes[1] == ttnn.bfloat16 else b
    if broadcast == "row":
        a, b = a[:1, :], b[:1, :]
    elif broadcast == "column":
        a, b = a[:, :1], b[:, :1]
    p = ttnn.from_torch(predicate, dtype=predicate_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    x = ttnn.from_torch(a, dtype=value_dtypes[0], layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.from_torch(b, dtype=value_dtypes[1], layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.from_torch(
        torch.full((32, 32), 99, dtype=torch.int32),
        dtype=output_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    float_out = ttnn.from_torch(torch.zeros((32, 32)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    for lhs, rhs, host_lhs, host_rhs in [(x, y, a, b), (y, x, b, a)]:
        expected = torch.where(predicate.bool(), host_lhs, host_rhs)
        control = ttnn.where(p, lhs, rhs, output_tensor=float_out)
        assert torch.equal(ttnn.to_torch(control).float(), expected)
        result = ttnn.where(p, lhs, rhs, output_tensor=out)
        assert result.buffer_address() == out.buffer_address()
        # Integer outputs convert like ttnn.typecast of the FP32 control, whatever its UINT16 rounding mode;
        # INT32 also has to truncate like torch.
        converted = ttnn.to_torch(ttnn.typecast(control, output_dtype)).to(torch.int32)
        if output_dtype == ttnn.int32:
            assert torch.equal(converted, torch.trunc(expected).to(torch.int32))
        assert torch.equal(ttnn.to_torch(result).to(torch.int32), converted)
