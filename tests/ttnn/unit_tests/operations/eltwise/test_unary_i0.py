# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn


_TOLS = {
    ttnn.float32: dict(rtol=1e-4, atol=1e-6),
    ttnn.bfloat16: dict(rtol=1e-2, atol=1e-3),
}


def _assert_close(name, got, ref, dtype):
    tols = _TOLS[dtype]
    got_f32 = got.float()
    ref_f32 = ref.float()
    if not torch.allclose(got_f32, ref_f32, rtol=tols["rtol"], atol=tols["atol"]):
        diff = (got_f32 - ref_f32).abs()
        rel = diff / (ref_f32.abs() + tols["atol"])
        idx = rel.argmax()
        raise AssertionError(
            f"{name}: not allclose (rtol={tols['rtol']}, atol={tols['atol']}); "
            f"worst rel_err={rel.flatten()[idx].item():.2e} "
            f"ref={ref_f32.flatten()[idx].item():.4e} got={got_f32.flatten()[idx].item():.4e}"
        )


@pytest.mark.parametrize(
    "shapes",
    [
        [1, 1, 32, 32],
        [4, 2, 96, 192],
        [64, 64],
    ],
)
def test_i0_range(device, shapes):
    torch.manual_seed(0)

    low, high = -7.0, 7.0
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * (high - low) + low
    torch_output_tensor = torch.special.i0(torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.float32,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.allclose(output_tensor, torch_output_tensor, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_ood(device, dtype):
    torch.manual_seed(0)

    shapes = [4, 2, 96, 192]
    torch_input_tensor_a = torch.rand(shapes, dtype=torch.float32) * 100.0 - 50.0
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    ref_input = torch_input_tensor_a.to(torch_dtype).to(torch.float32)
    torch_output_tensor = torch.special.i0(ref_input)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    _assert_close("test_i0_ood", output_tensor, torch_output_tensor, dtype)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16])
def test_i0_clamp_boundary(device, dtype):
    boundaries = torch.tensor(
        [-100.0, -88.5, -88.0, -7.5, -7.0, -6.5, 6.5, 7.0, 7.5, 88.0, 88.5, 100.0],
        dtype=torch.float32,
    )
    expected_input = torch.clamp(boundaries, min=-88.5, max=88.5)
    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    expected_input = expected_input.to(torch_dtype).to(torch.float32)
    torch_output_tensor = torch.special.i0(expected_input)

    padded = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    padded[0, 0, 0, : boundaries.numel()] = boundaries

    input_tensor_a = ttnn.from_torch(
        padded,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.i0(input_tensor_a, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)[0, 0, 0, : boundaries.numel()]

    _assert_close("test_i0_clamp_boundary", output_tensor, torch_output_tensor, dtype)
