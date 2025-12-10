import ttnn
import pytest
import torch

from tests.ttnn.utils_for_testing import assert_with_ulp, assert_equal


@pytest.mark.parametrize("val", [128.5, -128.5])
def test_from_torch_typecast(val, device):
    ttnn_dtype = ttnn.bfloat16

    shape = [20, 20]
    # torch_input_fp32 = torch.arange(shape[0] * shape[1], dtype=torch.float32)

    torch_input_fp32 = torch.full(shape, val, dtype=torch.float32)
    torch_input_fp32 = torch.reshape(torch_input_fp32, shape)
    torch_golden_bf16 = torch_input_fp32.to(torch.bfloat16)

    ttnn_input_fp32 = ttnn.from_torch(torch_input_fp32, device=device, layout=ttnn.TILE_LAYOUT)
    ttnn_input_bf16 = ttnn.typecast(ttnn_input_fp32, dtype=ttnn_dtype)

    torch_calculated_bf16 = ttnn.to_torch(ttnn_input_bf16)

    assert_with_ulp(torch_golden_bf16, torch_calculated_bf16, ulp_threshold=0)


@pytest.mark.parametrize("val", [128.5, -128.5])
def test_from_torch_to_dytpe(val, device):
    ttnn_dtype = ttnn.bfloat16

    shape = [20, 20]
    # torch_input_fp32 = torch.arange(shape[0] * shape[1], dtype=torch.float32)

    torch_input_fp32 = torch.full(shape, val, dtype=torch.float32)
    torch_input_fp32 = torch.reshape(torch_input_fp32, shape)
    torch_golden_bf16 = torch_input_fp32.to(torch.bfloat16)

    ttnn_input_fp32 = ttnn.from_torch(torch_input_fp32, layout=ttnn.TILE_LAYOUT)
    ttnn_input_bf16 = ttnn.to_dtype(ttnn_input_fp32, dtype=ttnn_dtype)

    torch_calculated_bf16 = ttnn.to_torch(ttnn_input_bf16)

    assert_with_ulp(torch_golden_bf16, torch_calculated_bf16, ulp_threshold=0)


@pytest.mark.parametrize("val", [128.5, -128.5])
def test_from_torch_with_dtype(device, val):
    torch_dtype = torch.float32
    ttnn_dtype = ttnn.bfloat16

    shape = [20, 20]
    # torch_input_fp32 = torch.arange(shape[0] * shape[1], dtype=torch.float32)
    torch_input_fp32 = torch.full(shape, 128.5, dtype=torch.float32)
    torch_input_fp32 = torch.reshape(torch_input_fp32, shape)
    torch_golden_bf16 = torch_input_fp32.to(torch.bfloat16)

    ttnn_calculated_bf16 = ttnn.from_torch(torch_input_fp32, dtype=ttnn_dtype, device=device, layout=ttnn.TILE_LAYOUT)
    torch_calculated_bf16 = ttnn.to_torch(ttnn_calculated_bf16)

    assert_with_ulp(torch_golden_bf16, torch_calculated_bf16, ulp_threshold=0)
