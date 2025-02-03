import torch, ttnn, pytest, torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc
import ttnn


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_silu_alone(device, use_program_cache, reset_seeds):
    torch_input_tensor = torch.randn(1, 32, 56, 56)
    act = nn.SiLU(inplace=True)
    torch_x = act(torch_input_tensor)

    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_x = ttnn.from_torch(
        ttnn_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    ttnn_x = ttnn.silu(ttnn_x)
    ttnn_x = ttnn.to_torch(ttnn_x).reshape(1, 56, 56, 32).permute(0, 3, 1, 2)

    assert_with_pcc(torch_x, ttnn_x, 0.99999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_slice_alone(device, use_program_cache, reset_seeds):
    torch_input_tensor = torch.randn(1, 64, 28, 28)
    torch_y1, torch_y2 = torch_input_tensor.chunk(2, 1)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_x = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_x = ttnn.to_layout(ttnn_x, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_y1 = ttnn_x[:, :, :, :32]
    ttnn_y2 = ttnn_x[:, :, :, 32:64]
    ttnn_y1 = ttnn.to_torch(ttnn_y1).reshape(1, 28, 28, 32).permute(0, 3, 1, 2)

    assert_with_pcc(torch_y1, ttnn_y1, 0.99999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_silu_layout_issue(device, use_program_cache, reset_seeds):
    torch_input_tensor = torch.randn(1, 64, 28, 28)
    act = nn.SiLU(inplace=True)
    torch_x = act(torch_input_tensor)
    torch_y1, torch_y2 = torch_input_tensor.chunk(2, 1)

    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))

    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_x = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_x = ttnn.silu(ttnn_x)
    ttnn_x = ttnn.to_layout(ttnn_x, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_y1 = ttnn_x[:, :, :, :32]
    ttnn_y2 = ttnn_x[:, :, :, 32:64]
    # ttnn_x = ttnn.to_layout(ttnn_x, ttnn.ROW_MAJOR_LAYOUT)
    # ttnn_x = ttnn.reshape(ttnn_x, (1, 28, 28, 64))

    # ttnn_y1, ttnn_y2 = ttnn.split(ttnn_x, 2, 3)
    ttnn_y1 = ttnn.to_torch(ttnn_y1).reshape(1, 28, 28, 32).permute(0, 3, 1, 2)

    assert_with_pcc(torch_y1, ttnn_y1, 0.99999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_split_dumped(device, use_program_cache, reset_seeds):  # 0.005 for fp8 and 099 for fp16
    torch_input_tensor = torch.randn(1, 64, 28, 28)
    torch_y1, torch_y2 = torch_input_tensor.chunk(2, 1)

    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_x = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_x = ttnn.to_layout(ttnn_x, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_x = ttnn.reshape(ttnn_x, (1, 28, 28, 64))
    ttnn_y1 = ttnn_x[:, :, :, :32]
    ttnn_y2 = ttnn_x[:, :, :, 32:64]

    assert_with_pcc(torch_y1, ttnn.to_torch(ttnn_y1).reshape(1, 28, 28, 32).permute(0, 3, 1, 2), 0.99999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_dtype_issue(device):
    a = torch.randn((1, 256, 1, 49), dtype=torch.bfloat16)
    a_ttnn = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    print("bfp8", a_ttnn.dtype)
    a_ttnn = ttnn.to_layout(a_ttnn, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    print("bfp16", a_ttnn.dtype)
    ttnn_output = ttnn.to_torch(a_ttnn)
    assert_with_pcc(a, ttnn_output, 0.99999)
