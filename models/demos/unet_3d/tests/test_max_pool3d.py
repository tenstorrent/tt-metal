import torch

import ttnn
from models.demos.unet_3d.ttnn_impl.max_pool3d import max_pool3d
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_single_case(device, N, D, H, W, C, k):
    # Create random input tensor
    torch_tensor = torch.randn((N, C, D, H, W), dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor.permute(0, 2, 3, 4, 1), dtype=ttnn.bfloat16, device=device)

    # Perform max pooling using torch
    pool3d_torch = torch.nn.MaxPool3d(
        kernel_size=k,
        stride=2,
        padding=1,
    ).to(dtype=torch.bfloat16)
    pool3d_torch.eval()
    torch_output = pool3d_torch(torch_tensor).permute(0, 2, 3, 4, 1)
    torch_output = torch_output.reshape(N, -1)

    # Perform max pooling using ttnn
    tt_output = max_pool3d(
        ttnn_tensor,
        kernel_size=k,
        stride=2,
        padding=1,
    )
    tt_output = ttnn.to_layout(tt_output, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_torch = ttnn.to_torch(tt_output.cpu()).reshape(N, -1)

    # Verify outputs are similar
    assert_with_pcc(torch_output, ttnn_torch, pcc=0.99)


def test_maxpool3d():
    N, D, H, W = 1, 64, 64, 64
    C = 64
    k = 2
    device = ttnn.open_device(
        device_id=0,
        dispatch_core_config=ttnn.device.DispatchCoreConfig(),
        l1_small_size=8192 * 4 * 4,
    )
    run_single_case(device, N, D, H, W, C, k)


if __name__ == "__main__":
    test_maxpool3d()
