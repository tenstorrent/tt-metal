import pytest
import torch
import ttnn

from ....utils.tensor import bf16_tensor
from ....utils.check import assert_quality


def run_test_linear(device, M, K, N):
    torch_dtype = torch.float32

    torch_model = torch.nn.Linear(K, N, bias=False).to(dtype=torch_dtype)
    torch_model.eval()

    torch_input = torch.randn((M, K), dtype=torch_dtype)

    # Prepare TT tensors
    tt_input = bf16_tensor(torch_input, device=device)
    # ttnn.linear expects weight shaped (K, N)
    torch_weight_t = torch_model.weight.transpose(0, 1)
    tt_weight = bf16_tensor(torch_weight_t, device=device)
    tt_bias = None
    if torch_model.bias is not None:
        tt_bias = bf16_tensor(torch_model.bias.reshape(1, -1), device=device)

    with torch.no_grad():
        torch_output = torch_model(torch_input)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,  # NOTE: True can improve correctness, tiny performance cost
    )
    # core_grid = device.compute_with_storage_grid_size()
    # core_grid = ttnn.CoreGrid(y=core_grid.y, x=core_grid.x)
    core_grid = ttnn.CoreCoord(1, 1)
    matmul_config = ttnn.MinimalMatmulConfig(compute_with_storage_grid_size=core_grid)

    tt_output = ttnn.experimental.minimal_matmul(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=None,
        compute_kernel_config=compute_config,
        config=matmul_config,
    )

    # Compare outputs
    tt_output = ttnn.to_torch(tt_output)
    check_result = assert_quality(torch_output, tt_output)

    return check_result


@pytest.mark.parametrize(
    "M, K, N",
    [(32, 32, 32)],
)
def test_linear(device, M, K, N):
    check_result = run_test_linear(device, M, K, N)
    assert check_result["pcc"] > 0.999_500
    assert check_result["relative_rmse"] < 0.02
