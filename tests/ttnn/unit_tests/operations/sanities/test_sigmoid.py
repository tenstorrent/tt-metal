import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest


@pytest.mark.parametrize("input_source", ["file", "random"])
def test_sigmoid_my_case(device, input_source):
    if input_source == "file":
        torch_input_tensor = torch.load("sig_ip.pt")
    elif input_source == "random":
        torch_input_tensor = torch.randn(1, 1, 256, 256)

    print("torch_input_tensor=>", torch_input_tensor)

    torch_output_tensor = torch.sigmoid(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.sigmoid(input_tensor)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output)
