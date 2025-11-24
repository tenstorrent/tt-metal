import pytest
import torch
import ttnn
import torch.nn as nn
from tests.ttnn.utils_for_testing import assert_with_pcc

import torch.nn.functional as F


@pytest.mark.parametrize(
    "torch_input_tensor",
    [
        torch.randn(1, 14, 39, 39),
        torch.randn(1, 14, 39, 39) * 50 + 10,
    ],
)
def test_softmax_my_case(device, torch_input_tensor):
    torch.manual_seed(0)
    torch_output_tensor = F.softmax(torch_input_tensor, dim=-1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.softmax(input_tensor, dim=-1)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
