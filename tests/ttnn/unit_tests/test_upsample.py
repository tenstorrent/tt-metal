import pytest

import torch
import torch.nn as nn
import ttnn


from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0
from models.utility_functions import torch_random


@skip_for_wormhole_b0()
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("f", [320])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("scale_h", [2])
@pytest.mark.parametrize("scale_w", [2])
def test_upsample(device, batch_size, f, h, w, scale_h, scale_w):
    input = torch.arange(1, 5, dtype=torch.float32).view(1, 1, 2, 2)
    print(input)
    scale_factor = (scale_h, scale_w)
    m = nn.Upsample(scale_factor=scale_factor, mode="nearest")
    torch_result = m(input)
    print(torch_result)
    print(torch_result.shape)

    input_tensor = ttnn.from_torch(input, device=device)
    # input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.upsample(input_tensor, scale_factor)

    # print(output)
    # print(output.shape)
    assert_with_pcc(torch_result, output_tensor, 0.997)
