import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_dtype_issue(device):
    a = torch.randn((1, 256, 1, 49), dtype=torch.bfloat16)
    a_ttnn = ttnn.from_torch(
        a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    print("bf8", a_ttnn)
    a_ttnn = ttnn.to_memory_config(a_ttnn, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    print(a_ttnn.get_dtype())
    assert a_ttnn.get_dtype() == ttnn.bfloat16
    ttnn_output = ttnn.to_torch(a_ttnn)
    assert_with_pcc(a, ttnn_output, 0.99999)
