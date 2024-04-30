import pytest
import ttnn
import torch

import ttnn.operations.binary


@pytest.mark.eager_package_silicon
def test_ttnn_import(reset_seeds):
    with ttnn.manage_device(device_id=0) as device:
        pass


@pytest.mark.eager_package_silicon
@pytest.mark.skip("Tries to open device twice")
def test_ttnn_add(reset_seeds):
    with ttnn.manage_device(device_id=0) as device:
        a_torch = torch.ones((5, 7))
        b_torch = torch.ones((1, 7))

        a = ttnn.from_torch(a_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        b = ttnn.from_torch(b_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        output = a + b
        output = ttnn.to_torch(output)
