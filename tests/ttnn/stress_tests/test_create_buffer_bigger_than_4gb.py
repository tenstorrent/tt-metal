import ttnn
import torch
import pytest
import os


@pytest.mark.parametrize(
    "shape",
    [
        (2**6, 2**13, 2**13 + 32),  # (4Gb + 32) * 2 (sizeof(bfloat16))
        (96 * 96, 1, 32 * 228),  # 4Gb + 8Mb after padding
    ],
)
@pytest.mark.parametrize(
    "fast_dispatch",
    [False],  # TODO: add fast dispatch fix
)
@pytest.mark.slow
def test_large_tensor_creation(device, shape, fast_dispatch):
    original_dispatch = os.environ.get("TT_METAL_SLOW_DISPATCH_MODE")
    try:
        shape = (2**6, 2**13, 2**13 + 32)
        if fast_dispatch:
            os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
        else:
            os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = "1"

        torch_input = torch.full(shape, 1).bfloat16()
        torch_output = torch_input

        input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.from_device(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor).bfloat16()

        assert torch_output.shape == output_tensor.shape
        assert torch.all(torch_output == output_tensor)
    finally:
        # Make sure to restore original dispatch value
        if original_dispatch is None:
            os.environ.pop("TT_METAL_SLOW_DISPATCH_MODE", None)
        else:
            os.environ["TT_METAL_SLOW_DISPATCH_MODE"] = original_dispatch
