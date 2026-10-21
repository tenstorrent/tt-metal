# Scratch, not for merge: ttnn.softmax numeric_stable (ROW MAX over the row's resident tiles through the reduce helper), bf16, interleaved.
import pytest
import torch
import ttnn

SHAPES = {
    "1x1x2048x1024": (1, 1, 2048, 1024),
    "1x8x512x512": (1, 8, 512, 512),
    "1x1x64x8192": (1, 1, 64, 8192),
    "1x1x1024x2048": (1, 1, 1024, 2048),
    "1x1x32x32000": (1, 1, 32, 32000),
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("neg", [0, 1])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_softmax(device, shape, neg):
    torch.manual_seed(61433)
    t = (torch.randn(SHAPES[shape], dtype=torch.float32) * 3 - 40 * neg).to(torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.to_torch(ttnn.softmax(x, dim=-1, numeric_stable=True))
