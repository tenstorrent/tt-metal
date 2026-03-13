# Tests that pass — simple ttnn operations
import torch
import pytest
import ttnn


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 4, 32, 32],
        [2, 2, 64, 64],
    ],
)
def test_add(device, shape):
    a = torch.randn(shape).to(torch.bfloat16)
    b = torch.randn(shape).to(torch.bfloat16)

    ta = ttnn.from_torch(a, device=device, layout=ttnn.TILE_LAYOUT)
    tb = ttnn.from_torch(b, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.add(ta, tb)
    result_torch = ttnn.to_torch(result)

    expected = a + b
    assert torch.allclose(result_torch, expected, atol=0.1)


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 4, 32, 32],
    ],
)
def test_exp(device, shape):
    x = torch.randn(shape).to(torch.bfloat16)

    tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.exp(tx)
    result_torch = ttnn.to_torch(result)

    expected = torch.exp(x)
    pcc = torch.corrcoef(torch.stack([result_torch.flatten(), expected.flatten()]))[0, 1]
    assert pcc > 0.99, f"PCC {pcc} < 0.99"
