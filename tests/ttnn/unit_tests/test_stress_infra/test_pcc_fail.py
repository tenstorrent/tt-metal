# Tests that fail PCC — operations run on device but results don't match expected
import torch
import pytest
import ttnn


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 4, 32, 32],
    ],
)
def test_wrong_expected(device, shape):
    """Computes exp on device, compares against sin (deliberately wrong)."""
    x = torch.randn(shape).to(torch.bfloat16)

    tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.exp(tx)
    result_torch = ttnn.to_torch(result)

    # Deliberately wrong expected — sin instead of exp
    expected = torch.sin(x)
    pcc = torch.corrcoef(torch.stack([result_torch.flatten(), expected.flatten()]))[0, 1]
    assert pcc > 0.99, f"PCC {pcc} < 0.99 — numerical mismatch"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
    ],
)
def test_atol_too_tight(device, shape):
    """Computes exp correctly but uses absurdly tight tolerance."""
    x = torch.randn(shape).to(torch.bfloat16)

    tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.exp(tx)
    result_torch = ttnn.to_torch(result)

    expected = torch.exp(x)
    # bfloat16 exp can't possibly match to 1e-10
    assert torch.allclose(result_torch, expected, atol=1e-10), "Tolerance too tight for bfloat16"
