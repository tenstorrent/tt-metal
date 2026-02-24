# Mixed test file — some pass, some fail PCC, within the same file.
# With -x (stop on first failure), only tests before the first failure run.
# Without -x, all tests run and you get a mix of pass/fail.
import torch
import pytest
import ttnn


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 4, 32, 32],
        [2, 2, 64, 64],
        [1, 8, 32, 32],
    ],
)
def test_add_pass(device, shape):
    """Always passes — simple add."""
    a = torch.randn(shape).to(torch.bfloat16)
    b = torch.randn(shape).to(torch.bfloat16)
    ta = ttnn.from_torch(a, device=device, layout=ttnn.TILE_LAYOUT)
    tb = ttnn.from_torch(b, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.to_torch(ttnn.add(ta, tb))
    expected = a + b
    assert torch.allclose(result, expected, atol=0.1)


@pytest.mark.parametrize(
    "op_name,tolerance",
    [
        ("exp", 0.99),  # passes — exp has good PCC
        ("sin", 0.0001),  # FAILS — comparing exp result against sin threshold makes no sense
        ("cos", 0.99),  # passes — cos has good PCC when compared correctly
    ],
)
def test_unary_pcc(device, op_name, tolerance):
    """Tests different unary ops. The 'sin' parametrization deliberately
    uses a nonsensical tolerance that will fail."""
    shape = [1, 2, 32, 32]
    x = torch.randn(shape).to(torch.bfloat16)
    tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)

    op_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)

    result = ttnn.to_torch(op_fn(tx))
    expected = torch_fn(x)

    pcc = torch.corrcoef(torch.stack([result.flatten(), expected.flatten()]))[0, 1]
    assert pcc > tolerance, f"{op_name}: PCC {pcc:.6f} < {tolerance}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 2, 32, 32],
        [1, 4, 32, 32],
    ],
)
def test_multiply_pass(device, shape):
    """Always passes — simple multiply."""
    a = torch.randn(shape).to(torch.bfloat16)
    b = torch.randn(shape).to(torch.bfloat16)
    ta = ttnn.from_torch(a, device=device, layout=ttnn.TILE_LAYOUT)
    tb = ttnn.from_torch(b, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.to_torch(ttnn.multiply(ta, tb))
    expected = a * b
    assert torch.allclose(result, expected, atol=0.2)


@pytest.mark.parametrize(
    "wrong_op",
    [
        "exp",  # FAILS — comparing neg result against exp
        "relu",  # FAILS — comparing neg result against relu
    ],
)
def test_neg_wrong_compare(device, wrong_op):
    """Computes neg on device, compares against a different op. Always fails."""
    shape = [1, 1, 32, 32]
    x = torch.randn(shape).to(torch.bfloat16)
    tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    result = ttnn.to_torch(ttnn.neg(tx))

    wrong_fn = getattr(torch, wrong_op)
    expected = wrong_fn(x)

    pcc = torch.corrcoef(torch.stack([result.flatten(), expected.flatten()]))[0, 1]
    assert pcc > 0.99, f"neg vs {wrong_op}: PCC {pcc:.6f} — deliberate mismatch"
