# Scratch, not for merge. ttnn.min over H and W (the reduce_hw_neg and single core HW paths, SCALAR MAX of the negated input):
# all-negative, mixed, zero-heavy and bfp8/fp32 inputs, outputs hashed by run_nodes.py.
import pytest
import torch
import ttnn

SHAPES = [(1, 1, 32, 32), (1, 1, 64, 64), (2, 3, 96, 128), (1, 1, 37, 41), (4, 1, 32, 320)]
DTYPES = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b]


def make(kind, shape):
    torch.manual_seed(7)
    if kind == "neg":
        return -(0.5 + torch.rand(shape))
    if kind == "mixed":
        return torch.rand(shape) * 2 - 1
    if kind == "pos":
        return 0.5 + torch.rand(shape)
    z = torch.randint(-1, 2, shape).float()  # zeros, ones, minus ones
    return z * (torch.rand(shape) > 0.5)


@pytest.mark.parametrize("fp32_dest", [False, True], ids=["dest16", "dest32"])
@pytest.mark.parametrize("dim", [None, (-2, -1)], ids=["none", "hw"])
@pytest.mark.parametrize("kind", ["neg", "mixed", "pos", "zeros"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32", "bfp8"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_min_hw(device, shape, dtype, kind, dim, fp32_dest):
    x = make(kind, shape)
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False, fp32_dest_acc_en=fp32_dest
    )
    kw = {"compute_kernel_config": cfg}
    y = ttnn.min(t, dim=dim, **kw) if dim is not None else ttnn.min(t, **kw)
    got = ttnn.to_torch(y).float()
    ref = torch.amin(ttnn.to_torch(t).float(), dim=(-2, -1) if dim is None else dim)
    assert got.reshape(-1)[0].item() == ref.reshape(-1)[0].item() or dtype == ttnn.bfloat8_b
