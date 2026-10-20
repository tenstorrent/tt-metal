# Scratch, not for merge: the production callers of REDUCE_COL SUM/AVG (generic reduce over H, moreh sum, mean and softmax over H).
# Each test runs the op once; the output goes through ttnn.to_torch so the arms' results can be hashed.
import pytest
import torch
import ttnn

SHAPES = {
    "1x1x1024x1024": (1, 1, 1024, 1024),
    "1x1x4096x128": (1, 1, 4096, 128),
    "4x1x2048x256": (4, 1, 2048, 256),
    "1x1x32x8192": (1, 1, 32, 8192),
    "1x1x256x2048": (1, 1, 256, 2048),
    "1x1x128x128": (1, 1, 128, 128),
}
DT = {"bf16": ttnn.bfloat16, "f32": ttnn.float32}


def _x(device, shape, dt):
    torch.manual_seed(58706)
    t = torch.randn(shape, dtype=torch.float32)
    return ttnn.from_torch(t, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("dt", list(DT))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_sum_h(device, shape, dt):
    ttnn.to_torch(ttnn.sum(_x(device, SHAPES[shape], dt), dim=-2, keepdim=True))


@pytest.mark.parametrize("dt", list(DT))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_mean_h(device, shape, dt):
    ttnn.to_torch(ttnn.mean(_x(device, SHAPES[shape], dt), dim=-2, keepdim=True))


@pytest.mark.parametrize("dt", ["bf16"])
@pytest.mark.parametrize("shape", ["1x1x1024x1024", "4x1x2048x256", "1x1x256x2048"])
def test_moreh_sum_h(device, shape, dt):
    ttnn.to_torch(ttnn.operations.moreh.sum(_x(device, SHAPES[shape], dt), -2, keepdim=True))


@pytest.mark.parametrize("dt", ["bf16"])
@pytest.mark.parametrize("shape", ["1x1x1024x1024", "4x1x2048x256", "1x1x256x2048"])
def test_moreh_mean_h(device, shape, dt):
    ttnn.to_torch(ttnn.operations.moreh.mean(_x(device, SHAPES[shape], dt), dim=-2, keepdim=True))


@pytest.mark.parametrize("dt", ["bf16"])
@pytest.mark.parametrize("shape", ["1x1x1024x1024", "4x1x2048x256", "1x1x256x2048"])
def test_moreh_softmax_h(device, shape, dt):
    ttnn.to_torch(ttnn.operations.moreh.softmax(_x(device, SHAPES[shape], dt), -2))
