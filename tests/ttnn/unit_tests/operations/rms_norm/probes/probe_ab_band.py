import pytest, torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

_ML = ttnn.TensorMemoryLayout


def _cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


CASES = [
    ((1, 1, 224, 3072), _ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT, "TILE_width_w3072"),
    ((1, 1, 224, 3072), _ML.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT, "BAND_width_w3072"),
    ((1, 1, 224, 3072), _ML.BLOCK_SHARDED, ttnn.TILE_LAYOUT, "TILE_block_w3072"),
    ((1, 1, 224, 3072), _ML.BLOCK_SHARDED, ttnn.ROW_MAJOR_LAYOUT, "BAND_block_w3072"),
    ((1, 1, 256, 512), _ML.WIDTH_SHARDED, ttnn.TILE_LAYOUT, "TILE_width_w512"),
    ((1, 1, 256, 512), _ML.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT, "BAND_width_w512"),
    ((1, 1, 256, 512), _ML.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, "RM_height_w512"),
]


@pytest.mark.parametrize("shape,ml,layout,label", CASES, ids=[c[3] for c in CASES])
def test_ab(device, shape, ml, layout, label):
    torch.manual_seed(42)
    W = shape[-1]
    mc = auto_shard_config(list(shape), ml, layout=layout, dtype=ttnn.bfloat16, device=device)
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc
    )
    gt = torch.randn(W, dtype=torch.bfloat16)
    g = ttnn.from_torch(
        gt.reshape(1, 1, 1, W) if layout == ttnn.TILE_LAYOUT else gt, dtype=ttnn.bfloat16, layout=layout, device=device
    )
    out = rms_norm(x, gamma=g, compute_kernel_config=_cfg(), memory_config=mc)
    assert tuple(out.shape) == tuple(shape)
