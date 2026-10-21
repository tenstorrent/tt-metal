# Scratch, not for merge. Float32 and Int32 inputs through the welford var/std (hw and w) paths; Int32 may be rejected by the op, which is fine.
import pytest
import torch
import ttnn


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.int32])
@pytest.mark.parametrize("op", [ttnn.var, ttnn.std])
@pytest.mark.parametrize("dim", [None, -1, -2, (-2, -1)])
def test_var_std_types(device, dtype, op, dim):
    torch.manual_seed(0)
    t = (torch.randn(1, 1, 64, 96) * 10).to(torch.float32 if dtype == ttnn.float32 else torch.int32)
    x = ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        out = op(x, dim=dim, keepdim=True) if dim is not None else op(x)
    except RuntimeError as e:
        if dtype == ttnn.int32:
            pytest.skip(f"int32 rejected: {str(e)[:80]}")
        raise
    ttnn.to_torch(out)
