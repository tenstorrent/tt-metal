import torch, ttnn
from eval.sharding import auto_shard_config
from tests.ttnn.utils_for_testing import comp_pcc
from ttnn.operations.rms_norm import rms_norm

WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


def ref(x, g):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    return (xf / rms * g.to(torch.float32).reshape(-1)).to(x.dtype)


device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 32, 64)  # sub-tile Ws=8, WIDTH, 8-core group
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16)
    mc = auto_shard_config(list(shape), WIDTH, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xin = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    # TILE gamma (the 5c case)
    gin = ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm(xin, gamma=gin, compute_kernel_config=cfg(), memory_config=xin.memory_config())
    actual = ttnn.to_torch(out).reshape(shape).to(torch.float32)
    expected = ref(x, g).to(torch.float32)
    ok, pcc = comp_pcc(expected, actual, 0.995)
    print("PCC:", pcc, "ok:", ok)
    print("sample expected[0,0,0,:8]:", expected[0, 0, 0, :8].tolist())
    print("sample actual  [0,0,0,:8]:", actual[0, 0, 0, :8].tolist())
finally:
    ttnn.close_device(device)
