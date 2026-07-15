import torch, ttnn
from eval.sharding import auto_shard_config
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def _cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def ref(x, gamma=None, eps=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def run(shape, dtype, with_gamma):
    torch.manual_seed(0)
    tdt = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}[dtype]
    x = torch.randn(shape, dtype=tdt)
    g = torch.randn(shape[-1], dtype=tdt) if with_gamma else None
    mem = auto_shard_config(list(shape), WIDTH, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    ncores = mem.shard_spec.grid.num_cores()
    bb = mem.shard_spec.grid.bounding_box()
    nx = int(bb.end.x) - int(bb.start.x) + 1
    ny = int(bb.end.y) - int(bb.start.y) + 1
    ragged = ncores != nx * ny
    print(
        f"shape={shape} dtype={dtype} gamma={with_gamma} shard={list(mem.shard_spec.shape)} ncores={ncores} bbox={nx}x{ny} RAGGED={ragged}"
    )
    xin = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
    gin = (
        ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        if with_gamma
        else None
    )
    out = rms_norm(xin, gamma=gin, compute_kernel_config=_cfg(), memory_config=xin.memory_config())
    act = ttnn.to_torch(out).reshape(shape)
    exp = ref(x, g)
    pcc_ok = True
    try:
        assert_with_pcc(exp.to(torch.float32), act.to(torch.float32), 0.99)
        print("   PCC PASS")
    except AssertionError as e:
        print("   PCC FAIL:", str(e)[:200])
        pcc_ok = False
    return pcc_ok


device = ttnn.open_device(device_id=0)
try:
    # bf16 W=50: 7 cores (7x1) -> NOT ragged -> mcast path
    run((1, 1, 32, 50), ttnn.bfloat16, False)
    run((1, 1, 32, 50), ttnn.bfloat16, True)
    # fp32 W=50: 13 cores (8x2) -> RAGGED -> unicast path
    run((1, 1, 32, 50), ttnn.float32, False)
    run((1, 1, 32, 50), ttnn.float32, True)
finally:
    ttnn.close_device(device)
