import torch, ttnn, math
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
    grid = mem.shard_spec.grid
    ncores = grid.num_cores()
    bb = grid.bounding_box()
    nx = int(bb.end.x) - int(bb.start.x) + 1
    ny = int(bb.end.y) - int(bb.start.y) + 1
    ragged = ncores != nx * ny
    Hs = int(mem.shard_spec.shape[0])
    rounds = math.ceil(Hs / 32)
    tag = f"{shape} {str(dtype).split('.')[-1]} g={int(with_gamma)} nc={ncores} {nx}x{ny} ragged={int(ragged)} rounds={rounds}"
    try:
        xin = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem)
        gin = (
            ttnn.from_torch(g.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            if with_gamma
            else None
        )
        out = rms_norm(xin, gamma=gin, compute_kernel_config=_cfg(), memory_config=xin.memory_config())
        act = ttnn.to_torch(out).reshape(shape)
        exp = ref(x, g)
        assert_with_pcc(exp.to(torch.float32), act.to(torch.float32), 0.99)
        print(f"PASS  {tag}")
    except AssertionError as e:
        print(f"FAIL  {tag}  :: {str(e)[:120]}")
    except Exception as e:
        print(f"ERR   {tag}  :: {type(e).__name__}: {str(e)[:120]}")


device = ttnn.open_device(device_id=0)
try:
    shapes = [
        (2, 1, 128, 100),
        (128, 100),
        (4, 8, 32, 47),
        (1, 1, 17, 50),
        (2, 1, 100, 47),
        (1, 32, 50),
        (4, 128, 47),
        (1, 1, 64, 17),
        (32, 17),
    ]
    for s in shapes:
        for dt in (ttnn.bfloat16, ttnn.float32):
            for wg in (False, True):
                run(s, dt, wg)
finally:
    ttnn.close_device(device)
