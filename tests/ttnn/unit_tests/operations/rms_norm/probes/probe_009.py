import torch, ttnn
from tests.ttnn.utils_for_testing import comp_pcc
from ttnn.operations.rms_norm import rms_norm


def cfg():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = True
    c.math_approx_mode = False
    return c


def ref(x, g=None, eps=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    o = xf / rms
    if g is not None:
        o = o * g.to(torch.float32).reshape(-1)
    return o


device = ttnn.open_device(device_id=0)
try:
    cases = [
        ("RM   bf16 Hnonalign nogamma", (1, 1, 17, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, False),
        ("RM   fp32 Hnonalign nogamma", (1, 1, 17, 64), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, False),
        ("RM   fp32 bothnonalign gamma", (2, 1, 100, 47), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, True),
        ("RM   bf16 bothnonalign gamma", (2, 1, 100, 47), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True),
        ("RM   bf16 Hnonalign gamma", (4, 8, 47, 256), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True),
        ("RM   bf16 aligned gamma", (1, 1, 64, 128), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True),
        ("TILE bf16 bothnonalign gamma", (2, 1, 100, 47), ttnn.bfloat16, ttnn.TILE_LAYOUT, True),
        ("RM   bf16 wide gamma", (1, 1, 32, 4096), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True),
    ]
    pcc_by = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
    for name, shape, dt, layout, wg in cases:
        torch.manual_seed(42)
        tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
        x = torch.randn(shape, dtype=tdt)
        W = shape[-1]
        ti = ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        g = None
        tg = None
        if wg:
            tg = torch.randn(W, dtype=tdt)
            g = ttnn.from_torch(
                tg.reshape(1, 1, 1, W),
                dtype=dt,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        exp = ref(x, tg, 1e-6)
        try:
            out = rms_norm(ti, gamma=g, epsilon=1e-6, compute_kernel_config=cfg())
            act = ttnn.to_torch(out).reshape(exp.shape).to(torch.float32)
            ok, msg = comp_pcc(exp.to(torch.float32), act, pcc_by[dt])
            print(f"{'PASS' if ok else 'FAIL'} {name}  {msg}")
        except Exception as e:
            print(f"ERR  {name}  {str(e)[:150]}")
finally:
    ttnn.close_device(device)
