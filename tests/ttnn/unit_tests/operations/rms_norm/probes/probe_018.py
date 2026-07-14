import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.metrics import check_output, CheckOutputError


def ref(x, gamma=None, eps=1e-6):
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(torch.bfloat16) if x.dtype == torch.bfloat16 else out.to(x.dtype)


def cfg(acc=True):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


TOL = {ttnn.bfloat16: (0.995, 0.04), ttnn.float32: (0.999, 0.02), ttnn.bfloat8_b: (0.99, 0.10)}


def run(name, shape, in_dtype, layout=ttnn.TILE_LAYOUT, gdt=ttnn.bfloat16, glay=ttnn.TILE_LAYOUT, acc=True):
    torch.manual_seed(0)
    W = shape[-1]
    td = torch.float32 if in_dtype == ttnn.float32 else torch.bfloat16
    ti = torch.randn(shape, dtype=td)
    xi = ttnn.from_torch(ti, dtype=in_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tg = None
    xg = None
    if gdt is not None:
        gtd = torch.float32 if gdt == ttnn.float32 else torch.bfloat16
        tg = torch.randn(W, dtype=gtd)
        xg = ttnn.from_torch(
            tg.reshape(1, 1, 1, W), dtype=gdt, layout=glay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    exp = ref(ti, gamma=tg)
    try:
        out = rms_norm(xi, gamma=xg, epsilon=1e-6, compute_kernel_config=cfg(acc))
        check_output(out, exp, shape=shape, dtype=in_dtype, expected_layout=layout, tolerance=TOL[in_dtype])
        print(f"[{name}] {shape} {in_dtype} -> PASS")
    except CheckOutputError as e:
        print(f"[{name}] {shape} {in_dtype} -> FAIL {str(e)[:130]}")
    except Exception as e:
        print(f"[{name}] {shape} {in_dtype} -> EXC {type(e).__name__}: {str(e)[:100]}")


device = ttnn.open_device(device_id=0)
try:
    # the 3 WIDE loose cases (bf16 TILE + bf16 TILE gamma)
    run("WIDE_16384", (1, 1, 32, 16384), ttnn.bfloat16)
    run("WIDE_32768", (1, 1, 32, 32768), ttnn.bfloat16)  # was the failure
    run("WIDE_12288", (1, 1, 64, 12288), ttnn.bfloat16)
    # regression spot-checks
    run("bf16_small", (1, 1, 64, 128), ttnn.bfloat16)
    run("bf16_4096", (1, 1, 32, 4096), ttnn.bfloat16)
    run("bf16_noacc", (1, 1, 64, 128), ttnn.bfloat16, acc=False)
    run("fp32_8192", (1, 1, 32, 8192), ttnn.float32, gdt=ttnn.float32)
    run("fp32_16384", (1, 1, 32, 16384), ttnn.float32, gdt=ttnn.float32)
    run("bf8b_256", (1, 1, 64, 256), ttnn.bfloat8_b, gdt=ttnn.bfloat8_b)
    run("bf8b_wide4096", (1, 1, 32, 4096), ttnn.bfloat8_b, gdt=ttnn.bfloat8_b)
    run("bf16_RM", (1, 1, 64, 128), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, glay=ttnn.ROW_MAJOR_LAYOUT)
finally:
    ttnn.close_device(device)
