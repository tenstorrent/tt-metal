import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)


def ref(x, g):
    x = x.float()
    r = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)
    return r * g.float().reshape(-1) if g is not None else r


def run(
    shape,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    gdt=ttnn.bfloat16,
    gl=ttnn.TILE_LAYOUT,
    fp32=False,
    fid=ttnn.MathFidelity.HiFi4,
    gamma=True,
    mc=ttnn.DRAM_MEMORY_CONFIG,
):
    torch.manual_seed(0)
    x = torch.randn(shape).bfloat16()
    g = torch.randn(shape[-1]).bfloat16() if gamma else None
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=dev, memory_config=mc)
    tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=gdt, layout=gl, device=dev) if gamma else None
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fid
    c.fp32_dest_acc_en = fp32
    c.math_approx_mode = False
    try:
        y = ttnn.to_torch(rms_norm(tx, gamma=tg, compute_kernel_config=c)).float()
    except Exception as ex:
        print(f"{shape} {dtype} g={gdt} fp32={fp32}: Error {ex}"[:300])
        return
    e = ref(ttnn.to_torch(tx).float(), ttnn.to_torch(tg).float().reshape(-1) if gamma else None)
    pcc = torch.corrcoef(torch.stack([y.flatten(), e.flatten()]))[0, 1].item()
    rms = ((y - e).pow(2).mean().sqrt() / e.std()).item()
    print(
        f"{shape} {dtype} {layout} g={gdt if gamma else None} fp32={fp32} {fid}: pcc={pcc:.6f} rrms={rms:.4f} ratio={(y/e)[e.abs()>0.5].median():.4f}",
        flush=True,
    )


run((1, 1, 256, 7168))
run((1, 1, 256, 7168), dtype=ttnn.bfloat8_b, gdt=ttnn.bfloat8_b)
run((1, 1, 8192, 7168), fid=ttnn.MathFidelity.HiFi2)
run((1, 1, 64, 1024))
run((1, 1, 50, 47))
run((1, 1, 50, 47), layout=ttnn.ROW_MAJOR_LAYOUT, gl=ttnn.ROW_MAJOR_LAYOUT)
run((1, 1, 32, 4095))
run((1, 1, 64, 1024), dtype=ttnn.bfloat8_b, gamma=False)
run((1, 1, 64, 1024), fp32=True)
ttnn.close_device(dev)
