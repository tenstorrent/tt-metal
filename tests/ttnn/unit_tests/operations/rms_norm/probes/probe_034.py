import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, default_compute_kernel_config

device = ttnn.open_device(device_id=0)


def cfg_loose():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def run(
    shape,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    glayout=ttnn.TILE_LAYOUT,
    gamma=True,
    cfg=None,
    levers=None,
    tag="",
):
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)
    g = None
    gt = None
    if gamma:
        gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
        g = ttnn.from_torch(gt, dtype=dtype, layout=glayout, device=device)
    out = rms_norm(x, gamma=g, compute_kernel_config=cfg or cfg_loose(), _levers=levers)
    got = ttnn.to_torch(out).float()
    ref = t / torch.sqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6)
    if gamma:
        ref = ref * gt
    a, b = got.flatten(), ref.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    print(f"{tag or shape} pcc={pcc:.6f}")


run((1, 1, 32, 7168), tag="focus G32")
run((1, 1, 32, 7168), levers=dict(w_split=0), tag="focus G1  ")
run((1, 1, 32, 1024), tag="decode_1024")
run((1, 1, 32, 2304), tag="decode_2304 G12")
run((1, 1, 32, 5120), tag="decode_5120 G20")
run((1, 1, 32, 7168), gamma=False, tag="focus no_gamma G16")
run((1, 1, 32, 7168), glayout=ttnn.ROW_MAJOR_LAYOUT, tag="grid_starved RMgamma")
run((1, 1, 8192, 7168), tag="prefill_7168 G4")
run((1, 1, 32, 4095), tag="w_nonalign G1")
run((1, 1, 100, 736), tag="h_nonalign G1")
run((1, 1, 1024, 7168), layout=ttnn.ROW_MAJOR_LAYOUT, glayout=ttnn.ROW_MAJOR_LAYOUT, tag="RM input 1024x7168")
run((1, 1, 32, 7168), dtype=ttnn.float32, cfg=default_compute_kernel_config(), tag="fp32 focus")
run((1, 1, 32, 7168), dtype=ttnn.bfloat8_b, tag="bf8b focus")

ttnn.close_device(device)
