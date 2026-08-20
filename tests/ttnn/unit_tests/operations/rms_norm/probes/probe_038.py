import torch, ttnn
from ttnn.operations.rms_norm import rms_norm, default_compute_kernel_config

device = ttnn.open_device(device_id=0)


def loose():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def run(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, glayout=ttnn.TILE_LAYOUT, cfg=None, levers=None, tag=""):
    print("CASE start:", tag, flush=True)
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    g = ttnn.from_torch(gt, dtype=dtype, layout=glayout, device=device)
    out = rms_norm(x, gamma=g, compute_kernel_config=cfg or loose(), _levers=levers)
    got = ttnn.to_torch(out).float()
    ref = t / torch.sqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * gt
    print(
        f"CASE done: {tag:34s} pcc={torch.corrcoef(torch.stack([got.flatten(),ref.flatten()]))[0,1].item():.6f}",
        flush=True,
    )
    ttnn.deallocate(x)
    ttnn.deallocate(g)
    ttnn.deallocate(out)


S = (1, 1, 32, 7168)
RM = ttnn.ROW_MAJOR_LAYOUT
for lv, tag in [
    (dict(noc_split=0), "B9 off (P5 -> G=1)"),
    (dict(noc_split=0, w_split=0), "B9 off + w_split=0"),
    (dict(w_split=0), "w_split off"),
    (dict(w_group=56), "w_group=56"),
    (dict(w_group=28), "w_group=28"),
    (dict(w_group=16), "w_group=16"),
    (dict(w_group=8), "w_group=8"),
    (dict(block_ht=1, dest_block=1), "compute_block_size off"),
    (dict(coarse_chunk=0), "coarse_chunk off"),
    (dict(wt_block=8), "wt_block=8"),
    (dict(acc_narrow=0), "acc_narrow off"),
    (dict(reduce_via_add=0), "reduce_via_add off"),
    (dict(stub_dm=1), "stub_dm"),
    (dict(stub_compute=1), "stub_compute"),
    (dict(stub_dm=1, stub_compute=1), "stub_both"),
]:
    run(S, levers=lv, tag=tag)
run((1, 1, 100, 7168), tag="h_non_aligned wide G16")
run((1, 1, 1024, 7168), layout=RM, glayout=RM, tag="RM 1024x7168 G4")
run((1, 1, 8192, 1024), layout=RM, glayout=RM, tag="RM 8192x1024 G1")
run((1, 1, 32, 32768), tag="very wide 32768")
run((32, 7168), tag="rank2")
run((1, 32, 7168), tag="rank3")
run((1, 1, 64, 12288), tag="64x12288")
run((1, 1, 32, 4095), tag="w_nonalign")
run((1, 1, 100, 736), tag="h_nonalign")
run((32, 17), tag="smallest")
run(S, dtype=ttnn.float32, cfg=default_compute_kernel_config(), tag="fp32 focus G32")
run(S, dtype=ttnn.bfloat8_b, tag="bf8b focus G32")
ttnn.close_device(device)
