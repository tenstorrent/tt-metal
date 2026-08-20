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
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    g = ttnn.from_torch(gt, dtype=dtype, layout=glayout, device=device)
    out = rms_norm(x, gamma=g, compute_kernel_config=cfg or loose(), _levers=levers)
    got = ttnn.to_torch(out).float()
    ref = t / torch.sqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * gt
    a, b = got.flatten(), ref.flatten()
    print(f"{tag:40s} pcc={torch.corrcoef(torch.stack([a,b]))[0,1].item():.6f}")
    ttnn.deallocate(x)
    ttnn.deallocate(g)
    ttnn.deallocate(out)


run((1, 1, 32, 7168), levers=dict(w_group=56), tag="focus w_group=56")
run((1, 1, 32, 7168), levers=dict(w_group=2), tag="focus w_group=2")
run((1, 1, 32, 7168), levers=dict(active_cores=32), tag="focus A0 off (39 cores)")
run((1, 1, 32, 7168), levers=dict(row_wise=0), tag="focus A1 off")
run((1, 1, 32, 7168), levers=dict(double_buffer=0), tag="focus C16 off")
run((1, 1, 32, 7168), levers=dict(coalesce=0), tag="focus B5/B6 off")
run((1, 1, 32, 7168), levers=dict(barrier_per_block=0), tag="focus B7 off")
run((1, 1, 32, 7168), levers=dict(noc_split=0), tag="focus B9 off")
run((1, 1, 32, 7168), levers=dict(block_ht=1, dest_block=1), tag="focus compute_block_size off")
run((1, 1, 32, 7168), levers=dict(coarse_chunk=0), tag="focus coarse_chunk off")
run((1, 1, 32, 7168), levers=dict(wt_block=8), tag="focus wt_block=8")
run((1, 1, 32, 7168), levers=dict(acc_narrow=0), tag="focus acc_narrow off")
run((1, 1, 32, 7168), levers=dict(reduce_via_add=0), tag="focus reduce_via_add off")
run((1, 1, 100, 7168), tag="h_non_aligned wide G16")
run((1, 1, 1024, 7168), layout=ttnn.ROW_MAJOR_LAYOUT, glayout=ttnn.ROW_MAJOR_LAYOUT, tag="RM 1024x7168 G4")
run((1, 1, 32, 32768), tag="very wide (1,1,32,32768)")
run((32, 7168), tag="rank2 (32,7168)")
run((1, 32, 7168), tag="rank3 (1,32,7168)")
run((1, 1, 64, 12288), tag="(1,1,64,12288)")
ttnn.close_device(device)
