import sys, torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def loose():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def run(shape, levers=None, tag=""):
    print("CASE start:", tag, flush=True)
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gt = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)
    g = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = rms_norm(x, gamma=g, compute_kernel_config=loose(), _levers=levers)
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
run(S, dict(w_group=56), "w_group=56")
run(S, dict(w_group=2), "w_group=2")
run(S, dict(active_cores=32), "A0 off")
run(S, dict(row_wise=0), "A1 off")
run(S, dict(double_buffer=0), "C16 off")
run(S, dict(coalesce=0), "B5/B6 off")
run(S, dict(barrier_per_block=0), "B7 off")
run(S, dict(noc_split=0), "B9 off")
run(S, dict(block_ht=1, dest_block=1), "compute_block_size off")
run(S, dict(coarse_chunk=0), "coarse_chunk off")
run(S, dict(wt_block=8), "wt_block=8")
run(S, dict(acc_narrow=0), "acc_narrow off")
run(S, dict(reduce_via_add=0), "reduce_via_add off")
ttnn.close_device(device)
