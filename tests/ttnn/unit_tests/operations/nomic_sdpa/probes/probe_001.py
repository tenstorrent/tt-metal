import torch, ttnn
from eval.golden_tests.nomic_sdpa.helpers import make_qkv, reference
from ttnn.operations.nomic_sdpa import nomic_sdpa

device = ttnn.open_device(device_id=0)
try:
    B, S = 8, 512
    q, k, v = make_qkv((B, 12, S, 64), "real", seed=0)
    ref = reference(q, k, v, 1.0).reshape(B, 1, S, 768)
    tq, tk, tv = (
        ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        for t in (q, k, v)
    )
    for fid in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi4):
        for approx in (True, False):
            cfg = ttnn.ComputeConfigDescriptor(
                math_fidelity=fid, fp32_dest_acc_en=False, math_approx_mode=approx, dst_full_sync_en=False
            )
            out = nomic_sdpa(tq, tk, tv, compute_kernel_config=cfg)
            got = ttnn.to_torch(out).reshape(B, 1, S, 768).double()
            err = got - ref
            rel = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
            # bias: does the output shrink?
            ratio = ((got * ref).sum() / (ref * ref).sum()).item()
            print(f"CFG fid={fid} approx={approx} rel_rms={rel:.5f} gain={ratio:.5f}")
            ttnn.deallocate(out)
    # emulate: bf16 P exact exp, fp64 matmul
    sc = (q.double() @ k.double().transpose(-1, -2)).bfloat16().double()
    m = sc.amax(-1, keepdim=True)
    P = torch.exp(sc - m).bfloat16().double()
    O = (P @ v.double()) / P.sum(-1, keepdim=True)
    O = O.permute(0, 2, 1, 3).reshape(B, 1, S, 768)
    e = O - ref
    print("CFG emul bf16-scores bf16-P fp64-acc rel_rms", (e.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item())
finally:
    ttnn.close_device(device)
