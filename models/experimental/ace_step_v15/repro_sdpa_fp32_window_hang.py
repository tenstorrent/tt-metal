#!/usr/bin/env python3
"""One SDPA compute-kernel-config case per process, so a stall cannot contaminate the next.

    python sdpa_matrix.py <fidelity> <fp32_dest_acc> <packer_l1_acc> <window|none>
e.g. python sdpa_matrix.py HiFi2 1 0 256
"""
import sys, time, torch, ttnn

NQ, NKV, D, S = 16, 8, 128, 128
fid_s, fp32_s, packer_s, win_s = sys.argv[1:5]
t0 = time.time()

dev = ttnn.open_device(device_id=0, l1_small_size=65536)
cfg = ttnn.init_device_compute_kernel_config(
    dev.arch(),
    math_fidelity=getattr(ttnn.MathFidelity, fid_s),
    math_approx_mode=False,
    fp32_dest_acc_en=bool(int(fp32_s)),
    packer_l1_acc=bool(int(packer_s)),
)

torch.manual_seed(0)
to = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
q, k, v = to(torch.randn(1, NQ, S, D)), to(torch.randn(1, NKV, S, D)), to(torch.randn(1, NKV, S, D))

kw = dict(is_causal=False, scale=D**-0.5, compute_kernel_config=cfg)
if win_s != "none":
    kw["sliding_window_size"] = int(win_s)

print(f"CASE fid={fid_s} fp32={fp32_s} packer={packer_s} window={win_s}", flush=True)
try:
    r = ttnn.transformer.scaled_dot_product_attention(q, k, v, **kw)
    print("  enqueued; syncing", flush=True)
    h = ttnn.to_torch(r)
    print(f"  RESULT OK sum={h.float().sum().item():.3f} ({time.time()-t0:.2f}s)", flush=True)
except Exception as e:
    print(f"  RESULT RAISED {type(e).__name__}: {str(e)[:200]}", flush=True)
finally:
    ttnn.close_device(dev)
