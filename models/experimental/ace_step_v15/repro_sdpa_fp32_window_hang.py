#!/usr/bin/env python3
"""Minimal reproducer for TRAP-13: SDPA deadlocks on `fp32_dest_acc_en` + `sliding_window_size`.

One case per process, because a stall's SIGTERM contaminates the next run.

    export TT_METAL_HOME=<tt-metal>  PYTHONPATH=<tt-metal>
    python repro_sdpa_fp32_window_hang.py <fidelity> <fp32_dest_acc> <packer_l1_acc> <window|none>

    timeout 100 python -u repro_sdpa_fp32_window_hang.py HiFi2 1 0 256    # HANGS (exit 124)
    timeout 100 python -u repro_sdpa_fp32_window_hang.py HiFi2 0 0 256    # ok
    timeout 100 python -u repro_sdpa_fp32_window_hang.py HiFi2 0 1 256    # ok
    timeout 100 python -u repro_sdpa_fp32_window_hang.py HiFi2 1 1 none   # ok
    timeout 100 python -u repro_sdpa_fp32_window_hang.py HiFi4 1 0 256    # HANGS (exit 124)

Measured on Wormhole b0, q[1,16,128,128] / k,v[1,8,128,128], bf16, DRAM-interleaved,
is_causal=False, program_config=None. It is specifically the fp32-dest-acc x window interaction:
`packer_l1_acc` and `math_fidelity` are irrelevant, and either knob alone is safe.

The op enqueues fine; the **readback** of its output never returns, and `TT_METAL_WATCHER`
reports no stuck core, no assert and no pending NOC transaction. Consistent with a
circular-buffer wait that is never satisfied (a core correctly blocked on a semaphore is not
something the watcher flags), though that mechanism is **unverified**.

⚠ **`tt-smi -r` between invocations.** A killed run leaves the card degraded-but-openable:
`open_device` still returns in ~0.8 s while ops then hang at the first sync, so batching these
cases in a loop produces one real data point followed by garbage.

    ( source /localdev/acicovic/tt-xla/venv/bin/activate && tt-smi -r )

Downstream fix: `sdpa_compute_config` in `tt/ttnn_ace_step_common.py` sets
`fp32_dest_acc_en=False`. The window cannot be dropped instead -- TRAP-1 shows that silently
degrades PCC to 0.762. Not yet reported upstream; draft issue text in
model-bringup/ace_step_1_5/ISSUE_sdpa_fp32_window_hang.md.
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
