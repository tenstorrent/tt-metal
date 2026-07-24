# Self-contained repro for the SDPA KV-reuse multicast intermittent hang.
# Run via:  TTNN_SDPA_KV_MCAST=1 scripts/tt-probe.sh --dev scaled_dot_product_attention < repro_mcast_hang.py
# Needs only this outer clone (no eval submodule / golden harness). torch.randn inputs
# are sufficient — the hang is in the mcast handshake, independent of the data.
import os, torch, ttnn

N = int(os.environ.get("STRESS_ITERS", "40"))
dev = ttnn.open_device(device_id=0)
try:
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa
    B, H, S, D = 1, 10, 9472, 128           # b*H_q = 10 == grid_rows -> mcast-eligible on 11x10 BH
    torch.manual_seed(1234)
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)   # fp32 reference
    to = lambda x: ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tq, tk, tv = to(q), to(k), to(v)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    print(f"KV_MCAST={os.environ.get('TTNN_SDPA_KV_MCAST')} iters={N} shape={(B,H,S,D)}", flush=True)
    for i in range(N):
        out = sdpa(tq, tk, tv, compute_kernel_config=cfg)
        ttnn.synchronize_device(dev)
        if i in (0, N - 1):
            ot = ttnn.to_torch(out).to(torch.float32)
            pcc = torch.corrcoef(torch.stack([ref.flatten(), ot.flatten()]))[0, 1].item()
            print(f"  iter {i}: PCC={pcc:.6f}", flush=True)
        else:
            print(f"  iter {i}: ok", flush=True)
    print(f"STRESS_DONE: {N} iterations, NO HANG", flush=True)
finally:
    ttnn.close_device(dev)
