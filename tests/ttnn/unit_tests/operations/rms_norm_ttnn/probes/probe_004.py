import torch, ttnn
from ttnn.operations import normalization as N
dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 3456, 1024, dtype=torch.float32).to(torch.bfloat16)
g = torch.randn(1, 1, 1, 1024, dtype=torch.float32).to(torch.bfloat16)
L1I = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
def run(op, mc, label):
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.ReadDeviceProfiler(dev)
    out = op(tx, epsilon=1e-5, weight=tg, memory_config=mc)
    ttnn.synchronize_device(dev); ttnn.ReadDeviceProfiler(dev)
    per = ttnn.get_latest_programs_perf_data()
    tot = sum(float(pr.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration)
              for v in (per or {}).values() for pr in v
              if "DEVICE KERNEL DURATION [ns]" in (pr.program_analyses_results or {}))
    t = ttnn.to_torch(out).to(torch.float32)
    ref = x.to(torch.float32); ref = ref*torch.rsqrt(ref.pow(2).mean(-1,keepdim=True)+1e-5)*g.to(torch.float32)
    pcc = float(torch.corrcoef(torch.stack([t.flatten(), ref.flatten()]))[0,1])
    print(f"CASE {label:36s} {tot:9.0f} ns   PCC {pcc:.6f}", flush=True)
    out.deallocate(); tx.deallocate(); tg.deallocate(); return tot
a = run(N._native_rms_norm, L1I, "native    L1-interleaved")
b = run(ttnn.rms_norm,      L1I, "generated L1-interleaved (FIXED)")
print(f"ratio native/generated L1 = {a/b:.3f}x", flush=True)
c = run(N._native_rms_norm, ttnn.DRAM_MEMORY_CONFIG, "native    DRAM")
d = run(ttnn.rms_norm,      ttnn.DRAM_MEMORY_CONFIG, "generated DRAM (must be unchanged)")
print(f"DRAM ratio native/generated = {c/d:.3f}x", flush=True)
ttnn.close_device(dev); print("PROBE_OK", flush=True)
