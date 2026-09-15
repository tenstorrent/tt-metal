# D41's residual arm: two streams in ONE read loop, so the transaction cap halves
# the group in `w` (4 instead of 8).  Untested by the golden/nightly suites with an
# L1 input, so exercise it here -- correctness first, then the timing.
import torch, ttnn

dev = ttnn.open_device(device_id=0)
torch.manual_seed(1234)
N_, C, H, W = 1, 9, 384, 1024
EPS = 1e-2
MC = {
    "L1": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
    "DRAM": ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
}
x = torch.rand((N_, C, H, W)) * 2 - 0.95
r = torch.rand((N_, C, H, W)) * 2 - 0.95

for place in ("L1", "DRAM"):
    mc = MC[place]
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    tr = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    ttnn.ReadDeviceProfiler(dev)
    out = ttnn.rms_norm(tx, epsilon=EPS, residual_input_tensor=tr, memory_config=MC["L1"])
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    per = ttnn.get_latest_programs_perf_data()
    ns = sum(
        float(pr.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration)
        for v in (per or {}).values()
        for pr in v
        if "DEVICE KERNEL DURATION [ns]" in (pr.program_analyses_results or {})
    )
    t = ttnn.to_torch(out).to(torch.float32)
    ref = (x + r).to(torch.float32)
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + EPS)
    pcc = float(torch.corrcoef(torch.stack([t.flatten(), ref.flatten()]))[0, 1])
    print(f"RESIDRESULT in={place:4s} {ns:9.0f} ns  PCC {pcc:.6f}  maxabs {(t-ref).abs().max():.5f}", flush=True)
    for tt in (out, tx, tr):
        tt.deallocate()
ttnn.close_device(dev)
print("PROBE_OK", flush=True)
