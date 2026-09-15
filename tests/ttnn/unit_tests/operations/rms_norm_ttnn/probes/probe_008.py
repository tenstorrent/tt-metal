# rms_norm_ttnn barrier-cadence A/B bench.
# Four cases x {native, generated}, ONE dispatch each (some cases accumulate a
# residual, so no trial loop), shape/dtype/eps/allocation-order copied from
# tests/ttnn/nightly/unit_tests/operations/fused/test_rmsnorm.py.
import hashlib
import os
import torch
import ttnn
from ttnn.operations import normalization as N

dev = ttnn.open_device(device_id=0)
torch.manual_seed(1234)

N_, C, H, W = 1, 9, 384, 1024
EPS = 1e-2
L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DR = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

x = torch.rand((N_, C, H, W)) * 2 - 0.95
gamma = torch.rand(1, 1, 1, W) * 2 - 1
beta = torch.rand(1, 1, 1, W) * 2.0 - 1.1


def device_ns():
    per = ttnn.get_latest_programs_perf_data()
    return sum(
        float(pr.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration)
        for v in (per or {}).values()
        for pr in v
        if "DEVICE KERNEL DURATION [ns]" in (pr.program_analyses_results or {})
    )


def run(op, label, in_mc, out_mc, gb):
    # test_rmsnorm.py allocates gamma/beta BEFORE x; the writeup measured a 5.8us
    # swing from L1 allocation order alone, so keep the test's order exactly.
    tg = tb = None
    if gb:
        tg = ttnn.from_torch(
            gamma.expand(1, 1, 32, W).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=in_mc,
        )
        tb = ttnn.from_torch(
            beta.expand(1, 1, 32, W).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=in_mc,
        )
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=in_mc)
    ttnn.ReadDeviceProfiler(dev)
    kw = dict(epsilon=EPS, memory_config=out_mc)
    if gb:
        kw["weight"] = tg
        kw["bias"] = tb
    out = op(tx, **kw)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    ns = device_ns()
    t = ttnn.to_torch(out).to(torch.float32)

    ref = x.to(torch.float32)
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + EPS)
    if gb:
        ref = ref * gamma.to(torch.float32) + beta.to(torch.float32)
    pcc = float(torch.corrcoef(torch.stack([t.flatten(), ref.flatten()]))[0, 1])
    h = hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]  # t is the lossless f32 view of a bf16 output
    print(f"RESULT {label:34s} {ns:9.0f} ns  PCC {pcc:.6f}  sha {h}", flush=True)
    for tt in (out, tx, tg, tb):
        if tt is not None:
            tt.deallocate()
    return ns


CASES = [
    ("RMSN     L1->L1  ", L1, L1, False),
    ("RMSN     L1->DRAM", L1, DR, False),
    ("RMSN_GB  L1->L1  ", L1, L1, True),
    ("RMSN     DRAM->L1", DR, L1, False),
]
tag = os.environ.get("BENCH_TAG", "run")
res = {}
for name, imc, omc, gb in CASES:
    nat = run(N._native_rms_norm, f"native    {name}", imc, omc, gb)
    gen = run(ttnn.rms_norm, f"{tag:9s} {name}", imc, omc, gb)
    res[name] = (nat, gen)
print(flush=True)
print(f"{'case':20s} {'native':>9s} {tag:>9s} {'ratio nat/gen':>14s}", flush=True)
for name, (nat, gen) in res.items():
    print(f"{name:20s} {nat:9.0f} {gen:9.0f} {nat/gen:14.3f}", flush=True)
ttnn.close_device(dev)
print("PROBE_OK", flush=True)
