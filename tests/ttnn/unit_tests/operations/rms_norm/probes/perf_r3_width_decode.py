# Refinement 3 probe: device-kernel ns of the four WIDTH_SHARDED decode perf loose cases at their exact config
# (bf16 / HiFi2 / fp32_dest_acc_en=False / TILE bf16 γ, pinned shard geometry) + PCC at the 0.9995 gate.
# Run: TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
#      scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rms_norm/probes/perf_r3_width_decode.py -s
import os
import pytest, torch, ttnn
from ttnn.operations.rms_norm import rms_norm

ML = ttnn.TensorMemoryLayout
CASES = [  # rows, W, shard [h, w], grid (x, y), achievable_ns @1350 MHz
    (32, 1024, (32, 128), (8, 1), 4110),
    (32, 2304, (32, 256), (9, 1), 4617),
    (32, 5120, (32, 160), (8, 4), 5267),
    (32, 7168, (32, 256), (7, 4), 5481),
]


def _ns(device):
    ttnn.ReadDeviceProfiler(device)
    data = ttnn.get_latest_programs_perf_data()
    out = []
    for _c, progs in (data or {}).items():
        for p in progs:
            r = p.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
            if r is not None:
                out.append((int(r.duration), int(getattr(p, "core_count", 0) or 0)))
    return out


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}x{c[1]}")
def test_perf(device, case):
    rows, W, sh, (gx, gy), ach = case
    torch.manual_seed(0)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    mc = ttnn.MemoryConfig(ML.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(grid, list(sh), ttnn.ShardOrientation.ROW_MAJOR))
    x = torch.randn((1, 1, rows, W)).to(torch.bfloat16)
    g = torch.randn((1, 1, 1, W)).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False)
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=mc)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * g.float()
    y = ttnn.to_torch(out).float()
    pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].item()
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        ttnn.synchronize_device(device)
        _ns(device)
        rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=mc)
        ttnn.synchronize_device(device)
        r = _ns(device)
        print(f"\nPERFR3 {rows}x{W} grid={gx}x{gy} ns={r[-1] if r else None} achievable@1350={ach} pcc={pcc:.6f}")
    assert pcc >= 0.9995, pcc
