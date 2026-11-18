# Refinement 2 probe: device-kernel ns on the feature_spec SHARDED perf loose cases at their exact config
# (bf16 / HiFi2 / fp32_dest_acc_en=False / TILE bf16 γ, pinned shard geometry). Baseline for Refinements 3/4.
# Run: TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
#      scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rms_norm/probes/perf_sharded_probe.py -s
import pytest, torch, ttnn
from ttnn.operations.rms_norm import rms_norm

ML = ttnn.TensorMemoryLayout
CASES = [  # rows, W, layout, shard [h, w], grid (x, y), achievable_ns @1350 MHz
    (32, 1024, ML.WIDTH_SHARDED, (32, 128), (8, 1), 4110),
    (32, 2304, ML.WIDTH_SHARDED, (32, 256), (9, 1), 4617),
    (32, 5120, ML.WIDTH_SHARDED, (32, 160), (8, 4), 5267),
    (32, 7168, ML.WIDTH_SHARDED, (32, 256), (7, 4), 5481),
    (640, 6144, ML.BLOCK_SHARDED, (64, 704), (9, 10), 19039),
    (640, 7168, ML.BLOCK_SHARDED, (64, 1024), (7, 10), 21669),
    (7168, 1024, ML.BLOCK_SHARDED, (736, 128), (8, 10), 18871),
    (7168, 2048, ML.BLOCK_SHARDED, (736, 192), (11, 10), 25947),
    (8192, 1024, ML.BLOCK_SHARDED, (1024, 128), (8, 8), 25640),
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
    rows, W, ml, sh, (gx, gy), ach = case
    torch.manual_seed(0)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    mc = ttnn.MemoryConfig(ml, ttnn.BufferType.L1, ttnn.ShardSpec(grid, list(sh), ttnn.ShardOrientation.ROW_MAJOR))
    x = torch.randn((1, 1, rows, W)).to(torch.bfloat16)
    g = torch.randn((1, 1, 1, W)).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False)
    rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=mc)
    ttnn.synchronize_device(device)
    _ns(device)
    rms_norm(tx, gamma=tg, compute_kernel_config=cfg, memory_config=mc)
    ttnn.synchronize_device(device)
    r = _ns(device)
    print(f"\nPERFPROBE {rows}x{W} {str(ml).split('.')[-1]} ns={r[-1] if r else None} achievable@1350={ach}")
