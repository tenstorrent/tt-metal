# Refinement 3 guard probe: device-kernel ns on multi-block / other-placement shapes, to check the W_SPLIT levers
# (scoped finalize, face-0/2 gather, handshake-free mcast, x·γ prescale) do not regress them. `xg` A/Bs the
# prescale knob (XG_PRESCALE) against its parked (pre-R3 schedule) value.
# Run: TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
#      scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rms_norm/probes/perf_r3_guard.py -s
import os
import pytest, torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

ML = ttnn.TensorMemoryLayout
CASES = [  # rows, W, layout, placement, shard [h, w], grid (x, y)
    (640, 6144, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, (64, 704), (9, 10)),
    (640, 7168, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, (64, 1024), (7, 10)),
    (7168, 1024, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, (736, 128), (8, 10)),
    (7168, 2048, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, (736, 192), (11, 10)),
    (8192, 1024, ttnn.TILE_LAYOUT, ML.BLOCK_SHARDED, (1024, 128), (8, 8)),
    (1024, 4096, ttnn.TILE_LAYOUT, ML.HEIGHT_SHARDED, (32, 4096), (8, 4)),
    (32, 4096, ttnn.ROW_MAJOR_LAYOUT, ML.WIDTH_SHARDED, (32, 512), (8, 1)),
    (32, 1024, ttnn.TILE_LAYOUT, ML.INTERLEAVED, None, None),
    (32, 4096, ttnn.TILE_LAYOUT, ML.INTERLEAVED, None, None),
    (32, 7168, ttnn.TILE_LAYOUT, ML.INTERLEAVED, None, None),
    (32, 4096, ttnn.ROW_MAJOR_LAYOUT, ML.INTERLEAVED, None, None),
    (8192, 7168, ttnn.TILE_LAYOUT, ML.INTERLEAVED, None, None),
    (7168, 1024, ttnn.TILE_LAYOUT, ML.INTERLEAVED, None, None),
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


@pytest.mark.parametrize("xg", [True, False], ids=["xg", "noxg"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}x{c[1]}-{str(c[2]).split('.')[-1]}-{str(c[3]).split('.')[-1]}")
def test_guard(device, case, xg):
    rows, W, layout, ml, sh, grid = case
    saved = getattr(pd, "XG_PRESCALE", True)
    pd.XG_PRESCALE = xg
    torch.manual_seed(0)
    if ml == ML.INTERLEAVED:
        mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        gx, gy = grid
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        mc = ttnn.MemoryConfig(ml, ttnn.BufferType.L1, ttnn.ShardSpec(crs, list(sh), ttnn.ShardOrientation.ROW_MAJOR))
    x = torch.randn((1, 1, rows, W)).to(torch.bfloat16)
    g = torch.randn((1, 1, 1, W)).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
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
        print(f"\nPERFGUARD {rows}x{W} {str(layout).split('.')[-1]} {str(ml).split('.')[-1]} xg={pd.XG_PRESCALE} ns={r[-1] if r else None} pcc={pcc:.6f}")
    pd.XG_PRESCALE = saved
    assert pcc >= 0.999, pcc
