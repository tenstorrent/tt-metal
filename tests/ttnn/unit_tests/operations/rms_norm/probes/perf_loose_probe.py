# Verifier probe: device-kernel ns on the feature_spec perf loose-case shapes (interleaved), at the
# nearest SUPPORTED config (fp32_dest_acc_en=True, HiFi2) -- a headroom estimate, not the target config.
import os
import pytest, torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import OpGeometry, select_blocking

SHAPES = [(32, 1024), (32, 2304), (32, 5120), (32, 7168), (640, 6144), (640, 7168),
          (7168, 1024), (7168, 2048), (8192, 1024), (8192, 2304), (8192, 5120), (8192, 7168)]


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


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_perf(device, shape):
    rows, W = shape
    torch.manual_seed(0)
    x = torch.randn((1, 1, rows, W)).to(torch.bfloat16)
    g = torch.randn((1, 1, 1, W)).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False)
    rms_norm(tx, gamma=tg, compute_kernel_config=cfg)
    ttnn.synchronize_device(device)
    _ns(device)
    best = None
    for _ in range(3):
        rms_norm(tx, gamma=tg, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        r = _ns(device)
        if r:
            best = r[-1] if best is None or r[-1][0] < best[0] else best
    print(f"\nPERFPROBE {rows}x{W} ns={best}")
