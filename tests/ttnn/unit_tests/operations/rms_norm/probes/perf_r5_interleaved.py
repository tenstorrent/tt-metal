# Refinement 5 probe: device-kernel ns on the 12 interleaved perf loose-case shapes at their EXACT config
# (bf16, HiFi2, fp32_dest_acc_en=False, TILE bf16 γ). Knob overrides: EVAL_R5_KNOBS="NAME=val,NAME=val" (module constants
# of rms_norm_program_descriptor). EVAL_R5_SHAPES="32x1024,8192x7168" filters.
# Run: TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
#      scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/rms_norm/probes/perf_r5_interleaved.py -s
import os
import pytest, torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

SHAPES = [(32, 1024), (32, 2304), (32, 5120), (32, 7168), (640, 6144), (640, 7168),
          (7168, 1024), (7168, 2048), (8192, 1024), (8192, 2304), (8192, 5120), (8192, 7168)]
ACH = dict(zip(SHAPES, [9149, 17003, 75825, 14894, 101968, 119307, 89392, 172892, 96744, 211345, 738307, 1032281]))
_flt = os.environ.get("EVAL_R5_SHAPES")
if _flt:
    SHAPES = [s for s in SHAPES if f"{s[0]}x{s[1]}" in _flt.split(",")]


def _knobs():
    for kv in filter(None, os.environ.get("EVAL_R5_KNOBS", "").split(",")):
        k, v = kv.split("=")
        setattr(pd, k, type(getattr(pd, k))(eval(v)))


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
    _knobs()
    rows, W = shape
    torch.manual_seed(0)
    x = torch.randn((1, 1, rows, W)).to(torch.bfloat16)
    g = torch.randn((1, 1, 1, W)).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False)
    out = rms_norm(tx, gamma=tg, compute_kernel_config=cfg)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * g.float()
    y = ttnn.to_torch(out).float()
    pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].item()
    ns = None
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        ttnn.synchronize_device(device)
        _ns(device)
        rms_norm(tx, gamma=tg, compute_kernel_config=cfg)
        ttnn.synchronize_device(device)
        r = _ns(device)
        ns = r[-1] if r else None
    gq = pd.OpGeometry(is_tile=True, has_gamma=True, gamma_is_rm=False, t_in=2048, t_g=2048, es_g=2, t_int=2048)
    grid = device.compute_with_storage_grid_size()
    b = pd.select_blocking(gq, rows // 32, W // 32, grid.x * grid.y)
    print(f"\nR5PERF {rows}x{W} ns={ns} ach={ACH[shape]} pcc={pcc:.6f} k={b.w_split} cw={b.core_w_tiles} "
          f"lines={b.num_lines} br={b.block_rows} knobs={os.environ.get('EVAL_R5_KNOBS','')}")
    assert pcc >= 0.9995, pcc
