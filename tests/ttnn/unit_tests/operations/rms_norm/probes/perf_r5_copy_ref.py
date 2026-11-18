# Refinement 5 probe: DRAM-traffic reference — device-kernel ns of ttnn.clone / ttnn.relu (read x once, write y once:
# the same DRAM bytes as rms_norm) on the interleaved perf shapes. Bounds what a pure-DM op reaches on this board.
import pytest, torch, ttnn

SHAPES = [(640, 6144), (7168, 1024), (8192, 1024), (8192, 5120), (8192, 7168)]


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


@pytest.mark.parametrize("op", ["clone", "relu"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_copy_ref(device, shape, op):
    rows, W = shape
    x = torch.randn((1, 1, rows, W)).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    f = (lambda t: ttnn.clone(t)) if op == "clone" else (lambda t: ttnn.relu(t))
    f(tx)
    ttnn.synchronize_device(device)
    _ns(device)
    f(tx)
    ttnn.synchronize_device(device)
    r = _ns(device)
    ns = r[-1] if r else None
    gbps = 2 * rows * W * 2 / ns[0] if ns else 0
    print(f"\nR5COPY {op} {rows}x{W} ns={ns} GB/s={gbps:.0f}")
