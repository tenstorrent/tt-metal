import time, statistics, torch, ttnn
from ttnn.operations.g6_zeros_like import g6_zeros_like
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd

device = ttnn.open_device(device_id=0, trace_region_size=8 << 20)
CONFIGS = {
    "head0": dict(MEM_ZEROS_HEAD_PAGES=0),
    "head2": dict(MEM_ZEROS_HEAD_PAGES=2),
    "headNone": dict(MEM_ZEROS_HEAD_PAGES=None),
}
N = 200
try:
    x = ttnn.from_torch(
        torch.zeros((1, 12, 128, 128)),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    target = ttnn.allocate_tensor_on_device(x.spec, device)
    for rep in range(3):
        for name, knobs in CONFIGS.items():
            for k, v in knobs.items():
                setattr(pd, k, v)
            g6_zeros_like(x, optional_tensor=target)
            ttnn.synchronize_device(device)
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            for _ in range(N):
                g6_zeros_like(x, optional_tensor=target)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.synchronize_device(device)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(device)
                ts.append((time.perf_counter() - t0) / N * 1e6)
            ttnn.release_trace(device, tid)
            print(f"CFG rep{rep} {name}: {statistics.median(ts):.3f} us/call (min {min(ts):.3f})")
    assert torch.count_nonzero(ttnn.to_torch(target)) == 0
finally:
    ttnn.close_device(device)
