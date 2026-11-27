import time, statistics, torch, ttnn
from pathlib import Path
from ttnn.operations.g6_zeros_like import g6_zeros_like
from ttnn.operations.g6_zeros_like import g6_zeros_like_program_descriptor as pd

ABL = str(Path("tests/ttnn/unit_tests/operations/g6_zeros_like/probes/ablate_no_writes_writer.cpp").resolve())
REAL = pd.WRITER_KERNEL
device = ttnn.open_device(device_id=0, trace_region_size=8 << 20)
CONFIGS = {
    "head0": dict(MEM_ZEROS_HEAD_PAGES=0, WRITER_KERNEL=REAL),
    "headNone": dict(MEM_ZEROS_HEAD_PAGES=None, WRITER_KERNEL=REAL),
    "no_payload": dict(MEM_ZEROS_HEAD_PAGES=None, WRITER_KERNEL=ABL),
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
    for rep in range(2):
        for name, knobs in CONFIGS.items():
            for k, v in knobs.items():
                setattr(pd, k, v)
            pd._PROGRAM_CACHE.clear()
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
finally:
    ttnn.close_device(device)
