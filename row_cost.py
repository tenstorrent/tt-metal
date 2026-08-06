"""Does op cost track rows or bytes? The answer decides how the fused band kernel must be shaped.

Evidence that prompted this: in op_floor.py, 331212 rows took 2.824 ms in fp32 and 2.803 ms in
bfloat16 -- the same time for half the bytes. If cost tracks rows rather than bytes, then the audio
tail is penalised not for being large but for being *narrow*: at C=8 a row is 32 bytes, and the decode
does hundreds of thousands of them per op.

Holds total elements fixed and varies C, so every row below moves the same aggregate data with a
different row count. If time falls as C grows, cost is per-row and the kernel must widen rows (fold
timesteps into channels) as well as reduce op count.
"""

import os
import statistics
import time

import torch

import ttnn

TOTAL = 2 * 331212 * 8  # elements, matching the s6 tail
WIDTHS = [8, 16, 32, 64, 128, 256]
ITERS = 5


def main():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        for dtype, name in ((ttnn.float32, "float32"), (ttnn.bfloat16, "bfloat16")):
            print(f"\n=== {name}: {TOTAL} elements held constant, C varied ===")
            print(f"{'C':>5} {'rows':>9} {'ms':>8} {'ns/row':>8} {'GB/s':>7}")
            base = None
            for C in WIDTHS:
                rows = TOTAL // C
                x = torch.randn(1, rows, C) * 0.3
                try:
                    xd = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
                    ttnn.add(xd, xd)
                    ttnn.synchronize_device(device)
                    ts = []
                    for _ in range(ITERS):
                        s = time.perf_counter()
                        ttnn.add(xd, xd)
                        ttnn.synchronize_device(device)
                        ts.append((time.perf_counter() - s) * 1e3)
                    ms = statistics.median(ts)
                except Exception as exc:  # noqa: BLE001
                    print(f"{C:>5} {rows:>9}  FAILED {str(exc).splitlines()[0][:44]}")
                    continue
                esz = 4 if dtype == ttnn.float32 else 2
                gbs = 3 * TOTAL * esz / 1e9 / (ms / 1e3)
                if base is None:
                    base = ms
                print(f"{C:>5} {rows:>9} {ms:>8.3f} {ms * 1e6 / rows:>8.1f} {gbs:>7.1f}   x{base / ms:.2f} vs C=8")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
