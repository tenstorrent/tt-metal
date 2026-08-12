"""Is the ~170 us/op fixed cost real serial device work, or an artefact of syncing after every op?

An earlier probe timed one op at a time with a synchronize in between, which serializes host and device
and folds round-trip latency into every measurement. That number (170-180 us) happens to match the
decode's 1401 ms / 6955 ops = 201 us, but agreement is not proof -- the decode could be paying something
else.

**Answered 2026-08-12: it is sync latency, not device work.** Measured chained 141.9 / independent 125.7
/ per-op-sync 138.8 us/op -- all three equal, so this microbenchmark is host-*issue*-bound and cannot
see device time at all. The "6955 ops x 180 us = 1254 ms floor" that followed from it is void; real
per-op device cost is ~37 us. Kept as the standing counter-evidence, since that floor is quoted in older
notes. See ITEM1_RESULT.md.

Three measurements, each issuing N ops and synchronizing ONCE at the end:

  chained      op i+1 consumes op i's output, so the device cannot overlap them
  independent  N ops on the same input, free to pipeline
  per-op-sync  the per-op-synchronize methodology, for comparison

If independent is far cheaper than per-op-sync, the floor is sync latency and the real per-op cost is
lower. If chained ~= per-op-sync, the cost is genuine serial work and op count is the wall.

Run with TT_VISIBLE_DEVICES=0 to ask the same questions of a single-chip cluster.
"""

import os
import statistics
import time

import torch

import ttnn

ROWS = int(os.environ.get("PIPE_ROWS", "2048"))
C = 8
N = int(os.environ.get("PIPE_N", "50"))
REPS = int(os.environ.get("PIPE_REPS", "5"))


def main():
    visible = os.environ.get("TT_VISIBLE_DEVICES", "(all)")
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        n_devices = device.get_num_devices() if hasattr(device, "get_num_devices") else "?"
        print(f"TT_VISIBLE_DEVICES={visible}  mesh devices={n_devices}  rows={ROWS} C={C} N={N}")
        x = torch.randn(2, ROWS, C) * 0.3
        xd = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=xd_dev(device))

        def chained():
            y = xd
            for _ in range(N):
                y = ttnn.add(y, xd)
            return y

        def independent():
            for _ in range(N):
                ttnn.add(xd, xd)

        def per_op_sync():
            for _ in range(N):
                ttnn.add(xd, xd)
                ttnn.synchronize_device(device)

        for label, fn in (("chained", chained), ("independent", independent), ("per-op-sync", per_op_sync)):
            fn()
            ttnn.synchronize_device(device)
            ts = []
            for _ in range(REPS):
                s = time.perf_counter()
                fn()
                ttnn.synchronize_device(device)
                ts.append((time.perf_counter() - s) * 1e3)
            ms = statistics.median(ts)
            print(f"  {label:<13} {ms:>8.2f} ms total  ->  {ms * 1e3 / N:>7.1f} us/op")
    finally:
        ttnn.close_mesh_device(device)


def xd_dev(d):
    return d


if __name__ == "__main__":
    main()
