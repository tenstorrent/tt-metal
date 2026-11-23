# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E1 PCIe microbenchmark for the Hy4 host tier (HY4_PORT_PLAN P1-b; agent hy4cap). NEEDS A DEVICE (4x8 mesh); no model weights. Run it in the product container on a free slot (host .45 / .46), TERM-only stop:

    python -m models.demos.blackhole.hy4_preview.tools.pcie_bw --sizes-gb 0.5,1,3.2 --chips 1,8 --reps 3 [--decode-trace] [--cq 2]

Measures, per chip size X (GB of DRAM per chip; 3.2 GB = one chip's share of a 433k-token chain [C]: 25.4 GB / 8 chips) and per chip count n in {1, 8} (the n chips of mesh row 0 in parallel, one host thread per chip):
  d2h  : ttnn.to_torch of the DRAM tensor of every chip (uint32, row major)       -> GB/s per chip and aggregate over the n chips
  h2d  : ttnn.copy_host_to_device_tensor of a host tensor into the DRAM tensor    -> GB/s per chip and aggregate
Modes: idle device, and --decode-trace: a stand-in decode trace (a chain of bf8 matmuls + all-reduce-free eltwise on every chip, captured on CQ0, replayed in a loop by a background thread) is running while the
transfers run; reported: transfer GB/s AND the decode iteration time with / without the transfers (the interference the host tier must not cause). --cq 2 opens the mesh with two command queues and issues the
transfers on CQ1 (default: CQ0 = the CQ of the decode trace).
Output: one line per cell ``PCIE_BW mode=<idle|trace> cq=<n> chips=<n> size_gb=<x> dir=<d2h|h2d> gbps_chip=<x> gbps_total=<x> best_s=<x>`` and ``PCIE_TRACE iter_ms idle=<x> during_d2h=<x> during_h2d=<x>``.
UNVERIFIED: never run (hy4cap has no device by order); the API calls follow ttnn usage elsewhere in this repo (tt/hy4_xrow.py, tests/dev_common.py).
"""
import argparse
import threading
import time


def log(s):
    print(s, flush=True)


def build_buffers(md, size_gb, chips):
    """one DRAM uint32 RM tensor per mesh device of row 0 (``chips`` of them): a replicated mesh tensor, then per-device views."""
    import torch

    import ttnn

    words = int(size_gb * 1e9) // 4
    width = 1024  # words per row: 4 KB pages
    rows = words // width
    t = ttnn.from_torch(
        torch.zeros(1, 1, rows, width, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    devs = ttnn.get_device_tensors(t)[:chips]  # row 0 = the first mesh row (8 chips)
    return t, devs, rows * width * 4


def timed_threads(fns):
    """run one callable per thread concurrently, return the wall time of the slowest start-to-end"""
    ths = [threading.Thread(target=f) for f in fns]
    t0 = time.perf_counter()
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    return time.perf_counter() - t0


def make_decode_trace(md, cq=0):
    """stand-in decode: 40 x (bf8 matmul 32x4096x4096 + add) on every chip; captured once, replayed by ``replayer``."""
    import torch

    import ttnn

    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )

    def body():
        y = x
        for _ in range(40):
            y = ttnn.matmul(y, w)
            y = ttnn.add(y, x)
        return y

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=cq)
    out = body()
    ttnn.end_trace_capture(md, tid, cq_id=cq)
    ttnn.synchronize_device(md)
    return tid, out


class Replayer:
    def __init__(self, md, tid, cq=0):
        self.md, self.tid, self.cq = md, tid, cq
        self.stop = False
        self.iters = []  # (t_end, dt)
        self.th = threading.Thread(target=self.run)

    def run(self):
        import ttnn

        while not self.stop:
            t0 = time.perf_counter()
            ttnn.execute_trace(self.md, self.tid, cq_id=self.cq, blocking=True)
            self.iters.append((time.perf_counter(), time.perf_counter() - t0))

    def start(self):
        self.th.start()

    def finish(self):
        self.stop = True
        self.th.join()

    def mean_ms(self, t_from, t_to):
        xs = [dt for te, dt in self.iters if t_from <= te <= t_to]
        return 1e3 * sum(xs) / len(xs) if xs else float("nan")


def bench(md, size_gb, chips, reps, mode, cq, rep=None):
    import torch

    import ttnn

    t, devs, nbytes = build_buffers(md, size_gb, chips)
    host = ttnn.from_torch(
        torch.ones(1, 1, nbytes // 4096, 1024, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    res = {}
    for d in ("d2h", "h2d"):
        best = 1e9
        w0 = time.perf_counter()
        for _ in range(reps):
            if d == "d2h":
                fns = [lambda dv=dv: ttnn.to_torch(dv, **({"cq_id": cq} if cq else {})) for dv in devs]
            else:
                fns = [
                    lambda dv=dv: ttnn.copy_host_to_device_tensor(host, dv, **({"cq_id": cq} if cq else {}))
                    for dv in devs
                ]
            best = min(best, timed_threads(fns))
        w1 = time.perf_counter()
        gb = nbytes / 1e9
        log(
            f"PCIE_BW mode={mode} cq={cq} chips={chips} size_gb={size_gb} dir={d} gbps_chip={gb / best:.2f} gbps_total={gb * chips / best:.2f} best_s={best:.3f}"
        )
        if rep is not None:
            res[d] = rep.mean_ms(w0, w1)
    for dv in devs:
        pass
    ttnn.deallocate(t)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-gb", default="0.5,1,3.2")
    ap.add_argument("--chips", default="1,8")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--decode-trace", action="store_true")
    ap.add_argument(
        "--cq", type=int, default=1, help="1: transfers on CQ0 (default); 2: open 2 CQs and transfer on CQ1"
    )
    a = ap.parse_args()
    import ttnn
    from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    if a.cq == 2:
        md = ttnn.open_mesh_device(
            ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=200_000_000, num_command_queues=2
        )
    else:
        md = open_mesh(trace=200_000_000)
    cq = 1 if a.cq == 2 else 0
    try:
        for size in [float(x) for x in a.sizes_gb.split(",")]:
            for chips in [int(x) for x in a.chips.split(",")]:
                bench(md, size, chips, a.reps, "idle", cq)
        if a.decode_trace:
            tid, _ = make_decode_trace(md, 0)
            rep = Replayer(md, tid, 0)
            rep.start()
            time.sleep(2.0)
            t0, t1 = time.perf_counter() - 1.5, time.perf_counter()
            idle_ms = rep.mean_ms(t0, t1)
            ms = {}
            for size in [float(x) for x in a.sizes_gb.split(",")]:
                for chips in [int(x) for x in a.chips.split(",")]:
                    ms[(size, chips)] = bench(md, size, chips, a.reps, "trace", cq, rep)
            rep.finish()
            for (size, chips), r in ms.items():
                log(
                    f"PCIE_TRACE size_gb={size} chips={chips} iter_ms idle={idle_ms:.2f} during_d2h={r['d2h']:.2f} during_h2d={r['h2d']:.2f}"
                )
    finally:
        if a.cq == 2:
            ttnn.close_mesh_device(md)
        else:
            close_mesh(md)


if __name__ == "__main__":
    main()
