# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E1b (hy4fix): d2h / h2d host path of the Hy4 host tier: effect of the DRAM page size and of the read pattern. DEVICE needed (4x8 mesh).
 python -m models.demos.blackhole.hy4_preview.tools.pcie_bw2 [--mb 128]
Cells: PCIE2 pg=<page bytes> pat=<single|row8_nb|row8_seq|mesh32> dir=<d2h|h2d> gbps_total=<x> s=<x>  (bytes counted = those the host tier needs: one chip, or the 8 chips of one row; mesh32 reads 4x that)"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=128)
    ap.add_argument("--pages", default="704,32768")
    a = ap.parse_args()
    import torch

    import ttnn
    from models.demos.blackhole.hy4_preview.tests.dev_common import open_mesh

    md = open_mesh(trace=200_000_000)
    comp = ttnn.ConcatMesh2dToTensor(md, dims=(0, 1), mesh_shape=tuple(md.shape))
    try:
        for pg in [int(x) for x in a.pages.split(",")]:
            w = pg // 4
            rows = a.mb * 1_000_000 // pg
            nb = rows * pg
            t = ttnn.from_torch(
                torch.arange(rows * w, dtype=torch.int32).reshape(1, 1, rows, w),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=md,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(md),
            )
            devs = ttnn.get_device_tensors(t)
            row0 = devs[:8]

            def tm(f, reps=2):
                b = 1e9
                for _ in range(reps):
                    t0 = time.perf_counter()
                    f()
                    b = min(b, time.perf_counter() - t0)
                return b

            def out(pat, d, s, nbytes):
                print(f"PCIE2 pg={pg} pat={pat} dir={d} gbps_total={nbytes / s / 1e9:.2f} s={s:.3f}", flush=True)

            out("single", "d2h", tm(lambda: ttnn.to_torch(devs[0])), nb)
            out("row8_seq", "d2h", tm(lambda: [ttnn.to_torch(d) for d in row0]), 8 * nb)

            def nbf():
                hs = [ttnn.from_device(d, blocking=False) for d in row0]
                ttnn.synchronize_device(md)
                return [ttnn.to_torch(h) for h in hs]

            out("row8_nb", "d2h", tm(nbf), 8 * nb)
            import threading

            def thr(fn, devs_):
                errs = []

                def w(d):
                    try:
                        fn(d)
                    except Exception as e:
                        errs.append(repr(e)[:80])

                ths = [threading.Thread(target=w, args=(d,)) for d in devs_]
                [x.start() for x in ths]
                [x.join() for x in ths]
                if errs:
                    print("PCIE2 thread errors", errs[:2], flush=True)

            out("row8_thr_fromdev", "d2h", tm(lambda: thr(lambda d: ttnn.from_device(d), row0)), 8 * nb)
            out("row8_thr_tot", "d2h", tm(lambda: thr(lambda d: ttnn.to_torch(d), row0)), 8 * nb)
            out("mesh32", "d2h", tm(lambda: ttnn.to_torch(t, mesh_composer=comp), 1), 8 * nb)  # useful bytes = one row
            host = ttnn.from_torch(
                torch.ones(1, 1, rows, w, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            out("single", "h2d", tm(lambda: ttnn.copy_host_to_device_tensor(host, devs[0])), nb)
            out("row8_seq", "h2d", tm(lambda: [ttnn.copy_host_to_device_tensor(host, d) for d in row0]), 8 * nb)
            out("row8_thr", "h2d", tm(lambda: thr(lambda d: ttnn.copy_host_to_device_tensor(host, d), row0)), 8 * nb)
            ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
