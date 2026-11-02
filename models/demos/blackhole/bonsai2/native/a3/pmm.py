# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A3 at M>1: short-prefill ternary matmul straight from the resident A3 weights (kernels/pmm_*.cpp).

y [M, N] = x' [M, K] @ (codes * scales) for M <= 128 rows of an already-rotated x'. Each code block is
dequantized in L1 right before its matmul, so no [K, N] scratch is written or read back (that scratch is
what the short-prefill path spends its time on: ~3.3 ms per GDN layer at S=128).

Self-check: python -m models.demos.blackhole.bonsai2.native.a3.pmm [--seqs 32,128]
"""

from __future__ import annotations

import os

import numpy as np
import torch

import ttnn

KDIR = "models/demos/blackhole/bonsai2/native/a3/kernels/"
MAX_ROWS = 128


def _one(core, args):
    ra = ttnn.RuntimeArgs()
    ra[core.x][core.y] = args
    return ra


def pmm(x, w, s, k: int, n: int, gu: bool = False):
    """x: [1, 1, M, K] bf16 DRAM interleaved (rotated). w, s: a3.weights_for_device tensors.
    n: real output columns (gu: per half). Returns [1, 1, M, n] bf16 DRAM (gu: (gate, up))."""
    from .prefill import codes_view, scale_tiles, scales_view

    wt, wtb, cpb, sx = codes_view(w)
    device = x.device()
    st, sl = scales_view(s, ((k // 1024) + 3) // 4)
    stb, sfmt, s16 = scale_tiles(s)
    m = x.shape[-2]
    assert m % 32 == 0 and m <= MAX_ROWS and k % 1024 == 0
    mt, kt = m // 32, k // 32
    ng = kt // 32
    s_tiles = (ng + 3) // 4
    xtb = ttnn.Tile([32, 32]).get_tile_size(ttnn.bfloat16)
    ncols = (2 if gu else 1) * n // 32
    outs = [ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, m, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device,
                                           ttnn.DRAM_MEMORY_CONFIG) for _ in range(2 if gu else 1)]
    g = device.compute_with_storage_grid_size()
    mc = os.environ.get("BONSAI_PMM_MC", "1") == "1"
    if mc:
        # x' multicast: one sender at logical (g.x-1, 0) reads x' once and multicasts it to a worker rectangle
        # of width g.x-1, so all workers share one x' stream instead of each reading it from DRAM.
        # Worker rectangle: logical columns 0..g.x-2 (the sender sits in the last column). It spans Blackhole's
        # non-Tensix NOC columns 8-9, which is fine for multicast. The 18:37 hang was the corner order:
        # BRISC kernels use NOC1 here, so pmm_xsend.cpp swaps the corners for NOC1.
        wx = int(os.environ.get("BONSAI_PMM_WX", g.x - 1))
        per = -(-ncols // (wx * g.y))
        used = -(-ncols // per)
        h = -(-used // wx)
        cores = [ttnn.CoreCoord(cx, cy) for cy in range(h) for cx in range(wx)]
        sender = ttnn.CoreCoord(g.x - 1, 0)
        wrect = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(wx - 1, h - 1))])
        allr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(wx - 1, h - 1)),
                                  ttnn.CoreRange(sender, sender)])
        p0 = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
        p1 = device.worker_core_from_logical_core(ttnn.CoreCoord(wx - 1, h - 1))
        ps = device.worker_core_from_logical_core(sender)
        crs = wrect
    else:
        cores = [ttnn.CoreCoord(cx, cy) for cy in range(g.y) for cx in range(g.x)]
        # Every core streams all of x', so x traffic = cores x |x'| and binds at M=128 (down: 80 cores x 4.4 MB).
        # ~160/MT cores balances it (sweep at S=128: down 4 cols/core 0.78 ms vs 2/core 1.43; gate|up 16/core 0.84 vs 10/core 1.25).
        per = max(-(-ncols // len(cores)), -(-ncols // (160 // mt)), int(os.environ.get("BONSAI_PMM_PER", "0")))
        cores = cores[: -(-ncols // per)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        allr = crs
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        c0 = min(i * per, ncols)
        nc = max(0, min(per, ncols - c0))
        rd[c.x][c.y] = [x.buffer_address(), wt.buffer_address(), s.buffer_address(), c0, nc]
        cp[c.x][c.y] = [nc, c0]
        wr[c.x][c.y] = [outs[0].buffer_address(), outs[-1].buffer_address(), c0, nc]

    def cb(idx, fmt, page, nt):
        return ttnn.CBDescriptor(total_size=nt * page, core_ranges=allr, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)])

    kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
        kernel_source=KDIR + src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
        compile_time_args=ct, runtime_args=rt, config=cfg)
    modes = [ttnn.UnpackToDestMode.Default] * 64
    modes[4] = ttnn.UnpackToDestMode.UnpackToDestFp32  # fp32 partials reload exactly
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("BONSAI_PMM_FID", "HiFi2")), fp32_dest_acc_en=True,
                                       math_approx_mode=False)
    cfg.unpack_to_dest_mode = modes
    kernels = [
            kd("pmm_reader.cpp", [ng, mt, kt, cpb, wtb, s_tiles, xtb, int(mc), ps.x if mc else 0, ps.y if mc else 0, 0, 1] + sl + [stb],
               rd, ttnn.ReaderConfigDescriptor()),
            kd("pmm_writer.cpp", [mt, n // 32, int(gu), xtb], wr, ttnn.WriterConfigDescriptor()),
            kd("pmm_compute.cpp", [ng, mt, int(os.environ.get("BONSAI_PMM_DBG", "0")), sx, cpb, sl[0], s_tiles] + sl[1:] + [s16], cp, cfg),
    ]
    if mc:
        kernels.append(ttnn.KernelDescriptor(
            kernel_source=KDIR + "pmm_xsend.cpp", source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(sender, sender)]),
            compile_time_args=[ng, mt, kt, xtb, len(cores), p0.x, p0.y, p1.x, p1.y, 0, 1],
            runtime_args=_one(sender, [x.buffer_address()]), config=ttnn.WriterConfigDescriptor()))
    prog = ttnn.ProgramDescriptor(
        kernels=kernels,
        semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=allr, initial_value=0),
                    ttnn.SemaphoreDescriptor(id=1, core_ranges=allr, initial_value=0)],
        cbs=[cb(0, ttnn.bfloat16, xtb, 2 * mt * 32), cb(1, ttnn.DataType.BFLOAT2_B, wtb, max(64, 2 * per * 32)),
             cb(2, sfmt, 4096, max(2, 2 * per)),
             cb(3, ttnn.bfloat16, xtb, 32), cb(4, ttnn.float32, 4096, per * mt), cb(16, ttnn.bfloat16, xtb, 2 * mt)])
    ttnn.generic_op([x, wt, st] + outs, prog)
    return tuple(outs) if gu else outs[0]


def _selfcheck():
    import argparse
    import os
    import time

    from models.demos.blackhole.bonsai2.native import cpu_ref
    from models.demos.blackhole.bonsai2.native.a3 import linear as al
    from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
    from models.demos.blackhole.bonsai2.native.a3.mlp import Shared, TernaryMLP
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_ternary

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="32,128")
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)

    def timed(f, n=5):
        r = f()
        [ttnn.deallocate(t) for t in (r if isinstance(r, tuple) else (r,))]
        ttnn.synchronize_device(dev)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            r = f()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
            [ttnn.deallocate(t) for t in (r if isinstance(r, tuple) else (r,))]
        return float(np.median(ts))

    try:
        sh = Shared(dev)
        mlp = TernaryMLP(sh, 0, os.path.join(os.environ["TT_CACHE_PATH"], "a3_mlp"))
        cache = os.path.join(os.environ["TT_CACHE_PATH"], "a3_linear")
        gi = al.TernaryLinear(sh, "blk.0.gdn_in", cache, rows=lambda: al.gdn_in_rows(sh, 0), shape=al.gdn_in_shape(sh, 0))
        go = al.TernaryLinear(sh, "blk.0.ssm_out.weight", cache)
        t = {nm: read_ternary(sh.gguf, f"blk.0.ffn_{nm}.weight") for nm in ("gate", "up", "down")}
        dense = lambda tt: tt.codes.astype(np.float64) * np.repeat(tt.scales.astype(np.float64), 128, axis=1)  # noqa: E731
        wref = {"gate": dense(t["gate"]), "up": dense(t["up"]), "down": dense(t["down"]),
                "gdn_in": np.asarray(al.gdn_in_rows(sh, 0)[0], np.float64) * np.repeat(
                    np.asarray(al.gdn_in_rows(sh, 0)[1], np.float64), 128, axis=1),
                "gdn_out": dense(read_ternary(sh.gguf, "blk.0.ssm_out.weight"))}
        cases = [("gu", 5120, None), ("down", 17408, None), ("gdn_in", 5120, gi), ("gdn_out", 6144, go)]
        for srows in [int(v) for v in args.seqs.split(",")]:
            for name, k, tl in cases:
                xh = torch.randn(1, 1, srows, k).bfloat16()
                x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
                xf = xh.float().numpy().reshape(srows, k)
                if name == "gu":
                    f_new = lambda: pmm(x, mlp.w_gu, mlp.s_gu, k, 17408, gu=True)  # noqa: E731
                    f_old = lambda: tuple(ttnn.linear(x, wt, compute_kernel_config=pf._LOFI)  # noqa: E731
                                          for wt in pf.dequant(mlp.w_gu, mlp.s_gu, k, 17408, gu=True))
                    got = [ttnn.to_torch(o).float().numpy().reshape(srows, -1) for o in f_new()]
                    pcs = [cpu_ref.pcc(got[0], xf @ wref["gate"].T), cpu_ref.pcc(got[1], xf @ wref["up"].T)]
                elif name == "down":
                    f_new = lambda: pmm(x, mlp.w_d, mlp.s_d, k, 5120)  # noqa: E731
                    f_old = lambda: ttnn.linear(x, pf.dequant(mlp.w_d, mlp.s_d, k, 5120), compute_kernel_config=pf._LOFI)  # noqa: E731
                    pcs = [cpu_ref.pcc(ttnn.to_torch(f_new()).float().numpy().reshape(srows, -1), xf @ wref["down"].T)]
                else:
                    f_new = lambda: pmm(x, tl.w, tl.s, k, tl.n)  # noqa: E731
                    f_old = lambda: ttnn.linear(x, pf.dequant(tl.w, tl.s, k, tl.n), compute_kernel_config=pf._LOFI)  # noqa: E731
                    pcs = [cpu_ref.pcc(ttnn.to_torch(f_new()).float().numpy().reshape(srows, -1), xf @ wref[name].T)]
                got_all = f_new()
                for o in (got_all if isinstance(got_all, tuple) else (got_all,)):
                    assert torch.isfinite(ttnn.to_torch(o).float()).all(), "non-finite pmm output"
                o0 = ttnn.to_torch(got_all[0] if isinstance(got_all, tuple) else got_all).float().numpy().reshape(srows, -1)
                r0 = xf @ (wref["gate"] if name == "gu" else wref[name]).T
                rel = float(np.abs(o0 - r0).max() / np.abs(r0).max())  # scale-sensitive: PCC isn't (SCALE16 bug)
                assert rel < 0.05, f"{name}: max rel err {rel}"
                pcs.append(rel)
                print(f"S={srows} {name}: pmm pcc vs float64 {' '.join(f'{p:.6f}' for p in pcs)}; "
                      f"ms pmm {timed(f_new):.3f} vs dequant+matmul {timed(f_old):.3f}", flush=True)
                ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    _selfcheck()
