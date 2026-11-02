# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill rotation x' = H(D x) for M rows as one kernel (kernels/rot_*.cpp).

H_1024 = H32 (x) H32, so output tile i of 1024-block b is sum_j x_j @ (h(j,i) diag(d_bj) H32 / 32): 32 tile
matmuls against 64 small resident tiles per block, with x read once and x' written once. The stock-op
Rotator (prefill.py) spends most of its time moving data (transposes + a batched auto matmul).

Self-check: python -m models.demos.blackhole.bonsai2.native.a3.rot [--seqs 128,640,2048]
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn

KDIR = "models/demos/blackhole/bonsai2/native/a3/kernels/"


def pn_tensor(device, signs: np.ndarray):
    """[nb*64*32, 32] bfp8 (+-1/32 is exact): tile b*64 + 2j + neg = (-1)^neg diag(d_bj) H32 / 32."""
    from models.demos.blackhole.bonsai2.native.hadamard import sylvester

    h32 = np.sign(sylvester(32)).astype(np.float32) / 32
    k = signs.shape[0]
    d = signs.astype(np.float32).reshape(k // 1024, 32, 32, 1)  # [b, j, row, 1]
    p = d * h32[None, None]  # [b, j, 32, 32]
    t = np.stack([p, -p], axis=2).reshape(-1, 32)  # [b, j, neg, 32, 32] -> rows
    return ttnn.from_torch(torch.from_numpy(t), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


class KRotator:
    def __init__(self, device, had, widths=(5120, 6144, 17408)):
        self.device = device
        self.pn = {k: pn_tensor(device, had.signs[k]) for k in widths}

    def __call__(self, x, mem=ttnn.DRAM_MEMORY_CONFIG):
        s, k = x.shape[-2], x.shape[-1]
        x4 = ttnn.reshape(x, [1, 1, s, k])
        if x4.dtype != ttnn.bfloat16:
            x4 = ttnn.typecast(x4, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif x4.is_sharded() or x4.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x4 = ttnn.to_memory_config(x4, ttnn.DRAM_MEMORY_CONFIG)
        y = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, s, k]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device,
                                           ttnn.DRAM_MEMORY_CONFIG)
        pn = self.pn[k]
        rows, kt, nb = s // 32, k // 32, k // 1024
        units = rows * nb
        g = self.device.compute_with_storage_grid_size()
        cores = [ttnn.CoreCoord(cx, cy) for cy in range(g.y) for cx in range(g.x)]
        per = -(-units // len(cores))
        cores = cores[: -(-units // per)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, c in enumerate(cores):
            u0, nu = i * per, min(per, units - i * per)
            rd[c.x][c.y] = [x4.buffer_address(), pn.buffer_address(), u0, nu]
            cp[c.x][c.y] = [u0, nu]
            wr[c.x][c.y] = [y.buffer_address(), u0, nu]
        xtb = ttnn.Tile([32, 32]).get_tile_size(ttnn.bfloat16)
        ptb = ttnn.Tile([32, 32]).get_tile_size(ttnn.bfloat8_b)

        def cb(idx, fmt, page, n):
            return ttnn.CBDescriptor(total_size=n * page, core_ranges=crs, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)])

        kd = lambda src, ct, rt, cfg: ttnn.KernelDescriptor(  # noqa: E731
            kernel_source=KDIR + src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=crs,
            compile_time_args=ct, runtime_args=rt, config=cfg)
        prog = ttnn.ProgramDescriptor(
            kernels=[
                kd("rot_reader.cpp", [rows, kt, xtb, ptb], rd, ttnn.ReaderConfigDescriptor()),
                kd("rot_writer.cpp", [rows, kt, xtb], wr, ttnn.WriterConfigDescriptor()),
                kd("rot_compute.cpp", [rows], cp, ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)),
            ],
            semaphores=[],
            cbs=[cb(0, ttnn.bfloat16, xtb, 64), cb(1, ttnn.bfloat8_b, ptb, 128), cb(16, ttnn.bfloat16, xtb, 4)])
        ttnn.generic_op([x4, pn, y], prog)
        if x4 is not x and x4.buffer_address() != x.buffer_address():
            ttnn.deallocate(x4)
        if mem != ttnn.DRAM_MEMORY_CONFIG:
            y2 = ttnn.to_memory_config(y, mem)
            ttnn.deallocate(y)
            y = y2
        return y


def _selfcheck():
    import argparse
    import os
    import time

    from models.demos.blackhole.bonsai2.native import cpu_ref
    from models.demos.blackhole.bonsai2.native.a3 import prefill as pf
    from models.demos.blackhole.bonsai2.native.gguf_ternary import read_gguf
    from models.demos.blackhole.bonsai2.native.hadamard import BlockHadamard

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="128,640,2048")
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=int(os.environ.get("BONSAI_DEVICE", 0)), l1_small_size=24576)
    try:
        from models.demos.blackhole.bonsai2.native.a3.mlp import GGUF

        had = BlockHadamard.from_metadata(read_gguf(GGUF).kv)
        kr, old = KRotator(dev, had), pf.Rotator(dev, had)

        def timed(f, n=5):
            ttnn.deallocate(f())
            ttnn.synchronize_device(dev)
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                r = f()
                ttnn.synchronize_device(dev)
                ts.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(r)
            return float(np.median(ts))

        for s in [int(v) for v in args.seqs.split(",")]:
            for k in (5120, 6144, 17408):
                xh = torch.randn(1, s, k).bfloat16()
                x = ttnn.from_torch(xh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ref = had.rotate(xh.float().numpy().reshape(s, k))
                yk = ttnn.to_torch(kr(x)).float().numpy().reshape(s, k)
                yo = ttnn.to_torch(old(x)).float().numpy().reshape(s, k)
                assert np.isfinite(yk).all(), "non-finite kernel output"
                print(f"S={s} K={k}: kernel pcc {cpu_ref.pcc(yk, ref):.7f} max|d| {np.abs(yk - ref).max():.3e}; "
                      f"stock pcc {cpu_ref.pcc(yo, ref):.7f} max|d| {np.abs(yo - ref).max():.3e}; "
                      f"ms kernel {timed(lambda: kr(x)):.3f} stock {timed(lambda: old(x)):.3f}", flush=True)
                ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    _selfcheck()
