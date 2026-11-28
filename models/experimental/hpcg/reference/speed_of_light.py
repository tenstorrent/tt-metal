# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Speed of light (SoL) of one HPCG CG iteration on a P150a, from first principles.

Roofline per op (scaling book, inference chapter): T = max(FLOPs / peak FLOP/s, bytes / DRAM bandwidth).
The SoL of an iteration is the sum over HPCG's operations, each counted the way the HPCG rules require:
  * every matrix term is used in every sweep (spec section 6 FAQ 3: "you may not in any way eliminate terms"),
  * matrix storage is unstructured: 4-byte fp32 value + 4-byte column index per nonzero (HPCG's local_int_t),
  * symmetry is not exploited (FAQ 2),
  * nnz = the true nonzero count (zero-valued padding an implementation adds is its own cost).
Bytes = the minimum DRAM traffic of each op on its own: every input read once, every output written once. A matrix
too big for on-chip SRAM is re-read on each of the two SYMGS sweeps; one that fits is read once.
Vectors that are known to be zero (x entering the pre-smoother, after HPCG's ZeroVector) need no DRAM read.
FLOPs = HPCG's own count (ReportResults.cpp lines 81-97), used for both T_math and the GFLOP/s rating.
usage: python -m models.experimental.hpcg.reference.speed_of_light nx [measured_ms_per_iteration]
"""

import json
import sys

BW = 512e9  # P150a: GDDR6 at 16 Gbps (tt-smi dram_speed "16G") x 256-bit bus / 8 = 512 GB/s
CORES = 130  # 13 x 10 Tensix grid of this P150a (compute_with_storage_grid_size)
CLK = 1.35e9  # asic_fmax 1350 MHz (tt-smi limits)
PEAK = CORES * 32 * 2 * CLK  # fp32 FMA on the 32-lane SFPU of each core = 11.2 TFLOP/s
SRAM = CORES * 1.4e6  # L1 per core usable for data, about the sparse-row op's CB budget = 182 MB
V, I, W = 4, 4, 4  # bytes: matrix value, column index, vector word (fp32 port of HPCG)


def level(nx):
    return nx**3, (3 * nx - 2) ** 3  # rows, true nonzeros of a 27-point stencil on an nx^3 grid


def roof(name, cnt, fl, rd, wr):
    tm, tc = fl / PEAK, (rd + wr) / BW
    return dict(
        op=name,
        per_iter=cnt,
        flops=fl,
        read_B=rd,
        write_B=wr,
        t_math_us=tm * 1e6,
        t_comms_us=tc * 1e6,
        sol_us=max(tm, tc) * 1e6,
    )


def ops(nx):
    out = []
    n0, nnz0 = level(nx)
    for l in range(4):
        n, nnz = level(nx >> l)
        passes = 1 if nnz * (V + I) + 2 * n * W <= SRAM else 2
        mat = passes * nnz * (V + I)
        if l < 3:
            nc = level(nx >> (l + 1))[0]
            out.append(roof(f"SYMGS pre  L{l}", 1, 4 * nnz, mat + n * W, n * W))  # read A, r; write x
            out.append(roof(f"SpMV resid L{l}", 1, 2 * nnz, nnz * (V + I) + n * W, n * W))
            out.append(roof(f"restrict   L{l}", 1, 0, 2 * nc * W, nc * W))  # rc = r - Ax at nc points
            out.append(roof(f"prolong    L{l}", 1, 0, 2 * nc * W, nc * W))  # x += xc at nc points
            out.append(roof(f"SYMGS post L{l}", 1, 4 * nnz, mat + 2 * n * W, n * W))  # read A, r, x; write x
        else:
            out.append(roof(f"SYMGS      L{l}", 1, 4 * nnz, mat + n * W, n * W))
    out.append(roof("SpMV Ap", 1, 2 * nnz0, nnz0 * (V + I) + n0 * W, n0 * W))
    out.append(roof("DDOT", 3, 2 * n0, 2 * n0 * W, 0))
    out.append(roof("WAXPBY", 3, 2 * n0, 2 * n0 * W, n0 * W))
    return out


def hpcg_flops(nx):
    n0, nnz0 = level(nx)
    return 6 * n0 + 6 * n0 + 2 * nnz0 + sum(10 * level(nx >> l)[1] for l in range(3)) + 4 * level(nx >> 3)[1]


def hpcg_start_flops(nx):
    """Flops of the CG start (r = b - A x, |r|), which ReportResults.cpp credits once per CG set: one DDOT, one
    WAXPBY and one SpMV."""
    n0, nnz0 = level(nx)
    return 2 * n0 + 2 * n0 + 2 * nnz0


if __name__ == "__main__":
    nx = int(sys.argv[1])
    ms = float(sys.argv[2]) if len(sys.argv) > 2 else None
    o = ops(nx)
    sol_us = sum(r["per_iter"] * r["sol_us"] for r in o)
    rd = sum(r["per_iter"] * r["read_B"] for r in o)
    wr = sum(r["per_iter"] * r["write_B"] for r in o)
    fl = hpcg_flops(nx)
    res = dict(
        nx=nx,
        peak_tflops=round(PEAK / 1e12, 3),
        bw_gbps=BW / 1e9,
        critical_intensity=round(PEAK / BW, 2),
        sol_ms_per_iter=round(sol_us / 1e3, 4),
        read_MB=round(rd / 1e6, 1),
        write_MB=round(wr / 1e6, 1),
        read_share_pct=round(100 * rd / (rd + wr), 2),
        hpcg_flops_per_iter=fl,
        sol_gflops=round(fl / (sol_us * 1e-6) / 1e9, 2),
    )
    if ms:
        res.update(
            measured_ms_per_iter=ms,
            measured_gflops=round(fl / (ms * 1e-3) / 1e9, 2),
            pct_of_sol=round(100 * sol_us / 1e3 / ms, 1),
        )
    res["ops"] = [
        dict(
            op=r["op"],
            per_iter=r["per_iter"],
            sol_us=round(r["sol_us"], 2),
            math_us=round(r["t_math_us"], 3),
            MB=round((r["read_B"] + r["write_B"]) / 1e6, 2),
        )
        for r in o
    ]
    print(json.dumps(res, indent=1))
