# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Can ``ttnn.linear`` read the width-sharded DRAM weight after all?  Yes.

The stage's first pass rejected ``ttnn.linear`` for prefill on a single API error,
``MatmulMultiCoreProgramConfig: Input B memory layout must be INTERLEAVED``
(``matmul_device_operation.cpp:1233``).  That error comes from the *auto-selected*
fallback program config, and ``$optimize`` is explicit that a first API error is
not a rejection.  Reading the validator instead of the error message:

* ``validate_matmul_mcast2d_config`` (``:1368``) accepts a ``WIDTH_SHARDED``
  ``input_tensor_b`` **in DRAM** (``:1541-1553``);
* the extra "non-DRAM width-sharded input B requires input A interleaved or
  height-sharded, and ``per_core_N`` must equal the in1 shard width" clause is
  gated on ``buffer_type() != DRAM`` (``:1525``), so it does not constrain this
  weight;
* the only width-shard clause that does apply is that the in1 shard grid's
  bounding box is one row tall, which the 8-DRAM-bank weight already satisfies.

So an *explicit* 2D-multicast program config is legal over exactly the tensor this
stage already ships, with DRAM-interleaved activations and output -- which is the
prefill contract.

**One undocumented constraint, found the hard way.**  With a width-sharded DRAM
in1 the program config's core-*column* count must equal the DRAM bank count (8 on
this part).  At ``gx = 9`` or ``gx = 11`` the op validates, runs, and returns
``inf`` in tens of thousands of elements -- a silent miscompute, not an error.
The same grids are correct with a DRAM-*interleaved* in1, which isolates it to the
width-sharded in1 reader.  ``--repro`` prints the minimal case; see
``README.md`` "A TTNN bug this stage found".  Every candidate here is therefore
gated on a finite-output + PCC check before its latency is reported at all.

    python .../bench/prefill_mcast_probe.py                 # the sweep
    python .../bench/prefill_mcast_probe.py --repro         # the gx > banks bug
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    DEFAULT_PRECISION,
    dram_sharded_weight_memcfg,
    minimal_matmul_blocks,
)

TILE = 32

#: ``role -> (K, N)``; ``mlp_gate`` and ``mlp_up`` share a shape.
SHAPES = {
    "wqkv": (6656, 4608),
    "attn_gate": (6656, 4096),
    "o_proj": (4096, 6656),
    "mlp_gate": (6656, 19968),
    "mlp_down": (19968, 6656),
}

ROWS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)

#: Core-row counts to try.  The core-*column* count is pinned to the DRAM bank
#: count by the constraint above, so this is the only free grid dimension.
GRID_Y = (2, 4, 8, 10)

#: PCC floor for a candidate to be reported as a timing result at all.  These are
#: single random-weight matmuls against a float64 reference, so the bar is about
#: catching corruption (the ``inf`` bug above scores ``nan``), not about model
#: accuracy -- that is the test suite's job.
MIN_PCC = 0.99


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def out_subblock(per_core_m: int, per_core_n: int) -> tuple[int, int]:
    """Largest legal ``(h, w)`` with ``h * w <= 8``, preferring a wide subblock."""
    best = (1, 1)
    for h in range(1, min(per_core_m, 8) + 1):
        for w in range(1, min(per_core_n, 8 // h) + 1):
            if per_core_m % h == 0 and per_core_n % w == 0 and h * w > best[0] * best[1]:
                best = (h, w)
    return best


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    if not bool(torch.isfinite(a).all()):
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def timed(fn, mesh, reps: int, rounds: int) -> tuple[float, list[float]]:
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    per_round = []
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        per_round.append((time.perf_counter() - t0) / reps * 1e3)
    return min(per_round), per_round


def mcast2d_candidates(rows: int, k: int, n: int, banks: int, grid_y=GRID_Y, top_bw: int = 0, out_blocks=((0, 0),)):
    """Legal 2D-multicast candidates with ``gx == banks``.

    ``out_blocks`` are candidate ``out_block_h`` values (``0`` = leave it at
    ``per_core_M``).  This is the field that makes the large-row candidates legal
    at all: the L1 output block is ``out_block_h * out_block_w`` tiles, so at 8192
    rows a ``per_core_M`` of 26 needs bounding or the static circular buffers grow
    past the 1.5 MB budget ($optimize: "If an op runs out of L1 ... reduce
    ``in0_block_w``, ``out_subblock_h``, or ``out_subblock_w``" -- ``out_block_h``
    is the coarser version of the same knob and keeps the subblock wide).
    """
    m_tiles, k_tiles, n_tiles = rows // TILE, k // TILE, n // TILE
    per_core_n = math.ceil(n_tiles / banks)
    for gy in grid_y:
        per_core_m = math.ceil(m_tiles / gy)
        if per_core_m * (gy - 1) >= m_tiles and gy > 1:
            continue  # an entire core row would be idle; the smaller gy covers it
        legal = sorted({d for d in divisors(k_tiles) if d <= 26}, reverse=True)
        for out_block_h, out_block_w in out_blocks:
            block_h = per_core_m if not out_block_h else out_block_h
            block_w = per_core_n if not out_block_w else out_block_w
            if per_core_m % block_h or per_core_n % block_w:
                continue
            sub_h, sub_w = out_subblock(block_h, block_w)
            suffix = "" if not (out_block_h or out_block_w) else f"_ob{block_h}x{block_w}"
            for in0_block_w in legal[:top_bw] if top_bw else legal:
                yield (
                    f"mcast2d_{banks}x{gy}_bw{in0_block_w}{suffix}",
                    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(banks, gy),
                        in0_block_w=in0_block_w,
                        out_subblock_h=sub_h,
                        out_subblock_w=sub_w,
                        out_block_h=block_h,
                        out_block_w=block_w,
                        per_core_M=per_core_m,
                        per_core_N=per_core_n,
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=True,
                    ),
                )


def repro(mesh) -> None:
    """Minimal case for the ``gx > dram_banks`` silent-``inf`` bug."""
    k, n, rows = 6656, 4608, 128
    banks = mesh.dram_grid_size().x
    torch.manual_seed(0)
    w_t = torch.randn(1, 1, k, n) / math.sqrt(k)
    x_t = torch.randn(1, 1, rows, k)
    ref = (x_t.to(torch.float64) @ w_t.to(torch.float64)).float()
    x = ttnn.from_torch(
        x_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print(f"REPRO dram banks = {banks}; K={k} N={n} rows={rows}, in1 bfloat8_b TILE")
    for in1_label, memcfg in (
        ("width-sharded DRAM", dram_sharded_weight_memcfg(k, n, mesh)),
        ("interleaved DRAM", ttnn.DRAM_MEMORY_CONFIG),
    ):
        w = ttnn.from_torch(w_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=memcfg)
        for gx in (banks, banks + 1, 11):
            per_core_n = math.ceil((n // TILE) / gx)
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(gx, 4),
                in0_block_w=26,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=1,
                per_core_N=per_core_n,
                transpose_mcast=False,
                fused_activation=None,
                fuse_batch=True,
            )
            out = ttnn.linear(x, w, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=cfg)
            o = ttnn.to_torch(out).float()
            print(
                f"REPRO in1={in1_label:20s} grid=({gx}, 4) per_core_N={per_core_n:3d}  "
                f"non-finite={int(torch.isinf(o).sum()) + int(torch.isnan(o).sum()):8d}  pcc={pcc(o, ref):.6f}"
            )
            ttnn.deallocate(out)
        ttnn.deallocate(w)
    ttnn.deallocate(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="all")
    ap.add_argument("--rows", default=",".join(str(r) for r in ROWS))
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repro", action="store_true", help="print the gx > banks bug and exit")
    ap.add_argument("--grid-y", default=",".join(str(g) for g in GRID_Y))
    ap.add_argument("--top-bw", type=int, default=0, help="only the N largest legal in0_block_w (0 = all)")
    ap.add_argument(
        "--out-blocks",
        default="0x0",
        help="candidate out_block_h x out_block_w pairs, comma separated; 0 = per_core_M / per_core_N",
    )
    args = ap.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    try:
        if args.repro:
            repro(mesh)
            return
        banks = mesh.dram_grid_size().x
        grid = mesh.compute_with_storage_grid_size()
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=DEFAULT_PRECISION.prefill_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        names = list(SHAPES) if args.shapes == "all" else args.shapes.split(",")
        for name in names:
            k, n = SHAPES[name]
            dtype = DEFAULT_PRECISION.weight_dtype(name)
            torch.manual_seed(abs(hash(name)) % 10_000)
            w_t = torch.randn(1, 1, k, n) / math.sqrt(k)
            w = ttnn.from_torch(
                w_t,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=dram_sharded_weight_memcfg(k, n, mesh),
            )
            for rows in [int(r) for r in args.rows.split(",")]:
                x_t = torch.randn(1, 1, rows, k)
                ref = (x_t.to(torch.float64) @ w_t.to(torch.float64)).float()
                x = ttnn.from_torch(
                    x_t,
                    device=mesh,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                blocks = minimal_matmul_blocks(name, rows, dtype)
                mm_config = None
                if blocks is not None:
                    m_block, k_block, n_block = blocks
                    sub_h, sub_w = (2, 4) if n >= rows else (4, 2)
                    mm_config = ttnn.MinimalMatmulConfig(
                        M_block_size=m_block,
                        K_block_size=k_block,
                        N_block_size=n_block,
                        subblock_h=sub_h,
                        subblock_w=sub_w,
                        compute_with_storage_grid_size=grid,
                    )
                candidates: list[tuple[str, str, object]] = [("minimal_shipped", "minimal", mm_config)]
                candidates += [
                    (label, "linear", cfg)
                    for label, cfg in mcast2d_candidates(
                        rows,
                        k,
                        n,
                        banks,
                        grid_y=[int(g) for g in args.grid_y.split(",")],
                        top_bw=args.top_bw,
                        out_blocks=[tuple(int(v) for v in o.split("x")) for o in args.out_blocks.split(",")],
                    )
                ]

                for label, kind, cfg in candidates:

                    def run(kind=kind, cfg=cfg):
                        if kind == "minimal":
                            return ttnn.experimental.minimal_matmul(
                                x,
                                w,
                                dtype=ttnn.bfloat16,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                compute_kernel_config=ck,
                                config=cfg,
                            )
                        return ttnn.linear(
                            x,
                            w,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=ck,
                            program_config=cfg,
                        )

                    print(f"GROUP {args.reps} {name} {label}_r{rows}", flush=True)
                    try:
                        out = run()
                        p = pcc(ttnn.to_torch(out).float(), ref)
                        ttnn.deallocate(out)
                        if not (p >= MIN_PCC):
                            print(f"CORRUPT {name:10s} rows={rows:5d} {label:22s} pcc={p}", flush=True)
                            continue
                        ms, per_round = timed(run, mesh, args.reps, args.rounds)
                        print(
                            f"RESULT {name:10s} rows={rows:5d} {label:22s} {ms:9.4f} ms  pcc={p:.6f}  "
                            f"({'/'.join(f'{r:.4f}' for r in per_round)})",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
                        print(f"BLOCKED {name} rows={rows} {label}: {msg[:250]}", flush=True)
                ttnn.deallocate(x)
            ttnn.deallocate(w)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
