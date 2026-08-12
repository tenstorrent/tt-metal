# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where the decode SwiGLU / attention-gate multiplies actually spend their time.

The optimized decode step's two activation-folded multiplies cost more in absolute
terms than they did in the fused stage, even though everything around them got
faster:

============================  ==============  ==============
op                            fused (110 c)   optimized
============================  ==============  ==============
SwiGLU ``mul(..., SILU)``     14.23 us        40.47 us (26 c)
attn gate ``mul(..., SIGMOID)``  5.96 us       14.28 us (16 c)
plain residual add            --              1.72 us (16 c)
============================  ==============  ==============

The fused stage ran them on 110 DRAM-interleaved cores; this stage runs them on the
phase-specific width-sharded L1 grids the matmuls need.  The obvious hypothesis is
that the SFPU transcendental is the cost and more cores would fix it.  **This probe
refutes that**: dropping the activation entirely (the "floor, wrong math" row) moves
the SwiGLU case by ~12 us and the attention-gate case by ~0.4 us, so the multiply
itself on a narrow shard is nearly all of it -- and every wider-grid candidate is
worse, because resharding a 19968-wide tensor twice costs more than the multiply
saves.

Two candidates are worth separating, because they fail for different reasons:

1. fold the activation into the **matmul** (``fused_activation``).  Measured at
   whole-layer level and 4.4-4.6 % *slower*: that op has only 12 worker cores.
   See ``DECODE_FUSED_ACTIVATION`` in ``tt/optimized_decoder.py``.
2. keep it on the elementwise op but stop starving it -- either split the unary
   out, or move the multiply to a wider grid and pay the reshards.

This probe measures family 2 at the real decode shapes, in isolation, so the
reshard cost is explicit rather than buried.

**Read the absolute numbers as an upper bound with a floor under them.** Timing is
host wall clock around untraced dispatches, so every row carries the same launch
overhead -- the attention-gate rows land near 29 us where the committed device row
for the same op is 14 us. Differences *between* rows at the same shape are the
result; the absolute values are not comparable to a `tt-perf-report` row.

    python .../bench/decode_elementwise_probe.py
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import width_sharded_l1

TILE = 32

#: ``label -> (rows, width, shard cores, activation, operand)`` -- the two real
#: decode multiplies, each with the unary the layer actually folds into it and the
#: operand it folds it onto.  These differ: SwiGLU is ``silu(gate) * up`` (operand
#: a) and the attention gate is ``out * sigmoid(gate)`` (operand **b**).
CASES = {
    "swiglu_mlp": (32, 19968, 26, ttnn.UnaryOpType.SILU, "a"),
    "attn_gate": (32, 4096, 16, ttnn.UnaryOpType.SIGMOID, "b"),
}


def timed(fn, mesh, reps: int, rounds: int) -> float:
    for _ in range(3):
        out = fn()
        ttnn.deallocate(out)
    ttnn.synchronize_device(mesh)
    best = float("inf")
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        best = min(best, (time.perf_counter() - t0) / reps * 1e6)
    return best


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    if not bool(torch.isfinite(a).all()):
        return float("nan")
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    ttnn.SetDefaultDevice(mesh)
    try:
        grid = mesh.compute_with_storage_grid_size()
        wide = grid.x * grid.y
        for label, (rows, width, cores, act, operand) in CASES.items():
            torch.manual_seed(11)
            a_t = torch.randn(1, 1, rows, width)
            b_t = torch.randn(1, 1, rows, width)
            if operand == "a":
                ref = (torch.nn.functional.silu(a_t.to(torch.float64)) * b_t.to(torch.float64)).float()
                act_kwargs = {"input_tensor_a_activations": [act]}
            else:
                ref = (a_t.to(torch.float64) * torch.sigmoid(b_t.to(torch.float64))).float()
                act_kwargs = {"input_tensor_b_activations": [act]}
            memcfg = width_sharded_l1(rows, width, cores, grid)
            a = ttnn.from_torch(a_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memcfg)
            b = ttnn.from_torch(b_t, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=memcfg)

            def folded():
                return ttnn.mul(a, b, dtype=ttnn.bfloat16, memory_config=memcfg, **act_kwargs)

            def plain():
                return ttnn.mul(a, b, dtype=ttnn.bfloat16, memory_config=memcfg)

            def split_unary():
                unary = ttnn.silu if act == ttnn.UnaryOpType.SILU else ttnn.sigmoid
                if operand == "a":
                    applied = unary(a, memory_config=memcfg)
                    out = ttnn.mul(applied, b, dtype=ttnn.bfloat16, memory_config=memcfg)
                else:
                    applied = unary(b, memory_config=memcfg)
                    out = ttnn.mul(a, applied, dtype=ttnn.bfloat16, memory_config=memcfg)
                ttnn.deallocate(applied)
                return out

            candidates = {
                f"folded_mul_{cores}c (shipped)": folded,
                f"plain_mul_{cores}c (floor, wrong math)": plain,
                f"{'silu' if act == ttnn.UnaryOpType.SILU else 'sigmoid'}_then_mul_{cores}c": split_unary,
            }

            # Wider-grid candidates: reshard both inputs out, multiply, reshard back,
            # so the layout contract the matmuls need is preserved.
            for wide_cores in sorted({wide, 52, 104}):
                if (width // TILE) % wide_cores:
                    continue
                wide_memcfg = width_sharded_l1(rows, width, wide_cores, grid)

                def wider(wide_memcfg=wide_memcfg):
                    aw = ttnn.reshard(a, wide_memcfg)
                    bw = ttnn.reshard(b, wide_memcfg)
                    outw = ttnn.mul(aw, bw, dtype=ttnn.bfloat16, memory_config=wide_memcfg, **act_kwargs)
                    ttnn.deallocate(aw)
                    ttnn.deallocate(bw)
                    out = ttnn.reshard(outw, memcfg)
                    ttnn.deallocate(outw)
                    return out

                candidates[f"reshard_to_{wide_cores}c_mul_reshard_back"] = wider

            for name, fn in candidates.items():
                try:
                    us = timed(fn, mesh, args.reps, args.rounds)
                    out = fn()
                    p = pcc(ttnn.to_torch(out).float(), ref)
                    ttnn.deallocate(out)
                    print(f"EW {label:11s} {name:42s} {us:8.2f} us  pcc={p:.6f}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
                    print(f"EW {label:11s} {name:42s} BLOCKED: {msg[:200]}", flush=True)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
