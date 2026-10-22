# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Fused SwiGLU front half vs the split path, at the paper shape (one chip).

    python models/experimental/timestral/tests/perf/sweep_swiglu.py <out_dir> [--precision accuracy|performance]
        [--rows 98304]

Computes ``gate * silu(up)`` for ``x (rows, 768)``, up / gate ``768 -> 2048``:

    split    the model's current path: up linear (silu fused), gate linear, mul
    fused    ttnn.experimental.minimal_matmul(fuse_swiglu=True) on one 768 x 4096 weight whose column
             tiles alternate [up_t0, gate_t0, up_t1, ...] (the kernel emits silu(even) * odd)

Time per call is the median wall time of synchronized calls. Writes ``swiglu_sweep_<precision>.json``
and ``.md`` to ``out_dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch

TILE = 32
TIMED_RUNS = 5
D, HIDDEN = 768, 2048


def interleave_swiglu(w_silu: torch.Tensor, w_mul: torch.Tensor) -> torch.Tensor:
    """Two ``(in, n)`` weights -> ``(in, 2n)`` with alternating column tiles, silu half first."""
    k, n = w_silu.shape
    pair = torch.stack([w_silu.reshape(k, n // TILE, TILE), w_mul.reshape(k, n // TILE, TILE)], dim=2)
    return pair.reshape(k, 2 * n)


def _even_subblock(block_h: int, block_w: int, max_tiles: int = 8) -> tuple[int, int]:
    """Largest ``h * w <= max_tiles`` dividing the block with an even ``w`` (SwiGLU runs in the K loop)."""
    best = (1, 2)
    for h in range(1, max_tiles + 1):
        for w in range(2, max_tiles // h + 1, 2):
            if block_h % h == 0 and block_w % w == 0 and (h * w, w) > (best[0] * best[1], best[1]):
                best = (h, w)
    return best


def _time(fn, device) -> float:
    import ttnn

    times = []
    for _ in range(TIMED_RUNS):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        times.append(time.perf_counter() - start)
        ttnn.deallocate(out)
    return 1e3 * statistics.median(times)


def sweep(device, rows: int, precision) -> list[dict]:
    import ttnn

    from models.experimental.timestral.tt.common import linear, upload
    from tests.ttnn.utils_for_testing import comp_pcc

    torch.manual_seed(0)
    x = torch.randn(rows, D).to(torch.bfloat16).float()
    w_up = (torch.randn(HIDDEN, D) / math.sqrt(D)).to(torch.bfloat16).float()
    w_gate = (torch.randn(HIDDEN, D) / math.sqrt(D)).to(torch.bfloat16).float()
    expected = (x @ w_gate.t()) * torch.nn.functional.silu(x @ w_up.t())

    dtype = precision.weight_dtype()
    x_tt = upload(device, x.reshape(1, 1, rows, D))
    up_tt = upload(device, w_up.t(), dtype=dtype)
    gate_tt = upload(device, w_gate.t(), dtype=dtype)
    fused_tt = upload(device, interleave_swiglu(w_up.t(), w_gate.t()), dtype=dtype)
    kernel = precision.compute_kernel_config()
    grid = device.compute_with_storage_grid_size()

    def split():
        up = linear(x_tt, up_tt, precision, activation="silu")
        gate = linear(x_tt, gate_tt, precision)
        out = ttnn.mul(gate, up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)
        return out

    candidates = [("split (current)", split)]
    for mb in (4, 8, 16):
        for kb in (4, 6, 8, 12, 24):
            for nb in (2, 4, 6, 8, 14):
                if 2 * mb * kb + 2 * kb * nb + 2 * mb * nb > 720:
                    continue
                sh, sw = _even_subblock(mb, nb)
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=kb,
                    N_block_size=nb,
                    subblock_h=sh,
                    subblock_w=sw,
                    compute_with_storage_grid_size=grid,
                )

                def fused(cfg=cfg):
                    return ttnn.experimental.minimal_matmul(
                        x_tt,
                        fused_tt,
                        config=cfg,
                        fuse_swiglu=True,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=kernel,
                    )

                candidates.append((f"fused[{mb},{kb},{nb},sb={sh}x{sw}]", fused))

    results = []
    for name, fn in candidates:
        entry = {"candidate": name}
        try:
            out = fn()
            ttnn.synchronize_device(device)
            got = ttnn.to_torch(out).float().reshape(rows, HIDDEN)
            ttnn.deallocate(out)
            entry["pcc"] = comp_pcc(expected, got, 0.0)[1]
            entry["ms"] = _time(fn, device)
            print(f"  {name:32s} {entry['ms']:8.3f} ms  pcc {entry['pcc']:.5f}", flush=True)
        except Exception as e:  # noqa: BLE001 - illegal configs are recorded, not fatal
            entry["error"] = str(e).strip().splitlines()[0][:200]
            print(f"  {name:32s} FAILED: {entry['error']}", flush=True)
        results.append(entry)
    return results


def main() -> None:
    import ttnn

    from models.experimental.timestral.tt.common import TtTimestralPrecision

    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--precision", choices=["accuracy", "performance"], default="accuracy")
    parser.add_argument("--rows", type=int, default=1024 * 96)
    args = parser.parse_args()
    precision = TtTimestralPrecision.performance() if args.precision == "performance" else TtTimestralPrecision()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        results = sweep(device, args.rows, precision)
    finally:
        ttnn.close_mesh_device(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"swiglu_sweep_{args.precision}.json").write_text(json.dumps(results, indent=2) + "\n")
    ok = sorted((r for r in results if "ms" in r), key=lambda r: r["ms"])
    lines = [
        f"# Fused SwiGLU sweep ({args.precision}, rows {args.rows})",
        "",
        "| candidate | ms | PCC / error |",
        "|---|---|---|",
    ]
    lines += [f"| `{r['candidate']}` | {r['ms']:.3f} | {r['pcc']:.5f} |" for r in ok]
    lines += [f"| `{r['candidate']}` | — | FAILED: {r['error']} |" for r in results if "error" in r]
    (args.out_dir / f"swiglu_sweep_{args.precision}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
