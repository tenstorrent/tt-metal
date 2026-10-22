# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""QKV projection with split outputs vs the fused 2D linear, at the paper shape (one chip).

    python models/experimental/timestral/tests/perf/sweep_qkv_split.py <out_dir> [--precision accuracy|performance]
        [--rows 98304]

Computes ``x (rows, 768) @ Wqkv (768, 2304)``:

    2d linear  the model's current path: one fused (rows, 2304) output
    split      ttnn.experimental.minimal_matmul_split(chunks=3): three (rows, 768) outputs, the layout
               the fused head-split + RoPE post op reads

Time per call is the median wall time of synchronized calls. Writes ``qkv_split_sweep_<precision>.json``
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
D = 768


def _subblocks(block_h: int, block_w: int, max_tiles: int = 8) -> list[tuple[int, int]]:
    """The widest and the tallest largest-area subblocks dividing the block."""
    fits = [
        (h, w)
        for h in range(1, block_h + 1)
        for w in range(1, block_w + 1)
        if block_h % h == 0 and block_w % w == 0 and h * w <= max_tiles
    ]
    widest = max(fits, key=lambda s: (s[0] * s[1], s[1]))
    tallest = max(fits, key=lambda s: (s[0] * s[1], s[0]))
    return list(dict.fromkeys([widest, tallest]))


def _time(fn, device) -> float:
    import ttnn

    times = []
    for _ in range(TIMED_RUNS):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        outs = fn()
        ttnn.synchronize_device(device)
        times.append(time.perf_counter() - start)
        for out in outs if isinstance(outs, (list, tuple)) else [outs]:
            ttnn.deallocate(out)
    return 1e3 * statistics.median(times)


def sweep(device, rows: int, precision) -> list[dict]:
    import ttnn

    from models.experimental.timestral.tt.common import linear, upload
    from tests.ttnn.utils_for_testing import comp_pcc

    torch.manual_seed(0)
    x = torch.randn(rows, D).to(torch.bfloat16).float()
    w = (torch.randn(D, 3 * D) / math.sqrt(D)).to(torch.bfloat16).float()
    expected = x @ w

    x_tt = upload(device, x.reshape(1, 1, rows, D))
    w_tt = upload(device, w, dtype=precision.weight_dtype())
    kernel = precision.compute_kernel_config()
    grid = device.compute_with_storage_grid_size()

    candidates = [("2d linear (current)", lambda: linear(x_tt, w_tt, precision))]
    for mb in (4, 8, 16):
        for kb in (4, 6, 8, 12):
            for nb in (3, 4, 6, 8, 9, 12):
                if 2 * mb * kb + 2 * kb * nb + 2 * mb * nb > 720:
                    continue
                for sh, sw in _subblocks(mb, nb):
                    cfg = ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        subblock_h=sh,
                        subblock_w=sw,
                        compute_with_storage_grid_size=grid,
                    )

                    def split(cfg=cfg):
                        return ttnn.experimental.minimal_matmul_split(
                            x_tt,
                            w_tt,
                            chunks=3,
                            config=cfg,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=kernel,
                        )

                    candidates.append((f"split[{mb},{kb},{nb},sb={sh}x{sw}]", split))

    results = []
    for name, fn in candidates:
        entry = {"candidate": name}
        try:
            outs = fn()
            ttnn.synchronize_device(device)
            parts = outs if isinstance(outs, (list, tuple)) else [outs]
            got = torch.cat([ttnn.to_torch(t).float().reshape(rows, -1) for t in parts], dim=-1)
            for t in parts:
                ttnn.deallocate(t)
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
    (args.out_dir / f"qkv_split_sweep_{args.precision}.json").write_text(json.dumps(results, indent=2) + "\n")
    ok = sorted((r for r in results if "ms" in r), key=lambda r: r["ms"])
    lines = [
        f"# QKV split-output sweep ({args.precision}, rows {args.rows})",
        "",
        "| candidate | ms | PCC / error |",
        "|---|---|---|",
    ]
    lines += [f"| `{r['candidate']}` | {r['ms']:.3f} | {r['pcc']:.5f} |" for r in ok]
    lines += [f"| `{r['candidate']}` | — | FAILED: {r['error']} |" for r in results if "error" in r]
    (args.out_dir / f"qkv_split_sweep_{args.precision}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
