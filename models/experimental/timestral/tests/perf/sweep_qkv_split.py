# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Attention projections with split outputs, at the paper shape (one chip).

    python models/experimental/timestral/tests/perf/sweep_qkv_split.py <out_dir> [--precision accuracy|performance]
        [--rows 98304] [--chunks 3|4|2]

Computes ``x (rows, 768) @ W (768, chunks * 768)`` with
``ttnn.experimental.minimal_matmul_split(chunks)``, one (rows, 768) output per chunk, against a baseline:

    chunks 3  [Wq | Wk | Wv]       vs the fused 2D linear (one (rows, 2304) output)
    chunks 4  [Wq | Wk | Wv | Wg]  vs the 3-chunk split plus the separate wg + sigmoid linear
    chunks 2  [Wv | Wg]            vs the separate wv and wg + sigmoid linears (univariate V-path)

Time per call is the median wall time of synchronized calls. Writes ``<name>_sweep_<precision>.json``
and ``.md`` to ``out_dir`` (name: qkv_split, qkvg_split, vg_split).
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
NAMES = {3: "qkv_split", 4: "qkvg_split", 2: "vg_split"}


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


def _baseline(x_tt, w: torch.Tensor, chunks: int, precision, device):
    """``(name, fn, sigmoid_last)``: the path the model ran before packing these chunks."""
    from models.experimental.timestral.tt.common import linear, linear_split, upload

    dtype = precision.weight_dtype()
    if chunks == 3:
        w_tt = upload(device, w, dtype=dtype)
        return "2d linear", lambda: linear(x_tt, w_tt, precision), False
    head, wg = upload(device, w[:, :-D], dtype=dtype), upload(device, w[:, -D:], dtype=dtype)
    if chunks == 4:
        fn = lambda: [*linear_split(x_tt, head, precision, 3), linear(x_tt, wg, precision, activation="sigmoid")]
        return "split 3 + wg/sigmoid (current)", fn, True
    fn = lambda: [linear(x_tt, head, precision), linear(x_tt, wg, precision, activation="sigmoid")]
    return "wv + wg/sigmoid (current)", fn, True


def sweep(device, rows: int, precision, chunks: int) -> list[dict]:
    import ttnn

    from models.experimental.timestral.tt.common import upload
    from tests.ttnn.utils_for_testing import comp_pcc

    torch.manual_seed(0)
    x = torch.randn(rows, D).to(torch.bfloat16).float()
    w = (torch.randn(D, chunks * D) / math.sqrt(D)).to(torch.bfloat16).float()
    expected = x @ w

    x_tt = upload(device, x.reshape(1, 1, rows, D))
    w_tt = upload(device, w, dtype=precision.weight_dtype())
    kernel = precision.compute_kernel_config()
    grid = device.compute_with_storage_grid_size()

    name, fn, sigmoid_last = _baseline(x_tt, w, chunks, precision, device)
    baseline_expected = torch.cat([expected[:, :-D], expected[:, -D:].sigmoid()], -1) if sigmoid_last else expected
    candidates = [(name, fn, baseline_expected)]
    for mb in (4, 8, 16):
        for kb in (4, 6, 8, 12):
            for nb in (2, 3, 4, 6, 8, 12):
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
                            chunks=chunks,
                            config=cfg,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            compute_kernel_config=kernel,
                        )

                    candidates.append((f"split[{mb},{kb},{nb},sb={sh}x{sw}]", split, expected))

    results = []
    for name, fn, expected in candidates:
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
    parser.add_argument("--chunks", type=int, choices=sorted(NAMES), default=3)
    args = parser.parse_args()
    precision = TtTimestralPrecision.performance() if args.precision == "performance" else TtTimestralPrecision()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        results = sweep(device, args.rows, precision, args.chunks)
    finally:
        ttnn.close_mesh_device(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = NAMES[args.chunks]
    (args.out_dir / f"{name}_sweep_{args.precision}.json").write_text(json.dumps(results, indent=2) + "\n")
    ok = sorted((r for r in results if "ms" in r), key=lambda r: r["ms"])
    lines = [
        f"# {name} sweep: {args.chunks} chunks ({args.precision}, rows {args.rows})",
        "",
        "| candidate | ms | PCC / error |",
        "|---|---|---|",
    ]
    lines += [f"| `{r['candidate']}` | {r['ms']:.3f} | {r['pcc']:.5f} |" for r in ok]
    lines += [f"| `{r['candidate']}` | — | FAILED: {r['error']} |" for r in results if "error" in r]
    (args.out_dir / f"{name}_sweep_{args.precision}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:14]))


if __name__ == "__main__":
    main()
