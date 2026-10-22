# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Block / subblock sweep of the context-stream matmuls at the paper shape (one chip).

    python models/experimental/timestral/tests/perf/sweep_subblocks.py <out_dir> [--precision accuracy|performance]
        [--rows 65536] [--ops qkv_split,kv_split,...] [--configs "16,8,4,4,1;8,12,2,2,2"] [--batches 3] [--tag NAME]

``--configs`` replaces the block / subblock grid with the listed (M, K, N, h, w) configs (plus the shipped one).

Every op runs as the model calls it (``minimal_matmul_split``, ``dit_minimal_matmul_addcmul_fused`` with the
residual, ``minimal_matmul``, ``minimal_matmul(fuse_swiglu=True)``), on (rows, K) bf16 activations. Each
candidate is a ``MinimalMatmulConfig`` (M, K, N block, subblock h x w); the first row of every op is the
config the model ships. Time per call is the median over 3 batches of 10 back-to-back calls, so host
dispatch is amortized. Writes ``subblock_sweep_<precision>.json`` and ``.md`` to ``out_dir``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

D = 768
BATCHES, CALLS = 3, 10


@dataclass(frozen=True)
class Op:
    kind: str  # split | addcmul | minimal | swiglu
    k: int
    n: int  # weight columns
    chunks: int = 1
    blocks: tuple = ()  # (M, K, N) block candidates, the shipped one first


OPS = {
    "qkv_split": Op("split", D, 3 * D, 3, ((16, 12, 4), (16, 8, 4), (8, 12, 4), (16, 12, 6), (16, 12, 8))),
    "kv_split": Op("split", D, 2 * D, 2, ((16, 4, 6), (16, 8, 6), (16, 12, 4), (16, 12, 6), (8, 8, 8))),
    "wo_residual": Op("addcmul", D, D, blocks=((8, 8, 8), (16, 8, 4), (16, 8, 8), (8, 12, 8), (16, 12, 4))),
    "w_down_residual": Op("addcmul", 2048, D, blocks=((8, 8, 8), (16, 8, 4), (16, 8, 8), (8, 16, 8), (16, 16, 4))),
    "wv": Op("minimal", D, D, blocks=((8, 8, 8), (16, 8, 4), (16, 8, 8), (8, 12, 8), (16, 12, 4))),
    "swiglu": Op("swiglu", D, 4096, blocks=((8, 12, 2), (16, 12, 2), (8, 8, 2), (16, 6, 2), (8, 12, 4))),
    "patch_out": Op("minimal", 3072, D, blocks=((8, 8, 8), (16, 8, 4), (16, 12, 4))),
    "patch_skip": Op("minimal", 64, D, blocks=((8, 2, 8), (16, 2, 4))),
}


def _subblocks(block_h: int, block_w: int, current: tuple[int, int] | None, max_tiles: int = 8):
    """Every subblock dividing the block with at least 4 tiles (or the whole block if smaller), current first."""
    fits = [
        (h, w)
        for h in range(1, block_h + 1)
        for w in range(1, block_w + 1)
        if block_h % h == 0 and block_w % w == 0 and h * w <= max_tiles and h * w >= min(4, block_h * block_w)
    ]
    fits.sort(key=lambda s: (-s[0] * s[1], -s[1]))
    return list(dict.fromkeys(([current] if current in fits else []) + fits))


def _shipped(name: str, op: Op, rows: int, grid, precision):
    """The (M, K, N, sub_h, sub_w) the model builds for this op."""
    from models.experimental.timestral.tt import program_configs

    tiles = rows // 32, op.k // 32, op.n // 32
    if op.kind == "split":
        cfg = program_configs.split_config(grid, tiles[0], tiles[1], tiles[2] // op.chunks, op.chunks)
    elif op.kind == "swiglu":
        cfg = program_configs.swiglu_config(grid, tiles[0], tiles[1], tiles[2] // 2)
    else:
        cfg = program_configs.minimal_matmul_config(grid, *tiles)
    return cfg.M_block_size, cfg.K_block_size, cfg.N_block_size, cfg.subblock_h, cfg.subblock_w


def _time(fn, device) -> float:
    import ttnn

    per_call = []
    for _ in range(BATCHES):
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        outs = [fn() for _ in range(CALLS)]
        ttnn.synchronize_device(device)
        per_call.append((time.perf_counter() - start) / CALLS)
        for out in outs:
            for t in out if isinstance(out, (list, tuple)) else [out]:
                ttnn.deallocate(t)
    return 1e3 * statistics.median(per_call)


def sweep_op(device, name: str, op: Op, rows: int, precision, configs=None) -> list[dict]:
    import ttnn

    from models.experimental.timestral.tt.common import upload, upload_swiglu
    from tests.ttnn.utils_for_testing import comp_pcc

    torch.manual_seed(0)
    x = torch.randn(rows, op.k).to(torch.bfloat16).float()
    w = (torch.randn(op.k, op.n) / math.sqrt(op.k)).to(torch.bfloat16).float()
    dtype = precision.weight_dtype(residual=op.kind == "addcmul")
    x_tt = upload(device, x.reshape(1, 1, rows, op.k))
    kernel = precision.compute_kernel_config()
    grid = device.compute_with_storage_grid_size()
    extra = {}
    if op.kind == "swiglu":
        half = op.n // 2
        w_tt = upload_swiglu(device, w[:, :half].t(), w[:, half:].t(), dtype)
        expected = (x @ w[:, half:]) * torch.nn.functional.silu(x @ w[:, :half])
    else:
        w_tt = upload(device, w, dtype=dtype)
        expected = x @ w
    if op.kind == "addcmul":
        residual = torch.randn(rows, op.n).to(torch.bfloat16).float()
        extra["residual"] = upload(device, residual.reshape(1, 1, rows, op.n))
        extra["ones"] = upload(device, torch.ones(1, op.n))
        expected = expected + residual

    def run(cfg):
        if op.kind == "split":
            return ttnn.experimental.minimal_matmul_split(
                x_tt,
                w_tt,
                chunks=op.chunks,
                config=cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=kernel,
            )
        if op.kind == "addcmul":
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x_tt,
                w_tt,
                1.0,
                extra["residual"],
                extra["ones"],
                config=cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=kernel,
            )
        return ttnn.experimental.minimal_matmul(
            x_tt,
            w_tt,
            config=cfg,
            fuse_swiglu=op.kind == "swiglu",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=kernel,
        )

    shipped = _shipped(name, op, rows, (grid.x, grid.y), precision)
    candidates = [shipped]
    if configs is not None:
        candidates += [c for c in configs if c != shipped]
    else:
        for mb, kb, nb in dict.fromkeys([shipped[:3], *op.blocks]):
            current = shipped[3:] if (mb, kb, nb) == shipped[:3] else None
            candidates += [(mb, kb, nb, sh, sw) for sh, sw in _subblocks(mb, nb, current)]
        candidates = list(dict.fromkeys(candidates))
    results = []
    for mb, kb, nb, sh, sw in candidates:
        label = f"[{mb},{kb},{nb}] sb {sh}x{sw}" + (" (shipped)" if (mb, kb, nb, sh, sw) == shipped else "")
        entry = {"op": name, "candidate": label, "config": [mb, kb, nb, sh, sw]}
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=grid,
        )
        try:
            out = run(cfg)
            parts = out if isinstance(out, (list, tuple)) else [out]
            got = torch.cat([ttnn.to_torch(t).float().reshape(rows, -1) for t in parts], dim=-1)
            for t in parts:
                ttnn.deallocate(t)
            entry["pcc"] = comp_pcc(expected, got, 0.0)[1]
            entry["ms"] = _time(lambda: run(cfg), device)
            print(f"  {name:16s} {label:34s} {entry['ms']:7.3f} ms  pcc {entry['pcc']:.5f}", flush=True)
        except Exception as e:  # noqa: BLE001 - illegal configs are recorded, not fatal
            entry["error"] = str(e).strip().splitlines()[0][:200]
            print(f"  {name:16s} {label:34s} FAILED: {entry['error']}", flush=True)
        results.append(entry)
    for t in (x_tt, w_tt, *extra.values()):
        ttnn.deallocate(t)
    return results


def main() -> None:
    global BATCHES
    import ttnn

    from models.experimental.timestral.tt.common import TtTimestralPrecision

    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--precision", choices=["accuracy", "performance"], default="accuracy")
    parser.add_argument("--rows", type=int, default=1024 * 64)
    parser.add_argument("--ops", default=",".join(OPS))
    parser.add_argument("--configs", default=None)
    parser.add_argument("--batches", type=int, default=BATCHES)
    parser.add_argument("--tag", default="subblock_sweep")
    args = parser.parse_args()
    BATCHES = args.batches
    configs = None
    if args.configs:
        configs = [tuple(int(v) for v in c.split(",")) for c in args.configs.split(";")]
    precision = TtTimestralPrecision.performance() if args.precision == "performance" else TtTimestralPrecision()
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    results = []
    try:
        for name in args.ops.split(","):
            results += sweep_op(device, name, OPS[name], args.rows, precision, configs)
    finally:
        ttnn.close_mesh_device(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.tag}_{args.precision}.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [f"# Subblock sweep ({args.precision}, rows {args.rows}, {BATCHES} x {CALLS} calls per candidate)", ""]
    for name in args.ops.split(","):
        ok = sorted((r for r in results if r["op"] == name and "ms" in r), key=lambda r: r["ms"])
        shipped = next((r for r in ok if "(shipped)" in r["candidate"]), None)
        lines += [f"## {name}", "", "| candidate | ms | vs shipped | PCC |", "|---|---|---|---|"]
        for r in ok[:8] + ([shipped] if shipped and shipped not in ok[:8] else []):
            rel = f"{100 * (r['ms'] / shipped['ms'] - 1):+.1f} %" if shipped else "—"
            lines.append(f"| `{r['candidate']}` | {r['ms']:.3f} | {rel} | {r['pcc']:.5f} |")
        failed = [r for r in results if r["op"] == name and "error" in r]
        if failed:
            lines.append(f"| {len(failed)} configs failed | — | — | e.g. {failed[0]['error'][:80]} |")
        lines.append("")
    (args.out_dir / f"{args.tag}_{args.precision}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
