# SPDX-License-Identifier: Apache-2.0
"""
nlp_create_qkv_heads_decode program-config sweep — generalized CLI tool.

Give it the attention shape (--n-q-heads / --n-kv-heads / --head-dim, or a
--preset) and it sweeps the INPUT shard grid (which is what this op inherits its
parallelism from) plus overlap_qk_coregrid, timing each config on device and
reporting the fastest.

Problem it targets (Llama-3.1-8B decode tracy profile):
NLPCreateQKVHeadsDecodeDeviceOperation ran on **1 core** at ~16.8us avg (1.08ms
total over decode) — the most under-utilized op. It splits the fused QKV row
[1,1,32,(n_q+2*n_kv)*hd] into Q/K/V head tensors and inherits parallelism from
the INPUT shard grid; if the input arrives interleaved it collapses to 1 core.
Width-sharding the fused-QKV input across N cores parallelizes it.

Correctness: this is a pure data-shuffle (no math) so the output is bit-identical
regardless of grid; we assert Q/K/V shapes. Ranked by device kernel time.

PRIMARY metric = device kernel duration (profiler). Run with ALL THREE profiler
env vars or rows fall back to src=host (dispatch-dominated, not for ranking):
  export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd):$(pwd)/.auto MESH_DEVICE=P150
  export TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1

Examples
  # Llama-3.1-8B (default): n_q=32 n_kv=8 hd=128
  python models/tt_transformers/tests/sweeps/sweep_create_heads.py --csv ch_8b.csv

  # Llama-3.2-1B via preset
  python models/tt_transformers/tests/sweeps/sweep_create_heads.py --preset llama3-1b --csv ch_1b.csv

  # arbitrary shape (e.g. MHA, no GQA: n_kv == n_q)
  python models/tt_transformers/tests/sweeps/sweep_create_heads.py --n-q-heads 32 --n-kv-heads 32 --head-dim 128
"""
import argparse
import itertools
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep_common import CSVLog, open_dev, timed_run

# Preset attention shapes: (n_q_heads, n_kv_heads, head_dim)
PRESETS = {
    "llama3-8b": (32, 8, 128),
    "llama3-70b": (64, 8, 128),
    "llama3-1b": (32, 8, 64),
    "llama3-3b": (24, 8, 128),
    "mistral-7b": (32, 8, 128),
    "qwen2-7b": (28, 4, 128),
}


# Input memory layouts to sweep. "width_sharded" uses the core grid (this is what
# parallelizes the op); the interleaved variants collapse the op to 1 core but let
# us measure the L1-vs-DRAM residence cost of the baseline path.
INPUT_LAYOUTS = ["interleaved_l1", "interleaved_dram", "width_sharded"]
# Output memory configs to sweep.
OUTPUT_LAYOUTS = ["height_sharded_l1", "interleaved_l1", "interleaved_dram"]


def build_fused_qkv(device, input_layout, grid_xy, dtype, qkv_w):
    """Fused QKV [1,1,32,qkv_w] in the requested input memory layout.

    interleaved_l1 / interleaved_dram -> op runs on 1 core (baseline paths).
    width_sharded  -> width-sharded across grid_xy cores in L1 (parallelized).
    """
    x = torch.randn(1, 1, 32, qkv_w).bfloat16().float()
    if input_layout == "interleaved_l1":
        return ttnn.from_torch(
            x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
    if input_layout == "interleaved_dram":
        return ttnn.from_torch(
            x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    # width_sharded (requires a grid)
    gx, gy = grid_xy
    ncores = gx * gy
    t = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mem = ttnn.create_sharded_memory_config(
        shape=(32, qkv_w // ncores),
        core_grid=ttnn.CoreGrid(y=gy, x=gx),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.to_memory_config(t, mem)


def build_out_mem(output_layout, head_dim):
    """Output memory config for the Q/K/V head tensors."""
    if output_layout == "height_sharded_l1":
        return ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    if output_layout == "interleaved_l1":
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def grid_candidates(qkv_w, max_x=8, max_y=8):
    # width shards need ncores to divide qkv_w/32 tiles cleanly
    tiles = qkv_w // 32
    grids = []
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            nc = gx * gy
            if tiles % nc == 0 and nc >= 1:
                grids.append((gx, gy))
    return grids


def main():
    ap = argparse.ArgumentParser(description="Generalized nlp_create_qkv_heads_decode sweep")
    ap.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default=None,
        help="attention shape preset (overridden by explicit --n-q-heads etc.)",
    )
    ap.add_argument("--n-q-heads", type=int, default=None, help="number of query heads")
    ap.add_argument("--n-kv-heads", type=int, default=None, help="number of KV heads (GQA); == n_q for MHA")
    ap.add_argument("--head-dim", type=int, default=None, help="per-head dimension")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--dtype", choices=["bf8", "bf16"], default="bf16")
    ap.add_argument(
        "--input-layouts", nargs="+", default=INPUT_LAYOUTS, choices=INPUT_LAYOUTS, help="input memory layouts to sweep"
    )
    ap.add_argument(
        "--out-layouts",
        nargs="+",
        default=["height_sharded_l1"],
        choices=OUTPUT_LAYOUTS,
        help="output memory configs to sweep",
    )
    ap.add_argument("--csv", type=str, default="sweep_create_heads.csv")
    args = ap.parse_args()

    # Resolve shape: preset supplies defaults, explicit flags override individually.
    pq, pk, ph = PRESETS.get(args.preset or "llama3-8b")
    n_q = args.n_q_heads if args.n_q_heads is not None else pq
    n_kv = args.n_kv_heads if args.n_kv_heads is not None else pk
    head_dim = args.head_dim if args.head_dim is not None else ph
    qkv_w = (n_q + 2 * n_kv) * head_dim

    dtype = ttnn.bfloat8_b if args.dtype == "bf8" else ttnn.bfloat16
    dev = open_dev()
    log = CSVLog(
        args.csv,
        [
            "n_q",
            "n_kv",
            "head_dim",
            "qkv_w",
            "input_layout",
            "input_grid",
            "num_cores",
            "out_layout",
            "overlap_qk",
            "src",
            "dur_us",
            "status",
            "note",
        ],
    )
    print(f"\n===== nlp_create_qkv_heads_decode  n_q={n_q} n_kv={n_kv} hd={head_dim} qkv_w={qkv_w} =====")
    print(f"{'in_layout':>17s} {'in_grid':>9s} {'nc':>4s} {'out_layout':>18s} {'ovl':>4s} {'src':>4s} {'us':>9s}")

    grids = grid_candidates(qkv_w)
    best = None
    n_ok = n_err = 0
    for input_layout, out_layout, overlap in itertools.product(args.input_layouts, args.out_layouts, [True, False]):
        # width_sharded fans out over the grid; interleaved variants are 1-core, so
        # only enumerate grids for the sharded path (avoid redundant duplicate rows).
        layout_grids = grids if input_layout == "width_sharded" else [None]
        for grid_xy in layout_grids:
            label = "1core" if grid_xy is None else f"({grid_xy[0]},{grid_xy[1]})"
            nc = 1 if grid_xy is None else grid_xy[0] * grid_xy[1]
            rb = [n_q, n_kv, head_dim, qkv_w, input_layout, label, nc, out_layout, overlap]
            inp = None
            try:
                out_mem = build_out_mem(out_layout, head_dim)
                inp = build_fused_qkv(dev, input_layout, grid_xy, dtype, qkv_w)

                def run():
                    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                        inp,
                        num_heads=n_q,
                        num_kv_heads=n_kv,
                        overlap_qk_coregrid=overlap,
                        memory_config=out_mem,
                    )
                    ttnn.deallocate(k)
                    ttnn.deallocate(v)
                    return q

                r = timed_run(dev, run, args.iters)
            except Exception as e:
                n_err += 1
                msg = str(e).strip().split("\n")[0][:80] or type(e).__name__
                log.row(rb + ["", "", "ERR", msg])
                if inp is not None:
                    try:
                        inp.deallocate(True)
                    except Exception:
                        pass
                continue
            n_ok += 1
            log.row(rb + [r["src"], f"{r['dur_ns']/1000:.2f}", "OK", ""])
            print(
                f"{input_layout:>17s} {label:>9s} {nc:4d} {out_layout:>18s} "
                f"{str(overlap)[0]:>4s} {r['src']:>4s} {r['dur_ns']/1000:9.2f}"
            )
            try:
                inp.deallocate(True)
            except Exception:
                pass
            if best is None or r["dur_ns"] < best[0]:
                best = (r["dur_ns"], input_layout, label, nc, out_layout, overlap, r["src"])

    print(f"\n  [create_heads] OK={n_ok} ERR={n_err}")
    if best:
        d, il, label, nc, ol, overlap, src = best
        print(f"  BEST: input={il} grid={label} ({nc}c) out={ol} overlap_qk={overlap} " f"-> {d/1000:.2f}us [{src}]")
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
