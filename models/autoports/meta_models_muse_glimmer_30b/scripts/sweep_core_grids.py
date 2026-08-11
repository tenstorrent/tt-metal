#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep legal Blackhole core grids for the decoder's three SDPA call sites.

This is Blackhole (11 x 10 compute grid), not Wormhole, so the usual 8x8 / 8x4 program
configs are not a given: they are simply *one* legal option among many here. The sweep
measures the SDPA ops in isolation at the layer's real shapes (32 Q heads / 2 KV heads,
head_dim 128, sliding window 2048) so the signal is not buried under the MLP matmuls, and
writes both a CSV and a markdown summary under ``doc/functional_decoder/perf/``.

    python models/autoports/meta_models_muse_glimmer_30b/scripts/sweep_core_grids.py
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch

import ttnn

OUT_DIR = Path(__file__).resolve().parents[1] / "doc" / "functional_decoder" / "perf"

HEAD_DIM = 128
NUM_HEADS = 32
NUM_KV_HEADS = 2
SLIDING_WINDOW = 2048
PREFILL_SEQ = 8192
DECODE_CONTEXT = 4096
# The decode SDPA parallelises over (user, KV head) pairs, so the best grid depends on the
# batch: sweep the target serving batch as well as batch 1.
DECODE_BATCHES = (1, 32)
BLOCK_SIZE = 64
ITERS = 10
WARMUP = 3


def _grids(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    candidates = [
        (grid.x, grid.y),  # 11x10, the full Blackhole compute grid
        (grid.x, grid.y - 2),
        (grid.x - 1, grid.y),
        (grid.x - 3, grid.y),
        (grid.x, 5),
        (8, 8),  # the Wormhole-shaped default, measured rather than assumed
        (8, 4),
        (4, 4),
    ]
    legal = []
    for x, y in candidates:
        if 0 < x <= grid.x and 0 < y <= grid.y and (x, y) not in legal:
            legal.append((x, y))
    return legal


def _timed(fn):
    for _ in range(WARMUP):
        out = fn()
        out.deallocate(True)
    ttnn.synchronize_device(_timed.mesh)
    start = time.perf_counter()
    for _ in range(ITERS):
        out = fn()
        out.deallocate(True)
    ttnn.synchronize_device(_timed.mesh)
    return 1000.0 * (time.perf_counter() - start) / ITERS


def main() -> int:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    _timed.mesh = mesh
    mapper = ttnn.ReplicateTensorToMesh(mesh)
    # The same compute-kernel config FunctionalDecoder passes to every SDPA call. Measuring
    # under a different fidelity would rank grids under a policy the model does not use
    # (HiFi4 + fp32 dest accumulation roughly triples SDPA math time versus the op default).
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def dev(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(tensor, dtype=dtype, layout=layout, device=mesh, mesh_mapper=mapper)

    torch.manual_seed(0)
    rows: list[dict] = []

    # --- 1. sliding-window prefill SDPA (in-memory K/V, causal + window) ---------------
    q = dev(torch.randn(1, NUM_HEADS, PREFILL_SEQ, HEAD_DIM))
    k = dev(torch.randn(1, NUM_KV_HEADS, PREFILL_SEQ, HEAD_DIM))
    v = dev(torch.randn(1, NUM_KV_HEADS, PREFILL_SEQ, HEAD_DIM))
    for grid in _grids(mesh):
        for q_chunk, k_chunk in ((128, 128), (256, 128), (512, 128), (1024, 128), (128, 64), (256, 256), (512, 256)):
            program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
            try:
                ms = _timed(
                    lambda pc=program_config: ttnn.transformer.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        is_causal=True,
                        scale=HEAD_DIM**-0.5,
                        sliding_window_size=SLIDING_WINDOW,
                        program_config=pc,
                        compute_kernel_config=compute_kernel_config,
                    )
                )
                status = "ok"
            except Exception as exc:  # a grid/chunk combination the op rejects
                ms, status = float("nan"), type(exc).__name__
            rows.append(
                {
                    "call_site": "prefill_sdpa_sliding",
                    "grid": f"{grid[0]}x{grid[1]}",
                    "cores": grid[0] * grid[1],
                    "batch": 1,
                    "q_chunk": q_chunk,
                    "k_chunk": k_chunk,
                    "seq_len": PREFILL_SEQ,
                    "ms_per_call": ms,
                    "status": status,
                }
            )
            print(rows[-1], flush=True)
    for tensor in (q, k, v):
        tensor.deallocate(True)

    # --- 2. chunked prefill SDPA over the paged cache (full-attention layers) ---------
    blocks = PREFILL_SEQ // BLOCK_SIZE
    k_cache = dev(torch.randn(blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM))
    v_cache = dev(torch.randn(blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM))
    page_table = dev(
        torch.randperm(blocks).to(torch.int32).reshape(1, blocks), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    q_chunk_tensor = dev(torch.randn(1, NUM_HEADS, PREFILL_SEQ, HEAD_DIM))
    for grid in _grids(mesh):
        for q_chunk, k_chunk in ((128, 128), (256, 128), (512, 128), (1024, 128), (256, 256), (512, 256)):
            program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
            try:
                ms = _timed(
                    lambda pc=program_config: ttnn.transformer.chunked_scaled_dot_product_attention(
                        q_chunk_tensor,
                        k_cache,
                        v_cache,
                        page_table,
                        0,
                        program_config=pc,
                        compute_kernel_config=compute_kernel_config,
                    )
                )
                status = "ok"
            except Exception as exc:
                ms, status = float("nan"), type(exc).__name__
            rows.append(
                {
                    "call_site": "prefill_sdpa_chunked",
                    "grid": f"{grid[0]}x{grid[1]}",
                    "cores": grid[0] * grid[1],
                    "batch": 1,
                    "q_chunk": q_chunk,
                    "k_chunk": k_chunk,
                    "seq_len": PREFILL_SEQ,
                    "ms_per_call": ms,
                    "status": status,
                }
            )
            print(rows[-1], flush=True)
    for tensor in (q_chunk_tensor,):
        tensor.deallocate(True)

    # --- 3. paged decode SDPA ---------------------------------------------------------
    blocks_per_user = (DECODE_CONTEXT + BLOCK_SIZE) // BLOCK_SIZE
    for batch in DECODE_BATCHES:
        decode_blocks = blocks_per_user * batch
        k_dec = dev(torch.randn(decode_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM))
        v_dec = dev(torch.randn(decode_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM))
        pt_dec = dev(
            torch.randperm(decode_blocks).to(torch.int32).reshape(batch, blocks_per_user),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        cur_pos = dev(
            torch.full((batch,), DECODE_CONTEXT, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        q_dec = dev(torch.randn(1, batch, NUM_HEADS, HEAD_DIM))
        for grid in _grids(mesh):
            for k_chunk in (32, 64, 128, 256):
                program_config = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
                    q_chunk_size=32,
                    k_chunk_size=k_chunk,
                    exp_approx_mode=False,
                )
                for window in (SLIDING_WINDOW, None):
                    # A grid with fewer than batch * num_kv_heads cores passes the op's own
                    # validation but silently folds KV heads onto one core and returns wrong
                    # results, so it is not a legal candidate however fast it looks.
                    if grid[0] * grid[1] < batch * NUM_KV_HEADS:
                        rows.append(
                            {
                                "call_site": f"decode_sdpa_{'sliding' if window else 'full'}",
                                "grid": f"{grid[0]}x{grid[1]}",
                                "cores": grid[0] * grid[1],
                                "batch": batch,
                                "q_chunk": 32,
                                "k_chunk": k_chunk,
                                "seq_len": DECODE_CONTEXT,
                                "ms_per_call": float("nan"),
                                "status": "illegal_cores_lt_batch_times_kv_heads",
                            }
                        )
                        continue
                    try:
                        ms = _timed(
                            lambda pc=program_config, w=window, q=q_dec, k=k_dec, v=v_dec, cp=cur_pos, pt=pt_dec: (
                                ttnn.transformer.paged_scaled_dot_product_attention_decode(
                                    q,
                                    k,
                                    v,
                                    cur_pos_tensor=cp,
                                    page_table_tensor=pt,
                                    scale=HEAD_DIM**-0.5,
                                    sliding_window_size=w,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    program_config=pc,
                                    compute_kernel_config=compute_kernel_config,
                                    block_size=BLOCK_SIZE,
                                    num_kv_heads=NUM_KV_HEADS,
                                )
                            )
                        )
                        status = "ok"
                    except Exception as exc:
                        ms, status = float("nan"), type(exc).__name__
                    rows.append(
                        {
                            "call_site": f"decode_sdpa_{'sliding' if window else 'full'}",
                            "grid": f"{grid[0]}x{grid[1]}",
                            "cores": grid[0] * grid[1],
                            "batch": batch,
                            "q_chunk": 32,
                            "k_chunk": k_chunk,
                            "seq_len": DECODE_CONTEXT,
                            "ms_per_call": ms,
                            "status": status,
                        }
                    )
                    print(rows[-1], flush=True)
        for tensor in (k_dec, v_dec, pt_dec, cur_pos, q_dec):
            tensor.deallocate(True)

    ttnn.close_mesh_device(mesh)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "core_grid_sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Blackhole core-grid sweep — decoder SDPA call sites",
        "",
        "Every row uses the layer's own compute-kernel config (HiFi4, `math_approx_mode=False`,",
        "`fp32_dest_acc_en=True`, `packer_l1_acc=True`), so the ranking holds for the policy the",
        "model actually runs.",
        "",
        "Measured with `scripts/sweep_core_grids.py` on one Blackhole p300c chip",
        "(11 x 10 compute grid) at the layer's real shapes: 32 Q heads / 2 KV heads,",
        f"head_dim {HEAD_DIM}, sliding window {SLIDING_WINDOW}, prefill seq {PREFILL_SEQ},",
        f"decode context {DECODE_CONTEXT}, block size {BLOCK_SIZE}.",
        f"Wall clock over {ITERS} calls after {WARMUP} warmup calls, device synchronized.",
        "",
        "Best configuration per call site:",
        "",
        "| call site (batch) | grid | q_chunk | k_chunk | ms/call |",
        "|---|---|---|---|---|",
    ]
    ok_rows = [r for r in rows if r["status"] == "ok" and r["ms_per_call"] == r["ms_per_call"]]
    for call_site, batch in sorted({(r["call_site"], r["batch"]) for r in ok_rows}):
        subset = [r for r in ok_rows if r["call_site"] == call_site and r["batch"] == batch]
        best = min(subset, key=lambda r: r["ms_per_call"])
        lines.append(
            f"| {call_site} (batch {batch}) | {best['grid']} | {best['q_chunk']} | {best['k_chunk']} | "
            f"{best['ms_per_call']:.3f} |"
        )
    lines += [
        "",
        "Full grid (11x10) vs the Wormhole-shaped 8x8, same chunk sizes:",
        "",
        "| call site | q_chunk | k_chunk | 11x10 ms | 8x8 ms |",
        "|---|---|---|---|---|",
    ]
    for call_site in sorted({r["call_site"] for r in ok_rows}):
        for q_chunk, k_chunk in sorted({(r["q_chunk"], r["k_chunk"]) for r in ok_rows if r["call_site"] == call_site}):
            full = [
                r
                for r in ok_rows
                if r["call_site"] == call_site
                and r["grid"] == "11x10"
                and r["q_chunk"] == q_chunk
                and r["k_chunk"] == k_chunk
            ]
            small = [
                r
                for r in ok_rows
                if r["call_site"] == call_site
                and r["grid"] == "8x8"
                and r["q_chunk"] == q_chunk
                and r["k_chunk"] == k_chunk
            ]
            if full and small:
                lines.append(
                    f"| {call_site} | {q_chunk} | {k_chunk} | {full[0]['ms_per_call']:.3f} | "
                    f"{small[0]['ms_per_call']:.3f} |"
                )
    failures = [r for r in rows if r["status"] != "ok"]
    if failures:
        lines += ["", "Rejected combinations (op raised):", ""]
        lines += [
            f"* `{r['call_site']}` grid {r['grid']} q{r['q_chunk']}/k{r['k_chunk']}: {r['status']}" for r in failures
        ]
    lines += ["", f"Raw data: `{csv_path.name}`.", ""]
    (OUT_DIR / "core_grid_sweep.md").write_text("\n".join(lines))
    print(json.dumps({"csv": str(csv_path), "rows": len(rows), "failures": len(failures)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
