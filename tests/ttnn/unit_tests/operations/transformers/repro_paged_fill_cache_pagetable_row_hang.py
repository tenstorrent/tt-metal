#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone repro: paged_fill_cache wedges the device on a wide page-table row.

paged_fill_cache's writer kernel reads the whole page-table row in a single
noc.async_read, then blocks on async_read_barrier(). With a 16 KB row and a
bfloat8_b cache, that read intermittently never completes and the device hangs
with no error.

Needs BOTH a wide page-table row AND bfloat8_b:

    --blocks-per-seq 4096 --dtype bfp8   ->  HANG   (16,384 B row)
    --blocks-per-seq 4096 --dtype bf16   ->  pass
    --blocks-per-seq 2048 --dtype bfp8   ->  pass   ( 8,192 B row)

Cache size is irrelevant (8,192 blocks here); only the page-table row matters.
Hang position varies wildly run to run (seen at fill 2, 50 and 150).

Usage (single chip):

    TT_VISIBLE_DEVICES=0 \
    TT_MESH_GRAPH_DESC_PATH=<tt-metal>/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    TT_METAL_OPERATION_TIMEOUT_SECONDS=60 \
    python3 repro_paged_fill_cache_pagetable_row_hang.py --blocks-per-seq 4096 --dtype bfp8

Exits 0 on PASS; the device timeout aborts the process on HANG.
"""
import argparse
import sys

import torch
import ttnn

NUM_USERS = 32
NUM_KV_HEADS = 8
BLOCK_SIZE = 32
HEAD_DIM = 128
CHUNK = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks-per-seq", type=int, default=4096, help="page-table width (entries)")
    ap.add_argument("--dtype", choices=["bfp8", "bf16"], default="bfp8")
    ap.add_argument("--total-blocks", type=int, default=8192)
    ap.add_argument("--fills", type=int, default=300)
    args = ap.parse_args()

    n = args.blocks_per_seq
    dt = ttnn.bfloat8_b if args.dtype == "bfp8" else ttnn.bfloat16
    print(f"[repro] page_table={NUM_USERS}x{n} ({n * 4} B/row) dtype={args.dtype} "
          f"cache_blocks={args.total_blocks} fills={args.fills}", flush=True)

    device = ttnn.open_device(device_id=0)
    try:
        cache = ttnn.zeros(
            ttnn.Shape([args.total_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]),
            dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

        page_table = (torch.arange(NUM_USERS * n, dtype=torch.int32)
                      % args.total_blocks).reshape(NUM_USERS, n)
        page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

        x = torch.randn(NUM_USERS, NUM_KV_HEADS, CHUNK, HEAD_DIM).bfloat16().float()
        x_tt = ttnn.Tensor(x, dt).to(ttnn.TILE_LAYOUT).to(device)

        batch_idx = ttnn.Tensor(torch.arange(NUM_USERS, dtype=torch.int32), ttnn.int32).to(device)

        # Match how the tt-mlir runtime invokes this op: batch_idx as a tensor
        # (input_batch > 1) plus explicit mesh_coords, which selects the
        # MeshWorkloadFactory rather than the single-program factory.
        kwargs = {"batch_idx_tensor": batch_idx}
        try:
            kwargs["mesh_coords"] = set(cache.tensor_topology().mesh_coords())
        except Exception as e:  # noqa: BLE001
            print(f"[repro] mesh_coords unavailable ({e}); continuing without", flush=True)

        for i in range(args.fills):
            cache = ttnn.experimental.paged_fill_cache(cache, x_tt, page_table_tt, **kwargs)
            ttnn.synchronize_device(device)
            if i % 25 == 0:
                print(f"[repro] fill {i} ok", flush=True)

        print(f"[repro] RESULT: PASS ({args.fills} fills)", flush=True)
        return 0
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
