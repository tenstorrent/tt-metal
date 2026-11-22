# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The batch-spec build's KV pool budget: ResidentDFlashRows as Qwen36ResidentSpecRowsForCausalLM.allocate_kv_cache
builds it (8 users x 4 verify rows, the PP prefill, max_pos for max_model_len), on a small pool, then the DRAM left
on the chips, probed with 128 MB allocations on every chip until one fails. Prints the pool that fits:
probe blocks + free / (K + V bytes per block per chip), less a margin.

usage: python -m qwen36_resident.tests.spec_rows_mem [--max-model-len 262144] [--probe-blocks 1000] [--margin-gb 1]
"""
import argparse
import faulthandler

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

KV_BYTES_PER_BLOCK = 2 * 278528  # K + V per 64-token block per chip (the 20,008-block pool's K: 5,572,788,224 B)
STEP = 128 << 20


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-model-len", type=int, default=262144)
    ap.add_argument("--probe-blocks", type=int, default=1000)
    ap.add_argument("--margin-gb", type=float, default=1.0)
    a = ap.parse_args()
    mesh = open_mesh()
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        R, U = 4, 8
        eng = ResidentDFlashRows(
            mesh, ck, users=U, max_pos=a.max_model_len + 2 * R + 64, R=R, prefill=True, kv_pool_blocks=a.probe_blocks
        )
        print(f"built: drafter ring {eng.drafter_ring} tokens per user", flush=True)
        held, n = [], 0
        while True:
            try:
                held.append(
                    ttnn.empty(
                        [1, 1, 8192, 8192],
                        ttnn.bfloat16,
                        ttnn.TILE_LAYOUT,
                        mesh,
                        ttnn.DRAM_MEMORY_CONFIG,
                    )
                )
                n += 1
            except RuntimeError:
                break
        free = n * STEP
        for t in held:
            ttnn.deallocate(t)
        blocks = a.probe_blocks + int((free - a.margin_gb * 2**30) // KV_BYTES_PER_BLOCK)
        print(
            f"free DRAM per chip after the build: {free / 2**30:.2f} GB ({n} x 128 MB); pool that fits with a "
            f"{a.margin_gb} GB margin: {blocks} blocks = {blocks * 64:,} tokens",
            flush=True,
        )
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
