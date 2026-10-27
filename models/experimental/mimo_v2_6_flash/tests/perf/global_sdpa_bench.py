# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One global layer's prefill SDPA on Galaxy row 0, alone: the last 4096-token chunk of a long prompt.

``chunked_flash_mla_prefill`` with MiMo's shapes after the chip-pair split (TtMiMoAttention._prefill_pair_chunk):
8 query heads over one KV head per chip, K 192 / V 128 in separate BFP8 paged caches (block 64, a shuffled page
table), the chunk start as a device tensor, HiFi2 with bf16 dest. For each context and program config it reports
the time per call, TFLOP/s and the DRAM bytes the units' K/V reads imply (every (head, query chunk) unit streams the
whole cached context unless the config chains it), so the bound can be read off before a kernel change.

    python models/experimental/mimo_v2_6_flash/tests/perf/global_sdpa_bench.py [--contexts 131072,524288,1048576]
        [--configs grid:q:k[:gG][:sS],...]

A config is ``grid:q_chunk:k_chunk`` with optional fields ``gG`` (SDPAProgramConfig.q_heads_per_unit: G query heads
share each K/V chunk read) and ``sS`` (SDPAProgramConfig.k_splits: each unit's keys split over S work items, merged
after the program).

Prints one GLOBAL_SDPA_BENCH line per (context, config).
"""

import argparse
import time

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan, compute_config

HEADS, DK, DV, BLOCK, CHUNK = 8, 192, 128, 64, 4096
KV_BYTES = (DK + DV) * 1088 / 1024  # per cached position, BFP8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="131072,524288,1048576")
    parser.add_argument("--configs", default="8x8:256:256,8x8:128:256,8x8:256:128,8x8:64:256,full:256:256,full:128:256")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]

    mesh, row = common.open_row()
    try:
        plan = MeshPlan(row)
        rep = dict(device=row, mesh_mapper=plan.replicate())
        full = row.compute_with_storage_grid_size()
        sdpa_config = compute_config(ttnn.MathFidelity.HiFi2, fp32_acc=False, packer_l1_acc=False)
        max_ctx = max(contexts)
        blocks = (max_ctx + BLOCK - 1) // BLOCK
        g = torch.Generator().manual_seed(0)
        # Paged caches: random values (the time does not depend on them), a shuffled page table like vLLM's.
        k_cache = ttnn.from_torch(
            torch.randn(blocks, 1, BLOCK, DK, generator=g), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, **rep
        )
        v_cache = ttnn.from_torch(
            torch.randn(blocks, 1, BLOCK, DV, generator=g), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, **rep
        )
        perm = torch.randperm(blocks, generator=g).to(torch.int32).reshape(1, -1)
        page_table = ttnn.from_torch(perm, dtype=ttnn.int32, **rep)
        q = ttnn.from_torch(
            torch.randn(1, HEADS, CHUNK, DK, generator=g), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **rep
        )
        for ctx in contexts:
            start = ctx - CHUNK
            start_tt = ttnn.from_torch(torch.tensor([start], dtype=torch.int32), dtype=ttnn.int32, **rep)
            # Each query attends to the cache before the chunk plus the causal part of the chunk.
            flops = HEADS * CHUNK * (start + CHUNK / 2) * (DK + DV) * 2
            for spec in args.configs.split(","):
                grid_s, qc, kc, *flags = spec.split(":")
                heads = int(next((f[1:] for f in flags if f.startswith("g")), 1))
                splits = int(next((f[1:] for f in flags if f.startswith("s")), 1))
                assert all(f[0] in "gs" for f in flags), f"unknown config flags {flags}"
                grid = full if grid_s == "full" else ttnn.CoreCoord(*map(int, grid_s.split("x")))
                qc, kc = int(qc), int(kc)
                program = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid,
                    q_chunk_size=qc,
                    k_chunk_size=kc,
                    exp_approx_mode=False,
                    q_heads_per_unit=heads,
                    k_splits=splits,
                )

                def call():
                    return ttnn.transformer.chunked_flash_mla_prefill(
                        q,
                        k_cache,
                        DV,
                        page_table,
                        chunk_start_idx_tensor=start_tt,
                        scale=DK**-0.5,
                        program_config=program,
                        compute_kernel_config=sdpa_config,
                        input_tensor_v=v_cache,
                    )

                try:
                    call().deallocate(True)  # compile
                    ttnn.synchronize_device(row)
                    t0 = time.time()
                    for _ in range(args.iters):
                        call().deallocate(True)
                    ttnn.synchronize_device(row)
                    ms = (time.time() - t0) / args.iters * 1e3
                except Exception as e:  # a config that does not fit L1 raises at program creation
                    logger.warning(f"ctx {ctx} {spec}: {str(e).splitlines()[0][:160]}")
                    print(f"GLOBAL_SDPA_BENCH ctx={ctx} config={spec} status=failed", flush=True)
                    continue
                units = HEADS * CHUNK // qc // heads  # each unit streams the K/V prefix once
                kv_gb = units * (start + CHUNK) * KV_BYTES / 1e9
                cores = grid.x * grid.y
                logger.info(
                    f"ctx {ctx} grid {grid.x}x{grid.y} q {qc} k {kc} g {heads} s {splits}: {ms:.2f} ms, {flops / ms / 1e9:.1f} TFLOP/s, "
                    f"{units} units on {cores} cores, implied K/V reads {kv_gb:.1f} GB ({kv_gb / ms * 1e3:.0f} GB/s)"
                )
                print(
                    f"GLOBAL_SDPA_BENCH ctx={ctx} config={spec} cores={cores} units={units} ms={ms:.3f} "
                    f"tflops={flops / ms / 1e9:.1f} kv_gb={kv_gb:.2f} kv_gbs={kv_gb / ms * 1e3:.0f}",
                    flush=True,
                )
            start_tt.deallocate(True)
    finally:
        ttnn.close_mesh_device(row)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
