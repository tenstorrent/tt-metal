# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does decode cost scale with the ALLOCATED paged-cache / page-table size?

Full-model decode measured 2.573 ms/token for the 2-layer probe with a
202752-token cache versus 1.790 ms with a 4096-token cache, at the same decode
position (~130). Isolate which op pays: ``paged_update_cache``, the paged flash
MLA decode read, or the page-table tensor width itself.

    python models/autoports/zai_org_glm_4_7_flash/probe/decode_cache_scaling_probe.py
"""

import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.provenance import source_manifest

TILE = 32
KVPE = 576
KV_LORA = 512
NHEADS = 20
BLOCK = 64
SCALE = 256**-0.5


def bench(fn, n=30, warm=3):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(bench.dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ttnn.synchronize_device(bench.dev)
    return (time.perf_counter() - t0) / n * 1e6


def main():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    bench.dev = dev
    torch.manual_seed(0)
    grid = dev.compute_with_storage_grid_size()
    flash_pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=0,
        k_chunk_size=128,
        exp_approx_mode=False,
        max_cores_per_head_batch=8,
    )
    ck = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    kvpe_mem = ttnn.create_sharded_memory_config(
        shape=(TILE, KVPE),
        core_grid=ttnn.num_cores_to_corerangeset(1, grid, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    out_path = Path(__file__).resolve().parents[1] / "doc" / "full_model" / "cache_scaling.json"
    payload = {"source_manifest": source_manifest([__file__]), "rows": [], "rope_table_lookup_us": {}}
    print(f"{'ctx':>9} {'blocks':>7} {'update us':>10} {'flash us':>10} {'flash(pt64) us':>15}")
    for ctx in (4096, 16384, 65536, 202752):
        blocks = -(-ctx // BLOCK)
        cache = ttnn.zeros(
            (blocks, 1, BLOCK, KVPE),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pt_full = ttnn.from_torch(
            torch.arange(blocks, dtype=torch.int32).reshape(1, blocks),
            device=dev,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pt_small = ttnn.from_torch(
            torch.arange(64, dtype=torch.int32).reshape(1, 64),
            device=dev,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        pos = ttnn.from_torch(torch.tensor([130], dtype=torch.int32), device=dev, dtype=ttnn.int32)
        kvpe = ttnn.from_torch(
            torch.randn(1, 1, 1, KVPE),
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=kvpe_mem,
        )
        q = ttnn.from_torch(
            torch.randn(1, 1, NHEADS, KVPE),
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def upd():
            ttnn.experimental.paged_update_cache(cache, kvpe, update_idxs_tensor=pos, page_table=pt_full)

        def flash(pt):
            out = ttnn.transformer.paged_flash_multi_latent_attention_decode(
                q,
                cache,
                head_dim_v=KV_LORA,
                page_table_tensor=pt,
                cur_pos_tensor=pos,
                scale=SCALE,
                program_config=flash_pc,
                compute_kernel_config=ck,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(out)

        t_upd = bench(upd)
        t_flash = bench(lambda: flash(pt_full))
        t_flash_small = bench(lambda: flash(pt_small))
        print(f"{ctx:9d} {blocks:7d} {t_upd:10.1f} {t_flash:10.1f} {t_flash_small:15.1f}")
        payload["rows"].append(
            {
                "context": ctx,
                "blocks": blocks,
                "paged_update_cache_us": round(t_upd, 1),
                "flash_decode_us": round(t_flash, 1),
                "flash_decode_64_block_page_table_us": round(t_flash_small, 1),
            }
        )
        for t in (cache, pt_full, pt_small, pos, kvpe, q):
            ttnn.deallocate(t)
    # The other thing that turned out to scale with an *allocated* size: the
    # per-layer RoPE cos/sin lookup. A TILE-layout `ttnn.embedding` over a
    # `[context, 64]` table costs more the taller the table is, which is what
    # made 94 lookups per decode step cost 19.7 ms at the full context and is
    # why the model shares one ROW_MAJOR table (work log FM-005). Measured here
    # so that claim has an artifact (FM-019).
    idx = ttnn.from_torch(
        torch.zeros(1, 1, dtype=torch.int32), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    for ctx in (4096, 202752):
        host = torch.zeros(ctx, 64, dtype=torch.bfloat16)
        for layout, name in ((ttnn.TILE_LAYOUT, "tile"), (ttnn.ROW_MAJOR_LAYOUT, "row_major")):
            table = ttnn.from_torch(
                host, device=dev, dtype=ttnn.bfloat16, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            us = bench(lambda: ttnn.deallocate(ttnn.embedding(idx, table, layout=ttnn.TILE_LAYOUT)))
            payload["rope_table_lookup_us"][f"{name}_ctx{ctx}"] = round(us, 1)
            print(f"rope {name} table ctx={ctx}: {us:.1f} us")
            ttnn.deallocate(table)
    ttnn.deallocate(idx)
    ttnn.close_device(dev)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote", out_path)
    print("CACHE_SCALING_PROBE_OK")


if __name__ == "__main__":
    main()
