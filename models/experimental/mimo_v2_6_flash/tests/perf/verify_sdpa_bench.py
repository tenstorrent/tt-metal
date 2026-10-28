# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip's global-layer verify SDPA at long context (tt/attention.py VERIFY_GROUPED): the per-slot verify (each
candidate a decode row: _paged_mla_decode_lse over the pass's slots), the grouped verify (_verify_grouped:
spec_multi_pos_tiles, one KV scan per request; ``packed``: spec_multi_pos_packed, two candidates per compute tile) and a
plain decode step (one row per request, the floor). MiMo's
shapes: 16 heads, K 192 / V 128, BFP8 caches in 64-token blocks, ``--local`` keys per request (a pair-split chip holds
half the context). Prints one ``VERIFY_SDPA`` line.

    python models/experimental/mimo_v2_6_flash/tests/perf/verify_sdpa_bench.py --local 524288 --reqs 1
"""

import argparse
import time

import torch

import ttnn

NH, DK, DV, BLOCK, TILE = 16, 192, 128, 64, 32


def program(batch, table_tokens, cands=0, k_chunk_size=0):
    """MiMo's _split_sdpa_program (8 x 8 grid), and _verify_sdpa_program's limits from 8 candidates."""
    k_chunk = 256 if table_tokens % 256 == 0 else 128
    cores = max(16, min(64, 64 // batch))
    if cands >= 8:
        k_chunk, cores = 128, min(cores, 32)
    k_chunk = k_chunk_size or k_chunk
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=0,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
        max_cores_per_head_batch=cores,
    )


def timed(device, fn, n):
    outs = fn()
    ttnn.synchronize_device(device)
    for t in outs:
        t.deallocate(True)
    t0 = time.perf_counter()
    for _ in range(n):
        outs = fn()
        for t in outs:
            t.deallocate(True)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / n * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local", type=int, default=512 * 1024, help="cached keys per request on this chip")
    p.add_argument("--reqs", type=int, default=1)
    p.add_argument("--cands", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--grouped-fidelity", default="HiFi4", help="the grouped verify's math fidelity")
    p.add_argument("--grouped-k-chunk", type=int, default=0, help="the grouped verify's K chunk (0: the decode's)")
    a = p.parse_args()
    reqs, cands, n_local = a.reqs, a.cands, a.local
    slots, blocks = reqs * cands, n_local // BLOCK
    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        cfg_grouped = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, a.grouped_fidelity),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        dram = ttnn.DRAM_MEMORY_CONFIG
        k_cache = ttnn.from_torch(
            torch.randn(reqs * blocks, 1, BLOCK, DK), device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        v_cache = ttnn.from_torch(
            torch.randn(reqs * blocks, 1, BLOCK, DV), device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        table = torch.arange(reqs * blocks, dtype=torch.int32).reshape(reqs, blocks)
        group_table = ttnn.from_torch(table, device=device, dtype=ttnn.int32)
        slot_table = ttnn.from_torch(table.repeat_interleave(cands, dim=0), device=device, dtype=ttnn.int32)
        pos = [n_local - cands + j for _ in range(reqs) for j in range(cands)]
        slot_pos = ttnn.from_torch(torch.tensor(pos, dtype=torch.int32), device=device, dtype=ttnn.int32)
        last_pos = ttnn.from_torch(
            torch.tensor([n_local - 1] * reqs, dtype=torch.int32), device=device, dtype=ttnn.int32
        )

        def sharded_q(batch):
            mc = ttnn.create_sharded_memory_config(
                shape=(TILE, DK),
                core_grid=ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(8, 8), row_wise=True),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            return ttnn.from_torch(
                torch.randn(1, batch, NH, DK),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mc,
            )

        q_slots, q_decode = sharded_q(slots), sharded_q(reqs)
        q_grouped = ttnn.from_torch(
            torch.randn(1, reqs, cands * TILE, DK),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=dram,
        )
        tokens = n_local  # table width in tokens
        common = dict(head_dim_v=DV, scale=DK**-0.5, memory_config=ttnn.L1_MEMORY_CONFIG, return_lse=True)
        mla = ttnn.transformer.paged_flash_multi_latent_attention_decode

        def per_slot():
            return mla(
                q_slots,
                k_cache,
                v_cache,
                page_table_tensor=slot_table,
                cur_pos_tensor=slot_pos,
                program_config=program(slots, tokens),
                compute_kernel_config=cfg,
                **common,
            )

        def grouped():
            return mla(
                q_grouped,
                k_cache,
                v_cache,
                page_table_tensor=group_table,
                cur_pos_tensor=slot_pos,
                program_config=program(reqs, tokens, cands, a.grouped_k_chunk),
                spec_multi_pos_tiles=cands,
                compute_kernel_config=cfg_grouped,
                **common,
            )

        packed_ok = cands % 2 == 0 and "spec_multi_pos_packed" in (mla.__doc__ or "")

        def packed():
            return mla(
                q_grouped,
                k_cache,
                v_cache,
                page_table_tensor=group_table,
                cur_pos_tensor=slot_pos,
                program_config=program(reqs, tokens, cands, a.grouped_k_chunk),
                spec_multi_pos_tiles=cands,
                spec_multi_pos_packed=True,
                compute_kernel_config=cfg_grouped,
                **common,
            )

        def decode():
            return mla(
                q_decode,
                k_cache,
                v_cache,
                page_table_tensor=group_table,
                cur_pos_tensor=last_pos,
                program_config=program(reqs, tokens),
                compute_kernel_config=cfg,
                **common,
            )

        fns = [("per_slot", per_slot), ("grouped", grouped), ("decode", decode)]
        if packed_ok:
            fns.append(("packed", packed))
        res = {name: timed(device, fn, a.iters) for name, fn in fns}
        kv_gb = reqs * n_local * (DK + DV) * 1088 / 1024 / 1e9
        packed_ms = f"{res['packed']:.3f} ms" if packed_ok else "n/a"
        print(
            f"VERIFY_SDPA local {n_local} reqs {reqs} cands {cands} grouped {a.grouped_fidelity}: per-slot {res['per_slot']:.3f} ms, grouped "
            f"{res['grouped']:.3f} ms, packed {packed_ms}, plain decode {res['decode']:.3f} ms (KV {kv_gb:.2f} GB per scan: "
            f"{kv_gb / res['decode'] * 1e3:.0f} GB/s at the decode's time)",
            flush=True,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
