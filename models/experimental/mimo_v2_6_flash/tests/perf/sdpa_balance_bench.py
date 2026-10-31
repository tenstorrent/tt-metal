# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One chip's grouped verify SDPA (tt/attention.py _verify_grouped: spec_multi_pos_tiles, packed) when the requests'
contexts differ: the op splits its cores evenly over the groups, so the longest request sets the time. Compares
``--ctx`` (comma-separated local keys per request; 0: an idle group) run as

- grouped: one call over every group (each group's share of the 8 x 8 grid);
- active: one call over the groups with a request (attention.verify_variants);
- sequential: one call per request, each on the whole grid (64 cores).

MiMo's shapes: 16 heads, K 192 / V 128, BFP8 caches in 64-token blocks, ``--cands`` candidates per request. Prints
one ``SDPA_BALANCE`` line.

    python models/experimental/mimo_v2_6_flash/tests/perf/sdpa_balance_bench.py --ctx 400000,25000,25000,25000
"""

import argparse
import time

import torch

import ttnn

NH, DK, DV, BLOCK, TILE = 16, 192, 128, 64, 32


def program(batch, k_chunk=256, grid=(8, 8)):
    """MiMo's _split_sdpa_program."""
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        q_chunk_size=0,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
        max_cores_per_head_batch=max(16, min(64, grid[0] * grid[1] // batch)),
    )


def timed(device, fn, n):
    for t in fn():
        t.deallocate(True)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(n):
        for t in fn():
            t.deallocate(True)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / n * 1e3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", default="400000,25000,25000,25000", help="local keys per request (0: idle group)")
    p.add_argument("--cands", type=int, default=2)
    p.add_argument("--iters", type=int, default=20)
    a = p.parse_args()
    ctx = [int(x) for x in a.ctx.split(",")]
    groups, cands = len(ctx), a.cands
    width = max(-(-max(ctx) // BLOCK), 1)
    width = -(-width // 4) * 4  # 256-token K chunks
    device = ttnn.open_device(device_id=0, l1_small_size=65536)
    try:
        cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        )
        blocks = groups * width
        k_cache = ttnn.from_torch(
            torch.randn(blocks, 1, BLOCK, DK), device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        v_cache = ttnn.from_torch(
            torch.randn(blocks, 1, BLOCK, DV), device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        table = torch.arange(blocks, dtype=torch.int32).reshape(groups, width)
        pos = []
        for n in ctx:
            pos += [n - cands + j if n > 0 else -1 for j in range(cands)]

        def dev(t, dtype=ttnn.int32):
            return ttnn.from_torch(t, device=device, dtype=dtype)

        def q(n):
            return ttnn.from_torch(
                torch.randn(1, n, cands * TILE, DK),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        mla = ttnn.transformer.paged_flash_multi_latent_attention_decode
        common = dict(
            head_dim_v=DV,
            scale=DK**-0.5,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            return_lse=True,
            spec_multi_pos_tiles=cands,
            spec_multi_pos_packed=True,
            compute_kernel_config=cfg,
        )
        q_all, t_all, p_all = q(groups), dev(table), dev(torch.tensor(pos, dtype=torch.int32))
        live = [g for g, n in enumerate(ctx) if n > 0]
        q_act = q(len(live))
        t_act = dev(table[live])
        p_act = dev(torch.tensor([pos[g * cands + j] for g in live for j in range(cands)], dtype=torch.int32))
        singles = [
            (q(1), dev(table[g : g + 1]), dev(torch.tensor(pos[g * cands : (g + 1) * cands], dtype=torch.int32)))
            for g in live
        ]

        def grouped():
            return mla(
                q_all,
                k_cache,
                v_cache,
                page_table_tensor=t_all,
                cur_pos_tensor=p_all,
                program_config=program(groups),
                **common,
            )

        def active():
            return mla(
                q_act,
                k_cache,
                v_cache,
                page_table_tensor=t_act,
                cur_pos_tensor=p_act,
                program_config=program(len(live)),
                **common,
            )

        def sequential():
            outs = []
            for qq, tt, pp in singles:
                outs += list(
                    mla(
                        qq,
                        k_cache,
                        v_cache,
                        page_table_tensor=tt,
                        cur_pos_tensor=pp,
                        program_config=program(1),
                        **common,
                    )
                )
            return outs

        res = {
            name: timed(device, fn, a.iters)
            for name, fn in [("grouped", grouped), ("active", active), ("sequential", sequential)]
        }
        kv_gb = sum(ctx) * (DK + DV) * 1088 / 1024 / 1e9
        print(
            f"SDPA_BALANCE ctx {a.ctx} cands {cands}: grouped {res['grouped']:.3f} ms, active {res['active']:.3f} ms, "
            f"sequential {res['sequential']:.3f} ms (KV {kv_gb:.3f} GB: {kv_gb / min(res.values()) * 1e3:.0f} GB/s at "
            f"the best)",
            flush=True,
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
