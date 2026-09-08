# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone DRAM-bandwidth benchmark for ttnn.experimental.deepseek.moe.fused_experts.

Runs the op at the real DeepSeek-V4-Flash decode shapes (H=4096, I=2048, top_k=6) in a
tight loop and reports the achieved DRAM read bandwidth against the Blackhole spec
(512 GB/s), which is what the op is bound by: every step streams the selected experts'
gate_up [H, 2I] and down [I, H] Bfp4_b weights out of DRAM.

The batch sweep is what makes that bound visible. All B tokens of a step select the same
experts here, so the union the op iterates -- and therefore the bytes it reads -- is the same
at B=32 as at B=1, and the step time should barely move while the per-token cost falls by
~B. That is the batching payoff: an expert several tokens share is fetched once.

``--top-k`` is really "size of the union", so it also stands in for the opposite extreme: a batch
whose tokens select *disjoint* experts, where the union grows to B*top_k (192 at B=32, top_k=6). Those
runs need ``--experts-block`` to bound the activation block held in L1, since only a block's worth is
resident at a time; the DRAM traffic is unchanged by blocking, so the roofline column stays the
comparison. A block size that does not fit L1 fails at program build with the CB sizes in the message.
"""

import time

import torch
import ttnn

BH_NUM_DRAM_BANKS = 8
BH_DRAM_BW_GB_S = 512.0
FUSED_EXPERTS_NUM_CORES = 64
TILE = 32
BFP4_TILE_BYTES = 576  # 32x32 Bfp4_b tile (512 datum bytes + 64 exponent bytes)


def _swiglu_cols_per_core(intermediate):
    return TILE * max(1, (intermediate // TILE) // FUSED_EXPERTS_NUM_CORES)


def _nd_sharded_dram_memory_config(rows, cols, shard_width, dram_core_range_set):
    return ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            shard_shape=[rows, shard_width],
            grid=dram_core_range_set,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


def _interleave_gate_up(w, block):
    k, two_i = w.shape
    blocks = (two_i // 2) // block
    return w.reshape(k, 2, blocks, block).permute(0, 2, 1, 3).reshape(k, two_i).contiguous()


def main(
    hidden=4096,
    intermediate=2048,
    num_experts=64,
    top_ks=(6,),
    batches=(1, 8, 32),
    iters=50,
    experts_blocks=(0,),
):
    torch.manual_seed(0)
    two_i = 2 * intermediate
    device = ttnn.open_device(device_id=0)
    try:
        dram_core_range_set = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(BH_NUM_DRAM_BANKS)]
        )
        swiglu_cols = _swiglu_cols_per_core(intermediate)
        gate_up_mc = _nd_sharded_dram_memory_config(hidden, two_i, 2 * swiglu_cols, dram_core_range_set)
        down_mc = _nd_sharded_dram_memory_config(
            intermediate, hidden, hidden // FUSED_EXPERTS_NUM_CORES, dram_core_range_set
        )

        # One host weight pair, uploaded num_experts times: the benchmark only cares about
        # the DRAM traffic, and reusing the host tensor keeps setup fast.
        gu_host = _interleave_gate_up((torch.rand((hidden, two_i), dtype=torch.bfloat16) - 0.5).float(), swiglu_cols)
        dn_host = (torch.rand((intermediate, hidden), dtype=torch.bfloat16) - 0.5).float()
        gate_up_tt = [
            ttnn.from_torch(
                gu_host, dtype=ttnn.bfloat4_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=gate_up_mc
            )
            for _ in range(num_experts)
        ]
        down_tt = [
            ttnn.from_torch(
                dn_host, dtype=ttnn.bfloat4_b, device=device, layout=ttnn.TILE_LAYOUT, memory_config=down_mc
            )
            for _ in range(num_experts)
        ]

        gu_bytes = (hidden // 32) * (two_i // 32) * BFP4_TILE_BYTES
        dn_bytes = (intermediate // 32) * (hidden // 32) * BFP4_TILE_BYTES

        print(f"\n{'=' * 74}")
        print(f"H={hidden} I={intermediate} experts={num_experts}  ({iters} iters)")
        print(f"  gate_up/expert {gu_bytes / 1e6:.2f} MB   down/expert {dn_bytes / 1e6:.2f} MB")
        print(f"{'-' * 74}")
        print(
            f"{'top_k':>6} {'batch':>6} {'block':>6} {'MB/step':>9} {'us':>9} {'us/token':>9} "
            f"{'GB/s':>9} {'roofline us':>12} {'eff %':>7}"
        )

        for num_nonzero in top_ks:
            for batch in batches:
                x = (torch.rand((1, 1, batch, hidden), dtype=torch.bfloat16) - 0.5).float()
                x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

                # Every token selects the same experts, so the union stays top_k for any batch and
                # the bytes read per step do not change -- only the tokens they are amortized over.
                ids = torch.arange(num_nonzero, dtype=torch.int32).expand(1, 1, batch, num_nonzero)
                ids_tt = ttnn.from_torch(ids, dtype=ttnn.uint16, device=device, layout=ttnn.TILE_LAYOUT)
                scores = torch.ones((1, 1, batch, num_experts), dtype=torch.float32)
                scores_tt = ttnn.from_torch(scores, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

                for experts_block in experts_blocks:

                    def run(
                        x_tt=x_tt, ids_tt=ids_tt, scores_tt=scores_tt, num_nonzero=num_nonzero, block=experts_block
                    ):
                        return ttnn.experimental.deepseek.moe.fused_experts(
                            x_tt,
                            routing_indices=ids_tt,
                            routing_scores=scores_tt,
                            gate_up_weights=gate_up_tt,
                            down_weights=down_tt,
                            num_experts=num_nonzero,
                            intermediate_size=intermediate,
                            swiglu_limit=10.0,
                            top_k=num_nonzero,
                            experts_block_size=block,
                        )

                    for _ in range(3):
                        run()
                    ttnn.synchronize_device(device)

                    t0 = time.perf_counter()
                    for _ in range(iters):
                        run()
                    ttnn.synchronize_device(device)
                    elapsed_us = (time.perf_counter() - t0) * 1e6 / iters

                    total_bytes = num_nonzero * (gu_bytes + dn_bytes)
                    bw = total_bytes / (elapsed_us * 1e-6) / 1e9
                    roofline_us = total_bytes / (BH_DRAM_BW_GB_S * 1e9) * 1e6
                    print(
                        f"{num_nonzero:>6} {batch:>6} {experts_block or num_nonzero:>6} "
                        f"{total_bytes / 1e6:>9.2f} {elapsed_us:>9.1f} "
                        f"{elapsed_us / batch:>9.1f} {bw:>9.1f} {roofline_us:>12.1f} "
                        f"{bw / BH_DRAM_BW_GB_S * 100:>7.1f}"
                    )
        print(f"{'=' * 74}\n")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--intermediate", type=int, default=2048)
    ap.add_argument("--num-experts", type=int, default=64)
    ap.add_argument("--top-k", type=int, nargs="+", default=[6])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument(
        "--experts-block",
        type=int,
        nargs="+",
        default=[0],
        help="experts held in L1 at once (0 = all of them, the default single block)",
    )
    args = ap.parse_args()
    main(
        args.hidden,
        args.intermediate,
        args.num_experts,
        tuple(args.top_k),
        tuple(args.batch),
        args.iters,
        tuple(args.experts_block),
    )
