# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Time the pieces of one MiMo MoE prefill layer's experts on Galaxy row 0: the plan (DevicePlanSplit), the grouped
expert matmul (GroupedExperts) and the k combine (CombineK), each alone over ``--iters`` back-to-back calls with one
device synchronize, at T tokens with a chosen routing:

- ``router``: the layer's own router (fp32 top-8) on normalized random tokens (spread over all 256 experts);
- ``skew``: distinct top-8 drawn from a skewed popularity (a few hot experts, like moe_plan_bench.py);
- a path to a ``.pt`` file of int [T', 8] expert ids (real routing captured from the model, MIMO_DUMP_ROUTING);
  its first T rows are used.

Reports per piece ms, and for the grouped matmul model / padded TFLOP/s and the DRAM bytes it moves (expert weights
read once per job, x rows read once per padded row, y rows written once per pair) as GB/s. ``--defines A,B`` passes
preprocessor defines to the grouped kernels (ablations). Prints MOE_EXPERTS_BENCH lines.
"""

import argparse
import time

import torch

import ttnn
from models.experimental.mimo_v2_6_flash.tests.perf import moe_decode_common as common
from models.experimental.mimo_v2_6_flash.tt.common import TILE
from models.experimental.mimo_v2_6_flash.tt.moe_prefill import CombineK, DevicePlanSplit
from models.experimental.mimo_v2_6_flash.tt.profiling import zone

E, K = 256, 8


def skew_routing(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    popularity = torch.rand(E, generator=g) ** 3 + 0.05
    ids = torch.multinomial(popularity.expand(T, E), K, replacement=False, generator=g).to(torch.int32)
    w = torch.rand(T, K, generator=g)
    return ids, w / w.sum(-1, keepdim=True)


def timed(fn, iters, row, name="x"):
    fn()  # compile / warm
    ttnn.synchronize_device(row)
    t0 = time.perf_counter()
    with zone(f"bench_{name}"):  # under tracy (MIMO_PROFILE_ZONES=1): moe_bench_parse.py reads the device times
        for _ in range(iters):
            fn()
    ttnn.synchronize_device(row)
    return (time.perf_counter() - t0) * 1e3 / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--routing", default="router")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--defines", default="", help="comma list NAME or NAME=VALUE for the grouped kernels")
    parser.add_argument(
        "--configs", default="", help="'/'-separated define sets to time one after another ('base': none)"
    )
    parser.add_argument("--stages", default="plan,grouped,combine")
    parser.add_argument(
        "--ref", default="", help="combined output of the first config: compared if the file exists, else saved"
    )
    parser.add_argument(
        "--l1-floor",
        type=int,
        default=0,
        help="hold L1 buffers from the top of L1 down to this address (vLLM decode keeps them from ~1.21 MB)",
    )
    args = parser.parse_args()
    T = args.tokens

    mesh, row = common.open_row()
    try:
        cfg = common.config()
        moe = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b)
        H, I = moe.hidden, moe.inter_local
        g = torch.Generator().manual_seed(0)
        rows = torch.nn.functional.normalize(torch.randn(T, H, generator=g), dim=-1) * H**0.5
        rep = ttnn.ReplicateTensorToMesh(row)
        x = ttnn.from_torch(
            rows.reshape(1, 1, T, H).bfloat16(),
            device=row,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        tile_kw = dict(device=row, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
        if args.routing == "router":
            logits = moe._router_logits(x, prefill=True)
            indices, weights = moe._topk(logits)
            logits.deallocate(True)
            ids = ttnn.to_torch(ttnn.get_device_tensors(indices)[0]).reshape(T, K).long()
        else:
            if args.routing == "skew":
                ids, w = skew_routing(T)
            else:
                ids = torch.load(args.routing)[:T].to(torch.int32)
                assert ids.shape == (T, K), ids.shape
                w = torch.rand(T, K, generator=g)
                w = w / w.sum(-1, keepdim=True)
            indices = ttnn.from_torch(ids.reshape(1, 1, T, K), dtype=ttnn.uint32, **tile_kw)
            weights = ttnn.from_torch(w.reshape(1, 1, T, K).float(), dtype=ttnn.float32, **tile_kw)
            ids = ids.long()

        counts = torch.bincount(ids.reshape(-1), minlength=E)
        groups = (counts + TILE - 1) // TILE
        G = int(groups.sum())
        print(
            f"MOE_EXPERTS_BENCH routing={args.routing} T={T} pairs={T * K} groups={G} padded_rows={G * TILE} "
            f"pad={G * TILE / (T * K) - 1:.3f} tokens/expert min={int(counts.min())} median={int(counts.median())} "
            f"max={int(counts.max())} experts_used={int((counts > 0).sum())} groups/expert max={int(groups.max())}",
            flush=True,
        )

        held = []
        if args.l1_floor:
            while True:
                t = ttnn.empty(
                    [120 * 8, 1024],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=row,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                held.append(t)
                if t.buffer_address() <= args.l1_floor:
                    break
            print(f"MOE_EXPERTS_BENCH l1 held from {held[-1].buffer_address()}", flush=True)
        experts = moe.prefill_experts
        grouped = experts._grouped()
        assert grouped is not None
        plan_op = DevicePlanSplit(moe, grouped)
        x_rm = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (T, H))
        row_token, row_weight, row_dest, plan = plan_op(indices, weights)
        team_plan = ttnn.to_torch(ttnn.get_device_tensors(plan)[0]).reshape(grouped.num_teams, -1).to(torch.int64)
        loads = []
        jobs = 0
        for t in range(grouped.num_teams):
            n = int(team_plan[t, 0])
            jobs += n
            loads.append(sum(int(team_plan[t, 3 + 3 * j]) for j in range(n)))
        print(
            f"MOE_EXPERTS_BENCH teams={grouped.num_teams} jobs={jobs} groups/team min={min(loads)} max={max(loads)} "
            f"mean={sum(loads) / len(loads):.1f}",
            flush=True,
        )
        y = ttnn.empty(
            [K * T, H],
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=row,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        stages = args.stages.split(",")
        if "plan" in stages:

            def run_plan():
                for t in plan_op(indices, weights):
                    t.deallocate(True)

            print(f"MOE_EXPERTS_BENCH stage=plan ms={timed(run_plan, args.iters, row, 'plan'):.3f}", flush=True)

        model_flops = T * K * 3 * 2 * H * I
        padded_flops = G * TILE * 3 * 2 * H * I
        w_bytes_expert = 3 * grouped.kt * grouped.ikt * grouped.w_tile_bytes
        w_bytes = jobs * w_bytes_expert
        xy_bytes = G * TILE * H * 2 + T * K * H * 2
        configs = args.configs.split("/") if args.configs else [args.defines]
        if "grouped" in stages:
            combine = CombineK(moe, grouped.cores, grouped.ranges)
            hifi2 = grouped.compute_config
            ref_out = None
            for conf in configs:
                defs = []
                lofi = False
                for d in filter(None, conf.split(",")):
                    name, _, val = d.partition("=")
                    if name == "base":
                        continue
                    if name == "LOFI":  # pseudo-define: the grouped kernel at LoFi
                        lofi = True
                        continue
                    defs.append((name, val or "1"))
                grouped.defines = defs
                grouped.compute_config = hifi2
                if lofi:
                    grouped.compute_config = ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=False,
                    )
                op, op_plan = grouped, plan
                ms = timed(
                    lambda: op(x_rm, row_token, row_weight, row_dest, op_plan, y),
                    args.iters,
                    row,
                    f"grouped[{conf or '-'}]",
                )
                check = ""
                if not any(n.startswith("ABL_") for n, _ in defs):
                    out = combine(y, T, K)
                    got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
                    out.deallocate(True)
                    if ref_out is None:
                        ref_out = got
                        if args.ref:
                            import os

                            if os.path.exists(args.ref):
                                saved = torch.load(args.ref)
                                print(
                                    f"MOE_EXPERTS_BENCH vs {args.ref}: max_abs_diff="
                                    f"{(got - saved).abs().max().item():.3g} pcc="
                                    f"{torch.corrcoef(torch.stack([got.flatten(), saved.flatten()]))[0, 1].item():.6f}",
                                    flush=True,
                                )
                            else:
                                torch.save(got, args.ref)
                    diff = (got - ref_out).abs().max().item()
                    check = f" max_abs_diff_vs_first={diff:.3g} finite={bool(torch.isfinite(got).all())}"
                print(
                    f"MOE_EXPERTS_BENCH stage=grouped defines={conf or '-'} ms={ms:.3f} "
                    f"model_TFLOPs={model_flops / ms / 1e9:.1f} padded_TFLOPs={padded_flops / ms / 1e9:.1f} "
                    f"w_MB={w_bytes / 1e6:.0f} xy_MB={xy_bytes / 1e6:.0f} "
                    f"GB/s={(w_bytes + xy_bytes) / ms / 1e6:.0f}{check}",
                    flush=True,
                )
            grouped.defines = []
            grouped.compute_config = hifi2
        if "combine" in stages:
            combine = CombineK(moe, grouped.cores, grouped.ranges)
            ms = timed(lambda: combine(y, T, K).deallocate(True), args.iters, row, "combine")
            print(f"MOE_EXPERTS_BENCH stage=combine ms={ms:.3f} GB/s={(T * K * H * 2 + T * H * 2) / ms / 1e6:.0f}")
        if "router" in stages:

            def run_router():
                lg = moe._router_logits(x, prefill=True)
                i, w = moe._topk(lg)
                for t in (lg, i, w):
                    t.deallocate(True)

            print(f"MOE_EXPERTS_BENCH stage=router ms={timed(run_router, args.iters, row, 'router'):.3f}", flush=True)
        if "moe" in stages:
            print(
                f"MOE_EXPERTS_BENCH stage=moe ms={timed(lambda: moe.forward(x).deallocate(True), args.iters, row, 'moe'):.3f}"
            )
    finally:
        common.close(mesh, row)


if __name__ == "__main__":
    main()
