# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P0-B: M3 moe_reduce = ttnn.experimental.deepseek_prefill.post_combine_reduce on a (2,4) sub-mesh.

Only the fused weighted top-k sum that TtMiniMaxReduce (tt/moe/tt_reduce.py) runs, not its closing TP
reduce-scatter. Inputs per chip, as M3's forward hands them over:
  combine_output [1, 1, tokens, 4, 6144] bf16 ROW_MAJOR DRAM (random; every slot finite)
  weights        [1, tokens, 4] bf16 ROW_MAJOR -> unsqueezed to [1, 1, tokens, 4, 1] exactly like the module
  indices        [1, tokens, 4] uint16 ROW_MAJOR, 4 distinct random experts of 128 per token
  expert_dispatch_table  M3's table (create_dispatch_table(128, 2, 4)), sharded per dispatch group (mesh col)
so each column owns 32 experts and ~1 of the 4 slots per token is local (the rest skipped by compute).

bytes_moved = combine read (reader reads every slot) + output write + weights + indices.

--with-rs also times the TP reduce-scatter M3 runs right after it inside the moe_reduce zone
(MeshConfig.reduce_scatter: reduce_scatter_minimal_async on axis 1, Linear, default num_links) alone
("rs" row) and the pair back to back ("fused+rs" row, = the zone's device work). The plain row is "fused".

  python3 bench_moe_reduce.py --dry-run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

COLUMNS = [
    "part",
    "tokens_per_chip",
    "topk",
    "emb",
    "worst_ms",
    "mean_ms",
    "min_ms",
    "bytes_moved",
    "gbps_worst",
    "gbps_mean",
    "local_slot_frac",
    "cores_used",
] + bc.TIMING_COLUMNS


def bytes_moved(tokens, topk=bc.TOPK, emb=bc.EMB):
    return int(tokens * topk * emb * 2 + tokens * emb * 2 + tokens * topk * 2 * 2)


def main():
    p = bc.add_common_args(argparse.ArgumentParser(description=__doc__), "moe_reduce.csv")
    p.add_argument("--tokens", type=bc.int_list, default=[1024, 2048, 4096])
    p.add_argument("--with-rs", action="store_true", help="also time the TP reduce-scatter (rs, fused+rs rows)")
    args = p.parse_args()

    print(
        f"[bench_moe_reduce] sub-mesh {bc.SUBMESH}, topk {bc.TOPK}, emb {bc.EMB} bf16, "
        f"warmup {args.warmup} repeats {args.repeats}"
    )
    for t in args.tokens:
        assert t % 32 == 0
        print(
            f"  tokens/chip={t:5d}  bytes_moved={bytes_moved(t) / 1e6:8.1f} MB  (expected cores min({t // 32}, grid))"
        )
    if args.dry_run:
        return 0

    bc.setup_env()
    import torch
    import ttnn

    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping
    from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule

    galaxy, mesh = bc.open_submesh()
    out = bc.CsvOut(args.out, COLUMNS)
    try:
        dgs, ndg = bc.SUBMESH
        rep = ttnn.ReplicateTensorToMesh(mesh)
        table = ExpertMapping.create_dispatch_table(bc.N_EXPERTS, dgs, ndg)
        table_tt = TtDispatchModule.shard_expert_dispatch_table(mesh, table, dispatch_axis=0)
        rs_fn = None
        if args.with_rs:
            from models.demos.minimax_m3.config import MeshConfig
            from models.demos.minimax_m3.tt.ccl import CCLManager
            from models.demos.minimax_m3.utils.general_utils import get_default_num_links

            mesh_config = MeshConfig(tuple(mesh.shape), tp=mesh.shape[1])
            ccl = CCLManager(mesh, num_links=get_default_num_links(mesh), topology=ttnn.Topology.Linear)
            rs_fn = lambda t: mesh_config.reduce_scatter(t, ccl, dim=len(t.shape) - 1, axis=1)  # noqa: E731
        timer = bc.OpTimer(mesh)
        torch.manual_seed(0)
        for t in args.tokens:
            base = dict(tokens_per_chip=t, topk=bc.TOPK, emb=bc.EMB, bytes_moved=bytes_moved(t))
            try:
                combine = torch.randn(1, 1, t, bc.TOPK, bc.EMB, dtype=torch.bfloat16)
                w = torch.rand(1, t, bc.TOPK, dtype=torch.bfloat16)
                idx = (
                    torch.argsort(torch.rand(t, bc.N_EXPERTS), dim=-1)[:, : bc.TOPK]
                    .to(torch.int32)
                    .reshape(1, t, bc.TOPK)
                )
                # Fraction of slots whose expert is local to mesh column 0 (the others are similar).
                local = float((table[0, : bc.N_EXPERTS][idx.long()] >= 0).float().mean())

                combine_tt = ttnn.from_torch(
                    combine,
                    mesh_mapper=rep,
                    device=mesh,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                w_tt = ttnn.from_torch(
                    w,
                    mesh_mapper=rep,
                    device=mesh,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                idx_tt = ttnn.from_torch(
                    idx,
                    mesh_mapper=rep,
                    device=mesh,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    dtype=ttnn.uint16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                del combine
                # TtMiniMaxReduce.forward's weight reshaping, verbatim.
                if w_tt.shape[-1] != 1:
                    w_tt = ttnn.unsqueeze(w_tt, dim=-1)
                while len(w_tt.shape) < len(combine_tt.shape):
                    w_tt = ttnn.unsqueeze(w_tt, dim=0)

                def run():
                    return ttnn.experimental.deepseek_prefill.post_combine_reduce(
                        combine_tt,
                        w_tt,
                        idx_tt,
                        table_tt,
                        expert_dim=3,
                        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

                res = timer.measure(run, args.warmup, args.repeats)
                gb = bytes_moved(t) / 1e9
                out.row(
                    part="fused",
                    **base,
                    **res,
                    gbps_worst=gb / (res["worst_ms"] / 1e3),
                    gbps_mean=gb / (res["mean_ms"] / 1e3),
                    local_slot_frac=local,
                    status="OK",
                )
                if rs_fn is not None:
                    summed = run()
                    rs_bytes = t * bc.EMB * 2  # bf16 [t, emb] partial sum per chip into the scatter
                    res = timer.measure(lambda: rs_fn(summed), args.warmup, args.repeats)
                    out.row(part="rs", **dict(base, bytes_moved=rs_bytes), **res, local_slot_frac=local, status="OK")
                    ttnn.deallocate(summed)

                    def run_pair():
                        x = run()
                        y = rs_fn(x)
                        ttnn.deallocate(x)
                        return y

                    res = timer.measure(run_pair, args.warmup, args.repeats)
                    out.row(part="fused+rs", **base, **res, local_slot_frac=local, status="OK")
                for x in (combine_tt, w_tt, idx_tt):
                    ttnn.deallocate(x)
            except Exception as e:
                out.row(
                    part="fused",
                    **base,
                    status=f"ERROR:{type(e).__name__}:{str(e).splitlines()[0][:160] if str(e) else ''}",
                )
    finally:
        out.close()
        bc.close_mesh(galaxy)
    print(f"[bench_moe_reduce] DONE -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
