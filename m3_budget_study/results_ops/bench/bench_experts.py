# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P0-B: M3 routed-expert op on a (2,4) sub-mesh, 16 experts per chip, bf4 ND-sharded weights.

Runs the same TtRoutedExpert.forward M3 builds (tt/moe/tt_minimax_moe.py: SwiGluOai, bf8 activations,
LoFi, ROW_MAJOR bf16 dispatch buffer handed straight in), in two modes:
  nd      hybrid_token_threshold=None -> unified_routed_expert_moe for every expert
  hybrid  hybrid_token_threshold=T    -> experts with <= T tokens on moe_fused_swiglu, the rest on the composite
Both with weights_dram_nd_sharded=True. The weight upload inside TtRoutedExpert is replaced by prebuilt
device tensors (one random host tensor per projection shape, uploaded once per local expert), since the
real 128-expert conversion would cost minutes of host time and ~30 GB of fp32 staging.

Synthetic dispatch: the first `active` local experts of each chip get `tokens_per_expert` rows each, in
tile-aligned regions laid out like offset_cumsum (region k starts at sum of aligned counts before it);
the other local experts get count 0. counts / region offsets are (ndg, dgs, E) tables sharded with
TtRoutedExpert.shard_expert_token_counts, the global-expert table is M3's (get_ep_mesh_mapper, squeezed).

  python3 bench_experts.py --dry-run
  python3 bench_experts.py [--tokens 16,64,...] [--active 4,8,16] [--paths nd,hybrid]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_common as bc  # noqa: E402

COLUMNS = [
    "path",
    "tokens_per_expert",
    "active_experts",
    "worst_ms",
    "mean_ms",
    "min_ms",
    "weight_bytes_read",
    "flops",
    "total_tokens",
    "cores_used",
] + bc.TIMING_COLUMNS

EXPERTS_PER_CHIP = bc.N_EXPERTS // (bc.SP * bc.TP)  # 16


def grid(args):
    for path in args.paths:
        for a in args.active:
            for t in args.tokens:
                yield path, t, a


def weight_bytes(active):
    return int(active * 3 * bc.EMB * bc.MOE_INTER * bc.BPE["bf4"])


def flops(tokens, active):
    return 2 * tokens * active * 3 * bc.EMB * bc.MOE_INTER


def build_offsets_counts(tokens, active, dgs, ndg, epc, E):
    """(ndg, dgs, E) tables: counts[g, r, gid] / region offsets[g, r, gid] for every chip (r, g)."""
    import torch

    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import ExpertMapping

    table = ExpertMapping.create_global_expert_idx_table(epc, dgs, ndg)  # (ndg, dgs, epc)
    counts = torch.zeros((ndg, dgs, E), dtype=torch.int32)
    offsets = torch.zeros((ndg, dgs, E), dtype=torch.int32)
    aligned = (tokens + 31) // 32 * 32
    for g in range(ndg):
        for r in range(dgs):
            off = 0
            for e in range(epc):
                gid = int(table[g, r, e])
                c = tokens if e < active else 0
                counts[g, r, gid] = c
                offsets[g, r, gid] = off
                off += (c + 31) // 32 * 32
    return table, counts, offsets, aligned


def main():
    p = bc.add_common_args(argparse.ArgumentParser(description=__doc__), "experts.csv")
    p.add_argument("--tokens", type=bc.int_list, default=[16, 64, 128, 256, 512, 1024])
    p.add_argument("--active", type=bc.int_list, default=[4, 8, 16])
    p.add_argument("--paths", type=lambda s: s.split(","), default=["nd", "hybrid"])
    p.add_argument("--threshold", type=int, default=128, help="hybrid_token_threshold for the hybrid path")
    p.add_argument(
        "--seq-len-per-chip",
        type=int,
        default=2048,
        help="M3 ep_seq_len_per_chip: max_tokens = dgs*seq, dispatch buffer = compute_constants(..., factor=topk)",
    )
    args = p.parse_args()

    dgs, ndg = bc.SUBMESH
    max_tokens = dgs * args.seq_len_per_chip
    max_buf = dgs * args.seq_len_per_chip * bc.TOPK + 32 * (EXPERTS_PER_CHIP - 1)
    pts = list(grid(args))
    print(
        f"[bench_experts] sub-mesh {bc.SUBMESH}, {EXPERTS_PER_CHIP} experts/chip, emb {bc.EMB}, inter {bc.MOE_INTER}, "
        f"bf4 ND-sharded weights, max_tokens {max_tokens}, dispatch buffer {max_buf} rows, "
        f"threshold {args.threshold}, {len(pts)} points, warmup {args.warmup} repeats {args.repeats}"
    )
    for path, t, a in pts:
        need = a * ((t + 31) // 32 * 32)
        assert t <= max_tokens and need <= max_buf, f"point {path},{t},{a} does not fit the dispatch buffer"
        print(
            f"  {path:6s} tokens/expert={t:5d} active={a:2d}  W={weight_bytes(a)/1e6:8.1f} MB  GFLOP={flops(t, a)/1e9:8.1f}"
        )
    if args.dry_run:
        return 0

    bc.setup_env()
    import torch
    import ttnn

    from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_ep_mesh_mapper
    from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert, routed_expert_weight_memory_config

    galaxy, mesh = bc.open_submesh()
    out = bc.CsvOut(args.out, COLUMNS)
    try:
        torch.manual_seed(0)
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def host_w(k, n):
            w = (torch.randn(k, n, dtype=torch.float32) * 0.02).to(torch.bfloat16)
            return ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)

        host_gate, host_up, host_down = (
            host_w(bc.EMB, bc.MOE_INTER),
            host_w(bc.EMB, bc.MOE_INTER),
            host_w(bc.MOE_INTER, bc.EMB),
        )
        mc_gu = routed_expert_weight_memory_config(mesh, bc.MOE_INTER, dram_nd_sharded=True)
        mc_d = routed_expert_weight_memory_config(mesh, bc.EMB, dram_nd_sharded=True)
        gate = [ttnn.to_device(host_gate, mesh, memory_config=mc_gu) for _ in range(EXPERTS_PER_CHIP)]
        up = [ttnn.to_device(host_up, mesh, memory_config=mc_gu) for _ in range(EXPERTS_PER_CHIP)]
        down = [ttnn.to_device(host_down, mesh, memory_config=mc_d) for _ in range(EXPERTS_PER_CHIP)]
        print(
            f"[bench_experts] weights up: gate/up {gate[0].memory_config()} down {down[0].memory_config()}", flush=True
        )

        table, _, _, _ = build_offsets_counts(16, 0, dgs, ndg, EXPERTS_PER_CHIP, bc.N_EXPERTS)
        gidx = ttnn.from_torch(
            table, mesh_mapper=get_ep_mesh_mapper(mesh), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, dtype=ttnn.uint32
        )
        gidx = ttnn.squeeze(ttnn.squeeze(gidx, 0), 0)

        # TtRoutedExpert with the weight conversion swapped for the prebuilt tensors (same forward as M3).
        orig = TtRoutedExpert.__dict__["_convert_and_cache_expert_weights"]
        TtRoutedExpert._convert_and_cache_expert_weights = staticmethod(lambda *a, **k: (gate, up, down))
        try:
            dummy = {"gate_proj": None, "up_proj": None, "down_proj": None}
            expert = TtRoutedExpert(
                mesh_device=mesh,
                experts_per_chip=EXPERTS_PER_CHIP,
                global_expert_idx_table=gidx,
                emb_dim=bc.EMB,
                hidden_dim=bc.MOE_INTER,
                max_tokens=max_tokens,
                torch_weights=[dummy] * (mesh.get_num_devices() * EXPERTS_PER_CHIP),
                activations_dtype=ttnn.bfloat8_b,
                weights_dtype=ttnn.bfloat4_b,
                activation=ttnn.RoutedExpertActivation.SwiGluOai,
                hybrid_token_threshold=args.threshold,
                weights_dram_nd_sharded=True,
            )
        finally:
            TtRoutedExpert._convert_and_cache_expert_weights = orig

        # One ROW_MAJOR bf16 dispatch buffer (max_buf, emb), random rows; regions are picked by the offsets.
        x = torch.randn(max_buf, bc.EMB, dtype=torch.bfloat16)
        x_tt = ttnn.from_torch(x, mesh_mapper=rep, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, dtype=ttnn.bfloat16)
        del x

        timer = bc.OpTimer(mesh)
        for path, t, a in pts:
            base = dict(
                path=path,
                tokens_per_expert=t,
                active_experts=a,
                weight_bytes_read=weight_bytes(a),
                flops=flops(t, a),
                total_tokens=t * a,
            )
            try:
                _, counts, offsets, _ = build_offsets_counts(t, a, dgs, ndg, EXPERTS_PER_CHIP, bc.N_EXPERTS)
                counts_tt = TtRoutedExpert.shard_expert_token_counts(mesh, counts)
                offsets_tt = TtRoutedExpert.shard_expert_token_counts(mesh, offsets)
                expert.hybrid_token_threshold = None if path == "nd" else args.threshold
                res = timer.measure(lambda: expert(x_tt, counts_tt, offsets_tt), args.warmup, args.repeats)
                out.row(**base, **res, status="OK")
                ttnn.deallocate(counts_tt)
                ttnn.deallocate(offsets_tt)
            except Exception as e:  # keep sweeping; a hang is the launcher's timeout's job
                out.row(**base, status=f"ERROR:{type(e).__name__}:{str(e).splitlines()[0][:160] if str(e) else ''}")
    finally:
        out.close()
        bc.close_mesh(galaxy)
    print(f"[bench_experts] DONE -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
