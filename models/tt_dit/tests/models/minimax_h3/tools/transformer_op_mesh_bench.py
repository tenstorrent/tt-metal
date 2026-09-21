# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""One matmul-class op of the MiniMax-H3 transformer block on one TP ring of the 4x8 Wormhole mesh, exactly as
the model runs it -- shape, fusion, blocking, grid and compute config from the shared registry `minimax_h3_ops.py`
-- host-timed over back-to-back calls and checked against fp32 torch on every device of the ring. No Tracy, no
sweep harness: the minimal reproducer for kernel changes. Not a test; pytest leaves it alone.

    python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff1
    python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_qkv [--no-fusion]
    python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op to_out [--gate-broadcast]
    python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_mesh_bench.py --op ff2 [--with-rs]
    ... [--fp32-dest 0] [--blocks 8,7,16,2,4] [--fidelity LoFi] [--iters 10]

    to_qkv  AGMM, chunks=3 (the writer splits the output into q|k|v; compute epilogue is a plain copy)
    to_out  AGMM, fused addcmul epilogue  out = a + scalar * (x @ w) * b  (a, b [M, N]; --gate-broadcast: b [1, N])
    ff1     AGMM, fused SwiGLU on the tile-pair-interleaved gate|up weight
    ff2     minimal_matmul on the full 8x9 grid (per device), --with-rs adds the ring reduce-scatter the model runs next
    --no-fusion runs the same shape as a plain matmul (what the sweep harness's "plain" use case measures).

Bar (MiniMaxH3_wormhole_perf.md, ff1 numerics): pcc > 0.9995, rel-RMSE < 0.02. The golden is computed on the first
--check-rows rows only (matmul rows are independent), which keeps the host side to seconds.
Wrap in `timeout 600` when testing kernel changes: a hang shows up as the run never printing "ms per call", and a
ring hang wedges the ETH heartbeat (recovery: `tt-smi -r all`, then wait ~75 s).

Consecutive AGMM calls alternate two semaphore pairs and two gathered-in0 buffers the way the model's CCLManager
does. With one shared set (`--no-pingpong`, what the sweep harness does) back-to-back calls race on the ring
semaphores and hang; the harness only survives because it synchronizes after every timed call.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minimax_h3_ops import OPS_BY_NAME, golden, make_extra_inputs, output_parts, prepare_weight  # noqa: E402

from models.tt_dit.utils.sweep_mm_block_sizes import close_mesh, open_mesh, resolve_config  # noqa: E402

BAR_PCC, BAR_RR = 0.9995, 0.02


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--op", default="ff1", choices=list(OPS_BY_NAME))
    p.add_argument("--M", type=int, default=None, help="rows per device (default: the op's 15 s / 768P / 16:9 M)")
    p.add_argument("--fp32-dest", type=int, default=1)
    p.add_argument(
        "--blocks", default=None, help="M_block,K_block,N_block,subblock_h,subblock_w (default: what the model runs)"
    )
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--check-rows", type=int, default=2048, help="rows of the output compared with torch (0 = skip)")
    p.add_argument(
        "--no-fusion", "--no-swiglu", dest="no_fusion", action="store_true", help="plain matmul of the same shape"
    )
    p.add_argument(
        "--gate-broadcast", action="store_true", help="to_out: addcmul b as [1, N] (row broadcast) instead of [M, N]"
    )
    p.add_argument("--with-rs", action="store_true", help="ff2: include the ring reduce-scatter in every timed call")
    p.add_argument(
        "--no-pingpong",
        action="store_true",
        help="reuse ONE semaphore set and ONE persistent buffer for every call (the sweep harness's way; races and hangs "
        "when calls are enqueued back to back -- the model alternates two of each via CCLManager)",
    )
    p.add_argument("--sync-each", action="store_true", help="synchronize the mesh after every timed call")
    p.add_argument(
        "--fidelity", default="HiFi2", choices=["LoFi", "HiFi2", "HiFi4"], help="LoFi = delivery-floor diagnostic"
    )
    args = p.parse_args()

    spec = OPS_BY_NAME[args.op]
    fused = spec.has_fusion and not args.no_fusion
    if args.with_rs and spec.family != "mm+rs":
        p.error("--with-rs applies to ff2 only")
    if args.gate_broadcast and spec.addcmul_scalar is None:
        p.error("--gate-broadcast applies to to_out only")
    M = args.M or spec.M
    K, N = spec.K, spec.N
    mb, kb, nb, sh, sw = (int(v) for v in (args.blocks or spec.blocks_str()).split(","))

    cfg = resolve_config("wh_4x8_ring")
    log("opening mesh")
    parent, mesh = open_mesh(cfg, trace_region_size=0)
    sp_axis, tp_axis = cfg["sp_axis"], cfg["tp_axis"]
    mesh_shape = tuple(mesh.shape)  # the cluster submesh: one TP ring, SP extent 1
    tp = mesh_shape[tp_axis]
    log(
        f"submesh {mesh_shape}, tp_axis {tp_axis}, sp_axis {sp_axis}; op {spec.name} ({spec.fusion if fused else 'plain'})"
    )

    torch.manual_seed(0)
    # Hand ttnn bf16 tensors: from_torch converts an fp32 tensor of this size to bf16 tiles in ~160 s on the
    # host (0.1 s from bf16). The golden below uses the same bf16-rounded values.
    w = (torch.randn(K, N) * (1.0 / K**0.5)).to(torch.bfloat16)
    tw = ttnn.from_torch(prepare_weight(spec, w, fused), dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT)
    extras = make_extra_inputs(spec, M, fused, args.gate_broadcast)
    t_extras = {
        k: ttnn.from_torch(v, dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT) for k, v in extras.items()
    }
    compute = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
        math_approx_mode=True,
        fp32_dest_acc_en=bool(args.fp32_dest),
        packer_l1_acc=True,
    )
    mmcfg = ttnn.MinimalMatmulConfig(
        M_block_size=mb,
        K_block_size=kb,
        N_block_size=nb,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=ttnn.CoreCoord(*spec.grid),
    )
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    n_sets = 1 if args.no_pingpong else 2
    call_idx = [0]

    if spec.is_agmm:
        # K-sharded activation across the ring (K_local per device), weight replicated at the per-device width.
        x = (torch.randn(M * mesh_shape[sp_axis], K) * 0.5).to(torch.bfloat16)
        tx = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh_shape, dims=[sp_axis, tp_axis]),
        )
        # Ping-pong exactly like CCLManager.get_ag_ping_pong_semaphore / get_ag_ping_pong_buffer: consecutive AGMM
        # calls alternate between two semaphore pairs and two gathered-in0 buffers, so call i+1 on a device that is
        # ahead cannot signal semaphores that call i on a neighbour is still consuming.
        sem_sets = [[ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)] for _ in range(n_sets)]
        pbufs = [
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([M, K]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            for _ in range(n_sets)
        ]
        use_addcmul = fused and spec.addcmul_scalar is not None

        def run():
            i = call_idx[0] % n_sets
            call_idx[0] += 1
            return ttnn.experimental.all_gather_minimal_matmul_async(
                tx,
                tw,
                bias_tensor=None,
                compute_kernel_config=compute,
                config=mmcfg,
                persistent_output_buffer=pbufs[i],
                multi_device_global_semaphore=sem_sets[i],
                num_links=cfg["num_links"],
                topology=cfg["topology"],
                cluster_axis=cfg["cluster_axis"],
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=cfg["num_workers_per_link"],
                num_buffers_per_channel=48,
                chunks=spec.chunks if fused else 1,
                fuse_swiglu=spec.fuse_swiglu and fused,
                scalar=spec.addcmul_scalar if use_addcmul else None,
                addcmul_input_tensor1=t_extras["a"] if use_addcmul else None,
                addcmul_input_tensor2=t_extras["b"] if use_addcmul else None,
            )

        kind = "AGMM"
    else:
        # ff2: RowParallelLinear -- every device multiplies its own K shard on the full grid, then the ring
        # reduce-scatters the [M, N] partials to N/TP per device. Inputs are replicated here (same partial on
        # every device), so the reduce-scatter's result is TP x one slice of x @ w.
        x = (torch.randn(1, 1, M, K) * 0.5).to(torch.bfloat16)
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=mesh, layout=ttnn.TILE_LAYOUT)
        if args.with_rs:
            # CCLManager.reduce_scatter without a persistent buffer: barrier semaphore + 3 semaphores per set,
            # hyperparameters from CCLManager.get_rs_hyperparams; two sets alternated like the AGMM's.
            rs_sems = [[ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(3)] for _ in range(n_sets)]
            barriers = [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(n_sets)]

        def run():
            i = call_idx[0] % n_sets
            call_idx[0] += 1
            out = ttnn.experimental.minimal_matmul(tx, tw, compute_kernel_config=compute, config=mmcfg)
            if not args.with_rs:
                return out
            rs = ttnn.experimental.reduce_scatter_minimal_async(
                out,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=rs_sems[i],
                barrier_semaphore=barriers[i],
                num_links=cfg["num_links"],
                memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
                topology=cfg["topology"],
                cluster_axis=cfg["cluster_axis"],
                chunks_per_sync=2,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            ttnn.deallocate(out)
            return rs

        kind = "MM+RS" if args.with_rs else "MM"
    log("tensors on device")

    out = run()
    ttnn.synchronize_device(mesh)
    log("first call done (compile + run)")
    t0 = time.perf_counter()
    for _ in range(args.iters):
        out = run()
        if args.sync_each:
            ttnn.synchronize_device(mesh)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / args.iters
    tag = f"fused ({spec.fusion})" if fused else "plain"
    mode = f"{args.fidelity}, {'1 set' if args.no_pingpong else 'ping-pong'}{', sync each' if args.sync_each else ''}"
    log(
        f"{kind} {spec.name} {tag} blocks ({mb},{kb},{nb}) sb ({sh},{sw}) fp32_dest={bool(args.fp32_dest)} [{mode}]: "
        f"{dt * 1e3:.2f} ms per call, host-timed over {args.iters} back-to-back calls"
    )

    if args.check_rows:
        r = args.check_rows
        with torch.no_grad():
            xr = x.reshape(-1, K)[:r]
            er = {k: (v if v.shape[0] == 1 else v[:r]) for k, v in extras.items()}
            gold = golden(spec, xr, w, er, fused)
        worst_pcc, worst_rr = 1.0, 0.0
        parts = output_parts(spec, out, fused)
        per_part = [ttnn.get_device_tensors(t) for t in parts]
        for d in range(len(per_part[0])):
            o = torch.cat([ttnn.to_torch(pp[d]).float().reshape(-1, pp[d].shape[-1])[:r] for pp in per_part], dim=-1)
            g = gold
            if args.with_rs:  # replicated partials: device d holds TP x slice d of the product
                g = tp * gold[:, d * spec.N_out : (d + 1) * spec.N_out]
            pcc = torch.corrcoef(torch.stack([o.flatten(), g.flatten()]))[0, 1].item()
            rr = ((o - g).pow(2).mean().sqrt() / g.pow(2).mean().sqrt()).item()
            worst_pcc, worst_rr = min(worst_pcc, pcc), max(worst_rr, rr)
            log(f"  device {d}: pcc {pcc:.7f}  rel-rmse {rr:.5f}")
        verdict = "OK" if worst_pcc > BAR_PCC and worst_rr < BAR_RR else "FAILS THE BAR"
        log(
            f"worst: pcc {worst_pcc:.7f}  rel-rmse {worst_rr:.5f}  (bar: pcc > {BAR_PCC}, rel-rmse < {BAR_RR}) {verdict}"
        )
    close_mesh(parent)
    log("closed")


if __name__ == "__main__":
    main()
