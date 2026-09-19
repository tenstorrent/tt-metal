# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ff1 AGMM on one TP ring of the 4x8 Wormhole mesh exactly as the model runs it -- fused SwiGLU,
bias=False, HiFi2, fp32 dest, 8x8 grid, blocking (8, 7, 10) subblock 2x2 -- host-timed and checked
against fp32 torch on every device of the ring. No Tracy, no sweep harness: the minimal reproducer
for kernel changes to the AGMM compute kernel. Not a test; pytest leaves it alone.

    python models/tt_dit/tests/models/minimax_h3/tools/agmm_ff1_mesh_bench.py [--fp32-dest 0] [--blocks 8,7,10,2,2]

Bar (MiniMaxH3_wormhole_perf.md, ff1 numerics): pcc > 0.9995, rel-RMSE < 0.02. The golden is computed
on the first --check-rows rows only (matmul rows are independent), which keeps the host side to seconds.
Wrap in `timeout 600` when testing kernel changes: a hang shows up as the run never printing "ms per call".

Consecutive calls alternate two semaphore pairs and two gathered-in0 buffers the way the model's CCLManager
does. With one shared set (`--no-pingpong`, what the sweep harness does) back-to-back calls race on the ring
semaphores and hang; the harness only survives because it synchronizes after every timed call.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

import ttnn

sys.path.insert(0, ".")
from models.tt_dit.utils.sweep_mm_block_sizes import close_mesh, open_mesh, resolve_config  # noqa: E402
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu  # noqa: E402

M, K, N2 = 13664, 5376, 7168  # 15 s / 768P / 16:9 rows per device; hidden; packed gate|up


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fp32-dest", type=int, default=1)
    p.add_argument("--blocks", default="8,7,10,2,2", help="M_block,K_block,N_block,subblock_h,subblock_w")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--check-rows", type=int, default=2048, help="rows of the output compared with torch (0 = skip)")
    p.add_argument("--no-swiglu", action="store_true", help="plain AGMM (N = 7168 out) instead of fused SwiGLU")
    p.add_argument(
        "--no-pingpong",
        action="store_true",
        help="reuse ONE semaphore pair and ONE persistent buffer for every call (the sweep harness's way; races and hangs "
        "when calls are enqueued back to back -- the model alternates two of each via CCLManager)",
    )
    p.add_argument("--sync-each", action="store_true", help="synchronize the mesh after every timed call")
    p.add_argument(
        "--fidelity", default="HiFi2", choices=["LoFi", "HiFi2", "HiFi4"], help="LoFi = delivery-floor diagnostic"
    )
    args = p.parse_args()
    mb, kb, nb, sh, sw = (int(v) for v in args.blocks.split(","))

    cfg = resolve_config("wh_4x8_ring")
    log("opening mesh")
    parent, mesh = open_mesh(cfg, trace_region_size=0)
    sp_axis, tp_axis = cfg["sp_axis"], cfg["tp_axis"]
    mesh_shape = tuple(mesh.shape)  # the cluster submesh: one TP ring, SP extent 1
    log(f"submesh {mesh_shape}, tp_axis {tp_axis}, sp_axis {sp_axis}")

    torch.manual_seed(0)
    # Hand ttnn bf16 tensors: from_torch converts an fp32 tensor of this size to bf16 tiles in ~160 s on the
    # host (0.1 s from bf16). The golden below uses the same bf16-rounded values.
    x = (torch.randn(M * mesh_shape[sp_axis], K) * 0.5).to(torch.bfloat16)
    w = (torch.randn(K, N2) * (1.0 / K**0.5)).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=mesh_shape, dims=[sp_axis, tp_axis]),
    )
    tw = ttnn.from_torch(
        w if args.no_swiglu else prepare_for_fused_swiglu(w, ndev=1, gate_is_first=True),
        dtype=ttnn.bfloat16,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
    )
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    # Ping-pong exactly like CCLManager.get_ag_ping_pong_semaphore / get_ag_ping_pong_buffer: consecutive AGMM
    # calls alternate between two semaphore pairs and two gathered-in0 buffers, so call i+1 on a device that is
    # ahead cannot signal semaphores that call i on a neighbour is still consuming.
    n_sets = 1 if args.no_pingpong else 2
    sem_sets = [[ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)] for _ in range(n_sets)]
    pbufs = [
        ttnn.allocate_tensor_on_device(
            ttnn.Shape([M, K]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        for _ in range(n_sets)
    ]
    call_idx = [0]
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
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
    )
    log("tensors on device")

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
            fuse_swiglu=not args.no_swiglu,
        )

    out = run()
    ttnn.synchronize_device(mesh)
    log("first call done (compile + run)")
    t0 = time.perf_counter()
    for _ in range(args.iters):
        run()
        if args.sync_each:
            ttnn.synchronize_device(mesh)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / args.iters
    tag = "plain" if args.no_swiglu else "swiglu"
    mode = f"{args.fidelity}, {'1 set' if args.no_pingpong else 'ping-pong'}{', sync each' if args.sync_each else ''}"
    log(
        f"AGMM ff1 {tag} blocks ({mb},{kb},{nb}) sb ({sh},{sw}) fp32_dest={bool(args.fp32_dest)} [{mode}]: "
        f"{dt * 1e3:.2f} ms per call, host-timed over {args.iters} back-to-back calls"
    )

    if args.check_rows:
        r = args.check_rows
        with torch.no_grad():
            xr, wf = x[:r].float(), w.float()
            if args.no_swiglu:
                golden = xr @ wf
            else:
                g, u = torch.chunk(xr @ wf, 2, dim=-1)
                golden = torch.nn.functional.silu(g) * u
        worst_pcc, worst_rr = 1.0, 0.0
        out_t = out[0] if isinstance(out, (list, tuple)) else out  # the op returns a list (one entry per chunk)
        for i, t in enumerate(ttnn.get_device_tensors(out_t)):
            o = ttnn.to_torch(t).float().reshape(-1, golden.shape[1])[:r]
            pcc = torch.corrcoef(torch.stack([o.flatten(), golden.flatten()]))[0, 1].item()
            rr = ((o - golden).pow(2).mean().sqrt() / golden.pow(2).mean().sqrt()).item()
            worst_pcc, worst_rr = min(worst_pcc, pcc), max(worst_rr, rr)
            log(f"  device {i}: pcc {pcc:.7f}  rel-rmse {rr:.5f}")
        log(f"worst: pcc {worst_pcc:.7f}  rel-rmse {worst_rr:.5f}  (bar: pcc > 0.9995, rel-rmse < 0.02)")
    close_mesh(parent)
    log("closed")


if __name__ == "__main__":
    main()
