# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Single-device stand-in for the ff1 AGMM: `minimal_matmul` on one Wormhole chip at the per-device shape
(M=13664, K=5376 gathered, N=7168 packed gate|up), plain and with fused SwiGLU, host-timed and checked
against fp32 torch. The AGMM's matmul half is a near-verbatim copy of `minimal_matmul` (same relay-chain
data movement, same compute kernel structure, same `swiglu_block`), and the ring gather is fully hidden
(MiniMaxH3_wormhole_perf.md, "ff1 AGMM: where the other 49% goes", exp 3), so this is the fast loop for
compute-kernel and data-movement changes: ~25 s per case including the JIT compile, no mesh, no Tracy.
Not a test; pytest leaves it alone.

    python models/tt_dit/tests/models/minimax_h3/tools/agmm_ff1_single_device_bench.py
    python .../agmm_ff1_single_device_bench.py --cases "8,7,10,2,2,1;12,7,8,4,2,0" --fidelity LoFi,HiFi2

Each case is M_block,K_block,N_block,subblock_h,subblock_w,fp32_dest. Utilisation is against the 64-core
8x8 AGMM grid at 2048 FLOP/cycle/core (HiFi2) and 1.0 GHz; scale for other fidelities. Host timing includes
dispatch, so absolute numbers run ~1-2% high; A/B deltas between cases are what this measures.

Reference numbers (2026-09-18, HiFi2, blocks (8,7,10) sb 2x2 fp32 dest): plain 14.5 ms, SwiGLU 16.5 ms.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

import ttnn

sys.path.insert(0, ".")
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu  # noqa: E402

M, K, N2 = 13664, 5376, 7168
CORES, FLOP_PER_CYCLE_HIFI2, CLOCK = 64, 2048, 1.0e9
FIDELITY_CYCLES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cases", default="8,7,10,2,2,1;12,7,8,4,2,0", help="M_block,K_block,N_block,sb_h,sb_w,fp32_dest ; ..."
    )
    p.add_argument("--fidelity", default="HiFi2", help="comma list from LoFi,HiFi2,HiFi3,HiFi4")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--no-swiglu", action="store_true")
    p.add_argument("--no-plain", action="store_true")
    args = p.parse_args()

    dev = ttnn.open_device(device_id=args.device)
    grid = dev.compute_with_storage_grid_size()
    print(f"device {args.device}: grid {grid.x}x{grid.y}", flush=True)
    torch.manual_seed(0)
    x = torch.randn(M, K) * 0.5
    w = torch.randn(K, N2) * (1.0 / K**0.5)
    with torch.no_grad():
        full = x @ w
        gate, up = torch.chunk(full, 2, dim=-1)
        golden_swiglu = torch.nn.functional.silu(gate) * up
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
    tw_il = ttnn.from_torch(
        prepare_for_fused_swiglu(w, ndev=1, gate_is_first=True),
        dtype=ttnn.bfloat16,
        device=dev,
        layout=ttnn.TILE_LAYOUT,
    )

    def cfg(fp32: bool, fid: str):
        return ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, fid),
            math_approx_mode=True,
            fp32_dest_acc_en=fp32,
            packer_l1_acc=True,
        )

    def mmcfg(m, k, n, sh, sw):
        return ttnn.MinimalMatmulConfig(
            M_block_size=m,
            K_block_size=k,
            N_block_size=n,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        )

    def quality(out_tt, golden):
        o = ttnn.to_torch(out_tt).float()
        g = golden[: o.shape[0], : o.shape[1]]
        pcc = torch.corrcoef(torch.stack([o.flatten(), g.flatten()]))[0, 1].item()
        return pcc, ((o - g).pow(2).mean().sqrt() / g.pow(2).mean().sqrt()).item()

    def bench(name, fn, golden, fid):
        peak = CORES * FLOP_PER_CYCLE_HIFI2 * 2 / FIDELITY_CYCLES[fid] * CLOCK
        try:
            out = fn()
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                out = fn()
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / args.iters
            pcc, rr = quality(out, golden)
            print(
                f"{name:60s} {dt * 1e3:8.2f} ms  util {2.0 * M * K * N2 / dt / peak * 100:5.1f}%  pcc {pcc:.6f} rel-rmse {rr:.5f}",
                flush=True,
            )
            ttnn.deallocate(out)
        except Exception as e:  # L1 overflow for too-large blocks lands here; keep going
            print(f"{name:60s} FAILED: {str(e).splitlines()[0][:140]}", flush=True)

    for fid in args.fidelity.split(","):
        for case in args.cases.split(";"):
            m, k, n, sh, sw, fp32 = (int(v) for v in case.split(","))
            tag = f"{fid:5s} blk({m},{k},{n}) sb({sh},{sw}) fp32_dest={'on ' if fp32 else 'off'}"
            if not args.no_plain:
                bench(
                    f"minimal_matmul plain  {tag}",
                    lambda: ttnn.experimental.minimal_matmul(
                        tx, tw, compute_kernel_config=cfg(bool(fp32), fid), config=mmcfg(m, k, n, sh, sw)
                    ),
                    full,
                    fid,
                )
            if not args.no_swiglu:
                bench(
                    f"minimal_matmul swiglu {tag}",
                    lambda: ttnn.experimental.minimal_matmul(
                        tx,
                        tw_il,
                        compute_kernel_config=cfg(bool(fp32), fid),
                        config=mmcfg(m, k, n, sh, sw),
                        fuse_swiglu=True,
                    ),
                    golden_swiglu,
                    fid,
                )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
