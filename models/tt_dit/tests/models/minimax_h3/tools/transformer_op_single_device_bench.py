# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Single-device stand-in for one matmul-class op of the MiniMax-H3 transformer block: the per-device shape of
the op (from `minimax_h3_ops.py`) run on one Wormhole chip through the single-device op that carries the same
fusion, plain and fused, host-timed and checked against fp32 torch. The AGMM's matmul half is a near-verbatim
copy of `minimal_matmul` (same relay-chain data movement, same compute kernel structure, same epilogues) and the
ring gather is fully hidden (MiniMaxH3_wormhole_perf.md, "ff1 AGMM: where the other 49% goes", exp 3), so this is
the fast loop for compute-kernel changes: ~25 s per case including the JIT compile, no mesh, no Tracy. Not a test;
pytest leaves it alone.

    python models/tt_dit/tests/models/minimax_h3/tools/transformer_op_single_device_bench.py --op ff1
    python .../transformer_op_single_device_bench.py --op to_out --cases "8,8,6,2,2,1;16,8,6,4,2,0" --fidelity LoFi,HiFi2

    op      plain                    fused
    to_qkv  minimal_matmul           minimal_matmul_split(chunks=3)            (the AGMM's writer-side q|k|v split)
    to_out  minimal_matmul           dit_minimal_matmul_addcmul_fused          (a + scalar * (x @ w) * b)
    ff1     minimal_matmul           minimal_matmul(fuse_swiglu=True)
    ff2     minimal_matmul on 8x9    -- (the reduce-scatter is a mesh op; see transformer_op_mesh_bench.py --with-rs)

Each case is M_block,K_block,N_block,subblock_h,subblock_w,fp32_dest (default: the op's model blocking, fp32 on;
ff1 keeps its two historical cases). Utilisation is against the op's grid (64 cores for the AGMM stand-ins, 72
for ff2) at 2048 FLOP/cycle/core (HiFi2) and 1.0 GHz; scale for other fidelities. Host timing includes dispatch,
so absolute numbers run ~1-2% high; A/B deltas between cases are what this measures. Host tensors are bf16 and the
golden uses the same rounded values.

Reference numbers (2026-09-18, HiFi2, ff1 blocks (8,7,10) sb 2x2 fp32 dest): plain 14.5 ms, SwiGLU 15.7 ms.
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

FLOP_PER_CYCLE_HIFI2, CLOCK = 2048, 1.0e9
FIDELITY_CYCLES = {"LoFi": 1, "HiFi2": 2, "HiFi3": 3, "HiFi4": 4}
FF1_CASES = "8,7,10,2,2,1;12,7,8,4,2,0"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--op", default="ff1", choices=list(OPS_BY_NAME))
    p.add_argument("--M", type=int, default=None, help="rows per device (default: the op's 15 s / 768P / 16:9 M)")
    p.add_argument("--cases", default=None, help="M_block,K_block,N_block,sb_h,sb_w,fp32_dest ; ...")
    p.add_argument("--fidelity", default="HiFi2", help="comma list from LoFi,HiFi2,HiFi3,HiFi4")
    p.add_argument("--iters", type=int, default=8)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--no-fusion", "--no-swiglu", dest="no_fusion", action="store_true", help="skip the fused variant")
    p.add_argument("--no-plain", action="store_true", help="skip the plain variant")
    p.add_argument(
        "--gate-broadcast", action="store_true", help="to_out: addcmul b as [1, N] (row broadcast) instead of [M, N]"
    )
    args = p.parse_args()

    spec = OPS_BY_NAME[args.op]
    M, K, N = args.M or spec.M, spec.K, spec.N
    cases = args.cases or (FF1_CASES if spec.name == "ff1" else f"{spec.blocks_str()},1")
    run_fused = spec.has_fusion and not args.no_fusion
    if spec.has_fusion is False and not args.no_fusion:
        print(f"note: {spec.name} has no single-device fused variant; running plain only", flush=True)

    dev = ttnn.open_device(device_id=args.device)
    grid = dev.compute_with_storage_grid_size()
    print(
        f"device {args.device}: grid {grid.x}x{grid.y}; op {spec.name}, M={M} K={K} N={N}, grid {spec.grid}", flush=True
    )
    torch.manual_seed(0)
    x = (torch.randn(M, K) * 0.5).to(torch.bfloat16)
    w = (torch.randn(K, N) * (1.0 / K**0.5)).to(torch.bfloat16)
    extras = make_extra_inputs(spec, M, True, args.gate_broadcast) if run_fused else {}
    with torch.no_grad():
        golden_plain = golden(spec, x, w, {}, fused=False)
        golden_fused = golden(spec, x, w, extras, fused=True) if run_fused else None
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
    tw_fused = (
        ttnn.from_torch(prepare_weight(spec, w, True), dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT)
        if spec.fuse_swiglu
        else tw
    )
    t_extras = {
        k: ttnn.from_torch(v, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT) for k, v in extras.items()
    }

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
            compute_with_storage_grid_size=ttnn.CoreCoord(*spec.grid),
        )

    def fused_call(compute, config):
        if spec.chunks > 1:
            return ttnn.experimental.minimal_matmul_split(
                tx, tw, chunks=spec.chunks, dim=-1, compute_kernel_config=compute, config=config
            )
        if spec.addcmul_scalar is not None:
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                tx, tw, spec.addcmul_scalar, t_extras["a"], t_extras["b"], compute_kernel_config=compute, config=config
            )
        return ttnn.experimental.minimal_matmul(
            tx, tw_fused, compute_kernel_config=compute, config=config, fuse_swiglu=True
        )

    def quality(out, gold, fused):
        parts = output_parts(spec, out, fused)
        o = torch.cat([ttnn.to_torch(t).float().reshape(-1, t.shape[-1]) for t in parts], dim=-1)
        g = gold[: o.shape[0], : o.shape[1]]
        pcc = torch.corrcoef(torch.stack([o.flatten(), g.flatten()]))[0, 1].item()
        return pcc, ((o - g).pow(2).mean().sqrt() / g.pow(2).mean().sqrt()).item(), parts

    def bench(name, fn, gold, fid, fused):
        peak = spec.cores * FLOP_PER_CYCLE_HIFI2 * 2 / FIDELITY_CYCLES[fid] * CLOCK
        try:
            out = fn()
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                out = fn()
            ttnn.synchronize_device(dev)
            dt = (time.perf_counter() - t0) / args.iters
            pcc, rr, parts = quality(out, gold, fused)
            print(
                f"{name:64s} {dt * 1e3:8.2f} ms  util {2.0 * M * K * N / dt / peak * 100:5.1f}%  pcc {pcc:.6f} rel-rmse {rr:.5f}",
                flush=True,
            )
            for t in parts:
                ttnn.deallocate(t)
        except Exception as e:  # L1 overflow for too-large blocks lands here; keep going
            print(f"{name:64s} FAILED: {str(e).splitlines()[0][:140]}", flush=True)

    for fid in args.fidelity.split(","):
        for case in cases.split(";"):
            m, k, n, sh, sw, fp32 = (int(v) for v in case.split(","))
            tag = f"{fid:5s} blk({m},{k},{n}) sb({sh},{sw}) fp32_dest={'on ' if fp32 else 'off'}"
            compute, config = cfg(bool(fp32), fid), mmcfg(m, k, n, sh, sw)
            if not args.no_plain:
                bench(
                    f"{spec.name:6s} plain  minimal_matmul        {tag}",
                    lambda: ttnn.experimental.minimal_matmul(tx, tw, compute_kernel_config=compute, config=config),
                    golden_plain,
                    fid,
                    False,
                )
            if run_fused:
                bench(
                    f"{spec.name:6s} fused  {spec.fusion:21s} {tag}",
                    lambda: fused_call(compute, config),
                    golden_fused,
                    fid,
                    True,
                )
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
