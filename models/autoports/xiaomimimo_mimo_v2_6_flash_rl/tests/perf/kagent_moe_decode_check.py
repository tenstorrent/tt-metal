# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate for MiMo MoE decode changes (one layer on Galaxy row 0, 4 users per step).

1. BFP8 experts on real MoE inputs of the golden prompts: the device output against the fp32 reference
   evaluated with the device's own top-8 (so near-tie routing flips cannot mask expert math errors),
   PCC >= 0.9999, and top-8 set agreement with the fp32 router >= 95% (base: 97.9%).
2. BFP4 experts (the full model's format): the output must match the base implementation's saved output
   on the same inputs, PCC >= 0.999 (``--write-base`` records it; it lives outside the repository).
3. The dense 1D-multicast matmul shares its kernels with sparse_matmul, and the rest of the model uses it:
   a few shapes with in0- and in1-multicast program configs must match torch (PCC >= 0.999).

Prints MOE_DECODE_CORRECTNESS_PASS after every check passed.
"""

import argparse
import os

import torch

import ttnn
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.reference import model as ref
from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.perf import moe_decode_common as common

BASE_OUTPUT = os.environ.get("MIMO_MOE_DECODE_BASE", "/home/zni/models/mimo-golden/moe_decode_layer1_bfp4_base.pt")
STEPS = 24


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run_decode(row, moe, inputs):
    """Decode STEPS x 4 users; returns (outputs [rows, hidden], device top-8 ids [rows, 8], weights)."""
    outs, ids, weights = [], [], []
    for s in range(STEPS):
        rows = inputs[s * common.USERS : (s + 1) * common.USERS]
        x, mask = common.device_inputs(row, rows)
        dense = moe.route(x, mask)  # the device's routing, for the expert-math reference
        out = moe.forward_decode(x, mask)
        d = ttnn.to_torch(ttnn.get_device_tensors(dense)[0]).float()[0, 0, : common.USERS]
        o = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0, : common.USERS]
        w, i = d.topk(8, dim=-1)
        outs.append(o)
        ids.append(i)
        weights.append(w)
        for t in (x, mask, dense, out):
            t.deallocate(True)
    return torch.cat(outs), torch.cat(ids), torch.cat(weights)


def check_dense_matmuls(row):
    """Dense matmuls through the 1D mcast program (same kernels as sparse_matmul) vs torch."""
    from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tt.common import compute_config

    rep = ttnn.ReplicateTensorToMesh(row)
    worst = 1.0
    torch.manual_seed(3)
    for m, k, n, mcast_in0 in ((32, 4096, 2048, True), (128, 2048, 4096, True), (1024, 512, 256, False)):
        a = torch.randn(1, 1, m, k)
        b = torch.randn(1, 1, k, n) * 0.05
        grid = (8, 4) if mcast_in0 else (4, 8)
        cores = grid[0] * grid[1]
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1 if mcast_in0 else m // 32 // cores,
            out_block_w=n // 32 // cores if mcast_in0 else n // 32,
            per_core_M=m // 32 if mcast_in0 else m // 32 // cores,
            per_core_N=n // 32 // cores if mcast_in0 else n // 32,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=mcast_in0,
        )
        a_tt = ttnn.from_torch(a, device=row, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
        b_tt = ttnn.from_torch(b, device=row, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep)
        out = ttnn.matmul(a_tt, b_tt, program_config=pc, compute_kernel_config=compute_config(ttnn.MathFidelity.HiFi2))
        got = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
        worst = min(worst, pcc(got, a @ b))
    print(f"dense 1D-mcast matmuls: worst PCC {worst:.6f}", flush=True)
    return worst >= 0.999


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--write-base", action="store_true", help="record the BFP4 output of this code as the base")
    args = parser.parse_args()

    cfg = common.config()
    ckpt = common.checkpoint(cfg)
    inputs = common.golden_moe_inputs(cfg, ckpt)[: STEPS * common.USERS]
    router = ckpt.router(common.LAYER)
    mesh, row = common.open_row()
    try:
        ok = True
        if not args.write_base:
            moe8 = common.build_moe(row, cfg, "layer_bfp8", ttnn.bfloat8_b, ckpt=ckpt)
            got, ids, w = run_decode(row, moe8, inputs)
            experts = ref.LazyExperts(ckpt, common.LAYER, torch.bfloat16)
            from models.autoports.xiaomimimo_mimo_v2_6_flash_rl.tests.test_moe_tp8_dp4 import reference_with_routing

            with torch.no_grad():
                forced = reference_with_routing(inputs, experts, ids, w)
            ref_idx, _ = ref.route(inputs, router, cfg)
            match = sum(set(a.tolist()) == set(b.tolist()) for a, b in zip(ref_idx, ids)) / len(ids)
            p = pcc(got, forced)
            print(f"bfp8 experts: PCC vs reference with device routing {p:.6f}; top-8 match {match:.3f}", flush=True)
            ok &= p >= 0.9999 and match >= 0.95  # routing near-ties may flip a token or two

        moe4 = common.build_moe(row, cfg, "model_bfp4experts", ttnn.bfloat4_b)
        got4, _, _ = run_decode(row, moe4, inputs)
        if args.write_base:
            torch.save(got4, BASE_OUTPUT)
            print(f"wrote base output {BASE_OUTPUT}", flush=True)
            return
        p4 = pcc(got4, torch.load(BASE_OUTPUT))
        print(f"bfp4 experts: PCC vs base implementation {p4:.6f}", flush=True)
        ok &= p4 >= 0.999
        ok &= check_dense_matmuls(row)
    finally:
        common.close(mesh, row)
    if ok:
        print("MOE_DECODE_CORRECTNESS_PASS", flush=True)
    else:
        raise RuntimeError("MoE decode correctness check failed")


if __name__ == "__main__":
    main()
