# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: vision QKV linear ([1,1,256,1024] x [1024,3072]) with the model's 2x1-grid program config
(MatmulMultiCoreReuseMultiCastProgramConfig, K split into blocks), packer L1 accumulation on and off.

usage: linear_kblocks.py [exact]   exact: only the model's call (whole 8x48 per-core output in one block)"""
import sys
import time

import torch
import ttnn


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def program_config(in0_block_w, out_block_w=24):
    # What quasar_config.fit_matmul_config produces for this linear on a 2x1 grid.
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(2, 1),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=8,
        out_block_w=out_block_w,
        per_core_M=8,
        per_core_N=48,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    x, w = torch.randn(1, 1, 256, 1024), torch.randn(1024, 3072) / 32
    up = lambda t: ttnn.from_torch(t.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    tx, tw = up(x), up(w)
    ref = x.bfloat16().float() @ w.bfloat16().float()
    # (packer_l1_acc, in0_block_w, out_block_w); the model's QKV call is (True, 1, 48).
    variants = [(True, 1, 48), (False, 1, 48)]
    if sys.argv[1:] != ["exact"]:
        variants += [(acc, k, 24) for acc in (False, True) for k in (1, 32)]
    for packer_l1_acc, in0_block_w, out_block_w in variants:
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=packer_l1_acc,
        )
        t0 = time.time()
        out = ttnn.linear(
            tx,
            tw,
            compute_kernel_config=ckc,
            program_config=program_config(in0_block_w, out_block_w),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        got = ttnn.to_torch(out).float()
        print(
            f"packer_l1_acc={packer_l1_acc!s:5s} in0_block_w={in0_block_w:2d} out_block_w={out_block_w} "
            f"pcc={pcc(got, ref):.5f} std_ratio={(got.std() / ref.std()).item():.3f} {time.time() - t0:.1f}s",
            flush=True,
        )
        ttnn.deallocate(out)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
