# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar: decode RMSNorm (width-sharded L1, sharded program config) before and after other ops have used L1.

In the e2e the decode norm is off by a constant factor per row after prefill, but correct in a fresh process.
"""
import torch
import ttnn

DIM = 2560


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().double(), b.flatten().double()]))[0, 1].item()


def main():
    torch.manual_seed(0)
    dev = ttnn.open_device(device_id=0)
    g = dev.compute_with_storage_grid_size()
    n = g.x * g.y
    x, w = torch.randn(1, 1, 32, DIM), torch.randn(DIM)
    ref = (
        x.bfloat16().float()
        * torch.rsqrt(x.bfloat16().float().pow(2).mean(-1, keepdim=True) + 1e-6)
        * w.bfloat16().float()
    )
    ws = ttnn.create_sharded_memory_config(
        shape=(32, DIM // n),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), subblock_w=1, block_h=1, block_w=DIM // 32 // n, inplace=False
    )
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
    )
    tw = ttnn.from_torch(w.bfloat16().reshape(1, 1, DIM // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

    def norm(label):
        tx = ttnn.from_torch(x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ws)
        out = ttnn.rms_norm(
            tx, epsilon=1e-6, weight=tw, program_config=prog, memory_config=ws, compute_kernel_config=ckc
        )
        got = ttnn.to_torch(out).float()
        ratio = (got[0, 0, 0, :8] / ref[0, 0, 0, :8]).tolist()
        print(f"{label:28s} pcc={pcc(got, ref):.5f} row0 got/ref={[round(r, 3) for r in ratio]}", flush=True)

    norm("fresh process")
    # Prefill-like work that fills circular buffers with data: interleaved norms and a matmul.
    big = ttnn.from_torch((torch.randn(1, 1, 128, DIM) * 50).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    big_w = ttnn.from_torch(w.bfloat16().reshape(1, 1, DIM // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    for _ in range(2):
        ttnn.rms_norm(big, epsilon=1e-6, weight=big_w, compute_kernel_config=ckc)
        ttnn.linear(big, ttnn.from_torch(torch.randn(DIM, 1024).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev))
    norm("after interleaved norm+matmul")
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
