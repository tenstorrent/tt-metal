# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Quasar Q17: a tiled ttnn.slice leaves dirty DM-cache lines that later corrupt a width-sharded RMSNorm.

The slice reader keeps its index odometer in a Scratchpad written through the DM cache. Those dirty L2 lines
outlive the program; the sharded RMSNorm writer's reduce-scaler fill flushes the cached alias of its tile and
writes them back over the fresh scaler, so E[x^2] reduces to 0 and the output is x * gamma / sqrt(eps).
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
    x, w = torch.randn(1, 1, 32, DIM) * 0.03, torch.randn(DIM)
    xf = x.bfloat16().float()
    ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.bfloat16().float()
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
        ratio = (got[0, 0, 0, :4] / ref[0, 0, 0, :4]).tolist()
        print(f"{label:24s} pcc={pcc(got, ref):.5f} row0 got/ref={[round(r, 3) for r in ratio]}", flush=True)

    norm("fresh process")
    # The prefill tail: keep the last 32 rows of the hidden state (tile-layout interleaved slice).
    h = ttnn.from_torch(torch.randn(1, 1, 128, DIM).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.slice(h, (0, 0, 96, 0), (1, 1, 128, DIM))
    norm("after tiled slice")
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
