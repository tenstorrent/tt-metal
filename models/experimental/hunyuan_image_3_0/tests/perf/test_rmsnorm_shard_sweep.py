# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolation go/no-go sweep for width-sharding the Hunyuan backbone RMSNorm.

The big input/post-attn/ln_f RMSNorms run [1, Sd, 4096] (Sd = per-device seq).
At Sd=32 (1 row-tile) the interleaved ttnn.rms_norm kernel parallelizes over the
row dim and lands on a SINGLE core -> 62 us in the AR perf report. This sweep asks:
does width-sharding the 4096 hidden across cores beat the 1-core interleaved norm,
INCLUDING the I2S-in / S2I-out reshards the model would pay?

  pytest models/experimental/hunyuan_image_3_0/tests/perf/test_rmsnorm_shard_sweep.py -s

Matches the in-model norm config: bf16 act, bf16 gamma (ROW_MAJOR [1,1,H/32,32]),
HiFi2 + fp32_dest_acc_en=True + packer_l1_acc=True (models/common/rmsnorm.py).
"""
from __future__ import annotations

import time

import pytest
import torch
import ttnn

H = 4096
_TILE = 32
EPS = 1e-5


def _bench(dev, fn, it=50):
    for _ in range(5):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(it):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / it * 1e6


def _pcc(a, b):
    a = a.float().flatten() - a.float().mean()
    b = b.float().flatten() - b.float().mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _divisors_leq(x, cap):
    return [d for d in range(cap, 1, -1) if x % d == 0]


def _torch_rmsnorm(x, w, eps=EPS):
    x = x.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w.float()


@pytest.mark.parametrize("S", [32, 64, 96], ids=["seq32_mt1", "seq64_mt2", "seq96_mt3"])
def test_rmsnorm_shard_sweep(device, S):
    _run_sweep(device, S, gx_only=None)


# Ship validation: benchmark the ACTUAL config rmsnorm_shard_config() returns (2-D
# block-shard, gy for rows / gx for width) vs the interleaved baseline, across the
# recaption Mt range (1-9) and into the large-Mt prefill/denoise regime (where the
# helper should return None => interleaved). This is what verifies the 2-D OOM fix.
@pytest.mark.parametrize(
    "S",
    [32, 64, 96, 128, 160, 224, 288, 512, 1024, 2048],
    ids=lambda s: f"seq{s}_mt{s // 32}",
)
def test_rmsnorm_ship(device, S):
    from models.experimental.hunyuan_image_3_0.tt.parallel_utils import rmsnorm_shard_config

    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    Mt = S // _TILE
    torch.manual_seed(0)
    x = torch.randn(1, S, H, dtype=torch.bfloat16) * 0.05
    w = (torch.randn(H, dtype=torch.bfloat16) * 0.02 + 1.0).to(torch.bfloat16)
    ref = _torch_rmsnorm(x.reshape(-1, H), w).reshape(1, S, H)
    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    wt = ttnn.from_torch(
        w.view(1, 1, H // _TILE, _TILE),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def norm_il():
        return ttnn.rms_norm(xt, epsilon=EPS, weight=wt, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)

    base_us = _bench(device, norm_il)
    cfg = rmsnorm_shard_config(device, S, H)
    if cfg is None:
        print(f"\n[ship] S={S} Mt={Mt}: helper=None (interleaved) baseline={base_us:.1f}u  <- fallback")
        return
    pc, shard_mc = cfg
    gx, gy = pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y
    bh, bw = pc.block_h, pc.block_w

    def full():
        xs = ttnn.interleaved_to_sharded(xt, shard_mc)
        ys = ttnn.rms_norm(
            xs, epsilon=EPS, weight=wt, program_config=pc, memory_config=shard_mc, compute_kernel_config=ckc
        )
        ttnn.deallocate(xs)
        yi = ttnn.sharded_to_interleaved(ys, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(ys)
        return yi

    try:
        out = ttnn.to_torch(full()).reshape(1, S, H)
        fl_us = _bench(device, full)
        print(
            f"\n[ship] S={S} Mt={Mt}: grid=({gx}x{gy})={gx * gy}c bh={bh} bw={bw}  "
            f"interleaved={base_us:.1f}u  sharded-full={fl_us:.1f}u  ({base_us / fl_us:.2f}x)  PCC={_pcc(out, ref):.6f}"
        )
    except Exception as e:
        print(f"\n[ship] S={S} Mt={Mt}: grid=({gx}x{gy}) bh={bh} bw={bw}  FAIL {str(e)[:70]}")


def _run_sweep(device, S, gx_only=None):
    device.enable_program_cache()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    grid = device.compute_with_storage_grid_size()
    Mt, Nt = S // _TILE, H // _TILE  # Nt = 128

    torch.manual_seed(0)
    x = torch.randn(1, S, H, dtype=torch.bfloat16) * 0.05
    w = (torch.randn(H, dtype=torch.bfloat16) * 0.02 + 1.0).to(torch.bfloat16)
    ref = _torch_rmsnorm(x.reshape(-1, H), w).reshape(1, S, H)

    xt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    # gamma as ROW_MAJOR [1,1,H/32,32], exactly like models/common/rmsnorm.py
    wt = ttnn.from_torch(
        w.view(1, 1, H // _TILE, _TILE),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ---- baseline: interleaved (program_config=None) -> 1 core at Mt=1 ----
    def norm_il():
        return ttnn.rms_norm(xt, epsilon=EPS, weight=wt, compute_kernel_config=ckc, memory_config=ttnn.L1_MEMORY_CONFIG)

    base_out = ttnn.to_torch(norm_il()).reshape(1, S, H)
    base_us = _bench(device, norm_il)
    print(f"\n=== RMSNorm [1,{S},{H}]  Mt={Mt} Nt={Nt}  grid=({grid.x},{grid.y}) ===")
    print(f"  {'variant':40} {'norm-only':>11} {'full+resh':>11} {'reshard':>9}  {'PCC':>9}")
    print(f"  {'interleaved (baseline, 1core@Mt=1)':40} {base_us:9.1f}u {'':>11} {'':>9}  {_pcc(base_out, ref):.6f}")

    # ---- width-sharded candidates: gx cores in a single row, block_w = Nt/gx ----
    gx_opts = [gx_only] if gx_only else [g for g in _divisors_leq(Nt, min(grid.x, Nt)) if g >= 4][:3]
    for gx in gx_opts:
        block_w = Nt // gx
        shard_mc = ttnn.create_sharded_memory_config(
            shape=(1, 1, S, H),
            core_grid=ttnn.CoreGrid(y=1, x=gx),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        sbw_opts = [4] if gx_only else sorted({block_w, 8, 4, 2, 1}, reverse=True)
        for sbw in sbw_opts:
            if block_w % sbw:
                continue
            try:
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(gx, 1),
                    subblock_w=sbw,
                    block_h=Mt,
                    block_w=block_w,
                    inplace=False,
                )
            except Exception as e:
                print(f"  gx={gx} bw={block_w} sbw={sbw:<2}  cfg FAIL {str(e)[:50]}")
                continue

            # pre-sharded input for the norm-only timing
            xs_persist = ttnn.interleaved_to_sharded(xt, shard_mc)

            def norm_only(pc=pc, smc=shard_mc):
                return ttnn.rms_norm(
                    xs_persist, epsilon=EPS, weight=wt, program_config=pc, memory_config=smc, compute_kernel_config=ckc
                )

            def full(pc=pc, smc=shard_mc):
                xs = ttnn.interleaved_to_sharded(xt, smc)
                ys = ttnn.rms_norm(
                    xs, epsilon=EPS, weight=wt, program_config=pc, memory_config=smc, compute_kernel_config=ckc
                )
                ttnn.deallocate(xs)
                yi = ttnn.sharded_to_interleaved(ys, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(ys)
                return yi

            def reshard_only(smc=shard_mc):
                xs = ttnn.interleaved_to_sharded(xt, smc)
                yi = ttnn.sharded_to_interleaved(xs, ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(xs)
                return yi

            try:
                out = ttnn.to_torch(full()).reshape(1, S, H)
                no_us = _bench(device, norm_only)
                fl_us = _bench(device, full)
                rs_us = _bench(device, reshard_only)
                tag = f"width gx={gx} bw={block_w} sbw={sbw}"
                print(
                    f"  {tag:40} {no_us:9.1f}u {fl_us:9.1f}u {rs_us:7.1f}u  {_pcc(out, ref):.6f}"
                    f"   ({base_us/fl_us:.2f}x full, {base_us/no_us:.2f}x norm)"
                )
            except Exception as e:
                print(f"  gx={gx} bw={block_w} sbw={sbw:<2}  run FAIL {str(e)[:60]}")
            finally:
                ttnn.deallocate(xs_persist)
