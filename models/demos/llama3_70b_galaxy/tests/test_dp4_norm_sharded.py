# Residual add + RMSNorm over full 8192-wide rows (the DP4 x TP8 sequence-parallel residual, R = M/8 rows per chip) on
# one Blackhole chip: today's interleaved add + 5-op norm vs one block-sharded ttnn.rms_norm(residual_input_tensor=...)
# (the interleaved rms_norm overflows L1 at this width). Time and accuracy vs torch.
#   pytest models/demos/llama3_70b_galaxy/tests/test_dp4_norm_sharded.py -s       env: NS_R (512,2048)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H = 8192
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _time(device, fn, n=20):
    o = fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn()
        for z in out if isinstance(out, (list, tuple)) else [out]:
            ttnn.deallocate(z)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / n * 1e3, o


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dp4_norm_sharded(device):
    grid = device.compute_with_storage_grid_size()
    for R in [int(r) for r in (os.environ.get("NS_R") or "512,2048").split(",")]:
        torch.manual_seed(0)
        x = torch.randn(1, 1, R, H)
        x[..., 17] *= 100
        r = torch.randn(1, 1, R, H)
        g = torch.rand(H) + 0.5
        s = x + r
        gold = s / torch.sqrt(s.pow(2).mean(-1, keepdim=True) + 1e-5) * g
        tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
        tr = ttnn.from_torch(r, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=DRAM)
        ones = ttnn.from_torch(
            torch.full((1, 1, H, 1), 1.0 / H), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        tg = ttnn.from_torch(
            g.reshape(1, 1, H // 32, 32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
        )
        ck = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )

        def today():
            s1 = ttnn.add(tx, tr, dtype=ttnn.bfloat16, memory_config=DRAM)
            sq = ttnn.mul(s1, s1, dtype=ttnn.bfloat16, memory_config=DRAM)
            ms = ttnn.matmul(sq, ones, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=DRAM)
            ttnn.deallocate(sq)
            me = ttnn.add(ms, 1e-5, memory_config=DRAM)
            ttnn.deallocate(ms)
            rr = ttnn.rsqrt(me, memory_config=DRAM)
            ttnn.deallocate(me)
            o = ttnn.mul(s1, rr, dtype=ttnn.bfloat8_b, memory_config=DRAM)  # gamma folded into the next weight
            ttnn.deallocate(rr)
            return [s1, o]

        ms_today, outs = _time(device, today)
        gold_nog = s / torch.sqrt(s.pow(2).mean(-1, keepdim=True) + 1e-5)
        logger.info(f"NORMSH R={R} today add+5op: {ms_today:.3f} ms pcc {_pcc(gold_nog, ttnn.to_torch(outs[1])):.5f}")

        Rt, Wt = R // 32, H // 32
        for gy, gx in ((8, 8), (4, 8), (8, 4), (2, 8), (grid.y, grid.x)):
            if Rt % gy or Wt % gx:
                continue
            block_h, block_w = Rt // gy, Wt // gx
            shard = ttnn.create_sharded_memory_config(
                shape=(R // gy, H // gx),
                core_grid=ttnn.CoreGrid(y=gy, x=gx),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            sw = max(d for d in (1, 2, 4, 8) if block_w % d == 0 and d <= 4)
            pcfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                subblock_w=sw,
                block_h=block_h,
                block_w=block_w,
                inplace=False,
            )
            try:
                xs = ttnn.to_memory_config(tx, shard)
                rs = ttnn.to_memory_config(ttnn.typecast(tr, ttnn.bfloat16), shard)

                def fused():
                    return ttnn.rms_norm(
                        xs,
                        residual_input_tensor=rs,
                        epsilon=1e-5,
                        weight=tg,
                        program_config=pcfg,
                        memory_config=shard,
                        compute_kernel_config=ck,
                    )

                ms_f, o = _time(device, fused)
                got = ttnn.to_torch(ttnn.to_memory_config(o, DRAM)).float()
                # interleaved -> sharded -> interleaved conversions the layer would need around it
                ms_in, _ = _time(device, lambda: ttnn.to_memory_config(tx, shard))
                ms_out, _ = _time(device, lambda: ttnn.to_memory_config(o, DRAM))
                logger.info(
                    f"NORMSH R={R} sharded {gy}x{gx} fused add+norm: {ms_f:.3f} ms pcc {_pcc(gold, got):.5f} "
                    f"(to-sharded {ms_in:.3f}, to-dram {ms_out:.3f})"
                )
                ttnn.deallocate(xs), ttnn.deallocate(rs)
            except Exception as e:
                logger.warning(f"NORMSH R={R} sharded {gy}x{gx} ERROR {str(e)[:250]}")
