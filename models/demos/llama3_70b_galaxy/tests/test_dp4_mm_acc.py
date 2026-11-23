# DP4 x TP8 matmul speed / accuracy by accumulation and block config, one Blackhole chip:
#   FF13+SwiGLU [M, 8192] bf8 @ [8192, 7168] bf4 LoFi (out bf8), QKV [M, 8192] bf8 @ [8192, 1280] bf8 LoFi (out bf16)
# fp32 dest accumulation (subblock 1x4) vs bf16 dest (1x8), block sizes. PCC vs torch on the same quantized inputs.
#   pytest models/demos/llama3_70b_galaxy/tests/test_dp4_mm_acc.py -s       env: MA_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

H = 8192
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    n = min(a.numel(), 4_000_000)
    return torch.corrcoef(torch.stack([a[:n], b[:n]]))[0, 1].item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dp4_mm_acc(device):
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    w13 = torch.randn(H, 7168) * 0.02
    wq = torch.randn(H, 1280) * 0.02
    tw13 = ttnn.from_torch(
        prepare_for_fused_swiglu(w13, ndev=1, gate_is_first=True).reshape(1, 1, H, 7168),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=DRAM,
    )
    twq = ttnn.from_torch(
        wq.reshape(1, 1, H, 1280), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=DRAM
    )
    w13q = ttnn.to_torch(ttnn.from_torch(w13, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    wqq = ttnn.to_torch(ttnn.from_torch(wq, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
    for M in [int(m) for m in (os.environ.get("MA_M") or "4096,16384").split(",")]:
        x = torch.randn(1, 1, M, H)
        tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=DRAM)
        xq = ttnn.to_torch(tx).float()[0, 0]
        gold13 = torch.nn.functional.silu(xq @ w13q[:, :3584]) * (xq @ w13q[:, 3584:])
        goldq = xq @ wqq
        for name, tw, swiglu, odt, gold in (
            ("ff13", tw13, True, ttnn.bfloat8_b, gold13),
            ("qkv", twq, False, ttnn.bfloat16, goldq),
        ):
            for fp32 in [x == "1" for x in (os.environ.get("MA_FP32") or "1,0").split(",")]:
                ck = ttnn.init_device_compute_kernel_config(
                    device.arch(),
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=fp32,
                    packer_l1_acc=True,
                )
                blocks = [
                    tuple(int(v) for v in b.split("x"))
                    for b in (os.environ.get("MA_BLOCKS") or "8x8x8x1x8,8x16x8x1x8,16x8x8x1x8,8x8x16x1x8").split(",")
                ]
                for mb, kb, nb, sbh, sbw in blocks:
                    if fp32 and sbh * sbw > 4:
                        sbh, sbw = 1, 4
                    if nb % sbw or mb % sbh:
                        continue
                    cfg = ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        subblock_h=sbh,
                        subblock_w=sbw,
                        compute_with_storage_grid_size=grid,
                    )
                    f = lambda: ttnn.experimental.minimal_matmul(
                        input_tensor=tx,
                        weight_tensor=tw,
                        config=cfg,
                        compute_kernel_config=ck,
                        memory_config=DRAM,
                        dtype=odt,
                        fuse_swiglu=swiglu,
                    )
                    try:
                        o = f()
                        p = _pcc(gold, ttnn.to_torch(o).float()[0, 0])
                        ttnn.deallocate(o)
                        ttnn.synchronize_device(device)
                        t0 = time.perf_counter()
                        for _ in range(10):
                            ttnn.deallocate(f())
                        ttnn.synchronize_device(device)
                        ms = (time.perf_counter() - t0) / 10 * 1e3
                        logger.info(
                            f"MMACC M={M} {name} fp32={fp32} mb{mb} kb{kb} nb{nb} sb{sbh}x{sbw}: {ms:.3f} ms pcc {p:.5f}"
                        )
                    except Exception as e:
                        logger.warning(f"MMACC M={M} {name} fp32={fp32} mb{mb} kb{kb} nb{nb} ERROR {str(e)[:200]}")
        ttnn.deallocate(tx)
