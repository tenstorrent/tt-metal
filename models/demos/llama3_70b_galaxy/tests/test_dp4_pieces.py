# Single-chip checks of the DP4 x TP8 prefill building blocks (tt/prefill_dp4tp8.py) against torch:
#   norm_col1 / norm_col32: x / rms(x) via sum-of-squares matmul against a ones column (width 1 vs 32 + slice)
#   swiglu: minimal_matmul(fuse_swiglu=True) on prepare_for_fused_swiglu(gate_is_first=True) weights vs silu(xW1)*(xW3)
#   pytest models/demos/llama3_70b_galaxy/tests/test_dp4_pieces.py -s
import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

DRAM = ttnn.DRAM_MEMORY_CONFIG


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dp4_pieces(device):
    torch.manual_seed(0)
    R, H, F = 512, 8192, 3584
    x = torch.randn(1, 1, R, H)
    x[..., 17] *= 300  # outlier channel like the Llama residual
    gold = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ck = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
    )
    for width in (1, 32):
        ones = ttnn.from_torch(
            torch.full((1, 1, H, width), 1.0 / H), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        sq = ttnn.mul(tx, tx, dtype=ttnn.bfloat16)
        ms = ttnn.matmul(sq, ones, compute_kernel_config=ck, dtype=ttnn.bfloat16)
        if width == 32:
            ms = ms[:, :, :, :1]
        r = ttnn.rsqrt(ttnn.add(ms, 1e-5))
        logger.info(
            f"PIECES norm_col{width}: ms shape {tuple(ms.shape)} padded {tuple(ms.padded_shape)} r {tuple(r.shape)}"
        )
        out = ttnn.to_torch(ttnn.mul(tx, r, dtype=ttnn.bfloat8_b)).float()
        logger.info(f"PIECES norm_col{width}: pcc {_pcc(gold, out):.5f} max_abs {(gold - out).abs().max():.3f}")

    w1 = torch.randn(H, F) * 0.02
    w3 = torch.randn(H, F) * 0.02
    import os

    R = int(os.environ.get("PIECES_M") or R)
    xs = torch.randn(1, 1, R, H)
    gold = torch.nn.functional.silu(xs @ w1) * (xs @ w3)
    txs = ttnn.from_torch(xs, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    full = device.compute_with_storage_grid_size()
    logger.info(f"PIECES device grid {full.x}x{full.y}")
    for gx, gy in ((full.x, full.y), (12, 10), (11, 10), (7, 10), (8, 8)):
        grid = ttnn.CoreCoord(gx, gy)
        for gate_first, label in ((True, "gate_first"), (False, "up_first_input")):
            if not gate_first:
                continue
            packed = torch.cat([w1, w3], dim=-1)
            wp = prepare_for_fused_swiglu(packed, ndev=1, gate_is_first=gate_first)
            tw = ttnn.from_torch(
                wp.reshape(1, 1, H, 2 * F), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b
            )
            for sbw in (4,):
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=sbw,
                    compute_with_storage_grid_size=grid,
                )
                for fp32 in (True,):
                    ckm = ttnn.init_device_compute_kernel_config(
                        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=fp32, packer_l1_acc=True
                    )
                    try:
                        o = ttnn.experimental.minimal_matmul(
                            input_tensor=txs,
                            weight_tensor=tw,
                            config=cfg,
                            compute_kernel_config=ckm,
                            memory_config=DRAM,
                            dtype=ttnn.bfloat8_b,
                            fuse_swiglu=True,
                        )
                        ot = ttnn.to_torch(o).float().reshape(R, -1)
                        logger.info(
                            f"PIECES swiglu grid {gx}x{gy} {label} sbw={sbw} fp32={fp32}: out {tuple(o.shape)} pcc {_pcc(gold.reshape(R, -1), ot):.5f}"
                        )
                    except Exception as e:
                        logger.warning(f"PIECES swiglu {label} sbw={sbw} fp32={fp32} ERROR {str(e)[:200]}")
