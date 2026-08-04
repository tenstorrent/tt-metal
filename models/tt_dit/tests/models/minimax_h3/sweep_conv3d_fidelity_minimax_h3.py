#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""conv3d **compute-kernel-config** sweep for the MiniMax-H3 encoder.

The blocking sweep (``sweep_conv3d_minimax_h3.py``) tuned *how* the conv is decomposed but
never touched *how the FPU is configured*. ``MiniMaxH3CausalConv3d`` has always used

    math_fidelity     = HiFi2          (2 FPU passes per bf16 multiply)
    fp32_dest_acc_en  = True           (fp32 accumulate: halves the dest-register tile budget)
    packer_l1_acc     = False

which was chosen when the encoder ran in **float32**. The encoder is bf16 now (amendment 47)
and those two settings are each worth up to 2x on the FPU, so they are the largest untested
lever left on the conv path -- which is ~80 % of encoder device time.

Method is the one that worked for SDPA (amendment 54): time the **op**, via trace so there is
no host dispatch in the measurement, and check PCC against torch for the same shape. Whole-
model wall clock cannot resolve a change this size (amendment 41).

    pytest models/tt_dit/tests/models/minimax_h3/sweep_conv3d_fidelity_minimax_h3.py -s
"""

from __future__ import annotations

import time

import pytest
import torch

import ttnn

from ....utils.conv3d import aligned_channels
from ..wan2_2.bruteforce_conv3d_sweep import TRACE_REGION_SIZE, _trace_us

# (name, C_in, C_out, stride, T, H, W, blocking) -- padded dims as handed to conv3d, and the
# swept blocking from _H3_ENCODER_BLOCKINGS. Ordered most-to-least compute, so a partial run
# still covers what dominates: b0_res x4 and b1_res x4 are ~85 % of encoder conv FLOPs.
LAYERS = [
    ("b0_res_128_128", 128, 128, (1, 1, 1), 19, 258, 258, (64, 128, 1, 16, 2)),
    ("b1_res1_256_256", 256, 256, (1, 1, 1), 19, 130, 130, (64, 128, 1, 16, 2)),
    ("b1_res0_128_256", 128, 256, (1, 1, 1), 19, 130, 130, (32, 256, 3, 16, 2)),
    ("b0_down_128_128", 128, 128, (1, 2, 2), 19, 257, 257, (64, 128, 1, 16, 2)),
    ("conv_in_32_128", 32, 128, (1, 1, 1), 19, 258, 258, (32, 128, 3, 2, 16)),
    ("b1_down_256_256", 256, 256, (2, 2, 2), 19, 129, 129, (64, 128, 1, 16, 2)),
    ("b2_res_256_256", 256, 256, (1, 1, 1), 11, 66, 66, (64, 128, 1, 16, 2)),
]

# (label, math_fidelity, fp32_dest_acc_en, packer_l1_acc)
CONFIGS = [
    ("HiFi2/fp32acc  (shipping)", ttnn.MathFidelity.HiFi2, True, False),
    ("HiFi2/bf16acc", ttnn.MathFidelity.HiFi2, False, False),
    ("HiFi2/bf16acc/packerL1", ttnn.MathFidelity.HiFi2, False, True),
    ("LoFi /bf16acc", ttnn.MathFidelity.LoFi, False, False),
    ("LoFi /bf16acc/packerL1", ttnn.MathFidelity.LoFi, False, True),
    ("LoFi /fp32acc", ttnn.MathFidelity.LoFi, True, False),
]

TRACE_ITERS = 8
KERNEL = (3, 3, 3)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(1, 1), {"trace_region_size": TRACE_REGION_SIZE, "l1_small_size": 65536}]],
    ids=["bh_1x1"],
    indirect=True,
)
def test_sweep_fidelity(mesh_device):
    device = mesh_device
    grid = device.compute_with_storage_grid_size()
    rows = []

    for name, c_in, c_out, stride, t, h, w, blocking in LAYERS:
        padded_cin = aligned_channels(c_in)
        cin_blk, cout_blk, t_blk, h_blk, w_blk = blocking
        torch.manual_seed(42)

        x = torch.randn(1, t, h, w, padded_cin, dtype=torch.float32) * 0.5
        weight = torch.randn(c_out, padded_cin, *KERNEL, dtype=torch.float32) * 0.05
        weight[:, c_in:] = 0.0
        bias = torch.randn(1, c_out, dtype=torch.float32) * 0.05

        # torch reference, fp32, in NCTHW
        ref = torch.nn.functional.conv3d(x.permute(0, 4, 1, 2, 3), weight, bias.flatten(), stride=stride).permute(
            0, 2, 3, 4, 1
        )

        tt_input = ttnn.from_torch(
            x, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt_w = ttnn.from_torch(weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
        tt_weight = ttnn.experimental.prepare_conv3d_weights(weight_tensor=tt_w, C_in_block=cin_blk, device=device)
        tt_bias = ttnn.from_torch(bias, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=0)
        cfg = ttnn.Conv3dConfig(
            weights_dtype=ttnn.bfloat16,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            T_out_block=t_blk,
            H_out_block=h_blk,
            W_out_block=w_blk,
            C_out_block=cout_blk,
            C_in_block=cin_blk,
            compute_with_storage_grid_size=grid,
        )

        print(f"\n=== {name}  C {c_in}->{c_out}  T{t} H{h} W{w} stride{stride} ===", flush=True)
        base = None
        for label, fidelity, fp32_acc, packer in CONFIGS:
            ckc = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_acc,
                packer_l1_acc=packer,
            )
            args = (device, tt_input, tt_weight, tt_bias, cfg, c_out, KERNEL, stride, (0, 0, 0), ckc)
            t0 = time.time()
            try:
                out = ttnn.experimental.conv3d(
                    input_tensor=tt_input,
                    weight_tensor=tt_weight,
                    bias_tensor=tt_bias,
                    config=cfg,
                    output_channels=c_out,
                    kernel_size=KERNEL,
                    stride=stride,
                    padding=(0, 0, 0),
                    padding_mode="zeros",
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=ckc,
                )
                pcc = _pcc(ttnn.to_torch(out)[..., :c_out].float(), ref)
                ttnn.deallocate(out)
                us = _trace_us(args, TRACE_ITERS, 5)
            except Exception as exc:  # a config the op rejects is a result, not a failure
                print(f"  {label:28s}  FAILED  {str(exc)[:90]}", flush=True)
                continue
            if base is None:
                base = us
            rows.append((name, label, us, pcc, base / us))
            print(
                f"  {label:28s}  {us:9.1f} us  pcc {pcc:.6f}  {base / us:5.2f}x" f"   [{time.time() - t0:.0f}s]",
                flush=True,
            )

        for tensor in (tt_input, tt_w, tt_weight, tt_bias):
            ttnn.deallocate(tensor)

    print("\n\n=== SUMMARY (speedup vs shipping HiFi2/fp32acc) ===", flush=True)
    by_config: dict[str, list[tuple[str, float, float]]] = {}
    for name, label, us, pcc, speed in rows:
        by_config.setdefault(label, []).append((name, us, pcc))
    shipping = {n: us for n, lbl, us, _, _ in rows if lbl == CONFIGS[0][0]}
    for label, entries in by_config.items():
        total = sum(us for _, us, _ in entries)
        base_total = sum(shipping.get(n, 0.0) for n, _, _ in entries)
        worst_pcc = min(p for _, _, p in entries)
        print(f"  {label:28s}  sum {total:9.1f} us  {base_total / total:5.2f}x  worst pcc {worst_pcc:.6f}", flush=True)
