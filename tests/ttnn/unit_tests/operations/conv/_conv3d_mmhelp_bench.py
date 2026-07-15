# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# mm_help profiling harness — isolated single conv3d for the main-vs-mm_help5 A/B.
# Underscore-prefixed so ambient `pytest tests/...` does NOT collect it; run EXPLICITLY
# under Tracy on each branch-build (A/B is cross-BUILD, not an in-build mode toggle):
#
#   C3_CIN=32 C3_COUT=384 C3_KD=3 C3_KH=3 C3_KW=3 C3_T=3 C3_H=23 C3_W=20 C3_GROUPS=1 \
#   C3_CIN_BLOCK=32 C3_COUT_BLOCK=128 C3_T_BLK=1 C3_H_BLK=8 C3_W_BLK=4 \
#   python -m tracy -r -p -v -m pytest tests/ttnn/unit_tests/operations/conv/_conv3d_mmhelp_bench.py
#
# Emits ttnn.experimental.conv3d twice (cold + warm) so the extractor takes the warm
# Conv3dDeviceOperation. torch-vs-ttnn PCC is checked every run (a mis-wired trace fails
# loudly). The why-fields come from the Tracy CSV ATTRIBUTES column with no perturbation.
#
# Env: shape  C3_CIN,C3_COUT,C3_KD,C3_KH,C3_KW,C3_T,C3_H,C3_W,C3_GROUPS
#      blocking C3_CIN_BLOCK,C3_COUT_BLOCK,C3_T_BLK,C3_H_BLK,C3_W_BLK  (C_in_block<=0 -> full C_in)
#      compute  C3_FIDELITY (HiFi2), C3_FP32_ACCUM (false), C3_L1_ACC (false),
#               C3_DTYPE (bfloat16), C3_WEIGHTS_DTYPE (bfloat16),
#               C3_PAD ("0,pH,pW" default derived as same-pad on H/W, 0 on D),
#               C3_PAD_MODE (zeros), C3_STRIDE ("1,1,1")
import os
import pytest
import torch
import torch.nn as nn
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import check_with_pcc

_DTYPE = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "float32": ttnn.float32}
_FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
ALIGNMENT = 32


def _e(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


def _bool(v):
    return str(v).lower() in ("1", "true", "yes")


def _triple(v):
    return tuple(int(x) for x in v.split(","))


def _out_size(in_size, pad, stride, k, dilation=1):
    effective_k = (dilation * (k - 1)) + 1
    return (in_size + 2 * pad - effective_k) // stride + 1


CFG = dict(
    C_in=_e("C3_CIN", 32, int),
    C_out=_e("C3_COUT", 384, int),
    kD=_e("C3_KD", 3, int),
    kH=_e("C3_KH", 3, int),
    kW=_e("C3_KW", 3, int),
    T=_e("C3_T", 3, int),
    H=_e("C3_H", 23, int),
    W=_e("C3_W", 20, int),
    groups=_e("C3_GROUPS", 1, int),
    C_in_block=_e("C3_CIN_BLOCK", 32, int),
    C_out_block=_e("C3_COUT_BLOCK", 128, int),
    T_out_block=_e("C3_T_BLK", 1, int),
    H_out_block=_e("C3_H_BLK", 8, int),
    W_out_block=_e("C3_W_BLK", 4, int),
    fidelity=_e("C3_FIDELITY", ttnn.MathFidelity.HiFi2, lambda v: _FID[v]),
    fp32_accum=_e("C3_FP32_ACCUM", False, _bool),
    l1_acc=_e("C3_L1_ACC", False, _bool),
    dtype=_e("C3_DTYPE", ttnn.bfloat16, lambda v: _DTYPE[v]),
    weights_dtype=_e("C3_WEIGHTS_DTYPE", ttnn.bfloat16, lambda v: _DTYPE[v]),
    stride=_e("C3_STRIDE", (1, 1, 1), _triple),
    pad=_e("C3_PAD", None, _triple),
    pad_mode=_e("C3_PAD_MODE", "zeros", str),
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv3d_mmhelp_bench(device):
    c = CFG
    torch.manual_seed(42)
    kernel_size = (c["kD"], c["kH"], c["kW"])
    stride = c["stride"]
    # default: same-pad on H/W (kH//2, kW//2), no pad on D (matches conv3d test convention (0,1,1))
    padding = c["pad"] if c["pad"] is not None else (0, c["kH"] // 2, c["kW"] // 2)
    C_in, C_out, groups = c["C_in"], c["C_out"], c["groups"]
    N, T, H, W = 1, c["T"], c["H"], c["W"]
    C_in_block = c["C_in_block"] if c["C_in_block"] > 0 else C_in

    grid = device.compute_with_storage_grid_size()
    logger.info(
        f"conv3d_bench Cin={C_in} Cout={C_out} k={kernel_size} THW={T}x{H}x{W} g={groups} "
        f"blk=(Cin{C_in_block},Cout{c['C_out_block']},T{c['T_out_block']},H{c['H_out_block']},W{c['W_out_block']}) "
        f"pad={padding} grid={grid.x}x{grid.y}"
    )

    input_tensor = torch.randn(N, C_in, T, H, W, dtype=torch.float32)
    conv3d_module = nn.Conv3d(
        C_in, C_out, groups=groups, kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=(1, 1, 1), bias=True, padding_mode=c["pad_mode"],
    )
    gt_output = conv3d_module(input_tensor)

    D_out = _out_size(T, padding[0], stride[0], kernel_size[0])
    H_out = _out_size(H, padding[1], stride[1], kernel_size[1])
    W_out = _out_size(W, padding[2], stride[2], kernel_size[2])

    # TTNN input: NCDHW -> NDHWC, align C to 32
    tt_in_torch = input_tensor.permute(0, 2, 3, 4, 1)
    if C_in % ALIGNMENT != 0:
        tt_in_torch = torch.nn.functional.pad(tt_in_torch, (0, ALIGNMENT - C_in % ALIGNMENT))
    tt_input = ttnn.from_torch(tt_in_torch, device=device, dtype=c["dtype"], layout=ttnn.ROW_MAJOR_LAYOUT)

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=c["fidelity"],
        math_approx_mode=False,
        fp32_dest_acc_en=c["fp32_accum"],
        packer_l1_acc=c["l1_acc"],
    )

    config = ttnn.Conv3dConfig(
        weights_dtype=c["weights_dtype"],
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=c["T_out_block"],
        W_out_block=c["W_out_block"],
        H_out_block=c["H_out_block"],
        C_out_block=c["C_out_block"],
        C_in_block=C_in_block,
        dilation=(1, 1, 1),
        compute_with_storage_grid_size=(grid.x, grid.y),
    )

    tt_weight = ttnn.from_torch(conv3d_module.weight.data, dtype=c["weights_dtype"], pad_value=0)
    tt_weight = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=tt_weight, groups=groups, C_in_block=config.C_in_block, alignment=ALIGNMENT, device=device
    )
    tt_bias = ttnn.from_torch(
        conv3d_module.bias.data.reshape(1, -1), device=device, dtype=c["weights_dtype"],
        layout=ttnn.TILE_LAYOUT, pad_value=0,
    )

    tt_output = None
    for _ in range(2):  # cold + warm; extractor takes the warm Conv3dDeviceOperation
        tt_output = ttnn.experimental.conv3d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            device=device,
            bias_tensor=tt_bias,
            dtype=c["dtype"],
            output_channels=C_out,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            dilation=(1, 1, 1),
            padding_mode=c["pad_mode"],
            config=config,
            compute_kernel_config=kernel_config,
        )

    out = ttnn.to_torch(tt_output, device=device, dtype=torch.float32)
    out = out.reshape(N, D_out, H_out, W_out, C_out).permute(0, 4, 1, 2, 3)
    assert out.shape == gt_output.shape, f"{out.shape} != {gt_output.shape}"
    pcc_passed, pcc_message = check_with_pcc(gt_output, out, pcc=0.999)
    logger.info(f"conv3d torch vs ttnn: {pcc_message}")
    assert pcc_passed, pcc_message
