# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Option B — end-to-end two-program split conv, with a PCC correctness gate.

The Quasar tilize 0x19 (per-tile MATH A2D datacopy never freeing the FPU dest-dvalid ring) is fixed by
UnpackToDestEn (TT_METAL_QSR_TILIZE_UNPACK_TO_DEST), but only in a tilize kernel WITHOUT an interleaved matmul
(the fused conv re-faults — dvalid-synced tilize + semaphore-synced matmul in one kernel). So the conv is split
into two Metal programs, orchestrated host-side in conv2d.cpp under TT_METAL_QSR_CONV_SPLIT_PROGRAM:
  - Program A: reader im2col gather + UnpackToDestEn tilize -> tilized activation tensor [M, K].
  - Program B: quasar matmul::linear(act_tilized, weights) -> conv output [M, N] (same GEMM the 1x1 path uses).

This runs the FULL split conv on the L1 path (pre-sharded L1 input, so no DRAM slicing / unported slice_write)
and checks the ACTUAL conv output against a torch golden — validating BOTH that the fix clears the 0x19 AND
that the unpack-to-dest tilize produced CORRECT data (fed through the matmul). Stem-like K=16-tile shape shrunk
to fit L1; act_block_h_override forces >=2 height blocks so the multi-block tilize (which faulted) is exercised.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_QSR_TILIZE_UNPACK_TO_DEST=1 \
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_split_program_e2e.py
"""

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99


def _run(mesh_device, *, with_bias_relu):
    device = mesh_device
    torch.manual_seed(0)

    batch_size = 1
    in_channels = 32  # groups(4) * C_aligned(8); K = 32*4*4 = 16 tiles (same as folded stem)
    out_channels = 64
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = (0, 0)
    out_h, out_w = 16, 32  # 16x32 = 512 sticks = 16 out-height tiles
    input_height = out_h + kernel_size[0] - 1  # 19
    input_width = out_w + kernel_size[1] - 1  # 35

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((out_channels,), dtype=torch.bfloat16).float() if with_bias_relu else None
    torch_golden = torch.nn.functional.conv2d(
        torch_input_nchw, torch_weight, bias=torch_bias, stride=stride, padding=padding
    )
    if with_bias_relu:
        torch_golden = torch.relu(torch_golden)

    # --- pre-shard the activation into L1 (height-sharded) so conv2d takes the L1 path (not DRAM slicing) ---
    nhw = batch_size * input_height * input_width
    flat = torch.permute(torch_input_nchw, (0, 2, 3, 1)).reshape(1, 1, nhw, in_channels).contiguous()
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = max(c for c in range(1, max_cores + 1) if nhw % c == 0)
    shard_h = nhw // num_cores
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, in_channels),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_input = tt_input.to(device, in_mem)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = (
        ttnn.from_torch(torch_bias.reshape(1, 1, 1, out_channels), dtype=ttnn.bfloat16) if with_bias_relu else None
    )

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        full_inner_dim=True,  # single K-block -> factory split_program_tilize_only eligibility + host split gate
        act_block_h_override=128,  # >=2 height blocks: exercises the multi-block tilize that faulted
        reshard_if_not_optimal=True,
        activation=(ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if with_bias_relu else None),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    # Two-program split: Program A tilize (UnpackToDestEn) + Program B matmul. Both flags set; no leak after.
    saved = {k: os.environ.get(k) for k in ("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "TT_METAL_QSR_TILIZE_UNPACK_TO_DEST")}
    os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = "1"
    os.environ["TT_METAL_QSR_TILIZE_UNPACK_TO_DEST"] = "1"
    try:
        out, [oh, ow], _wb = ttnn.experimental.quasar.conv2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=tt_bias,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=(1, 1),
            groups=1,
            device=device,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.bfloat16,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, oh, ow, tt_out.shape[-1])[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))
    print(f"split conv (Program A tilize + Program B matmul) completed. out shape={tuple(tt_out.shape)}")
    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_split_program_e2e_pure(mesh_device):
    # Primary gate: pure conv (no bias/relu) — isolates tilize+matmul correctness.
    _run(mesh_device, with_bias_relu=False)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_split_program_e2e_bias_relu(mesh_device):
    # Bonus: bias + RELU folded into Program B's matmul.
    _run(mesh_device, with_bias_relu=True)
