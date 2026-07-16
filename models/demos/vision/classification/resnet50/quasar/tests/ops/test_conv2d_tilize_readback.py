# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
TILIZE ISOLATION (Option 1) — read back the tilized activation and compare to a host golden, with NO matmul.

Why: the fused UnpackToDestEn conv (test_conv2d_unpack_to_dest.py) runs end-to-end but PCC ~= 0.001. The
per-block DPRINT probes (TZINIT / TZL1) proved the batched tilize-to-DEST INDEX MATH is exactly correct
(l1idx = y*FULL_CT*fpe, sst=1, srcz=2 dstz=64 soff0=8 — all match the LLK reference test). So the scramble is
NOT the index computation. It must be one of: (a) the UNP_DEST tilize MOP data production, (b) the packer
reading tile j out of DEST, or (c) the MATMUL consuming act_tilized. This test splits (a)+(b) from (c).

How: run ONLY the gather+tilize half in its own program (TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 selects
conv_tilize_only_metal2.cpp; TT_METAL_QSR_TILIZE_UNPACK_TO_DEST=1 injects the QSR_TILIZE_UNPACK_TO_DEST define
so it runs the SAME batched UNP_DEST tilize as the fused kernel). The factory binds OUT borrowed_from the
output tensor, so the op's OUTPUT IS the tilized activation. We read it back (to_torch untilizes it to
row-major) and compare against the host activation matrix.

Trick for an unambiguous golden: use a 1x1 / s1 / p0 conv, so the im2col gather is the IDENTITY — the tilize
input matrix A[m, k] is just the flattened NHWC input (m = output position = input position, k = channel). No
im2col reconstruction to get wrong. in_channels = 128 => K = 4 tiles => block_width = 4 and FULL_CT = 4, the
IDENTICAL tilize MOP config the failing 4x4 (in_ch=32) test hits. act_block_h_override forces >= 2 height
blocks so the multi-block tilize is exercised.

Verdict:
  * PASS (tilized readback == input)  -> the UNP_DEST tilize + pack are CORRECT; the fused PCC~0 is the MATMUL
                                         consuming act_tilized (or its CB/tensor_shape config), NOT the tilize.
  * FAIL (tilized readback scrambled) -> the UNP_DEST tilize MOP / pack-from-DEST is broken on-target -> LLK
                                         (tt-metal #49445); hand to /debug-kernel with the confirmed strides.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_QSR_TILIZE_UNPACK_TO_DEST=1 \
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_tilize_readback.py
"""

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_tilize_readback(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # 1x1 / s1 / p0 => im2col gather is the identity, so the tilize input matrix == the NHWC input (trivial
    # golden). in_channels = 128 => K = 4 tiles => block_width = 4 / FULL_CT = 4 (same MOP config as the failing
    # 4x4 in_ch=32 conv, which also has act_block_w = in_ch*kw/32 = 4).
    batch_size = 1
    in_channels = 128
    out_channels = 128  # unused by the tilize-only program (no matmul); kept == in_channels so nothing hinges on it
    kernel_size = (1, 1)
    stride = (1, 1)
    padding = (0, 0)
    out_h, out_w = 16, 32  # 16*32 = 512 sticks = 16 M-tiles; 1x1/s1/p0 => in_h/in_w == out_h/out_w
    input_height = out_h
    input_width = out_w

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    # 1x1 weight is required by the API for shape inference but the tilize-only program never binds/uses it.
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()

    # The tilize input matrix A [M, K]: for 1x1/s1/p0, M = nhw output positions, K = in_channels. This is exactly
    # the flattened NHWC input. After the device tilizes it and to_torch untilizes on readback, we must recover A.
    nhw = batch_size * input_height * input_width
    flat = torch.permute(torch_input_nchw, (0, 2, 3, 1)).reshape(1, 1, nhw, in_channels).contiguous()
    golden_A = flat.clone()  # [1, 1, nhw, in_channels] == [1, 1, 512, 128]

    # --- pre-shard the activation into L1 (height-sharded) so conv2d takes the L1 path (not DRAM slicing) ---
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

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        full_inner_dim=True,  # single K-block (in0_num_blocks_w == 1) — the split-program factory gate
        act_block_h_override=128,  # 4-tile height blocks => >= 2 height blocks: exercises the multi-block tilize
        reshard_if_not_optimal=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    # tilize-only program (conv_tilize_only_metal2.cpp) + the batched UNP_DEST tilize (QSR_TILIZE_UNPACK_TO_DEST).
    # Set both before the op (factory reads via std::getenv at program creation) and restore after so they don't
    # leak into other tests sharing the process.
    to_set = {
        "TT_METAL_QSR_CONV_SPLIT_PROGRAM": "1",
        "TT_METAL_QSR_TILIZE_UNPACK_TO_DEST": "1",
    }
    prev = {k: os.environ.get(k) for k in to_set}
    os.environ.update(to_set)
    try:
        out, [oh, ow], _wb = ttnn.experimental.quasar.conv2d(
            input_tensor=tt_input,
            weight_tensor=tt_weight,
            bias_tensor=None,
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
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # OUT is borrowed from the output tensor, so this IS the tilized activation. to_torch untilizes it to
    # row-major; flatten to [M, W] and compare the first K=in_channels columns to the host activation matrix A.
    tt_out = ttnn.to_torch(ttnn.from_device(out)).float()
    print(f"tilize-only readback: raw out shape={tuple(tt_out.shape)} (conv_out_hw=({oh},{ow}))")

    tt_flat = tt_out.reshape(-1, tt_out.shape[-1])  # [M_padded, W]
    m = golden_A.shape[2]
    k = golden_A.shape[3]
    assert tt_flat.shape[0] >= m, f"readback M {tt_flat.shape[0]} < expected {m}"
    assert tt_flat.shape[1] >= k, f"readback width {tt_flat.shape[1]} < expected K {k}"
    tt_A = tt_flat[:m, :k]
    golden_flat = golden_A.reshape(m, k)

    # Quick structural signal before the PCC: how many rows match vs are scrambled.
    row_match = torch.isclose(tt_A, golden_flat, atol=0.05).all(dim=1)
    print(
        f"tilized readback vs input matrix A[{m},{k}]: {int(row_match.sum())}/{m} rows match; "
        f"first mismatch row = {int((~row_match).nonzero()[0][0]) if (~row_match).any() else -1}"
    )
    assert_with_pcc(golden_flat, tt_A, pcc=PCC)
