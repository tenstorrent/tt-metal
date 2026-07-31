# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Step 1 validation for the UnpackToDestEn tilize fix — the REAL (fused) conv, end-to-end, with a PCC check.

Background: the Quasar regular tilize (tilize_block) faults with ERROR_TRISC1 0x19 because its per-tile MATH
A2D datacopy (MOVA2D) advances the FPU dest-dvalid ring but the semaphore-scheme section_done never frees it
(root-caused via LLK analysis). The fix (tt-metal #49445): route the tilize unpacker straight into DEST
(UNP_DEST) so the MOVA2D is bypassed — enabled by TT_METAL_QSR_TILIZE_UNPACK_TO_DEST=1, which the factory turns
into the QSR_TILIZE_UNPACK_TO_DEST compute define for any Quasar tilize_block path.

This test runs the PLAIN FUSED conv (NO split-program flags) — reader gather → in-kernel tilize → matmul → out
— on the L1 path (pre-sharded L1 input, so conv2d avoids DRAM slicing / the unported slice_write). Unlike the
split Program-A probe (which only checked completion on a throwaway tilized buffer), this checks the ACTUAL
conv output against a torch golden, so it validates BOTH that the fix clears the 0x19 AND that the
unpack-to-dest tilize produces CORRECT data. Same K=16-tile stem-like shape, shrunk to fit L1 on the emulator;
act_block_h_override forces >=2 height blocks so the multi-block tilize (which is what faulted) is exercised.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_QSR_TILIZE_UNPACK_TO_DEST=1 \
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_unpack_to_dest.py
"""

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.99


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_unpack_to_dest(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # Stem-like conv (same K = 32*4*4 = 16 tiles as the folded stem), shrunk to fit L1 on the emulator.
    batch_size = 1
    in_channels = 32
    out_channels = 64
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = (0, 0)
    out_h, out_w = 16, 32  # 16x32 = 512 sticks = 16 out-height tiles
    input_height = out_h + kernel_size[0] - 1  # s1/p0 => in = out + (k-1) = 19
    input_width = out_w + kernel_size[1] - 1  # 35

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    # Pure conv (no bias, no activation) — isolates the tilize+matmul correctness that the fix must preserve.
    torch_golden = torch.nn.functional.conv2d(torch_input_nchw, torch_weight, bias=None, stride=stride, padding=padding)

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

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        full_inner_dim=True,  # single K-block (in0_num_blocks_w == 1) — the fused kernel's tilize->matmul path
        act_block_h_override=128,  # 4-tile blocks => >=2 height blocks: exercises the multi-block tilize that faulted
        reshard_if_not_optimal=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    # Enable the UnpackToDestEn tilize fix; NO split-program flags (this is the plain fused conv). Set before the
    # op (factory reads it via std::getenv at program creation) and restore after so it doesn't leak.
    prev = os.environ.get("TT_METAL_QSR_TILIZE_UNPACK_TO_DEST")
    os.environ["TT_METAL_QSR_TILIZE_UNPACK_TO_DEST"] = "1"
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
        if prev is None:
            del os.environ["TT_METAL_QSR_TILIZE_UNPACK_TO_DEST"]
        else:
            os.environ["TT_METAL_QSR_TILIZE_UNPACK_TO_DEST"] = prev

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, oh, ow, tt_out.shape[-1])[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))
    print(f"UnpackToDestEn fused conv completed — no 0x19. out shape={tuple(tt_out.shape)}")
    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)
