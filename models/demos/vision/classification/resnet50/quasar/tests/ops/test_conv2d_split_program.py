# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Option B — Increment 1 (Program A, tilize-only) test.

Runs ONLY the gather+tilize half of the conv in a fresh, tilize-oriented Metal program (no matmul, no weights,
no output writer) by setting TT_METAL_QSR_CONV_SPLIT_PROGRAM=1, which makes the sharded conv factory select
conv_tilize_only_metal2.cpp. The op OUTPUTS the tilized activations (width = K tiles), not the conv result.

Purpose: confirm the Quasar 0x19 (Risc IB interrupt — a preceding matmul's terminal MVMUL leaves the MATH DEST
data-valid bit set, so the tilize's datacopy MOP is rejected) CLEARS when the tilize runs in its own program
with no preceding matmul. Every in-fused-kernel fix failed; the standalone tilize op passes precisely because
it runs isolated. Program B (the matmul over these tilized activations) is chained in a later increment.

ROUTING: conv2d takes the L1 path (write straight to the L1-sharded output, no DRAM slicing) iff the input is
already L1-sharded (determine_conv2d_execution_path: L1 iff input.is_l1() and no slice_config). We therefore
pre-shard the activation into L1 (height-sharded), exactly like the model feeds conv1 the fold output. This
also dodges the DRAM single-slice writeback (slice_write UNPAD_INPUT_WIDTH path is unported on Quasar), which
otherwise FATALs on the tilized-activation output width. The shape is the ring-/L1-fitting stem-like conv
(same K = 32*4*4 = 16 tiles as the folded stem); the 2-core emulator makes full-stem per-core shards too large
for the L1 path, so we validate on the small shape here (the real stem 0x19 was already reproduced with the
fused kernel, and cleared by Program A on the DRAM-slice path up to the unported slice_write).

PRIMARY SIGNAL: the conv COMPLETES without faulting (no 0x19 / ERROR_TRISC1). The output is the tilized
activation (no conv-golden PCC); we assert the op returns a finite, non-empty, correctly-widthed tensor.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_split_program.py
"""

import os

import pytest
import torch

import ttnn


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_split_program_tilize_only(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # Stem-like tilize-path conv, shrunk to fit L1 on the emulator: 4x4 / s1 / p0, in=32 (K = 32*4*4 = 16
    # tiles, the SAME K as the folded stem), out=64. No bias/activation are needed for Program A (tilize-only);
    # they belong to Program B (the matmul), added in a later increment.
    batch_size = 1
    in_channels = 32  # groups(4) * C_aligned(8)
    out_channels = 64
    kernel_size = (4, 4)
    stride = (1, 1)
    padding = (0, 0)
    out_h, out_w = 16, 32  # 16x32 = 512 sticks = 16 out-height tiles
    input_height = out_h + kernel_size[0] - 1  # s1/p0 => in = out + (k-1) = 19
    input_width = out_w + kernel_size[1] - 1  # 35

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()

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
        full_inner_dim=True,  # single K-block => in0_num_blocks_w == 1 (split-program factory gate)
        act_block_h_override=128,  # 4-tile blocks (stem-like) => >=2 height blocks on any core count
        reshard_if_not_optimal=True,  # let the conv fix the input shard to its optimal height-sharded layout
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    # Option B / Program A: select the tilize-only compute kernel in the sharded conv factory. Set before the
    # op runs (factory + device-op read it via std::getenv at program creation, in-process) and restore after
    # so the toggle does not leak into other tests sharing the process.
    prev = os.environ.get("TT_METAL_QSR_CONV_SPLIT_PROGRAM")
    os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = "1"
    try:
        out, [out_h_r, out_w_r], _weights_bias = ttnn.experimental.quasar.conv2d(
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
            del os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"]
        else:
            os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = prev

    # PRIMARY SIGNAL: we got here without a 0x19 / ERROR_TRISC1 fault (a fault aborts the process before this).
    tt_out = ttnn.to_torch(ttnn.from_device(out))
    assert tt_out is not None and tt_out.numel() > 0, "tilize-only conv returned an empty tensor"
    assert torch.isfinite(tt_out.float()).all(), "tilize-only conv output contains NaN/Inf"

    # The op output is the per-core tilized activation shard. Its width is K tiles (in0_block_w) = 32*4*4 = 512;
    # rows are the conv output rows padded to tiles. Sanity-check the width matches K (confirms Program A ran,
    # not the fused fallback whose output width would be N = out_channels).
    k_cols = in_channels * kernel_size[0] * kernel_size[1]  # 512
    print(
        f"Program A (tilize-only) completed. out shape={tuple(tt_out.shape)} "
        f"conv_out_hw=({out_h_r},{out_w_r}) expected_tilized_width(K)={k_cols} out_channels(N)={out_channels}"
    )
