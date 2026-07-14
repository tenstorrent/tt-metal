# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
HYPOTHESIS-VALIDATION test for the Quasar conv tilize<->matmul 0x19 race.

Background: the fused conv compute kernel (conv_bmm_tilize_metal2.cpp) interleaves, per height block,
tilize(block) -> matmul(block), so the compute engine ping-pongs between the tilize datacopy pipeline
and the matmul pipeline once PER height block. That transition has a Quasar MATH<->PACK DEST-handshake
race (watcher 0x19 / ERROR_TRISC1). Option C (conv_bmm_split_tilize_metal2.cpp) tilizes ALL height
blocks first, then matmuls them, so the engine transitions tilize->matmul exactly ONCE.

Option C can't run the real stem: its full per-core tilized activation (3.67 MB) blows past the 1 MB
uint16 DFB ring. This test uses a SMALL height-sharded conv whose WHOLE tilized activation fits the ring
(so Option C can run) while still having MANY height blocks each with enough K to rotate DEST banks (so
the fused kernel still has a chance to hit the race).

Getting to the compute kernel on the emulator required threading three needles:
  1. L1 path, not DRAM: conv2d routes to the DRAM-slicing path unless the input is already in L1
     (determine_conv2d_execution_path: L1 iff input.is_l1() and no slice_config). The single-slice DRAM
     writeback goes through sharded_to_interleaved, which is UNPORTED on Quasar (legacy DataMovementKernel
     -> TT_FATAL). So we pre-shard the activation into L1 (height-sharded) exactly like the model feeds
     conv1 the fold output; reshard_if_not_optimal lets the conv fix the shard to its optimal layout.
  2. Single K-block: the Option C factory gate needs in0_num_blocks_w == 1, so conv_config.full_inner_dim
     = True (keep the whole reduction dim in one block, same lever the stem's force_conv_no_spill uses).
  3. >=2 height blocks: act_block_h_override = 32 (1 tile) => num_blocks_act_h = per_core_out_height_tiles
     (>=2 on any core count), so the fused kernel does many tilize<->matmul transitions and Option C
     collapses them to one. padding=0 avoids the Quasar halo zero-pad stub (MEM_ZEROS is unported).

Ring-fit (the constraint the stem violated): Option C sizes ACT_TILIZED to hold ALL blocks =
per_core_out_height_tiles * K_tiles * 128 units, must stay < 65536. Here out 32x32 = 32 out-h tiles,
K = in_channels(32)*3*3/32 = 9 tiles => worst case (1 core) 32*9*128 = 36864 < 65536 (fits).

HOW TO READ THE RESULT (run BOTH variants -- the `fused` and `split` params):
  - fused faults/times-out/mismatches AND split passes => hypothesis VALIDATED: isolating the tilize
    phase avoids the race. Build Option B (tilized activation as a real tensor -> no ring limit).
  - both pass  => shape too small to trigger; raise in_channels / out size / act_block_h (keep ring-fit).
  - both fault => race is intrinsic to a SINGLE tilize->matmul transition; Option B won't help ->
    escalate to the LLK team (~/llk_conv_tilize_issue.md).

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  pytest models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_split_tilize_hypothesis.py
"""

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.97


def _run(mesh_device, *, use_split):
    device = mesh_device
    torch.manual_seed(0)

    # Small stem-like tilize-path conv: 3x3 / s1 / p0, in=32 (K=9 tiles), out=64, RELU + bias + packer_l1_acc.
    batch_size = 1
    in_channels = 32
    out_channels = 64
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    out_h = out_w = 32  # 32x32 = 1024 sticks = 32 out-height tiles (tile-aligned)
    input_height = out_h + kernel_size[0] - 1  # s1/p0 => in = out + (k-1) = 34
    input_width = out_w + kernel_size[1] - 1

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((1, 1, 1, out_channels), dtype=torch.bfloat16).float()
    torch_golden = torch.relu(
        torch.nn.functional.conv2d(
            torch_input_nchw, torch_weight, bias=torch_bias.reshape(-1), stride=stride, padding=padding
        )
    )

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
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        full_inner_dim=True,  # single K-block => in0_num_blocks_w == 1 (Option C gate)
        act_block_h_override=32,  # 1-tile blocks => many height blocks (>=2 on any core count)
        reshard_if_not_optimal=True,  # let the conv fix the input shard to its optimal height-sharded layout
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    prev = os.environ.get("TT_METAL_QSR_CONV_SPLIT_TILIZE")
    if use_split:
        os.environ["TT_METAL_QSR_CONV_SPLIT_TILIZE"] = "1"
    elif prev is not None:
        del os.environ["TT_METAL_QSR_CONV_SPLIT_TILIZE"]
    try:
        out, [oh, ow], [tt_weight, tt_bias] = ttnn.experimental.quasar.conv2d(
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
        if prev is None:
            os.environ.pop("TT_METAL_QSR_CONV_SPLIT_TILIZE", None)
        else:
            os.environ["TT_METAL_QSR_CONV_SPLIT_TILIZE"] = prev

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, oh, ow, tt_out.shape[-1])
    tt_out = tt_out[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))

    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("use_split", [False, True], ids=["fused", "split"])
def test_conv2d_split_tilize_hypothesis(mesh_device, use_split):
    _run(mesh_device, use_split=use_split)
