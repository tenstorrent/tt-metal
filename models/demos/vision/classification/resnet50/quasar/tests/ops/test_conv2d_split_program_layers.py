# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Ahead-of-model probe: does the Option-B two-program split (UnpackToDestEn tilize + matmul) scale to the
layer2/layer3 conv2 CHANNEL COUNTS, i.e. to larger K, ahead of running the full model?

The stem conv is in_ch=32 (K = 32*3*3-ish / for the folded 4x4 stem K = 32*4*4 = 16 tiles). The bottleneck
conv2's are 3x3 at growing widths:
  layer1 conv2: 64 -> 64   (K = 64*3*3  = 576  = 18 tiles)
  layer2 conv2: 128 -> 128 (K = 128*3*3 = 1152 = 36 tiles)
  layer3 conv2: 256 -> 256 (K = 256*3*3 = 2304 = 72 tiles)
Bigger K => more tilize blocks (the split's Program A tilizes M x full_K, num_blocks scaled by filter_h), which
is exactly what stresses the UNPACK_TO_DEST dvalid ring in the unpack-to-dest tilize. This probe runs those K
sizes through the split at a SMALL spatial size (out 8x8) so they fit the emulator, isolating K/channel scaling
from the spatial-size / L1 pressure.

ALL cases here are HEIGHT_SHARDED + full_inner_dim so they route to the split path (the factory
split_program_tilize_only gate needs height_sharded + in0_num_blocks_w==1). NOTE the model runs layer3/4 conv2
BLOCK_SHARDED — the split path does NOT cover block-sharding yet, so this probe deliberately forces
height-sharding to test the tilize/matmul K-scaling independently of that separate block-sharded gap.

DEPENDENCY: needs the split fix actually working end-to-end first (test_conv2d_split_program_e2e.py::..._pure
green). While that is blocked (e.g. the UNPACK_TO_DEST 0x19), these will fault too — they are the "does the fix
hold as K grows" gate, expected to come online right after the stem e2e passes.

Run (both split flags):
  TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_QSR_TILIZE_UNPACK_TO_DEST=1 \
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_split_program_layers.py
  # -k 128to128  (layer2) or -k 256to256 (layer3) to isolate one.
"""

import os

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.98


def _run_split_conv(mesh_device, *, in_channels, out_channels):
    device = mesh_device
    torch.manual_seed(0)
    batch_size = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (0, 0)
    out_h, out_w = 8, 8  # small spatial so the larger-K cases fit the emulator; K scaling is the point
    input_height = out_h + kernel_size[0] - 1  # 10
    input_width = out_w + kernel_size[1] - 1  # 10

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, *kernel_size), dtype=torch.bfloat16).float()
    torch_golden = torch.nn.functional.conv2d(torch_input_nchw, torch_weight, bias=None, stride=stride, padding=padding)

    # pre-shard activation to L1 height-sharded (L1 path, no DRAM slicing)
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
    tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(device, in_mem)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        full_inner_dim=True,  # single K-block -> factory split_program_tilize_only eligibility
        act_block_h_override=32,
        reshard_if_not_optimal=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    saved = {k: os.environ.get(k) for k in ("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "TT_METAL_QSR_TILIZE_UNPACK_TO_DEST")}
    os.environ["TT_METAL_QSR_CONV_SPLIT_PROGRAM"] = "1"
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
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    tt_out = ttnn.to_torch(ttnn.from_device(out)).reshape(batch_size, oh, ow, -1)[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))
    print(f"split conv {in_channels}->{out_channels} 3x3 completed. out shape={tuple(tt_out.shape)}")
    assert_with_pcc(torch_golden, tt_out.float(), pcc=PCC)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "in_channels, out_channels",
    [(64, 64), (128, 128), (256, 256)],
    ids=["64to64_K18t", "128to128_K36t", "256to256_K72t"],
)
def test_quasar_conv2d_split_program_layer_channels(mesh_device, in_channels, out_channels):
    _run_split_conv(mesh_device, in_channels=in_channels, out_channels=out_channels)
