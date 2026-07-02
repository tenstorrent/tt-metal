"""
Standalone repro of the resnet50/quasar STEM conv2d deadlock.

This isolates the first-conv call from
models/demos/vision/classification/resnet50/quasar/tt/ttnn_functional_resnet50.py
(the `resnet50.run()` stem: fold -> reshape -> conv2d). The fused conv compute kernel
`ttnn/cpp/ttnn/operations/experimental/quasar/conv2d/device/kernels/conv_bmm_tilize_metal2.cpp`
hangs on Quasar in a 3-thread MATH<->PACK<->UNPACK cycle over the matmul-partials compute
self-loop DFB (partials counter frozen at 96 = out-subblock 12; MATH_PACK sem stall).

To keep the repro minimal we build the conv INPUT directly in the exact shape/layout the fold
produces — [1, 1, batch*115*115, 16], HEIGHT_SHARDED — instead of running the fold op, then make
the identical conv2d call with the identical config. The conv's internal blocking (in0_num_blocks_w=4,
in0_num_subblocks=49, out_subblock_num_tiles=8, packer_l1_acc, fuse_bias) is derived from these
params, so this reproduces the same deadlock.

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so \
  TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
  TT_METAL_WATCHER_DISABLE_ASSERT=1 TT_METAL_WATCHER_DISABLE_PAUSE=1 TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 \
  pytest test_conv_hang.py::test_conv_hang

A healthy conv returns; the bug hangs in conv_bmm_tilize_metal2 (never reaches synchronize_device).
"""

import math

import pytest
import torch

import ttnn
from models.common.utility_functions import _nearest_y, nearest_32


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_conv_hang(mesh_device):
    # Full-grid repro (batch 16). Reproduces the stem-conv deadlock on the default 32-core descriptor.
    # OOMs on a tiny (e.g. 2-core) grid — use test_conv_hang_small_grid there.
    _run_stem_conv(mesh_device, batch_size=16)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_conv_hang_small_grid(mesh_device):
    # Small-grid repro (batch 1). Same conv op/config as test_conv_hang — only the batch is reduced so
    # the per-core output/matmul-partials DFB ring fits the uint16_t addressing limit on a 2-core grid.
    # Each DFB ring is addressed in uint16_t units of 16 B, so ring_bytes must stay < 65536*16 = 1 MB
    # (dataflow_buffer.cpp:632). The output shard IS the per-core sharded output and can't be blocked, so
    # it is sized by (N*H_out*W_out / num_cores_nhw) rounded to tiles * out-ch tiles:
    #   batch 2, 2 cores -> 2*112*112/2 = 12544 rows = 392 tiles * 2 wide = 784 tiles = 1.53 MB  -> OVERFLOW
    #   batch 1, 2 cores ->   112*112/2 =  6272 rows = 196 tiles * 2 wide = 392 tiles = 0.77 MB  -> fits
    # (batch 16 also OOMs the ~4 MB/bank L1 outright.) The deadlocking path is batch-independent:
    # kernel 4x4 (non-1x1 -> conv_bmm_tilize), in=16/out=64, packer_l1_acc + fused bias, and the
    # K-blocking (in0_num_blocks_w>1 -> matmul-partials spill) all come from the channels/kernel, not N.
    _run_stem_conv(mesh_device, batch_size=1)


def _run_stem_conv(mesh_device, batch_size):
    device = mesh_device
    num_devices = device.get_num_devices()

    # --- stem params, verbatim from resnet50.__init__ (kernel-4x4 stem class) ---
    first_conv_kernel_size = 3  # resnet50_first_conv_kernel_size
    first_conv_stride = 2  # resnet50_first_conv_stride
    input_shape = (batch_size * num_devices, 3, 224, 224)

    conv1_kernel_size = (4, 4)
    conv1_stride = (1, 1)
    conv1_padding = (0, 0)
    conv1_input_height = 115
    conv1_input_width = 115

    # fold output shape (fold pads by kernel, folds by stride):
    #   h,w = 224 + 3*2 = 230; folded = 230//2 = 115; C = nearest_y(3,4)=4; folded_C = 4*(2*2) = 16
    _, c, h, w = input_shape
    n = batch_size
    h += first_conv_kernel_size * 2
    w += first_conv_kernel_size * 2
    C = _nearest_y(c, 4)
    fold_output_shape = (n, h // first_conv_stride, w // first_conv_stride, C * (first_conv_stride * first_conv_stride))

    conv1_input_channels = fold_output_shape[-1]  # 16
    conv1_output_channels = 64

    # --- build the conv input directly in the fold-output layout: [1, 1, N*H*W, C_folded], HEIGHT_SHARDED ---
    # (mirrors resnet50.__init__ override_fold_mem_config: full grid clamped to device, height-sharded)
    compute_grid = device.compute_with_storage_grid_size()
    fold_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    fold_max_cores = compute_grid.x * compute_grid.y
    if fold_grid.num_cores() > fold_max_cores:
        fold_grid = ttnn.num_cores_to_corerangeset(fold_max_cores, compute_grid, row_wise=True)

    input_channels_padded = nearest_32(conv1_input_channels) if conv1_input_channels % 8 != 0 else conv1_input_channels
    if input_channels_padded % 8 != 0:
        input_channels_padded = ((input_channels_padded + 7) // 8) * 8

    tensor_height = conv1_input_width * conv1_input_height * batch_size  # N*H*W (211600 at batch 16, 26450 at batch 2)
    tensor_width = input_channels_padded  # 16
    num_cores = fold_grid.num_cores()
    shard_height = math.ceil(tensor_height / num_cores)

    fold_out_mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, tensor_width),
        core_grid=fold_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    conv_in_torch = torch.rand((1, 1, tensor_height, tensor_width), dtype=torch.bfloat16)
    conv_in = ttnn.from_torch(conv_in_torch, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    conv_in = conv_in.to(device, fold_out_mem_config)  # matches the model's tt_inputs_host.to(device, mem_config)

    # --- weights / bias (random; the deadlock is shape/config-driven, not value-driven) ---
    weight = ttnn.from_torch(
        torch.rand((conv1_output_channels, conv1_input_channels, *conv1_kernel_size), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
    )
    bias = ttnn.from_torch(torch.rand((1, 1, 1, conv1_output_channels), dtype=torch.bfloat16), dtype=ttnn.bfloat16)

    # --- the exact stem conv1 config (the deadlocking configuration) ---
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),  # fused RELU (fuse_bias path)
        deallocate_activation=False,
        reallocate_halo_output=True,
        act_block_h_override=0,  # quasar default -> full-height act block -> in0_num_subblocks=49
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        packer_l1_acc=True,  # packer_l1_acc + fused bias -> matmul-partials CB (the deadlock lynchpin)
    )

    # --- the conv2d call that hangs in conv_bmm_tilize_metal2 ---
    out, [out_h, out_w], [weight, bias] = ttnn.experimental.quasar.conv2d(
        input_tensor=conv_in,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=conv1_input_channels,
        out_channels=conv1_output_channels,
        batch_size=batch_size,
        input_height=conv1_input_height,
        input_width=conv1_input_width,
        kernel_size=conv1_kernel_size,
        stride=conv1_stride,
        padding=conv1_padding,
        dilation=(1, 1),
        groups=1,
        device=device,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )

    # If the conv is healthy this returns; the bug hangs above and never reaches here.
    ttnn.synchronize_device(device)
    assert out is not None
