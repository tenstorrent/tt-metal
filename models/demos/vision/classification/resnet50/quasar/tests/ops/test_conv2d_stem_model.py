# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
FAITHFUL standalone repro of the resnet50/quasar MODEL STEM (fold -> reshape -> on-device conv1).

WHY THIS EXISTS
---------------
The full model faults at run() line 995 (the on-device stem conv1) with a DM2
`reader_bmm_tile_layout_in0_sender_padding` assert. `test_conv2d_stem.py` does NOT reproduce it: it feeds
conv1 a synthetic HOST input with a hand-written conv_config, and PASSES. The model instead feeds conv1 the
ON-DEVICE `fold` output (folded small-face geometry) with the model's real `self.conv1_config`. This test
mirrors the model's ACTUAL stem path verbatim so the fault reproduces in seconds instead of a ~28-min run.

Mirrored verbatim from ttnn_functional_resnet50.py (__init__ stem setup + run() lines ~940-1004) and
resnet50_test_infra.setup_l1_sharded_input (Quasar branch):
  * input image (batch=1): torch (1,3,224,224) -> NHWC, host-padded C 3->nearest_y(3,8)=8 -> (1,224,224,8)
    bf16 ROW_MAJOR, interleaved L1.
  * fold: stride=2, padding=[3,3,3,3,0,5] (fold_pad_h/w=kernel_size=3, fold_pad_c=C-c=5),
    use_transpose_as_fold=False, input_is_nhwc=True, output_shape=(1,115,115,32), on the (device-clamped)
    8x8 fold compute grid.
  * reshape fold output (n,c,h,w) -> (1,1,n*c*h,w).
  * conv1: 32->64, 4x4, s1, p0, input 115x115, self.conv1_config (RELU, deallocate_activation,
    reallocate_halo_output=True, act_block_h_override=0 [Quasar], HEIGHT_SHARDED, reshard_if_not_optimal=False),
    packer_l1_acc compute config.

The golden is computed from the ACTUAL fold output (read back to torch) @ the folded weight + bias, RELU'd,
so it does not depend on re-deriving fold numerics; it validates conv1 whenever the op actually completes.

RUN (emulator, split-program stem path as the model uses, forced JIT):
  TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_QSR_TC_ISOLATE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_stem_model.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import _nearest_y
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_stem_model(mesh_device):
    device = mesh_device
    torch.manual_seed(0)

    # --- original stem params (resnet50_test_infra: input_shape=(bs,3,224,224), first-conv kernel_size=3,
    #     stride=2). batch=1 for the emulator. ---
    batch_size = 1
    c, h, w = 3, 224, 224
    fold_kernel_size = 3  # resnet50_first_conv_kernel_size
    fold_stride = 2  # resnet50_first_conv_stride

    # --- folded conv1 params (model __init__): 32->64, 4x4, s1, p0, folded input 115x115 ---
    conv1_in_channels = 32
    conv1_out_channels = 64
    conv1_kernel_size = (4, 4)
    conv1_stride = (1, 1)
    conv1_padding = (0, 0)
    conv1_input_height = conv1_input_width = 115

    # --- fold params (model __init__ ~lines 785-802) ---
    fold_stride_h = fold_stride_w = fold_stride
    hp = h + fold_kernel_size * 2  # 230
    wp = w + fold_kernel_size * 2  # 230
    C = _nearest_y(c, 8)  # 8 (Quasar aligns fold channels to 8)
    fold_pad_c = C - c  # 5
    fold_pad_h = fold_pad_w = fold_kernel_size  # 3
    fold_output_shape = (
        batch_size,
        hp // fold_stride_h,  # 115
        wp // fold_stride_w,  # 115
        C * (fold_stride_h * fold_stride_w),  # 32
    )

    # fold compute grid: model default 8x8, clamped to the device's real core count (emulator 1-2 cores).
    fold_compute_grid_size = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    _dev_grid = device.compute_with_storage_grid_size()
    _dev_max_cores = _dev_grid.x * _dev_grid.y
    if fold_compute_grid_size.num_cores() > _dev_max_cores:
        fold_compute_grid_size = ttnn.num_cores_to_corerangeset(_dev_max_cores, _dev_grid, row_wise=True)

    # --- input image: NHWC, host-padded C 3->8, bf16 ROW_MAJOR, interleaved L1 (test_infra Quasar branch) ---
    torch_input_nchw = torch.rand((batch_size, c, h, w), dtype=torch.float32)
    nhwc = torch_input_nchw.permute(0, 2, 3, 1).contiguous()  # (1,224,224,3)
    if C != c:
        nhwc = torch.nn.functional.pad(nhwc, (0, C - c))  # (1,224,224,8)
    tt_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(
        device, ttnn.L1_MEMORY_CONFIG
    )

    # --- folded conv1 weight/bias (model gets these preprocessed [out,in,4,4] from parameters; random here) ---
    torch_weight = torch.randn((conv1_out_channels, conv1_in_channels, *conv1_kernel_size), dtype=torch.bfloat16)
    torch_bias = torch.randn((1, 1, 1, conv1_out_channels), dtype=torch.bfloat16)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16)

    # --- conv1_config, verbatim from the model (Quasar: act_block_h_override=0) ---
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        deallocate_activation=True,
        reallocate_halo_output=True,
        act_block_h_override=0,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    # --- run() stem block: fold -> reshape -> conv1 ---
    fold_output_tensor = ttnn.experimental.quasar.fold(
        tt_input,
        fold_stride_h,
        fold_stride_w,
        use_transpose_as_fold=False,
        padding=[fold_pad_h, fold_pad_h, fold_pad_w, fold_pad_w, 0, fold_pad_c],
        grid_size=fold_compute_grid_size,
        input_is_nhwc=True,
        output_shape=ttnn.Shape(list(fold_output_shape)),
    )

    # golden from the ACTUAL fold output (NHWC) -> NCHW folded conv (4x4/s1/p0) + bias + RELU.
    fold_torch = ttnn.to_torch(ttnn.from_device(fold_output_tensor)).float().reshape(fold_output_shape)
    fold_nchw = fold_torch.permute(0, 3, 1, 2)  # (1,32,115,115)
    golden = torch.relu(
        torch.nn.functional.conv2d(
            fold_nchw,
            torch_weight.float(),
            bias=torch_bias.reshape(-1).float(),
            stride=conv1_stride,
            padding=conv1_padding,
        )
    )  # (1,64,112,112)

    n, cc, hh, ww = fold_output_tensor.shape
    fold_output_tensor = ttnn.experimental.quasar.reshape(fold_output_tensor, (1, 1, n * cc * hh, ww))

    out, [out_h, out_w], [tt_weight, tt_bias] = ttnn.experimental.quasar.conv2d(
        input_tensor=fold_output_tensor,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=conv1_in_channels,
        out_channels=conv1_out_channels,
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

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch_size, out_h, out_w, tt_out.shape[-1])[:, :, :, :conv1_out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))  # NHWC -> NCHW

    assert_with_pcc(golden, tt_out.float(), pcc=0.98)
