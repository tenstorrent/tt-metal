# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.conv2d`` — every distinct sliding-window conv the VAD-variant model ran
(ResNet-50 backbone at 384x640 + the FPN 3x3 output conv).

Shape source (kernel cache ``tt-metal-cache/3204133098312896700``)
----------------------------------------------------------------
Each case groups one ``conv_bmm_tilize`` compute variant with its activation reader
(``reader_conv_activations_*_v2``), weight sender/receiver
(``*_conv_weights_tiled_col_to_rm_blocks``), and ``halo_gather`` variants. The
compile-time args map via ``conv2d_op_sharded_program_factory.cpp`` (reader :852-934,
weights :963-1053, compute :1113-1167) and ``untilize_with_halo_program_factory.cpp:277-296``:

  * filter size, stride_w, dilation, ``W_in + 2*pad``  -> reader ct[0..9]      [H]
  * channels per shard (``c_read_bytes``), Cout/32     -> reader ct[3], weights ct[11]  [H]
  * act_block_h, output rows per core, NHW / C cores   -> compute + reader ct  [H]
  * fused ReLU (``PACK_RELU``), fused bias, bfp8_b weights, LoFi, no fp32 acc  [H]
  * input H and batch                                  -> inferred from rows/core x cores  [M]

1x1 stride-1 convs ran through the matmul path and are covered in ``test_linear.py`` /
``test_matmul.py``. Hashes in each id are the first 6 digits of the ``conv_bmm_tilize`` variant.

All convs: groups=1, dilation=1, weights+bias bfloat8_b (folded BN), LoFi, approx off,
fp32_dest_acc off, packer_l1_acc off; BLOCK_SHARDED convs use COL_MAJOR (``transpose_shards``).

Two tests:
  * ``test_conv2d`` — model shapes / dtypes / shard layout / act_block_h, auto core grid.
  * ``test_conv2d_model_grid`` (``blackhole_scale``) — additionally pins the model's
    20-core grid (``override_sharding_config`` + ``core_grid``), reproducing the cached
    per-core blocking.

Reference is ``torch.nn.functional.conv2d``. PCC 0.98 (bf16) / 0.97 (bfp8 activations or
reduction depth > 10000), matching the canonical ``run_conv`` thresholds.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.vad_variant_ops import op_utils as U

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED

# fmt: off
# (hw, n, in_ch, out_ch, h, w, kernel, stride, pad, relu, shard, act_block_h, in_dt, out_dt, grid, raw_in_ch)
#   hw        = input width (conftest emulator gating)
#   grid      = model core grid: ("hs", num_cores) or ("bs", grid_x) with 4 C-cores in y
#   raw_in_ch = stem's pre-pad channel count (RGB=3 padded to 16), else 0
CONV_CASES = [
    pytest.param(640, 1, 16, 64, 384, 640, 7, 2, 3, True, HS, 384, "bf16", "bf16", ("hs", 20), 3, id="C1-stem-474069"),
    pytest.param(160, 1, 64, 64, 96, 160, 3, 1, 1, True, HS, 384, "bfp8", "bfp8", ("hs", 20), 0, id="C2-layer1.conv2-505296"),
    pytest.param(160, 1, 128, 128, 96, 160, 3, 2, 1, True, HS, 192, "bfp8", "bfp8", ("hs", 20), 0, id="C3-layer2.0.conv2-092741"),
    pytest.param(80, 1, 128, 128, 48, 80, 3, 1, 1, True, HS, 192, "bfp8", "bfp8", ("hs", 20), 0, id="C4-layer2.conv2-092741"),
    pytest.param(160, 1, 256, 512, 96, 160, 1, 2, 0, False, HS, 96, "bfp8", "bfp8", ("hs", 20), 0, id="C5-layer2.0.downsample-016447"),
    pytest.param(40, 1, 256, 256, 24, 40, 3, 1, 1, True, HS, 96, "bfp8", "bfp8", ("hs", 10), 0, id="C6-layer3.conv2-216333"),
    pytest.param(80, 1, 256, 256, 48, 80, 3, 2, 1, True, BS, 192, "bfp8", "bfp8", ("bs", 5), 0, id="C7-layer3.0.conv2-881791"),
    pytest.param(80, 1, 512, 1024, 48, 80, 1, 2, 0, False, BS, 192, "bfp8", "bfp8", ("bs", 5), 0, id="C8-layer3.0.downsample-577604"),
    pytest.param(40, 1, 512, 512, 24, 40, 3, 2, 1, True, BS, 96, "bf16", "bf16", ("bs", 3), 0, id="C9-layer4.0.conv2-959893"),
    pytest.param(20, 1, 512, 512, 12, 20, 3, 1, 1, True, BS, 96, "bf16", "bf16", ("bs", 3), 0, id="C10-layer4.conv2-959893"),
    pytest.param(40, 1, 1024, 2048, 24, 40, 1, 2, 0, False, BS, 96, "bfp8", "bf16", ("bs", 3), 0, id="C11-layer4.0.downsample-475756"),
    pytest.param(20, 6, 256, 256, 12, 20, 3, 1, 1, False, BS, 96, "bf16", "bf16", ("bs", 5), 0, id="C12-fpn.conv3x3-064720"),
]
# fmt: on

_ARGNAMES = "hw, n, in_ch, out_ch, h, w, kernel, stride, pad, relu, shard, act_block_h, in_dt, out_dt, grid, raw_in_ch"


def _model_core_grid(mesh, grid):
    """The cached conv's core grid on the model's 5x4 worker grid."""
    U.model_grid(mesh)  # skips if the device can't provide 5x4
    kind, n = grid
    if kind == "hs":
        return ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(U.MODEL_GRID_X, U.MODEL_GRID_Y), row_wise=True)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, U.MODEL_GRID_Y - 1))})


def _run_conv(
    mesh,
    n,
    in_ch,
    out_ch,
    h,
    w,
    kernel,
    stride,
    pad,
    relu,
    shard,
    act_block_h,
    in_dt,
    out_dt,
    raw_in_ch,
    core_grid=None,
):
    src_ch = raw_in_ch if raw_in_ch else in_ch
    x_nchw = torch.randn(n, src_ch, h, w, dtype=torch.float32)
    if raw_in_ch:  # stem: RGB zero-padded to 16 channels
        x_nchw = torch.nn.functional.pad(x_nchw, (0, 0, 0, 0, 0, in_ch - raw_in_ch))
    weight = torch.randn(out_ch, in_ch, kernel, kernel, dtype=torch.float32)
    bias = torch.randn(1, 1, 1, out_ch, dtype=torch.float32)

    ref = torch.nn.functional.conv2d(x_nchw, weight, bias=bias.reshape(-1), stride=stride, padding=pad)
    if relu:
        ref = torch.relu(ref)
    out_h, out_w = ref.shape[2], ref.shape[3]
    ref_nhwc = ref.permute(0, 2, 3, 1).contiguous()

    # The model feeds the stem a ROW_MAJOR bf16 image and every later conv a DRAM TILE tensor.
    x_flat = x_nchw.permute(0, 2, 3, 1).reshape(1, 1, n * h * w, in_ch).contiguous()
    layout = ttnn.ROW_MAJOR_LAYOUT if raw_in_ch else ttnn.TILE_LAYOUT
    tt_input = U.to_tt(x_flat, mesh, dtype=U.DTYPES[in_dt], layout=layout)
    tt_weight = ttnn.from_torch(weight, dtype=ttnn.float32)
    tt_bias = ttnn.from_torch(bias, dtype=ttnn.float32)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=shard,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None,
        act_block_h_override=act_block_h,
        transpose_shards=shard == BS,
        # reshard_if_not_optimal and override_sharding_config are mutually exclusive.
        reshard_if_not_optimal=core_grid is None,
        deallocate_activation=False,
        config_tensors_in_dram=False,
    )
    if core_grid is not None:
        conv_config.override_sharding_config = True
        conv_config.core_grid = core_grid
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_out, [tt_out_h, tt_out_w] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=in_ch,
        out_channels=out_ch,
        device=mesh,
        kernel_size=(kernel, kernel),
        stride=(stride, stride),
        padding=(pad, pad),
        dilation=(1, 1),
        groups=1,
        batch_size=n,
        input_height=h,
        input_width=w,
        conv_config=conv_config,
        compute_config=compute_config,
        dtype=U.DTYPES[out_dt],
        return_output_dim=True,
    )
    assert (tt_out_h, tt_out_w) == (out_h, out_w), f"conv output dims {(tt_out_h, tt_out_w)} != torch {(out_h, out_w)}"

    out = ttnn.to_torch(ttnn.from_device(tt_out)).float()
    out = out.reshape(n, out_h, out_w, out.shape[-1])[:, :, :, :out_ch]

    low_precision = in_dt == "bfp8" or out_dt == "bfp8" or in_ch * kernel * kernel > 10000
    pcc = 0.97 if low_precision else 0.98
    passing, msg = U.comp_pcc(ref_nhwc, out, pcc)
    assert passing, f"conv2d PCC below {pcc}: {msg}"


@U.with_mesh_l1small(U.VAD_L1_SMALL_SIZE)
@pytest.mark.parametrize(_ARGNAMES, CONV_CASES)
def test_conv2d(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    n,
    in_ch,
    out_ch,
    h,
    w,
    kernel,
    stride,
    pad,
    relu,
    shard,
    act_block_h,
    in_dt,
    out_dt,
    grid,
    raw_in_ch,
):
    _run_conv(
        ttnn_mesh_device,
        n,
        in_ch,
        out_ch,
        h,
        w,
        kernel,
        stride,
        pad,
        relu,
        shard,
        act_block_h,
        in_dt,
        out_dt,
        raw_in_ch,
    )


@pytest.mark.blackhole_scale
@U.with_mesh_l1small(U.VAD_L1_SMALL_SIZE)
@pytest.mark.parametrize(_ARGNAMES, CONV_CASES)
def test_conv2d_model_grid(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    n,
    in_ch,
    out_ch,
    h,
    w,
    kernel,
    stride,
    pad,
    relu,
    shard,
    act_block_h,
    in_dt,
    out_dt,
    grid,
    raw_in_ch,
):
    """Same convs pinned to the model's cached core grid (20-core 5x4 worker grid)."""
    core_grid = _model_core_grid(ttnn_mesh_device, grid)
    _run_conv(
        ttnn_mesh_device,
        n,
        in_ch,
        out_ch,
        h,
        w,
        kernel,
        stride,
        pad,
        relu,
        shard,
        act_block_h,
        in_dt,
        out_dt,
        raw_in_ch,
        core_grid=core_grid,
    )
