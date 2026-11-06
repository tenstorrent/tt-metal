# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.conv2d`` — every distinct conv the OFT model ran on device.

Shape source (kernel cache ``tt-metal-cache-oft-e2e``, factories at main ``1088b10b014``)
--------------------------------------------------------------------------------------
Each case groups one ``conv_bmm_tilize`` compute variant with its activation reader
(``reader_conv_activations_padded_with_halo_3x3_weights_v2``), weight sender/receiver
(``reader_writer_tiled_out_1d_mcast_*_conv_weights_tiled_col_to_rm_blocks``) and
``halo_gather`` pair. From their compile-time args:

  * padded input width, channel read bytes, filter / stride            [H]
  * weight-matrix tile counts (Cin*k*k, Cout), fused bias, PACK_RELU    [H]
  * act_block_h, shard height per core (x 20 cores = NHW)               [H]
  * CB formats: weights bf16 everywhere; layer2 in/out float32; stem
    output bfloat8_b; head / NMS conv ROW_MAJOR output                  [H]
  * DRAM height slicing: stem 4 slices, head 2 slices (halo sizes)      [H]
  * input H for square / known inputs                                   [M]

Every conv: batch 1, groups 1, dilation 1, HEIGHT_SHARDED on the 20-core (5x4) grid,
HiFi2 + fp32_dest_acc (NMS conv: HiFi4), approx off, packer_l1_acc off, bias in bf16
(NMS conv: no bias). The frontend/topdown biases are GroupNorm-folded in the cached run.

The ``lat8`` 1x1 stride-1 conv ran through the matmul path
(``bmm_large_block_zm_fused_bias_activation/501803``: fp32 in0, K=128, per_core_M=12,
per_core_N=8, HiFi2 approx + fp32 acc, fused bias + ReLU); it is called here as the
``ttnn.conv2d`` it was.

ResNet layer3/layer4 and lat16/lat32 have no cache variants (they did not run on device).

Two tests:
  * ``test_conv2d`` — model shapes / dtypes / layouts / act_block_h / slicing, auto grid.
  * ``test_conv2d_model_grid`` (``blackhole_scale``) — additionally pins the model's
    20-core grid (``override_sharding_config`` + ``core_grid``).
Plus ``test_conv2d_small``: emulator-sized proxies with the same op configs.

Second cache (in-tree OFT run, factories at main ``628a4797bf7``): ``*_v2`` tests
---------------------------------------------------------------------------------
A second kernel cache (``tt-metal-cache930749440442549079``, fork of main 2026-04-16..24) matches
the in-tree model (``models/experimental/oft`` + ``models/tt_cnn`` builder) far more closely and
differs from the first run in ways the cases above do not exercise:

  * conv weights (and the prepared bias) are **bfloat8_b**; activations / outputs bf16    [H]
  * GroupNorm runs on device, so the ResNet / topdown convs have **no bias and no fused
    ReLU** (only the head conv has a bias)                                                 [H]
  * **packer_l1_acc on** (HiFi2 + fp32 dest acc) for the frontend, lat and head convs,
    off for the topdown convs; ``config_tensors_in_dram`` off (cache readers carry no
    DRAM config address, although the model config asks for it)                           [H]
  * the whole ResNet-18 runs on device: layer3 (24x80, 256 ch) and layer4 (12x40, 512 ch)
    including the 1x1 / s2 downsamples                                                     [H]
  * stem conv input is bf16 **TILE** (padded_slice tiled reader, 1 tile per row)           [H]
  * topdown at **256** channels, act_block_h 64, 4 DRAM height slices, ROW_MAJOR in / out;
    head 256 -> 9, act_block_h 128, 2 DRAM slices, TILE in / ROW_MAJOR out                 [H]
  * lat8 / lat16 / lat32 1x1 convs (bias, no ReLU, auto sharding) on the matmul path
    (``bmm_large_block_zm_fused_bias_activation`` 186165 / 138266 / 162197:
    FUSE_BIAS, FP32_DEST_ACC_EN, PACKER_L1_ACC, bfloat8_b in1 / bias)                     [H]

Everything is height sharded on the same 20-core (5x4) grid (1D weight mcast kernels only;
per-core blocks x 20 cores = NHW per slice). The NMS 5x5 conv (``conv_bmm_tilize/568080``) is
unchanged from case K and not repeated. The model-config sources say packer_l1_acc=False and
config_tensors_in_dram=True everywhere; the cache is treated as ground truth.
"""

import pytest
import torch

import ttnn
from models.experimental.ops.quasar.tests.oft_ops import op_utils as U

_CC = {
    # (math_fidelity, math_approx_mode)
    "hifi2": (ttnn.MathFidelity.HiFi2, False),
    "hifi2_approx": (ttnn.MathFidelity.HiFi2, True),
    "hifi4": (ttnn.MathFidelity.HiFi4, False),
}

_SHARD = {"hs": ttnn.TensorMemoryLayout.HEIGHT_SHARDED, None: None}

BEV = U.BEV

# fmt: off
# (hw, in_ch, out_ch, h, kernel, stride, pad, relu, bias, act_block_h, in_dt, in_rm, out_dt, out_rm, slices, cc)
#   hw     = input width (conftest emulator gating); batch is always 1
#   in_rm  = ROW_MAJOR input in DRAM (else TILE in DRAM, resharded by conv)
#   slices = DRAM height slices (0 = L1 full)
CONV_CASES = [
    pytest.param(1280, 3, 64, 384, 7, 2, 3, True, True, 128, "bf16", True, "bfp8", False, 4, "hifi2", id="A-stem.conv1-179189"),
    pytest.param(320, 64, 64, 96, 3, 1, 1, True, True, 32, "bfp8", False, "bf16", False, 0, "hifi2", id="B-layer1.0.conv1-bfp8in-897554"),
    pytest.param(320, 64, 64, 96, 3, 1, 1, True, True, 32, "bf16", False, "bf16", False, 0, "hifi2", id="B-layer1.1.conv1-897554"),
    pytest.param(320, 64, 64, 96, 3, 1, 1, False, True, 32, "bf16", False, "bf16", False, 0, "hifi2", id="C-layer1.conv2-317166"),
    pytest.param(320, 64, 128, 96, 3, 2, 1, True, True, 32, "bf16", False, "fp32", False, 0, "hifi2", id="D-layer2.0.conv1-113828"),
    pytest.param(320, 64, 128, 96, 1, 2, 0, False, True, 32, "bf16", False, "fp32", False, 0, "hifi2", id="E-layer2.0.downsample-696798"),
    pytest.param(160, 128, 128, 48, 3, 1, 1, True, True, 32, "fp32", False, "fp32", False, 0, "hifi2", id="F-layer2.1.conv1-510181"),
    pytest.param(160, 128, 128, 48, 3, 1, 1, False, True, 32, "fp32", False, "fp32", False, 0, "hifi2", id="G-layer2.conv2-114005"),
    pytest.param(160, 128, 256, 48, 1, 1, 0, True, True, 0, "fp32", False, "fp32", False, 0, "hifi2_approx", id="L-lat8-1x1-mm-501803"),
    pytest.param(BEV, 128, 128, BEV, 3, 1, 1, True, True, 32, "bf16", False, "bf16", False, 0, "hifi2", id="H-topdown.conv1-116319"),
    pytest.param(BEV, 128, 128, BEV, 3, 1, 1, False, True, 32, "bf16", False, "bf16", False, 0, "hifi2", id="I-topdown.conv2-898175"),
    pytest.param(BEV, 128, U.HEAD_OUT, BEV, 3, 1, 1, False, True, 64, "bf16", True, "bf16", True, 2, "hifi2", id="J-head-981391"),
    pytest.param(BEV, 1, 1, BEV, 5, 1, 2, False, False, 256, "bf16", True, "bf16", True, 0, "hifi4", id="K-encoder.nms_conv-357627"),
]

# Emulator-sized proxies: same op configs (dtypes, layouts, fusion, compute) at small H x W.
SMALL_CASES = [
    pytest.param(40, 64, 64, 24, 3, 1, 1, True, True, 32, "bf16", False, "bf16", False, 0, "hifi2", id="proxy-B-24x40"),
    pytest.param(40, 64, 128, 24, 3, 2, 1, True, True, 32, "bf16", False, "fp32", False, 0, "hifi2", id="proxy-D-24x40"),
    pytest.param(40, 128, 128, 12, 3, 1, 1, False, True, 32, "fp32", False, "fp32", False, 0, "hifi2", id="proxy-G-12x40"),
    pytest.param(32, 128, 128, 32, 3, 1, 1, True, True, 32, "bf16", False, "bf16", False, 0, "hifi2", id="proxy-H-32x32"),
    pytest.param(32, 1, 1, 32, 5, 1, 2, False, False, 0, "bf16", True, "bf16", True, 0, "hifi4", id="proxy-K-32x32"),
]
# fmt: on

_ARGNAMES = (
    "hw, in_ch, out_ch, h, kernel, stride, pad, relu, bias, act_block_h, in_dt, in_rm, out_dt, out_rm, slices, cc"
)

# ---------------------------------------------------------------------------------------------
# Second cache (in-tree model, main 628a4797bf7). Ids end with the conv_bmm_tilize hash (lat:
# bmm_large_block_zm_fused_bias_activation hash). Two extra columns:
#   w_dt  = weights dtype (bfloat8_b everywhere here)
#   l1acc = packer_l1_acc requested (conv drops it when Cin blocks < 3 without bias, e.g. 1x1)
# ---------------------------------------------------------------------------------------------
# fmt: off
# (hw, in_ch, out_ch, h, kernel, stride, pad, relu, bias, act_block_h, in_dt, in_rm, out_dt, out_rm, slices, cc, w_dt, l1acc)
CONV_CASES_V2 = [
    pytest.param(1280, 3, 64, 384, 7, 2, 3, False, False, 128, "bf16", False, "bf16", False, 4, "hifi2", "bfp8", True, id="M-stem.conv1-tilein-273496"),
    pytest.param(320, 64, 64, 96, 3, 1, 1, False, False, 32, "bfp8", False, "bf16", False, 0, "hifi2", "bfp8", True, id="N-layer1.0.conv1-bfp8in-280813"),
    pytest.param(320, 64, 64, 96, 3, 1, 1, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="N-layer1.conv-280813"),
    pytest.param(320, 64, 128, 96, 3, 2, 1, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="O-layer2.0.conv1-159448"),
    pytest.param(320, 64, 128, 96, 1, 2, 0, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="P-layer2.0.downsample-392590"),
    pytest.param(160, 128, 128, 48, 3, 1, 1, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="Q-layer2.conv-137991"),
    pytest.param(160, 128, 256, 48, 3, 2, 1, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="R-layer3.0.conv1-498246"),
    pytest.param(160, 128, 256, 48, 1, 2, 0, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="S-layer3.0.downsample-363953"),
    pytest.param(80, 256, 256, 24, 3, 1, 1, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="T-layer3.conv-481861"),
    pytest.param(80, 256, 512, 24, 3, 2, 1, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="U-layer4.0.conv1-147023"),
    pytest.param(80, 256, 512, 24, 1, 2, 0, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="V-layer4.0.downsample-555606"),
    pytest.param(40, 512, 512, 12, 3, 1, 1, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="W-layer4.conv-143907"),
    pytest.param(BEV, 256, 256, BEV, 3, 1, 1, False, False, 64, "bf16", True, "bf16", True, 4, "hifi2", "bfp8", False, id="X-topdown.conv-256ch-133849"),
    pytest.param(BEV, 256, U.HEAD_OUT, BEV, 3, 1, 1, False, True, 128, "bf16", False, "bf16", True, 2, "hifi2", "bfp8", True, id="Y-head-256ch-113779"),
]

# lat8 / lat16 / lat32: shard_layout=None (model "DEFAULT" config) -> conv picks the matmul path
# and its sharding (on the 5x4 grid: lat8 20-core height, lat16 4x4 block, lat32 15-core height).
# lat32 (12x40) is already emulator-sized; proxy-Z is the lat8 config at 12x40.
LAT_CASES_V2 = [
    pytest.param(160, 128, 256, 48, 1, 1, 0, False, True, 0, "bf16", False, "bf16", False, 0, "hifi2_approx", "bfp8", True, id="Z-lat8-auto-186165"),
    pytest.param(80, 256, 256, 24, 1, 1, 0, False, True, 0, "bf16", False, "bf16", False, 0, "hifi2_approx", "bfp8", True, id="Z-lat16-auto-138266"),
    pytest.param(40, 512, 256, 12, 1, 1, 0, False, True, 0, "bf16", False, "bf16", False, 0, "hifi2_approx", "bfp8", True, id="Z-lat32-auto-162197"),
    pytest.param(40, 128, 256, 12, 1, 1, 0, False, True, 0, "bf16", False, "bf16", False, 0, "hifi2_approx", "bfp8", True, id="proxy-Z-lat8-auto-12x40"),
]

SMALL_CASES_V2 = [
    pytest.param(48, 3, 64, 48, 7, 2, 3, False, False, 32, "bf16", False, "bf16", False, 4, "hifi2", "bfp8", True, id="proxy-M-48x48"),
    pytest.param(40, 64, 64, 24, 3, 1, 1, False, False, 32, "bfp8", False, "bf16", False, 0, "hifi2", "bfp8", True, id="proxy-N-bfp8in-24x40"),
    pytest.param(40, 128, 256, 24, 3, 2, 1, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="proxy-R-24x40"),
    pytest.param(40, 128, 256, 24, 1, 2, 0, False, False, 32, "bf16", False, "bf16", False, 0, "hifi2", "bfp8", True, id="proxy-S-24x40"),
    pytest.param(32, 256, 256, 32, 3, 1, 1, False, False, 32, "bf16", True, "bf16", True, 4, "hifi2", "bfp8", False, id="proxy-X-32x32"),
    pytest.param(32, 256, U.HEAD_OUT, 32, 3, 1, 1, False, True, 32, "bf16", False, "bf16", True, 2, "hifi2", "bfp8", True, id="proxy-Y-32x32"),
]
# fmt: on

_ARGNAMES_V2 = _ARGNAMES + ", w_dt, l1acc"


def _run_conv(
    mesh,
    hw,
    in_ch,
    out_ch,
    h,
    kernel,
    stride,
    pad,
    relu,
    bias,
    act_block_h,
    in_dt,
    in_rm,
    out_dt,
    out_rm,
    slices,
    cc,
    core_grid=None,
    w_dt="bf16",
    l1acc=False,
    shard="hs",
):
    """``w_dt``: weights dtype; ``l1acc``: packer_l1_acc; ``shard``: ``"hs"`` or ``None`` (conv picks)."""
    w = hw
    torch_dtype = torch.float32 if in_dt == "fp32" else torch.bfloat16
    x_nhwc = torch.randn(1, h, w, in_ch).to(torch_dtype)
    weight = torch.randn(out_ch, in_ch, kernel, kernel).to(torch.bfloat16)
    bias_t = torch.randn(1, 1, 1, out_ch).to(torch.bfloat16) if bias else None

    if in_rm:
        tt_input = U.to_tt(x_nhwc, mesh, dtype=U.DTYPES[in_dt], layout=ttnn.ROW_MAJOR_LAYOUT)
    else:
        tt_input = U.to_tt(x_nhwc.reshape(1, 1, h * w, in_ch), mesh, dtype=U.DTYPES[in_dt], layout=ttnn.TILE_LAYOUT)
    # Reference on what the device holds (bfloat8_b-quantized input where applicable).
    x_seen = U.seen(tt_input, x_nhwc, mesh).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x_seen.float(),
        weight.float(),
        bias=bias_t.float().reshape(-1) if bias else None,
        stride=stride,
        padding=pad,
    )
    if relu:
        ref = torch.relu(ref)
    out_h, out_w = ref.shape[2], ref.shape[3]
    ref_nhwc = ref.permute(0, 2, 3, 1).reshape(out_h * out_w, out_ch)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=U.DTYPES[w_dt],
        shard_layout=_SHARD[shard],
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU) if relu else None,
        output_layout=ttnn.ROW_MAJOR_LAYOUT if out_rm else ttnn.TILE_LAYOUT,
        # tt_cnn builder: reshard_if_not_optimal only with an explicit shard layout.
        reshard_if_not_optimal=shard is not None,
        deallocate_activation=False,
        config_tensors_in_dram=False,
    )
    if act_block_h:
        conv_config.act_block_h_override = act_block_h
    if core_grid is not None:
        # conv2d_utils rejects reshard_if_not_optimal together with override_sharding_config.
        conv_config.reshard_if_not_optimal = False
        conv_config.override_sharding_config = True
        conv_config.core_grid = core_grid
    fidelity, approx = _CC[cc]
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=fidelity,
        math_approx_mode=approx,
        fp32_dest_acc_en=True,
        packer_l1_acc=l1acc,
    )
    slice_config = (
        ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=slices)
        if slices
        else ttnn.Conv2dL1FullSliceConfig
    )

    tt_out, [tt_out_h, tt_out_w] = U.op("conv2d")(
        input_tensor=tt_input,
        weight_tensor=ttnn.from_torch(weight, dtype=ttnn.bfloat16),
        bias_tensor=ttnn.from_torch(bias_t, dtype=ttnn.bfloat16) if bias else None,
        in_channels=in_ch,
        out_channels=out_ch,
        device=mesh,
        kernel_size=(kernel, kernel),
        stride=(stride, stride),
        padding=(pad, pad),
        dilation=(1, 1),
        groups=1,
        batch_size=1,
        input_height=h,
        input_width=w,
        conv_config=conv_config,
        compute_config=compute_config,
        slice_config=slice_config,
        dtype=U.DTYPES[out_dt],
        return_output_dim=True,
    )
    assert (tt_out_h, tt_out_w) == (out_h, out_w), f"conv output dims {(tt_out_h, tt_out_w)} != torch {(out_h, out_w)}"

    out = U.from_tt(tt_out, mesh)
    out = out.reshape(-1, out.shape[-1])[: out_h * out_w, :out_ch]

    pcc = 0.98 if "bfp8" in (in_dt, out_dt, w_dt) else 0.99
    passing, msg = U.comp_pcc(ref_nhwc, out, pcc)
    assert passing, f"conv2d PCC below {pcc}: {msg}"


@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(_ARGNAMES, CONV_CASES)
def test_conv2d(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    in_ch,
    out_ch,
    h,
    kernel,
    stride,
    pad,
    relu,
    bias,
    act_block_h,
    in_dt,
    in_rm,
    out_dt,
    out_rm,
    slices,
    cc,
):
    _run_conv(
        ttnn_mesh_device,
        hw,
        in_ch,
        out_ch,
        h,
        kernel,
        stride,
        pad,
        relu,
        bias,
        act_block_h,
        in_dt,
        in_rm,
        out_dt,
        out_rm,
        slices,
        cc,
    )


@pytest.mark.blackhole_scale
@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(_ARGNAMES, CONV_CASES)
def test_conv2d_model_grid(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    in_ch,
    out_ch,
    h,
    kernel,
    stride,
    pad,
    relu,
    bias,
    act_block_h,
    in_dt,
    in_rm,
    out_dt,
    out_rm,
    slices,
    cc,
):
    """Same convs pinned to the model's 20-core (5x4) height-sharded grid."""
    _run_conv(
        ttnn_mesh_device,
        hw,
        in_ch,
        out_ch,
        h,
        kernel,
        stride,
        pad,
        relu,
        bias,
        act_block_h,
        in_dt,
        in_rm,
        out_dt,
        out_rm,
        slices,
        cc,
        core_grid=U.model_core_range_set(ttnn_mesh_device),
    )


@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(_ARGNAMES, SMALL_CASES)
def test_conv2d_small(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    in_ch,
    out_ch,
    h,
    kernel,
    stride,
    pad,
    relu,
    bias,
    act_block_h,
    in_dt,
    in_rm,
    out_dt,
    out_rm,
    slices,
    cc,
):
    _run_conv(
        ttnn_mesh_device,
        hw,
        in_ch,
        out_ch,
        h,
        kernel,
        stride,
        pad,
        relu,
        bias,
        act_block_h,
        in_dt,
        in_rm,
        out_dt,
        out_rm,
        slices,
        cc,
    )


# =============================================================================
# Second cache (in-tree model, main 628a4797bf7)
# =============================================================================


@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(_ARGNAMES_V2, CONV_CASES_V2)
def test_conv2d_v2(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    in_ch,
    out_ch,
    h,
    kernel,
    stride,
    pad,
    relu,
    bias,
    act_block_h,
    in_dt,
    in_rm,
    out_dt,
    out_rm,
    slices,
    cc,
    w_dt,
    l1acc,
):
    """bfloat8_b-weight, no-bias / no-ReLU (GroupNorm on device) height-sharded convs, auto grid.

    [H] op config (dtypes, layouts, act_block_h, slices, l1acc, fidelity) and per-core blocking;
    [M] input H x W (padded width and per-core rows x 20 cores read directly, H from the architecture).
    """
    _run_conv(
        ttnn_mesh_device,
        hw,
        in_ch,
        out_ch,
        h,
        kernel,
        stride,
        pad,
        relu,
        bias,
        act_block_h,
        in_dt,
        in_rm,
        out_dt,
        out_rm,
        slices,
        cc,
        w_dt=w_dt,
        l1acc=l1acc,
    )


@pytest.mark.blackhole_scale
@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(_ARGNAMES_V2, CONV_CASES_V2)
def test_conv2d_v2_model_grid(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    in_ch,
    out_ch,
    h,
    kernel,
    stride,
    pad,
    relu,
    bias,
    act_block_h,
    in_dt,
    in_rm,
    out_dt,
    out_rm,
    slices,
    cc,
    w_dt,
    l1acc,
):
    """Same convs pinned to the model's 20-core (5x4) height-sharded grid. [H]"""
    _run_conv(
        ttnn_mesh_device,
        hw,
        in_ch,
        out_ch,
        h,
        kernel,
        stride,
        pad,
        relu,
        bias,
        act_block_h,
        in_dt,
        in_rm,
        out_dt,
        out_rm,
        slices,
        cc,
        core_grid=U.model_core_range_set(ttnn_mesh_device),
        w_dt=w_dt,
        l1acc=l1acc,
    )


@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(_ARGNAMES_V2, LAT_CASES_V2)
def test_conv2d_v2_lat_auto(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    in_ch,
    out_ch,
    h,
    kernel,
    stride,
    pad,
    relu,
    bias,
    act_block_h,
    in_dt,
    in_rm,
    out_dt,
    out_rm,
    slices,
    cc,
    w_dt,
    l1acc,
):
    """Lateral 1x1 convs exactly as the model calls them: ``shard_layout=None`` (conv picks), bias, no ReLU.

    [H] K / N / per-core M, FUSE_BIAS, fp32 acc + packer_l1_acc, bfloat8_b weights and bias.
    [M] sharding: auto-chosen by conv; it matches the cache only on a 5x4 grid.
    """
    _run_conv(
        ttnn_mesh_device,
        hw,
        in_ch,
        out_ch,
        h,
        kernel,
        stride,
        pad,
        relu,
        bias,
        act_block_h,
        in_dt,
        in_rm,
        out_dt,
        out_rm,
        slices,
        cc,
        w_dt=w_dt,
        l1acc=l1acc,
        shard=None,
    )


@U.with_mesh_l1small(U.OFT_L1_SMALL_SIZE)
@pytest.mark.parametrize(_ARGNAMES_V2, SMALL_CASES_V2)
def test_conv2d_v2_small(
    ttnn_mesh_device,
    reset_seeds,
    hw,
    in_ch,
    out_ch,
    h,
    kernel,
    stride,
    pad,
    relu,
    bias,
    act_block_h,
    in_dt,
    in_rm,
    out_dt,
    out_rm,
    slices,
    cc,
    w_dt,
    l1acc,
):
    """Emulator-sized proxies of the v2 cases (act_block_h capped at 32 for the small H x W)."""
    _run_conv(
        ttnn_mesh_device,
        hw,
        in_ch,
        out_ch,
        h,
        kernel,
        stride,
        pad,
        relu,
        bias,
        act_block_h,
        in_dt,
        in_rm,
        out_dt,
        out_rm,
        slices,
        cc,
        w_dt=w_dt,
        l1acc=l1acc,
    )
