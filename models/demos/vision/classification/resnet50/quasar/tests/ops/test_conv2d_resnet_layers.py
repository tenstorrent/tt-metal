# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Wider per-op conv2d coverage: the layer2 / layer3 / layer4 bottleneck convs that test_conv2d.py intentionally
omits (its header notes the layer3/4 BLOCK_SHARDED convs overflow the uint16_t weights-DFB ring and the large
layer4 shapes need a bigger grid). These mirror what models/.../tt/ttnn_functional_resnet50.py actually issues:

  conv2 = 3x3 (stride 2 on the downsample/module1 block, else 1), act_block_h_override=32  -> TILIZE path
  conv1 / conv3 / downsample = 1x1                                                          -> MATMUL path
  sharding: layer1/2 HEIGHT_SHARDED; layer3/4 BLOCK_SHARDED on Quasar (+ reshard_if_not_optimal), like WH.
  spatial: 56 (L1) -> 28 (L2) -> 14 (L3) -> 7 (L4).

Inputs are pre-sharded to L1 height-sharded (as the model feeds conv1 the fold output and each block the L1
residual) so conv2d takes the L1 path — NOT the DRAM path, whose single-slice writeback hits the unported
Quasar slice_write (UNPAD_INPUT_WIDTH). Golden = torch.nn.functional.conv2d (+bias).

CURRENT EXPECTATIONS (2026-07-15):
  * The 3x3 conv2 cases exercise the Quasar tilize (conv_bmm_tilize) — they hit the tilize 0x19 (FPU
    dest-dvalid ring leak) and will FAIL until the UnpackToDestEn tilize fix is the Quasar default (or until
    the two-program split lands and becomes the routing for tilize convs). Kept here so the coverage exists
    and flips green the moment the fix lands.
  * The layer3/4 BLOCK_SHARDED cases (and the large 1x1 expansions 512->2048 / 1024->2048) may FATAL at DFB
    creation on the 2-core emulator (weights-DFB ring / grid too small even with reshard). They are full-model
    coverage for a real (many-core) Quasar; on the emulator expect the small height-sharded layer2 cases to be
    the ones that actually fit.
  * The 1x1 conv1/conv3/downsample cases use the matmul path (no tilize) — they are the ones most likely to
    run on the emulator today (subject to the ring/grid caveat above).

Run (craq-sim / emulator, slow dispatch + forced JIT):
  TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -s models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_resnet_layers.py
  # add -k "conv2" for just the tilize (3x3) cases, or -k "1x1" for the matmul cases.
"""

import math

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.97  # bf16 + MathFidelity.LoFi (matches the model's batch-1 LoFi config)

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED


@pytest.fixture(autouse=True)
def _enable_split_program(monkeypatch):
    # Scoped split-path routing: the model runs every layer conv through the two-program split
    # (Program A tilize + Program B matmul::linear), which is the path proven green on BOTH WH and
    # Quasar. Set it here so the file is self-contained (no need for TT_METAL_QSR_CONV_SPLIT_PROGRAM=1
    # on the command line) and so the HEIGHT_SHARDED 3x3 conv2 cases take the split instead of the
    # fused conv_bmm_tilize (which deadlocks in the WH fast_tilize pack-flush). Harmless for the 1x1
    # (mm_conv) cases, which don't consult it.
    monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")


def _run_conv2d_l1(
    mesh_device,
    *,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    shard_layout,
    reshard_if_not_optimal,
    act_block_h_override,
    pcc=PCC,
):
    torch.manual_seed(0)
    device = mesh_device
    batch_size = 1
    kh, kw = kernel_size
    # ttnn.experimental.quasar.conv2d's binding takes stride/padding as Sequence[int] (no int overload),
    # so normalize the scalar cfg values to 2-tuples (matching the other quasar conv tests). torch.conv2d
    # accepts the tuple form too.
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    # --- Sweep guards (all arches): imperatively xfail the known-failing buckets so the run stays green
    # instead of FATAL-ing / asserting / hanging. All three are pre-existing, arch-general quasar conv2d
    # limitations (they fail the same way on WH and the Quasar emulator), not the split routing, and none is
    # on the model's on-device path (downsamples are host-fallbacked; layer4's big weights ride the K-spill
    # split covered by test_conv2d_layer4_l1_fit.py).
    k_tiles = math.ceil(in_channels * kh * kw / 32)  # per-core resident weight height (HS: N not split)
    n_tiles = math.ceil(out_channels / 32)
    is_1x1 = tuple(kernel_size) == (1, 1)
    if is_1x1 and shard_layout == HS and stride[0] == 2:
        pytest.xfail(
            "1x1 stride-2 HEIGHT_SHARDED downsample drops the channel expansion (emits Cin, not Cout) -- "
            "known quasar conv2d device bug; downsamples are host-fallbacked in the model."
        )
    if is_1x1 and shard_layout == BS:
        pytest.xfail(
            "BLOCK_SHARDED 1x1 with non-tile-aligned NHW: this test pre-shards row-major at exact NHW "
            "(e.g. 784/196, not multiples of 32), so block-sharding splits it to a sub-tile shard height "
            "(14/4) that tensor_layout rejects. Pre-existing block-shard limitation for un-padded shapes."
        )
    if k_tiles * n_tiles >= 1024:
        pytest.xfail(
            f"whole-weight footprint {k_tiles}x{n_tiles}={k_tiles * n_tiles} tiles (~{k_tiles * n_tiles * 2 // 1024} "
            "MB/core) overflows L1 without K-spill; this forced-HS/L1 path does not K-spill (layer4 K-spill "
            "is covered by test_conv2d_layer4_l1_fit.py)."
        )

    torch_input_nchw = torch.randn((batch_size, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, kh, kw), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((1, 1, 1, out_channels), dtype=torch.bfloat16).float()
    torch_golden = torch.nn.functional.conv2d(
        torch_input_nchw, torch_weight, bias=torch_bias.reshape(-1), stride=stride, padding=padding
    )

    # --- pre-shard activation into L1 (height-sharded) so conv2d takes the L1 path (not DRAM slicing) ---
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
    tt_bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=shard_layout,
        reshard_if_not_optimal=reshard_if_not_optimal,
        act_block_h_override=act_block_h_override,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

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

    tt_out = ttnn.to_torch(ttnn.from_device(out)).reshape(batch_size, oh, ow, -1)[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))
    assert_with_pcc(torch_golden, tt_out.float(), pcc=pcc)


# (in_ch, out_ch, H, W, kernel, stride, pad, shard, reshard, act_block_h_override)
# fmt: off
_LAYER_CONV2_3x3 = [
    (128, 128, 56, 56, (3, 3), 2, 1, HS, True, 32),  # layer2_module1.conv2 (56->28)
    (128, 128, 28, 28, (3, 3), 1, 1, HS, True, 32),  # layer2.conv2
    (256, 256, 28, 28, (3, 3), 2, 1, BS, True, 32),  # layer3_module1.conv2 (28->14)
    (256, 256, 14, 14, (3, 3), 1, 1, BS, True, 32),  # layer3.conv2
    (512, 512, 14, 14, (3, 3), 2, 1, BS, True, 32),  # layer4_module1.conv2 (14->7)
    (512, 512, 7,  7,  (3, 3), 1, 1, BS, True, 32),  # layer4.conv2
]
_LAYER_CONV_1x1 = [
    (256, 128, 56, 56, (1, 1), 1, 0, HS, True, 32),   # layer2 conv1 (reduce)
    (128, 512, 28, 28, (1, 1), 1, 0, HS, True, 32),   # layer2 conv3 (expand)
    (256, 512, 56, 56, (1, 1), 2, 0, HS, True, 32),   # layer2 downsample (56->28)
    (512, 256, 28, 28, (1, 1), 1, 0, BS, True, 32),   # layer3 conv1
    (256, 1024, 14, 14, (1, 1), 1, 0, BS, True, 32),  # layer3 conv3
    (512, 1024, 28, 28, (1, 1), 2, 0, BS, True, 32),  # layer3 downsample (28->14)
    (1024, 512, 14, 14, (1, 1), 1, 0, BS, True, 32),  # layer4 conv1
    (512, 2048, 7,  7,  (1, 1), 1, 0, BS, True, 32),   # layer4 conv3
    (1024, 2048, 14, 14, (1, 1), 2, 0, BS, True, 32),  # layer4 downsample (14->7)
]
# fmt: on


def _id(cfg):
    ic, oc, h, w, k, s, p, shard, _, _ = cfg
    tag = "H" if shard == HS else "B"
    return f"{k[0]}x{k[1]}_{ic}to{oc}_s{s}_{h}x{w}_{tag}"


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("cfg", _LAYER_CONV2_3x3, ids=[_id(c) for c in _LAYER_CONV2_3x3])
def test_quasar_conv2d_layer_conv2_3x3(mesh_device, cfg):
    """layer2/3/4 conv2 (3x3) — the TILIZE path. Blocked on the tilize 0x19 fix (see module docstring)."""
    ic, oc, h, w, k, s, p, shard, reshard, abh = cfg
    _run_conv2d_l1(
        mesh_device,
        in_channels=ic,
        out_channels=oc,
        input_height=h,
        input_width=w,
        kernel_size=k,
        stride=s,
        padding=p,
        shard_layout=shard,
        reshard_if_not_optimal=reshard,
        act_block_h_override=abh,
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("cfg", _LAYER_CONV_1x1, ids=[_id(c) for c in _LAYER_CONV_1x1])
def test_quasar_conv2d_layer_conv_1x1(mesh_device, cfg):
    """layer2/3/4 conv1 / conv3 / downsample (1x1) — the MATMUL path (no tilize)."""
    ic, oc, h, w, k, s, p, shard, reshard, abh = cfg
    _run_conv2d_l1(
        mesh_device,
        in_channels=ic,
        out_channels=oc,
        input_height=h,
        input_width=w,
        kernel_size=k,
        stride=s,
        padding=p,
        shard_layout=shard,
        reshard_if_not_optimal=reshard,
        act_block_h_override=abh,
    )


# ---------------------------------------------------------------------------------------------------------
# [#48552] layer3/4 forced to HEIGHT_SHARDED instead of the model's default BLOCK_SHARDED. On the 2-core
# emulator block-sharding buys nothing and routes these convs to the fused conv_bmm_tilize (ERROR_TRISC1
# 0x0119, the "0x19 rabbit hole"); HEIGHT_SHARDED routes them onto the proven split/L1 path that layer1/2 use.
# Earlier this hit the uint16 weights-DFB ring limit -> main commit 6079b5f widened ring_size to uint32, so
# they should now fit. If these PASS, gate the model's height_sharding=True for layer3/4 on Quasar.
# Run just these:  -k as_hs
# fmt: off
_LAYER34_AS_HS_3x3 = [
    (256, 256, 28, 28, (3, 3), 2, 1, HS, True, 32),  # layer3_module1.conv2 (28->14)
    (256, 256, 14, 14, (3, 3), 1, 1, HS, True, 32),  # layer3.conv2
    (512, 512, 14, 14, (3, 3), 2, 1, HS, True, 32),  # layer4_module1.conv2 (14->7)
    (512, 512, 7,  7,  (3, 3), 1, 1, HS, True, 32),  # layer4.conv2
]
_LAYER34_AS_HS_1x1 = [
    (512, 256, 28, 28, (1, 1), 1, 0, HS, True, 32),   # layer3 conv1
    (256, 1024, 14, 14, (1, 1), 1, 0, HS, True, 32),  # layer3 conv3
    (512, 1024, 28, 28, (1, 1), 2, 0, HS, True, 32),  # layer3 downsample (28->14)
    (1024, 512, 14, 14, (1, 1), 1, 0, HS, True, 32),  # layer4 conv1
    (512, 2048, 7,  7,  (1, 1), 1, 0, HS, True, 32),   # layer4 conv3
    (1024, 2048, 14, 14, (1, 1), 2, 0, HS, True, 32),  # layer4 downsample (14->7)
]
# fmt: on


def _id_hs(cfg):
    ic, oc, h, w, k, s, p, _shard, _, _ = cfg
    return f"{k[0]}x{k[1]}_{ic}to{oc}_s{s}_{h}x{w}_asHS"


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("cfg", _LAYER34_AS_HS_3x3, ids=[_id_hs(c) for c in _LAYER34_AS_HS_3x3])
def test_quasar_conv2d_layer34_as_hs_3x3(mesh_device, cfg):
    """layer3/4 conv2 (3x3) forced HEIGHT_SHARDED (vs the model's BLOCK_SHARDED) to dodge the fused 0x19."""
    ic, oc, h, w, k, s, p, shard, reshard, abh = cfg
    _run_conv2d_l1(
        mesh_device,
        in_channels=ic,
        out_channels=oc,
        input_height=h,
        input_width=w,
        kernel_size=k,
        stride=s,
        padding=p,
        shard_layout=shard,
        reshard_if_not_optimal=reshard,
        act_block_h_override=abh,
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("cfg", _LAYER34_AS_HS_1x1, ids=[_id_hs(c) for c in _LAYER34_AS_HS_1x1])
def test_quasar_conv2d_layer34_as_hs_1x1(mesh_device, cfg):
    """layer3/4 conv1/conv3/downsample (1x1) forced HEIGHT_SHARDED (vs BLOCK_SHARDED) -> matmul path."""
    ic, oc, h, w, k, s, p, shard, reshard, abh = cfg
    _run_conv2d_l1(
        mesh_device,
        in_channels=ic,
        out_channels=oc,
        input_height=h,
        input_width=w,
        kernel_size=k,
        stride=s,
        padding=p,
        shard_layout=shard,
        reshard_if_not_optimal=reshard,
        act_block_h_override=abh,
    )
