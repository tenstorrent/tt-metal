# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Pre-full-model validation for the layer3/layer4 bring-up workarounds on the 2-core Quasar emulator (#48552).

Two independent workarounds are exercised here so they can be validated in ISOLATION before the e2e model run:

  1. LAYER 3 -> HEIGHT_SHARDED (test_quasar_conv2d_layer3_hs).
     Layer3 block-sharded conv2 hits the fused conv_bmm_tilize_metal2 0x0119 hang (first DEST bank reuse). Layer3
     weights are small enough to fit HEIGHT_SHARDED (per-core N * act_block_w(per-filter-row K) stays under the
     DFB/L1 budget -- unlike layer4), so HEIGHT_SHARDED both fits AND routes off the fused path. These cases
     should PASS.

  2. LAYER 4 -> DRAM height-slicing (test_quasar_conv2d_layer4_dram_sliced).
     Layer4 cannot be HEIGHT_SHARDED (full-N weights ~4.7 MB > the 4 MB L1 bank) and block-sharded on 2 cores hits
     the same fused 0x0119 hang. This test routes layer4 convs through the DRAM output-height-slicing path
     (slice_config = Conv2dSliceConfig(Conv2dDRAMSliceHeight, num_slices)), which re-enters conv2d_L1 per
     output-height slice.
     CAVEAT (this is the open question this test answers): DRAM HEIGHT-slicing shrinks the per-slice ACTIVATION
     footprint, NOT the weights (full-N x K resident in every slice) and NOT the per-slice sub-conv's shard
     scheme. So if layer4's blocker is the weights fit or the fused-conv hang, height-slicing will NOT resolve it
     (expect FATAL or hang), and layer4 needs a larger grid (num_cores_c >= 4) or the fused-hang LLK fix instead.
     If it PASSES, DRAM slicing is a viable layer4 route and we wire it into the model.

Run (craq-sim / emulator, forced JIT):
  TT_METAL_FORCE_JIT_COMPILE=1 \
  TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
  pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer34_workarounds.py

  -k layer3_hs           # just the layer3 HEIGHT_SHARDED workaround
  -k layer4_dram_sliced  # just the layer4 DRAM-slicing experiment
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc

PCC = 0.999
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _golden(torch_input_nchw, torch_weight, torch_bias, stride, padding):
    return torch.nn.functional.conv2d(
        torch_input_nchw, torch_weight, bias=torch_bias.reshape(-1), stride=stride, padding=padding
    )


def _make_conv_tensors(in_channels, out_channels, input_height, input_width, kh, kw):
    torch.manual_seed(0)
    torch_input_nchw = torch.randn((1, in_channels, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_channels, in_channels, kh, kw), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((1, 1, 1, out_channels), dtype=torch.bfloat16).float()
    return torch_input_nchw, torch_weight, torch_bias


def _run_conv(
    device,
    *,
    in_channels,
    out_channels,
    input_height,
    input_width,
    kernel_size,
    stride,
    padding,
    shard_layout,
    input_in_dram,
    slice_config=None,
    act_block_h_override=32,
    reshard_if_not_optimal=True,
    pcc=PCC,
):
    kh, kw = kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    torch_input_nchw, torch_weight, torch_bias = _make_conv_tensors(
        in_channels, out_channels, input_height, input_width, kh, kw
    )
    torch_golden = _golden(torch_input_nchw, torch_weight, torch_bias, stride, padding)

    nhw = input_height * input_width
    flat = torch.permute(torch_input_nchw, (0, 2, 3, 1)).reshape(1, 1, nhw, in_channels).contiguous()

    if input_in_dram:
        # DRAM-interleaved input -> conv2d takes the DRAM (slicing) path when slice_config is set.
        tt_input = ttnn.from_torch(
            flat,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        # Pre-shard into L1 (height-sharded) so conv2d takes the in-L1 path.
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
        batch_size=1,
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
        slice_config=slice_config,
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )

    tt_out = ttnn.to_torch(ttnn.from_device(out)).reshape(1, oh, ow, -1)[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))
    assert_with_pcc(torch_golden, tt_out.float(), pcc=pcc)


# ---------------------------------------------------------------------------
# Workaround 1: LAYER 3 -> HEIGHT_SHARDED (should PASS; fits + off the fused path)
#   (in_ch, out_ch, H, W, kernel, stride, pad)  -- every distinct layer3 conv shape
# ---------------------------------------------------------------------------
# fmt: off
_LAYER3_HS = [
    (512, 256, 28, 28, (1, 1), 1, 0),   # layer3_module1.conv1
    (256, 256, 28, 28, (3, 3), 2, 1),   # layer3_module1.conv2 (28->14)  <- the fused-0x19 case, dodged by HS
    (256, 1024, 14, 14, (1, 1), 1, 0),  # layer3.conv3 (expand)
    (1024, 256, 14, 14, (1, 1), 1, 0),  # layer3.conv1 (modules 2-6)
    (256, 256, 14, 14, (3, 3), 1, 1),   # layer3.conv2 (modules 2-6)
    # NOTE: layer3_module1.downsample (512->1024 s2 @28x28) is NOT here -- HEIGHT_SHARDED N-HALVES it (device out
    # = 512 ch, not 1024): the stride-2 1x1 downsample N-halving bug (NOT a DFB-ring overflow; ring is uint32 and
    # its weights are ~1MB). It stays BLOCK_SHARDED in the model; validated separately below.
]
# fmt: on


def _id3(cfg):
    ic, oc, h, w, k, s, p = cfg
    return f"{k[0]}x{k[1]}_{ic}to{oc}_s{s}_{h}x{w}_HS"


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("cfg", _LAYER3_HS, ids=[_id3(c) for c in _LAYER3_HS])
def test_quasar_conv2d_layer3_hs(mesh_device, cfg):
    """Layer3 convs forced HEIGHT_SHARDED (the workaround that both fits and dodges the fused conv_bmm 0x19)."""
    ic, oc, h, w, k, s, p = cfg
    # WH-only HANG on the 3x3 (conv2) cases: this test does NOT set the split env, so HEIGHT_SHARDED still runs
    # the FUSED conv_bmm_tilize (the "dodges the fused path" only holds with TT_METAL_QSR_CONV_SPLIT_PROGRAM set,
    # as the model does). On WH the fused kernel hits the fast_tilize->matmul cadence race (kRaceGuardSpin family;
    # MATH MWDD in matmul_block, PACK in program_packer_destination, SyncHalf, cb stuck; genuine, asserts-on) --
    # here even at act_block_h=32 (K=72), confirming it's inherent to the tilize<->matmul interleave, not abh.
    # Same family as relu_now_sfpu/stem_7x7. The 1x1 cases use the mm_conv (matmul-only) path -> left to run.
    if tuple(k) == (3, 3) and is_wormhole_b0():
        pytest.xfail(
            "WH fused conv_bmm_tilize fast_tilize->matmul cadence race (kRaceGuardSpin family) on layer3 3x3 HS "
            "(no split env -> fused path). Same family as relu_now_sfpu/stem_7x7; not the WH model path."
        )
    _run_conv(
        mesh_device,
        in_channels=ic,
        out_channels=oc,
        input_height=h,
        input_width=w,
        kernel_size=k,
        stride=s,
        padding=p,
        shard_layout=HS,
        input_in_dram=False,
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_quasar_conv2d_layer3_downsample_bs(mesh_device):
    """layer3_module1 downsample (512->1024 s2 @28x28) BLOCK_SHARDED -- the ONLY layer3 conv that can't be HS
    (HS N-halves it). Its weights fit block-sharded on 2 cores (~512 KB). This validates whether block-sharded
    produces CORRECT output (1024 ch) without hitting the fused conv_bmm 0x19. If it PASSES, layer3 is fully
    functional (5 HS convs + this 1 block downsample). If it hangs/N-halves, the downsample needs option A."""
    _run_conv(
        mesh_device,
        in_channels=512,
        out_channels=1024,
        input_height=28,
        input_width=28,
        kernel_size=(1, 1),
        stride=2,
        padding=0,
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        input_in_dram=False,
    )


# ---------------------------------------------------------------------------
# Workaround 2: LAYER 4 -> DRAM height-slicing (EXPERIMENT; see module docstring caveat)
#   (in_ch, out_ch, H, W, kernel, stride, pad, num_slices, shard_layout)
# ---------------------------------------------------------------------------
# fmt: off
_LAYER4_DRAM = [
    (512,  512,  14, 14, (3, 3), 2, 1, 2, HS),   # layer4_module1.conv2 (14->7)
    (512,  512,  7,  7,  (3, 3), 1, 1, 2, HS),   # layer4.conv2 (modules 2-3)
    (1024, 512,  14, 14, (1, 1), 1, 0, 2, HS),   # layer4.conv1
    (512,  2048, 7,  7,  (1, 1), 1, 0, 2, HS),   # layer4.conv3 (expand)
    (1024, 2048, 14, 14, (1, 1), 2, 0, 2, HS),   # layer4_module1.downsample (14->7)
]
# fmt: on


def _id4(cfg):
    ic, oc, h, w, k, s, p, ns, _ = cfg
    return f"{k[0]}x{k[1]}_{ic}to{oc}_s{s}_{h}x{w}_dram{ns}"


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("cfg", _LAYER4_DRAM, ids=[_id4(c) for c in _LAYER4_DRAM])
def test_quasar_conv2d_layer4_dram_sliced(mesh_device, cfg):
    """Layer4 convs via the DRAM output-height-slicing path. See the module docstring: this ANSWERS whether DRAM
    slicing is a viable layer4 route (it only helps activation footprint, not the weights fit or the fused hang)."""
    # 2-core Quasar-emulator experiment: DRAM slicing only trims the activation footprint, NOT the layer4 weight
    # residency, which overflows WH's ~1.5 MB/core L1 (DFB region 2.3-4.6 MB > 1.5 MB max). WH runs layer4 via
    # mainline conv2d resharded across the full grid. Quasar-emulator-scoped -> skip on WH (run on Quasar).
    if is_wormhole_b0():
        pytest.skip(
            "2-core Quasar-emulator layer4 DRAM-slicing experiment; layer4 weight residency overflows WH L1. Run on Quasar."
        )
    ic, oc, h, w, k, s, p, num_slices, shard = cfg
    slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=num_slices)
    _run_conv(
        mesh_device,
        in_channels=ic,
        out_channels=oc,
        input_height=h,
        input_width=w,
        kernel_size=k,
        stride=s,
        padding=p,
        shard_layout=shard,
        input_in_dram=True,
        slice_config=slice_config,
    )
