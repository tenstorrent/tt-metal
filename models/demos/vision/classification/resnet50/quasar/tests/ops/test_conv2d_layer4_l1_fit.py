# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone LAYER 4 L1-fit validation on the 2-core Quasar emulator (#48552).

GOAL: prove every distinct layer4 conv FITS in L1 and produces correct output, so layer4 can be wired into
the e2e model (layers 1-3 already pass). This is the pre-e2e gate for layer4, mirroring
test_conv2d_layer34_workarounds.py::test_quasar_conv2d_layer3_hs for layer3.

WHY LAYER 4 IS HARD (and how L1-fit is achieved):
  Layer4 convs cannot keep their FULL weight matrix ([full_K x full_N]) resident in L1 (layer4 conv2 is
  512->512 3x3 => full_K = 512*9/32 = 144 tiles, full_N = 16 tiles => ~4.6 MB > the L1 bank), and
  block-sharding them on 2 cores hits the fused conv_bmm_tilize_metal2 0x0119 hang. The L1-fit route is the
  HEIGHT_SHARDED **K-spill** path (the Quasar conv default -- i.e. WITHOUT TT_METAL_QSR_CONV_SPLIT_PROGRAM,
  which would force a single full-K block): the reduction dim K is sliced into blocks (act_block_w < full_K,
  num_blocks_act_w > 1) and the per-K-block partials are accumulated in DEST. Only ONE K-block of weights is
  L1-resident at a time, so the footprint is act_block_w*N instead of full_K*N -> it fits.

ASSUMPTION (explicit): that HEIGHT_SHARDED K-spill accumulation path routes through the Quasar matmul with
  weights streamed K-block-by-K-block -- exactly the capability validated by
  test_matmul_dram_weights_kspill.py. That matmul currently trips a HW tile-counter hazard (TILE_COUNTERS
  index 0x00010000 on the K-spill path), which has been handed to the HW/emulator team. This test is written
  ASSUMING that hazard is resolved. WORST CASE while it is pending: it can be masked by running with the
  matmul compute-kernel DPRINTs enabled (unset TT_METAL_LLK_ASSERTS; TT_METAL_DPRINT_CORES=all), which is a
  known-good masking configuration -- so this test is still runnable today for a functional (PCC) check.

WHAT EACH CASE EXERCISES:
  - The 3x3 convs (conv2, full_K = 144 tiles) are the true K-spill cases (num_blocks_act_w = filter_h = 3).
  - The 1x1 convs (conv1 / conv3 / downsample) are mm_conv matmuls; the large-N ones (conv3, downsample:
    N = 2048/32 = 64 tiles) are the large-weight cases that also depend on streaming weights rather than
    holding [K x full_N] resident.

Run (emulator, forced JIT):
  # assuming the K-spill matmul hazard is fixed:
  TT_METAL_FORCE_JIT_COMPILE=1 \
  TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
  pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer4_l1_fit.py

  # worst case, mask the pending K-spill hazard with the matmul DPRINTs:
  unset TT_METAL_LLK_ASSERTS
  TT_METAL_DPRINT_CORES=all TT_METAL_FORCE_JIT_COMPILE=1 \
  TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false, "enable_logging": true}' \
  pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer4_l1_fit.py

  -k conv2          # just the 3x3 K-spill convs
  -k downsample     # just the large-N 1x1 downsample
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc

# bf16 + LoFi accumulation over a deep K (up to 144) is noisy -> 0.98 (matches test_matmul_dram_weights_kspill).
PCC = 0.98
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
    act_block_h_override=32,
    pcc=PCC,
):
    """Run a single layer4 conv HEIGHT_SHARDED (the K-spill L1-fit route) and PCC-check vs torch."""
    kh, kw = kernel_size
    stride = (stride, stride) if isinstance(stride, int) else stride
    padding = (padding, padding) if isinstance(padding, int) else padding

    torch_input_nchw, torch_weight, torch_bias = _make_conv_tensors(
        in_channels, out_channels, input_height, input_width, kh, kw
    )
    torch_golden = _golden(torch_input_nchw, torch_weight, torch_bias, stride, padding)

    nhw = input_height * input_width
    flat = torch.permute(torch_input_nchw, (0, 2, 3, 1)).reshape(1, 1, nhw, in_channels).contiguous()

    # Pre-shard the activation into L1 (height-sharded) so conv2d takes the in-L1 (non-DRAM-slicing) path.
    # conv2d re-shards to its own optimal grid (reshard_if_not_optimal=True), so this shard split is only the
    # ingress layout, not the conv's compute grid.
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
        shard_layout=HS,
        reshard_if_not_optimal=True,
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
        return_output_dim=True,
        return_weights_and_bias=True,
        dtype=ttnn.bfloat16,
    )

    tt_out = ttnn.to_torch(ttnn.from_device(out)).reshape(1, oh, ow, -1)[:, :, :, :out_channels]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))
    assert_with_pcc(torch_golden, tt_out.float(), pcc=pcc)


# ---------------------------------------------------------------------------
# Every distinct layer4 conv shape (ResNet50 layer4 = 3 bottleneck modules).
#   (in_ch, out_ch, H, W, kernel, stride, pad)
# ---------------------------------------------------------------------------
# fmt: off
_LAYER4 = [
    # 1x1 reductions/expansions (mm_conv) -- large-N cases depend on streaming weights, not [K x full_N] resident
    (1024, 512,  14, 14, (1, 1), 1, 0),  # module1.conv1     (K=32)
    (2048, 512,  7,  7,  (1, 1), 1, 0),  # modules2-3.conv1  (K=64)
    (512,  2048, 7,  7,  (1, 1), 1, 0),  # conv3 expand      (N=64)
    (1024, 2048, 14, 14, (1, 1), 2, 0),  # module1.downsample (14->7, N=64)
    # 3x3 reductions (the true K-spill cases: full_K=144, num_blocks_act_w = filter_h = 3)
    (512,  512,  14, 14, (3, 3), 2, 1),  # module1.conv2 (14->7)
    (512,  512,  7,  7,  (3, 3), 1, 1),  # modules2-3.conv2 (7x7)
]
# fmt: on


def _id4(cfg):
    ic, oc, h, w, k, s, p = cfg
    tag = "conv2" if k == (3, 3) else ("downsample" if (ic, oc, s) == (1024, 2048, 2) else "conv1x1")
    return f"{tag}_{k[0]}x{k[1]}_{ic}to{oc}_s{s}_{h}x{w}_HS"


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("cfg", _LAYER4, ids=[_id4(c) for c in _LAYER4])
def test_quasar_conv2d_layer4_l1_fit(mesh_device, cfg):
    """Each distinct layer4 conv HEIGHT_SHARDED via the K-spill L1-fit route (see module docstring). PASS =>
    layer4 fits in L1 and is functionally correct, ready to wire into the e2e model."""
    # Designed for the 2-core Quasar emulator L1 budget (~3.7 MB/core). Layer4's weight-bound convs (full_K
    # K-spill residency) overflow WH's ~1.5 MB/core L1 (DFB region grows to 2.3-4.6 MB > 1.5 MB max). On WH,
    # layer4 runs via mainline ttnn.conv2d resharded across the full 8x8 grid, not this single-K-block K-spill
    # route. Quasar-emulator-scoped -> skip on WH (run it on Quasar).
    if is_wormhole_b0():
        pytest.skip("2-core Quasar-emulator layer4 K-spill L1-fit test; overflows WH's ~1.5 MB/core L1. Run on Quasar.")
    ic, oc, h, w, k, s, p = cfg
    _run_conv(
        mesh_device,
        in_channels=ic,
        out_channels=oc,
        input_height=h,
        input_width=w,
        kernel_size=k,
        stride=s,
        padding=p,
    )
