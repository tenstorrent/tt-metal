# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
FAST repro of the model's layer1/2 conv2 fault (ERROR_TRISC0 / UNPACK 0x19 in the split tilize).

WHY THIS EXISTS
---------------
test_conv2d_layer_conv2.py's layer1/layer2 (height-sharded split) PASS, but the FULL MODEL faults at
layer1_module2.conv2 after ~28 min. Same conv shape + same height-sharded split -> the trigger is a CONFIG
difference. The model's conv2 (ttnn_functional_resnet50.py:517-543) differs from the passing standalone test
in three ways:
    (1) act_block_h_override=32      <-- prime suspect: changes tilize block geometry
    (2) reallocate_halo_output=False
    (3) reshard_if_not_optimal=False (module2 default) vs True in the standalone

This test mirrors the model conv2 config and A/B toggles act_block_h_override (0 vs 32) so ONE quick run
isolates the trigger: if abh=0 passes and abh=32 faults, act_block_h_override=32 is the cause. Iterate the
LLK backport / any fix here (seconds) instead of the 28-min full model.

RUN (emulator, forced JIT):
  TT_METAL_QSR_TC_ISOLATE=1 TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer_conv2_modelcfg.py
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# (layer, in_ch, out_ch, spatial, id) — height-sharded 3x3 conv2 of layer1/2 (the split-path convs).
# NOTE: layer3 also uses the height split (f6b15a lets its K=72/N=8 weights fit) but its 14x14=196 output
# isn't tile-aligned so this test's naive reshape can't read it back -> validate layer3 via the clean
# matmul probe test_linear.py::test_layer3_conv2_matmul_wide_ring instead. layer4 (K=144/N=16 ~4.7MB) exceeds
# L1 so it can't do the height split at all (stays block-sharded / fused).
LAYER_CONV2 = [
    (1, 64, 64, 56, "layer1"),
    (2, 128, 128, 28, "layer2"),
]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("act_block_h_override", [0, 32], ids=["abh0", "abh32"])
@pytest.mark.parametrize("layer, in_ch, out_ch, spatial, tid", LAYER_CONV2, ids=[c[-1] for c in LAYER_CONV2])
def test_layer_conv2_modelcfg(mesh_device, layer, in_ch, out_ch, spatial, tid, act_block_h_override, monkeypatch):
    """layer1/2 conv2 with the MODEL's conv2 config. abh0 should pass (like the standalone test); abh32 mirrors
    the model. If abh32 faults with UNPACK 0x19, act_block_h_override=32 is the trigger."""
    device = mesh_device
    torch.manual_seed(0)

    # Force the split for the height-sharded conv2 (same as the model command).
    monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")  # monkeypatch auto-restores after the test

    batch = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    input_height = input_width = spatial

    torch_input = torch.randn((batch, in_ch, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_ch, in_ch, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((out_ch,), dtype=torch.bfloat16).float()
    torch_golden = torch.relu(
        torch.nn.functional.conv2d(torch_input, torch_weight, bias=torch_bias, stride=stride, padding=padding)
    )

    nhw = batch * input_height * input_width
    flat = torch.permute(torch_input, (0, 2, 3, 1)).reshape(1, 1, nhw, in_ch).contiguous()
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = max(c for c in range(1, max_cores + 1) if nhw % c == 0)
    shard_h = nhw // num_cores
    core_grid = ttnn.num_cores_to_corerangeset(num_cores, grid, True)
    in_mem = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_h, in_ch),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    tt_input = ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT).to(device, in_mem)
    tt_weight = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16)
    tt_bias = ttnn.from_torch(torch_bias.reshape(1, 1, 1, out_ch), dtype=ttnn.bfloat16)

    # Mirror the model's conv2 Conv2dConfig (ttnn_functional_resnet50.py:529-542) as closely as a standalone
    # test can: RELU, deallocate_activation, reallocate_halo_output=False, act_block_h_override, HEIGHT shard.
    # (reshard_if_not_optimal kept True so the fresh input reshards to the split's height layout; if abh32
    #  alone does NOT repro, flip this to False and pre-shard to match the model's no-reshard path.)
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=False,
        act_block_h_override=act_block_h_override,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    out, [oh, ow], _wb = ttnn.experimental.quasar.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        in_channels=in_ch,
        out_channels=out_ch,
        batch_size=batch,
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

    tt_out = ttnn.to_torch(ttnn.from_device(out))
    tt_out = tt_out.reshape(batch, oh, ow, tt_out.shape[-1])[:, :, :, :out_ch]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))
    assert_with_pcc(torch_golden, tt_out, 0.99)
