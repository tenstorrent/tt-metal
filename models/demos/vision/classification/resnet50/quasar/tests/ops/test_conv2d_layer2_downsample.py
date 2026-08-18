# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone repro of the resnet50/quasar layer2_module1 DOWNSAMPLE (the 1x1 shortcut projection).

WHY: the full model now clears stem -> maxpool -> ALL of layer1 -> layer2_module1 conv1/conv2/conv3, then
FATALs at the residual reshard `to_memory_config(ds_out, out.memory_config())`:
    "Shard width 512 must match physical width 256 for height sharded"
because the downsample output logged as (1,1,832,256) -- 256 channels -- while conv3 is (1,1,784,512).
The downsample is configured 256->512 (ds_conv_output_channels = weight.shape[0] = 512) but produced 256ch.
This isolates that exact op (1x1, 256->512, stride 2, 56x56 input, HEIGHT_SHARDED, verbatim from
run_downsample_if_req) so we can confirm the conv is dropping the channel expansion (out_ch=256 not 512)
without a full-model run. layer1's downsample was stride-1 64->256 and matched conv3 fine; this is the first
stride-2 channel-EXPANDING downsample.

RUN (emulator, forced JIT):
  TT_METAL_QSR_TC_ISOLATE=1 TT_METAL_QSR_CONV_SPLIT_PROGRAM=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer2_downsample.py
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "shard",
    [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED],
    ids=["height", "block"],
)
def test_quasar_layer2_module1_downsample(mesh_device, shard, monkeypatch):
    """layer2_module1 downsample: 1x1, in=256, out=512, stride 2, input 56x56. The output MUST be 512
    channels; if it comes back 256 the conv is dropping the channel expansion. The model uses HEIGHT here
    (input_height 56 != 28); the 'block' case tests whether BLOCK sharding gets the channel count right (if
    so, the fix is to force BLOCK for this stride-2 channel-expanding downsample)."""
    # Known layer-downsample bugs (same pair xfailed in test_conv2d_resnet_layers): HEIGHT drops the 1x1-stride-2
    # channel expansion (last-dim 256 vs 512); BLOCK is numerically wrong (PCC ~0.75). Arch-general device bugs;
    # downsamples are host-fallbacked in the model, so off the on-device path.
    if is_wormhole_b0():
        pytest.xfail(
            "known layer2 downsample bugs: HEIGHT drops the 1x1-s2 channel expansion (256 vs 512); BLOCK PCC ~0.75. "
            "Same as the test_conv2d_resnet_layers downsample xfails; downsamples host-fallbacked in the model."
        )
    device = mesh_device
    torch.manual_seed(0)

    monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")  # monkeypatch auto-restores after the test

    batch = 1
    in_ch = 256
    out_ch = 512
    input_height = input_width = 56  # layer2_module1 downsample input (layer1 output spatial)
    kernel_size = (1, 1)
    stride = (2, 2)
    padding = (0, 0)

    torch_input = torch.randn((batch, in_ch, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_ch, in_ch, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((out_ch,), dtype=torch.bfloat16).float()
    torch_golden = torch.nn.functional.conv2d(
        torch_input, torch_weight, bias=torch_bias, stride=stride, padding=padding
    )  # no activation on the downsample (RELU is fused in the residual add_)

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

    # Verbatim from run_downsample_if_req (input_height 56 != 28 -> HEIGHT_SHARDED).
    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=shard,
        deallocate_activation=True,
        reallocate_halo_output=True,
        act_block_h_override=32,
        reshard_if_not_optimal=True,
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
    # The whole point: assert the op kept all 512 output channels.
    assert tt_out.shape[-1] >= out_ch, f"downsample dropped channels: got last-dim {tt_out.shape[-1]}, want >= {out_ch}"
    tt_out = tt_out.reshape(batch, oh, ow, tt_out.shape[-1])[:, :, :, :out_ch]
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))  # NHWC -> NCHW
    assert_with_pcc(torch_golden, tt_out.float(), 0.99)
