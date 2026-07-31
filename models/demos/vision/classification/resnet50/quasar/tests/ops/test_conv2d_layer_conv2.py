# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone tests for the FOUR resnet50 layer conv2 ops (the 3x3 bottleneck convs) — one per layer.

WHY THESE FOUR
--------------
Every layer's 3x3 conv2 routes to the Quasar fused conv_bmm_tilize kernel, which hits a DEST-handshake
ERROR_TRISC1 0x19 (tilize<->matmul share DEST inside one program; the matmul's 2nd subblock faults). The
fix is the SPLIT path: Program A (tilize-only) + Program B (matmul::linear) run as SEPARATE device programs,
so each starts clean (its own compute_kernel_hw_startup) and there is no shared-DEST corruption. The
standalone matmul (Program B) is already proven (the stem split passes).

Sharding per layer (matches what the model uses with the split routing):
  layer1: 64->64,   56x56, K=18  -> HEIGHT_SHARDED  (weights 72 KB fit; height-sharded split works today)
  layer2: 128->128, 28x28, K=36  -> HEIGHT_SHARDED  (288 KB fit)
  layer3: 256->256, 14x14, K=72  -> BLOCK_SHARDED   (1.15 MB overflows height-sharded single-K-block ->
                                                     needs Program B as the 2D mcast matmul)
  layer4: 512->512, 7x7,   K=144 -> BLOCK_SHARDED   (4.6 MB -> block)
(module2+ shape: stride 1, pad 1. RELU + folded-BN bias mirrors the model's conv2.)

STATUS: layer1/layer2 use the (validated) height-sharded SPLIT path (Program A tilize-only + Program B
matmul) and PASS. layer3/layer4 are BLOCK_SHARDED, which fundamentally splits K across grid columns and
reduces across them (in0_num_blocks_w>1) -> incompatible with the single-K-block split -> they run the
ORIGINAL fused conv_bmm_tilize kernel and hit the Quasar DEST-handshake ERROR_TRISC1 0x19 (the tilize<->
matmul shared-DEST fault localized to the matmul's 2nd subblock). layer3/layer4 are the LLK-team repro of
that fused-conv 0x19; layer1/layer2 are the working reference. (SPLIT_PROGRAM is set ONLY for the
height-sharded split cases; the block cases run the pristine original block conv.)

RUN one layer (emulator, forced JIT):
  TT_METAL_QSR_TC_ISOLATE=1 TT_METAL_FORCE_JIT_COMPILE=1 \
    pytest -q models/demos/vision/classification/resnet50/quasar/tests/ops/test_conv2d_layer_conv2.py -k layer1
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# (layer, in_ch, out_ch, spatial, shard_layout, id) — the 3x3 conv2 of each resnet50 layer (module2+, s1, p1).
LAYER_CONV2 = [
    (1, 64, 64, 56, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, "layer1"),
    (2, 128, 128, 28, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, "layer2"),
    (3, 256, 256, 14, ttnn.TensorMemoryLayout.BLOCK_SHARDED, "layer3"),
    (4, 512, 512, 7, ttnn.TensorMemoryLayout.BLOCK_SHARDED, "layer4"),
]


@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "layer, in_ch, out_ch, spatial, shard_layout, tid", LAYER_CONV2, ids=[c[-1] for c in LAYER_CONV2]
)
def test_quasar_layer_conv2(mesh_device, layer, in_ch, out_ch, spatial, shard_layout, tid, monkeypatch):
    """One resnet50 layer's 3x3 conv2. layer1/2 (height-sharded) take the split path and must PCC ~1.0;
    layer3/4 (block-sharded) run the original fused conv_bmm_tilize and are the LLK 0x19 repro."""
    device = mesh_device
    torch.manual_seed(0)

    batch = 1
    kernel_size = (3, 3)
    stride = (1, 1)
    padding = (1, 1)
    input_height = input_width = spatial  # stride1 + pad1 + k3 keeps H, W

    torch_input = torch.randn((batch, in_ch, input_height, input_width), dtype=torch.bfloat16).float()
    torch_weight = torch.randn((out_ch, in_ch, *kernel_size), dtype=torch.bfloat16).float()
    torch_bias = torch.randn((out_ch,), dtype=torch.bfloat16).float()  # folded-BN bias + RELU, like the model
    torch_golden = torch.relu(
        torch.nn.functional.conv2d(torch_input, torch_weight, bias=torch_bias, stride=stride, padding=padding)
    )

    # Pre-shard the activation HEIGHT_SHARDED into L1; the conv reshards to shard_layout (reshard_if_not_optimal).
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

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=shard_layout,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, packer_l1_acc=True
    )

    # Height-sharded (layer1/2): engage the split path (Program A tilize-only + Program B matmul). Block-sharded
    # (layer3/4): run the ORIGINAL fused block conv (conv_bmm_tilize) -- block-sharding needs cross-column
    # K-reduction, incompatible with the single-K-block split -- so DO NOT set SPLIT_PROGRAM (this is the
    # fused-conv 0x19 repro for the LLK team).
    # monkeypatch auto-restores both branches after the test (no cross-test env leakage).
    if shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        monkeypatch.setenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", "1")
    else:
        monkeypatch.delenv("TT_METAL_QSR_CONV_SPLIT_PROGRAM", raising=False)

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
    tt_out = torch.permute(tt_out, (0, 3, 1, 2))  # NHWC -> NCHW
    assert_with_pcc(torch_golden, tt_out, 0.99)
