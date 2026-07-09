# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for fp32 accumulation precision of the conv2d cross-block partial reload.

conv2d reduces over C_in*filter_h*filter_w. When that reduction is split across blocks with
fp32_dest_acc_en=True and packer_l1_acc=False, the Float32 partials in the MATMUL_PARTIALS CB
are reloaded into DEST between blocks by conv_bmm_tilize's copy_block_matmul_partials. Unless the
CB is marked UnpackToDestFp32, the reload is routed through SrcA and truncated to TF32, so fp32
accumulation silently degrades.

A 3x3 kernel is used (1x1 stride-1 convs are lowered to matmul and would not exercise the conv
compute kernel). Inputs carry a large common offset and weights sum to zero over the full
(C_in, kh, kw) reduction, so the offset contributes nothing to the true result but makes the
mid-accumulation partials large: a TF32-truncated reload then catastrophically corrupts the
(small) true result. padding=0 keeps every reduction tap over real (offset+signal) data so the
cancellation is exact.
"""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "shard_layout, in_channels, out_channels, act_block_w_div",
    [
        # HEIGHT_SHARDED splits the C_in*kh*kw reduction via act_block_w_div (a few K-blocks).
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, 512, 64, 4),
        # WIDTH_SHARDED shards C_in across cores, so num_blocks == num_cores; a large C_in gives
        # many ring blocks and a strong signal (act_block_w_div must be 1 for width-sharded here).
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, 768, 768, 1),
    ],
    ids=["height_sharded", "width_sharded"],
)
def test_conv2d_fp32_crossblock_reload_precision(device, shard_layout, in_channels, out_channels, act_block_w_div):
    batch = 1
    height, width, kh, kw = 18, 18, 3, 3  # padding=0 -> 16x16 output
    torch.manual_seed(0)

    x_nchw = (torch.randn(batch, in_channels, height, width) + 1000.0).bfloat16()
    w = torch.randn(out_channels, in_channels, kh, kw)
    w = (w - w.mean(dim=(1, 2, 3), keepdim=True)).bfloat16()  # sum over (C_in, kh, kw) == 0

    ref_nchw = torch.nn.functional.conv2d(x_nchw.double(), w.double(), padding=0)
    ref = ref_nchw.permute(0, 2, 3, 1).reshape(-1, out_channels).float()

    tt_input = ttnn.from_torch(x_nchw.permute(0, 2, 3, 1), ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_weight = ttnn.from_torch(w, ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=shard_layout,
        act_block_w_div=act_block_w_div,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    tt_out = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=(kh, kw),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=batch,
        input_height=height,
        input_width=width,
        conv_config=conv_config,
        compute_config=compute_config,
        dtype=ttnn.float32,  # fp32 output -> the fp32 K-partials are what the reload must preserve
    )
    out = ttnn.to_torch(tt_out).reshape(-1, out_channels).float()[: ref.shape[0]]

    # A lossless fp32 reload keeps PCC high; the TF32-truncated reload drops it below 0.99.
    # HEIGHT_SHARDED has only a few K-blocks so the drop is modest (to ~0.985); WIDTH_SHARDED has
    # num_blocks == num_cores (many ring reloads) so the unfixed drop is large (to ~0.6).
    # allclose/ULP are not meaningful for a deliberately ill-conditioned bf16 conv, so only PCC and
    # Frobenius are checked.
    assert_numeric_metrics(
        ref,
        out,
        pcc_threshold=0.99,
        frobenius_threshold=0.5,
        check_allclose=False,
        check_pcc=True,
        check_frobenius=True,
        check_ulp=False,
    )
