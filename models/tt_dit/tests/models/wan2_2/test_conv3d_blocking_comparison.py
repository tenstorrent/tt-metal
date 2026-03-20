# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test to verify whether different C_in_block choices produce bit-identical
output on Blackhole (P100).  On Wormhole the bf16 truncation between partial
sums means different C_in_block values → different rounding → different hashes.
BH uses HiFi4 (no TF32 truncation), so the hypothesis is that the output is
invariant to C_in_block choice.

Run on a P100 (single BH chip, 1x1 mesh):

    pytest models/tt_dit/tests/models/wan2_2/test_conv3d_blocking_comparison.py -v -s --timeout=600
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole

from ....utils.conv3d import (
    aligned_channels,
    conv_pad_height,
    conv_pad_in_channels,
    conv_unpad_height,
    prepare_conv3d_weights,
)
from ....utils.tensor import typed_tensor_2dshard


def _run_conv3d(
    mesh_device,
    torch_weight,
    torch_bias,
    torch_input_BCTHW,
    C_in_block,
    C_out_block,
    H_out_block,
    W_out_block,
    kernel_size,
    stride,
    padding,
    dtype=ttnn.DataType.BFLOAT16,
):
    """Run a single conv3d with the given blocking and return the output as a torch tensor."""
    B, C_in_orig, T, H, W = torch_input_BCTHW.shape
    C_out_orig = torch_weight.shape[0]

    C_in = aligned_channels(C_in_orig)
    C_out = C_out_orig
    TILE_WIDTH = 32
    if C_out < TILE_WIDTH:
        C_out = TILE_WIDTH

    grid_size = mesh_device.compute_with_storage_grid_size()
    conv_config = ttnn.Conv3dConfig(
        weights_dtype=dtype,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=1,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=grid_size,
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4 if is_blackhole() else ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Prepare weights
    weight_padded = torch_weight
    bias_padded = torch_bias
    if C_out != C_out_orig:
        weight_padded = torch.nn.functional.pad(weight_padded, (0, 0, 0, 0, 0, 0, 0, 0, 0, C_out - C_out_orig))
        bias_padded = torch.nn.functional.pad(bias_padded, (0, C_out - C_out_orig))

    w_prepared, b_prepared = prepare_conv3d_weights(weight_padded, bias_padded, conv_config)

    # Put weights on device
    h_axis, w_axis = 0, 1
    tt_weight = typed_tensor_2dshard(
        w_prepared, mesh_device, shard_mapping={h_axis: 0, w_axis: 1}, layout=ttnn.TILE_LAYOUT, dtype=dtype
    )
    tt_bias = typed_tensor_2dshard(
        b_prepared, mesh_device, shard_mapping={h_axis: 0, w_axis: 1}, layout=ttnn.TILE_LAYOUT, dtype=dtype
    )

    # Prepare input: BCTHW -> BTHWC, pad channels, pad height
    tt_input = torch_input_BCTHW.permute(0, 2, 3, 4, 1)  # BTHWC
    tt_input = conv_pad_in_channels(tt_input)
    tt_input, logical_h = conv_pad_height(tt_input, 1)  # no height parallelism
    tt_input_dtype = ttnn.bfloat16 if dtype == ttnn.DataType.BFLOAT16 else ttnn.float32
    tt_input = typed_tensor_2dshard(
        tt_input, mesh_device, shard_mapping={h_axis: 2, w_axis: 3}, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=tt_input_dtype
    )

    # Run conv3d
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)

    tt_output = ttnn.experimental.conv3d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        bias_tensor=tt_bias,
        config=conv_config,
        output_channels=C_out,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode="zeros",
        dtype=dtype,
        compute_kernel_config=compute_kernel_config,
    )

    # Read back
    concat_dims = [None, None]
    concat_dims[h_axis] = 2
    concat_dims[w_axis] = 3
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=concat_dims),
    )
    tt_output_torch = conv_unpad_height(tt_output_torch, logical_h)
    tt_output_torch = tt_output_torch.permute(0, 4, 1, 2, 3)  # BTHWC -> BCTHW

    # Trim padded output channels
    if tt_output_torch.shape[1] != C_out_orig:
        tt_output_torch = tt_output_torch[:, :C_out_orig]

    return tt_output_torch


def _compare_outputs(name_a, out_a, name_b, out_b):
    """Compare two outputs and log detailed statistics."""
    diff = (out_a - out_b).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    nonzero = (diff > 0).sum().item()
    total = diff.numel()

    # PCC
    a_flat = out_a.flatten().double()
    b_flat = out_b.flatten().double()
    pcc = torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()

    bit_identical = max_abs == 0.0

    logger.info(f"--- {name_a} vs {name_b} ---")
    logger.info(f"  Bit-identical: {bit_identical}")
    logger.info(f"  PCC:           {pcc:.8f}")
    logger.info(f"  Max abs error: {max_abs:.6e}")
    logger.info(f"  Mean abs error:{mean_abs:.6e}")
    logger.info(f"  Nonzero diffs: {nonzero}/{total} ({100*nonzero/total:.2f}%)")

    return bit_identical, pcc, max_abs, mean_abs


# --- Test cases ---
# Each entry: (C_in, C_out, kernel_size, stride, padding, T, H, W,
#               good_blocking, aggressive_blocking)
# Blockings: (C_in_block, C_out_block, H_out_block, W_out_block)

BLOCKING_TEST_CASES = [
    pytest.param(
        96,
        96,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        1,
        32,
        32,
        (96, 32, 8, 4),  # GOOD: C_in_num_blocks=1
        (96, 32, 8, 4),  # AGGRESSIVE: same blocking (sanity check)
        id="96_96_no_reduction",
    ),
    pytest.param(
        96,
        96,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        1,
        32,
        32,
        (96, 32, 8, 4),  # GOOD: C_in_num_blocks=1 (C_in_block=96)
        (32, 32, 8, 4),  # AGGRESSIVE: C_in_num_blocks=3 (C_in_block=32)
        id="96_96_cin_block_32_vs_96",
    ),
]


@pytest.mark.parametrize(
    "C_in, C_out, kernel_size, stride, padding, T, H, W, good_blocking, aggressive_blocking",
    BLOCKING_TEST_CASES,
)
@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],
    ids=["1x1"],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_conv3d_cin_block_sensitivity(
    mesh_device,
    C_in,
    C_out,
    kernel_size,
    stride,
    padding,
    T,
    H,
    W,
    good_blocking,
    aggressive_blocking,
):
    """
    Run the same conv3d with two different C_in_block values and compare outputs.

    On BH (HiFi4, no TF32 truncation), the hypothesis is that the outputs are
    bit-identical regardless of C_in_block choice. On WH (HiFi2, TF32 truncation),
    the outputs will differ.
    """
    torch.manual_seed(42)

    C_in_block_good, C_out_block_good, H_block_good, W_block_good = good_blocking
    C_in_block_aggr, C_out_block_aggr, H_block_aggr, W_block_aggr = aggressive_blocking

    # Create random input and weights (use float32 torch tensors, conv3d will cast to bf16)
    torch_input = torch.randn(1, C_in, T, H, W, dtype=torch.float32)
    torch_model = torch.nn.Conv3d(C_in, C_out, kernel_size, stride, padding, bias=True)
    torch_weight = torch_model.weight.data  # (C_out, C_in, kD, kH, kW)
    torch_bias = torch_model.bias.data  # (C_out,)

    # Torch fp32 reference
    with torch.no_grad():
        torch_ref = torch_model(torch_input)

    logger.info(f"Conv3d: C_in={C_in}, C_out={C_out}, kernel={kernel_size}, input=({T},{H},{W})")
    logger.info(f"GOOD blocking:       C_in_block={C_in_block_good}, C_out_block={C_out_block_good}")
    logger.info(f"AGGRESSIVE blocking: C_in_block={C_in_block_aggr}, C_out_block={C_out_block_aggr}")
    logger.info(f"C_in_num_blocks: GOOD={C_in // C_in_block_good}, AGGRESSIVE={C_in // C_in_block_aggr}")

    # Run with GOOD blocking
    out_good = _run_conv3d(
        mesh_device,
        torch_weight,
        torch_bias,
        torch_input,
        C_in_block=C_in_block_good,
        C_out_block=C_out_block_good,
        H_out_block=H_block_good,
        W_out_block=W_block_good,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    # Run with AGGRESSIVE blocking
    out_aggr = _run_conv3d(
        mesh_device,
        torch_weight,
        torch_bias,
        torch_input,
        C_in_block=C_in_block_aggr,
        C_out_block=C_out_block_aggr,
        H_out_block=H_block_aggr,
        W_out_block=W_block_aggr,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    # Compare GOOD vs torch ref
    _, pcc_good_ref, max_good_ref, _ = _compare_outputs("GOOD", out_good, "Torch fp32", torch_ref)

    # Compare AGGRESSIVE vs torch ref
    _, pcc_aggr_ref, max_aggr_ref, _ = _compare_outputs("AGGRESSIVE", out_aggr, "Torch fp32", torch_ref)

    # Compare GOOD vs AGGRESSIVE (the key comparison)
    bit_identical, pcc_cross, max_cross, mean_cross = _compare_outputs("GOOD", out_good, "AGGRESSIVE", out_aggr)

    # Both configs should be reasonable vs torch
    assert pcc_good_ref > 0.999, f"GOOD vs torch PCC too low: {pcc_good_ref}"
    assert pcc_aggr_ref > 0.999, f"AGGRESSIVE vs torch PCC too low: {pcc_aggr_ref}"

    # The whole point: GOOD and AGGRESSIVE must produce bit-identical output.
    # If not, the fp32 intermediate CB fix is needed.
    assert bit_identical, (
        f"GOOD vs AGGRESSIVE are NOT bit-identical! "
        f"PCC={pcc_cross:.8f}, max_err={max_cross:.2e}, mean_err={mean_cross:.2e}. "
        f"C_in_block sensitivity exists — fp32 accumulation CB fix is required."
    )
