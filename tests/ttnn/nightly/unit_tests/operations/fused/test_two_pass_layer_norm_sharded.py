# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Full two-pass regression matrices; sanity retains a small representative sample."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.unit_tests.operations.fused.sharded_test_utils import (
    simple_size_params,
    create_sharded_mem_config,
    ttnn_layer_norm_sharded,
)

# LayerNorm is shift invariant: layer_norm(x + c) equals layer_norm(x) for any constant c.
# The accuracy budget therefore must not depend on the shared offset. FP32 input holds the
# spread to within one input ulp, which at an offset of 1e6 is 0.0625, or 0.001 of the spread
# of 64 used below, so statistics formed in FP32 after removing a shift stay close to the error
# this geometry reaches with no offset at all (about 0.008 max on the normalised output). The
# budget is that offset-free error with margin for the reciprocal-square-root step and the
# output write. A backend that reads the input through a 10-bit-mantissa TF32 operand instead
# cannot meet the budget once the offset passes about 1e4, because the operand resolution at
# that magnitude grows past the spread the statistics have to recover.
_LARGE_OFFSET_MAX_ABS_ERR = 0.05
_LARGE_OFFSET_PCC = 0.999

# Offsets span the range over which an FP32 operand still resolves the spread of 64 (1e3) up to
# the range where only a full-precision intake can (1e6).
_LARGE_OFFSET_BASES = [0.0, 1_000.0, 3_000.0, 10_000.0, 30_000.0, 100_000.0, 1_000_000.0]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32])
@pytest.mark.parametrize("norm, use_welford", [("layer", False), ("layer", True), ("rms", False)])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("two_stage", [False, True])
def test_sharded_norm_beta_only(device, dtype, norm, use_welford, has_residual, two_stage):
    device.enable_program_cache()
    # Fill the intermediate buffer: small shards can leave enough spare DFB
    # capacity to hide a producer reserving its own unconsumed input.
    h, w, cores_h, cores_w = (256, 320, 2, 5) if two_stage else (1024, 256, 4, 4)
    memory_config = create_sharded_mem_config(h, w, cores_h, cores_w, two_stage)
    torch.manual_seed(41)

    def make_tensor(values, sharded=False):
        return ttnn.from_torch(
            values,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config if sharded else ttnn.DRAM_MEMORY_CONFIG,
        )

    # Non-unit variance distinguishes (x - mean) + beta from LayerNorm + beta;
    # correlation alone can miss an omitted inverse-standard-deviation scale.
    input_tensor = make_tensor(5.0 + 3.0 * torch.randn(h, w), sharded=True)
    residual = make_tensor(torch.randn(h, w), sharded=True) if has_residual else None
    bias = make_tensor(torch.linspace(-0.25, 0.25, w))
    reference_input = ttnn.to_torch(input_tensor).double()
    if residual is not None:
        reference_input += ttnn.to_torch(residual).double()
    reference_bias = ttnn.to_torch(bias).double()
    epsilon = 1e-5
    if norm == "layer":
        reference = torch.nn.functional.layer_norm(reference_input, [w], bias=reference_bias, eps=epsilon)
    else:
        reference = reference_input * torch.rsqrt(reference_input.square().mean(dim=-1, keepdim=True) + epsilon)
        reference += reference_bias

    shard = memory_config.shard_spec
    config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        block_h=shard.shape[0] // 32,
        block_w=shard.shape[1] // 32,
        subblock_w=1,
        use_welford=use_welford,
        inplace=False,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    kwargs = {}
    if use_welford:
        kwargs["recip_tensor"] = ttnn.create_layer_norm_reciprocals(device, shard.grid, shard.shape[1])
    op = ttnn.layer_norm if norm == "layer" else ttnn.rms_norm
    for _ in range(2):
        output = op(
            input_tensor,
            residual_input_tensor=residual,
            bias=bias,
            epsilon=epsilon,
            memory_config=memory_config,
            program_config=config,
            compute_kernel_config=compute_config,
            **kwargs,
        )
        actual = ttnn.to_torch(output).double()
        assert torch.isfinite(actual).all()
        assert_numeric_metrics(
            reference,
            actual,
            rtol=0,
            atol=0.125 if dtype == ttnn.bfloat8_b else 0.05,
            frobenius_threshold=0.03 if dtype == ttnn.bfloat8_b else 0.01,
            pcc_threshold=0.999,
        )


@pytest.mark.parametrize("block_wt", [1, 2, 3])
@pytest.mark.parametrize("has_residual", [False, True])
def test_layer_norm_sharded_fp32_preserves_centred_low_bits(device, block_wt, has_residual):
    # Each pair differs in FP32 but collapses to one TF32 value. Comparing their
    # output difference isolates the centred-value reload from rsqrt accuracy.
    h, w = 128, 64 * block_wt
    values = torch.tensor([1.0001, 1.0002, -1.0001, -1.0002]).repeat(h, w // 4)
    memory_config = create_sharded_mem_config(h, w, 2, 2, two_stage=False)
    input_tensor = ttnn.from_torch(values, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    residual = (
        ttnn.from_torch(
            torch.full_like(values, 0.125), layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        if has_residual
        else None
    )
    config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    output = ttnn_layer_norm_sharded(
        device, input_tensor, True, block_ht=2, block_wt=block_wt, residual=residual, compute_kernel_config=config
    ).double()
    reference_input = (values + (0.125 if has_residual else 0.0)).double()
    reference = torch.nn.functional.layer_norm(reference_input, [w])
    torch.testing.assert_close(
        output[:, 1::4] - output[:, 0::4],
        reference[:, 1::4] - reference[:, 0::4],
        rtol=0.01,
        atol=1e-7,
    )


@pytest.mark.parametrize("base", _LARGE_OFFSET_BASES)
@pytest.mark.parametrize("two_stage", [False, True])
@pytest.mark.parametrize("has_residual", [False, True])
@pytest.mark.parametrize("has_gamma, has_beta", [(False, False), (True, False), (False, True), (True, True)])
def test_layer_norm_sharded_fp32_large_offset(device, base, two_stage, has_residual, has_gamma, has_beta):
    """Sharded FP32 LayerNorm accuracy must not degrade as a shared offset is added to every row.

    Each row holds a spread of 64 riding on the given offset. Because normalisation removes the
    row mean, the offset carries no information and must not cost accuracy. Sweeping it shows the
    offset at which a backend stops resolving the spread.
    """
    torch.manual_seed(41)
    h, w, num_cores_h, num_cores_w, block_ht, block_wt, subblock_wt = simple_size_params(two_stage)

    torch_input = base + 64.0 * torch.randn((h, w), dtype=torch.float32)
    torch_residual = base + 64.0 * torch.randn((h, w), dtype=torch.float32) if has_residual else None
    torch_weight = torch.linspace(0.75, 1.25, w, dtype=torch.float32) if has_gamma else None
    torch_bias = torch.linspace(-0.25, 0.25, w, dtype=torch.float32) if has_beta else None

    reference_input = torch_input.to(torch.float64)
    if torch_residual is not None:
        reference_input += torch_residual.to(torch.float64)
    reference = torch.nn.functional.layer_norm(
        reference_input,
        [w],
        weight=torch_weight.to(torch.float64) if torch_weight is not None else None,
        bias=torch_bias.to(torch.float64) if torch_bias is not None else None,
    )

    memory_config = create_sharded_mem_config(h, w, num_cores_h, num_cores_w, two_stage)
    input_tensor = ttnn.from_torch(
        torch_input,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    residual_tensor = (
        ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
        if torch_residual is not None
        else None
    )
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device) if torch_weight is not None else None
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device) if torch_bias is not None else None
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    output = ttnn_layer_norm_sharded(
        device,
        input_tensor,
        use_welford=True,
        block_ht=block_ht,
        block_wt=block_wt,
        subblock_w=subblock_wt,
        residual=residual_tensor,
        weight=weight,
        bias=bias,
        compute_kernel_config=compute_kernel_config,
    )
    actual = output.to(torch.float64)

    # Report the measured error at every offset, so a sweep over the offsets shows where the
    # selected backend stops resolving the spread rather than only the first offset that fails.
    print(
        f"[large_offset] base={base:<10g} two_stage={two_stage!s:<5} affine={has_gamma}"
        f" max_abs_err={(actual - reference).abs().max():.4g}",
        flush=True,
    )

    assert torch.isfinite(actual).all()
    assert_numeric_metrics(
        reference,
        actual,
        pcc_threshold=_LARGE_OFFSET_PCC,
        rtol=0,
        atol=_LARGE_OFFSET_MAX_ABS_ERR,
        frobenius_threshold=_LARGE_OFFSET_MAX_ABS_ERR,
    )
