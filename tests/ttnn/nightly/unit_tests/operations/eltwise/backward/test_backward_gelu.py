# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc, data_gen_with_range


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_gelu_default(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.gelu_bw(grad_tensor, input_tensor)
    tt_out = ttnn.to_torch(tt_output_tensor_on_device[0])

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    atol_value = 0.01
    assert torch.allclose(
        golden_tensor[0], tt_out, atol=atol_value
    ), f"gelu_bw default (approximate='none') mismatch beyond atol={atol_value} for shape {input_shapes}"


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "approximate, atol_value",
    (
        ("none", 0.01),
        ("tanh", 0.01),
    ),
)
def test_bw_gelu_opt_output(input_shapes, approximate, atol_value, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -5, 5, device)
    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data, approximate=approximate)
    tt_out = ttnn.to_torch(tt_output_tensor_on_device[0])

    assert torch.allclose(
        golden_tensor[0], tt_out, atol=atol_value
    ), f"gelu_bw(approximate={approximate!r}) mismatch beyond atol={atol_value} for optional output with shape {input_shapes}"


# =============================================================================
# Gelu_bw Tanh approximation Exhaustive FP32 ULP Distribution Test
# =============================================================================


def test_gelu_bw_fp32_exhaustive(device):
    """Exhaustive ULP distribution test for gelu_bw (approximate='tanh') on the FLOAT32 path.

    Generates all valid BF16 bit patterns as input then materialize the values as float32 inputs, uses grad=1.0, and measures
    ULP distance between device output and PyTorch float32 reference (tanh-approximate GELU derivative).
    """
    torch.manual_seed(0)

    # Generate all bf16 bit patterns, then materialize the values as float32 inputs.
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    vals_bf16 = all_bitpatterns.view(torch.bfloat16)

    # Filter to finite, non-zero, non-subnormal values (using the bf16 grid for the filter).
    min_abs = torch.finfo(torch.bfloat16).tiny
    max_abs = torch.finfo(torch.bfloat16).max
    vals_f32 = vals_bf16.to(torch.float32)
    vals_f32_abs = vals_f32.abs()
    mask = torch.isfinite(vals_f32) & (vals_f32_abs >= min_abs) & (vals_f32_abs <= max_abs) & (vals_f32_abs != 0)

    value_set = vals_f32[mask]  # same ~65k count, but float32
    N = value_set.numel()
    logger.debug(
        f"Testing gelu_bw (approximate=tanh, FP32) with {N} values in [{value_set.min().item():.2e}, {value_set.max().item():.2e}]"
    )

    # Pad to multiple of 32 for tile layout
    pad_size = (32 - (N % 32)) % 32
    if pad_size > 0:
        value_set_padded = torch.cat([value_set, torch.zeros(pad_size, dtype=torch.float32)])
    else:
        value_set_padded = value_set

    total_padded = value_set_padded.numel()
    value_set_2d = value_set_padded.reshape(1, total_padded)

    # Compute reference: GELU derivative via PyTorch autograd in float32
    x_f32 = value_set_2d.clone().requires_grad_(True)
    y = torch.nn.functional.gelu(x_f32, approximate="tanh")
    y.backward(torch.ones_like(y))
    z_torch = x_f32.grad.detach()

    # Run on device: gelu_bw with grad=1.0, float32 inputs -> exercises the fp32 kernel path
    grad_2d = torch.ones_like(value_set_2d)
    tt_input = ttnn.from_torch(value_set_2d, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_grad = ttnn.from_torch(grad_2d, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    results = ttnn.gelu_bw(tt_grad, tt_input, approximate="tanh")
    tt_out = ttnn.to_torch(results[0])

    # Trim padding
    z_torch = z_torch[:, :N]
    tt_out = tt_out[:, :N]

    # Filter out inf/nan results
    valid_mask = torch.isfinite(z_torch) & torch.isfinite(tt_out)
    assert torch.isfinite(
        tt_out[torch.isfinite(z_torch)]
    ).all(), "ttnn.gelu_bw(approximate='tanh', fp32) output is non-finite where the reference is finite"
    z_torch_valid = z_torch[valid_mask].contiguous()
    tt_out_valid = tt_out[valid_mask].contiguous()
    N_valid = z_torch_valid.numel()

    logger.debug(f"Valid (finite) outputs: {N_valid}/{N} ({N_valid/N*100:.2f}%)")

    # Ensure we are measuring float32 ULP (view(torch.int32) requires 4-byte floats)
    assert z_torch_valid.dtype == torch.float32, f"Reference must be float32 for ULP, got {z_torch_valid.dtype}"
    assert tt_out_valid.dtype == torch.float32, f"TTNN output must be float32 for ULP, got {tt_out_valid.dtype}"

    # Flush subnormal and max-normal/inf outputs to zero (float32: exp 0 + nonzero mantissa = subnormal;
    # exp 255 = max/inf) to model DAZ+FTZ before the ULP comparison.
    z_bits = z_torch_valid.view(torch.int32)
    tt_bits = tt_out_valid.view(torch.int32)
    subnormal_z = (((z_bits >> 23) & 0xFF) == 0) & ((z_bits & 0x7FFFFF) != 0)
    subnormal_tt = (((tt_bits >> 23) & 0xFF) == 0) & ((tt_bits & 0x7FFFFF) != 0)
    max_or_inf_z = ((z_bits >> 23) & 0xFF) == 255
    max_or_inf_tt = ((tt_bits >> 23) & 0xFF) == 255
    flush_mask = subnormal_z | subnormal_tt | max_or_inf_z | max_or_inf_tt
    z_torch_valid = torch.where(flush_mask, torch.zeros_like(z_torch_valid), z_torch_valid)
    tt_out_valid = torch.where(flush_mask, torch.zeros_like(tt_out_valid), tt_out_valid)

    # ULP check in float32 space using signed-magnitude representation.
    z_bits = z_torch_valid.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    tt_bits = tt_out_valid.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign_z = (z_bits & 0x80000000) != 0
    sign_tt = (tt_bits & 0x80000000) != 0
    z_ord = torch.where(sign_z, 0xFFFFFFFF - z_bits, z_bits + 0x80000000)
    tt_ord = torch.where(sign_tt, 0xFFFFFFFF - tt_bits, tt_bits + 0x80000000)

    ulp_dist = (z_ord - tt_ord).abs()

    max_ulp = ulp_dist.max().item()

    # ULP distribution
    ulp_0_count = (ulp_dist == 0).sum().item()
    ulp_1_count = (ulp_dist == 1).sum().item()
    ulp_2_count = (ulp_dist == 2).sum().item()
    ulp_3_to_10_count = ((ulp_dist >= 3) & (ulp_dist <= 10)).sum().item()
    ulp_11_to_100_count = ((ulp_dist >= 11) & (ulp_dist <= 100)).sum().item()
    ulp_above_100_count = (ulp_dist > 100).sum().item()

    mismatch_threshold = 2
    mismatch_mask = ulp_dist > mismatch_threshold
    total_mismatches = mismatch_mask.sum().item()
    mismatch_pct = (total_mismatches / N_valid) * 100 if N_valid > 0 else 0.0

    logger.debug(
        f"Max ULP (fp32 space): {max_ulp}, mismatches (ULP > {mismatch_threshold}): {total_mismatches}/{N_valid} ({mismatch_pct:.4f}%)"
    )
    logger.debug(f"\nULP Distribution (fp32 space):")
    logger.debug(f"  ULP = 0: {ulp_0_count:,} ({ulp_0_count/N_valid*100:.4f}%)")
    logger.debug(f"  ULP = 1: {ulp_1_count:,} ({ulp_1_count/N_valid*100:.4f}%)")
    logger.debug(f"  ULP = 2: {ulp_2_count:,} ({ulp_2_count/N_valid*100:.4f}%)")
    logger.debug(f"  ULP 3-10: {ulp_3_to_10_count:,} ({ulp_3_to_10_count/N_valid*100:.4f}%)")
    logger.debug(f"  ULP 11-100: {ulp_11_to_100_count:,} ({ulp_11_to_100_count/N_valid*100:.4f}%)")
    logger.debug(f"  ULP > 100: {ulp_above_100_count:,} ({ulp_above_100_count/N_valid*100:.4f}%)")

    # Verify counts sum correctly
    ulp_sum = ulp_0_count + ulp_1_count + ulp_2_count + ulp_3_to_10_count + ulp_11_to_100_count + ulp_above_100_count
    assert ulp_sum == N_valid, f"ULP counts don't sum to total valid: {ulp_sum} != {N_valid}"

    assert torch.allclose(
        z_torch_valid, tt_out_valid, rtol=1e-4, atol=1e-4
    ), "gelu_bw(approximate='tanh', fp32) output does not match float32 reference within tolerance"


def test_bw_gelu_program_cache_regression(device):
    """Program-cache regression guard for the gelu_bw descriptor migration.

    This test enables the program cache and invokes each mode repeatedly with newly allocated
    inputs, outputs, and non-unit gradients (distinct seeds -> different data + different buffer
    addresses), checks correctness on every call, and asserts the expected cache reuse/separation.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    shape = torch.Size([1, 1, 320, 384])
    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)

    def generate_inputs(seed):
        # Fresh allocations (distinct seed -> distinct data and buffer addresses) each call.
        in_data, input_tensor = data_gen_with_range(shape, -100, 100, device, True, seed=seed)
        grad_data, grad_tensor = data_gen_with_range(shape, -5, 5, device, seed=seed + 1000)
        return in_data, input_tensor, grad_data, grad_tensor

    # First call for each mode compiles a program; the two modes must NOT share a cache entry.
    seed = 0
    in_data, input_tensor, grad_data, grad_tensor = generate_inputs(seed=seed)

    approximate = "none"
    tt_out1 = ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)
    golden_tensor = golden_function(grad_data, in_data, approximate=approximate)
    assert compare_pcc(
        tt_out1, golden_tensor, pcc=0.999
    ), f"gelu_bw(approximate={approximate!r}) mismatch on cache-enabled run (seed={seed})"
    assert (
        device.num_program_cache_entries() == 1
    ), "first gelu_bw(approximate='none') must create exactly one cache entry"

    seed = 1
    in_data, input_tensor, grad_data, grad_tensor = generate_inputs(seed=seed)

    approximate = "tanh"
    tt_out2 = ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)
    golden_tensor = golden_function(grad_data, in_data, approximate=approximate)
    assert compare_pcc(
        tt_out2, golden_tensor, pcc=0.999
    ), f"gelu_bw(approximate={approximate!r}) mismatch on cache-enabled run (seed={seed})"
    assert device.num_program_cache_entries() == 2, (
        "gelu_bw(approximate='tanh') must create a SEPARATE cache entry from 'none' -- the "
        "approximation flag must be part of the program hash / select a distinct kernel."
    )

    # Re-run both modes with new buffers/data: must be cache HITS (no new entries) and still correct,
    # which only holds if the Buffer* bindings are re-patched on the fast path.
    seed = 42
    in_data, input_tensor, grad_data, grad_tensor = generate_inputs(seed=seed)
    approximate = "none"
    tt_out3 = ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)
    golden_tensor = golden_function(grad_data, in_data, approximate=approximate)
    assert compare_pcc(
        tt_out3, golden_tensor, pcc=0.999
    ), f"gelu_bw(approximate={approximate!r}) mismatch on cache-enabled run (seed={seed})"

    seed = 99
    in_data, input_tensor, grad_data, grad_tensor = generate_inputs(seed=seed)
    approximate = "tanh"
    tt_out4 = ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)
    golden_tensor = golden_function(grad_data, in_data, approximate=approximate)
    assert compare_pcc(
        tt_out4, golden_tensor, pcc=0.999
    ), f"gelu_bw(approximate={approximate!r}) mismatch on cache-enabled run (seed={seed})"

    assert device.num_program_cache_entries() == 2, (
        "re-running each mode with freshly allocated tensors must reuse the cached programs "
        "(no new entries). A new entry means the buffers/mode were wrongly folded into the hash."
    )

    logger.debug("gelu_bw program-cache regression: 2 entries, both modes correct across cache hits")

    device.disable_and_clear_program_cache()
