# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import compare_pcc


INPUT_SHAPES = (
    (torch.Size([32])),
    (torch.Size([25, 34])),  # not aligned by tile size
    (torch.Size([32, 32])),
    (torch.Size([1, 32, 32])),
    (torch.Size([1, 1, 32, 32])),
    (torch.Size([1, 1, 320, 384])),
    (torch.Size([1, 3, 320, 384])),
    (torch.Size([1, 3, 323, 389])),  # not aligned by tile size
)


def gen_data(input_shapes, low, high, device, required_grad=False, is_row_major=False, seed=213919):
    assert high > low, "Incorrect range provided"
    torch.manual_seed(seed)
    pt_tensor = torch.rand(input_shapes, requires_grad=required_grad).bfloat16() * (high - low) + low
    if is_row_major:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16)
        tt_tensor = ttnn.to_layout(tt_tensor, layout=ttnn.ROW_MAJOR_LAYOUT).to(device)
    else:
        tt_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16)
        tt_tensor = ttnn.to_layout(tt_tensor, layout=ttnn.TILE_LAYOUT).to(device)

    return pt_tensor, tt_tensor


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
@pytest.mark.parametrize(
    "approximate",
    (
        "none",
        "tanh",
    ),
)
def test_bw_gelu(input_shapes, approximate, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
def test_bw_gelu_default(input_shapes, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)

    tt_output_tensor_on_device = ttnn.gelu_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
@pytest.mark.parametrize(
    "approximate",
    (
        "none",
        "tanh",
    ),
)
def test_bw_gelu_opt_output(input_shapes, approximate, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)
    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    INPUT_SHAPES,
)
def test_bw_gelu_default_opt_output(input_shapes, device):
    in_data, input_tensor = gen_data(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = gen_data(input_shapes, -5, 5, device)
    input_grad = torch.zeros(input_shapes, dtype=torch.bfloat16)
    input_grad = ttnn.from_torch(
        input_grad, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    cq_id = 0
    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.gelu_bw(grad_tensor, input_tensor, input_grad=input_grad, queue_id=cq_id)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))

    tt_output_tensor_on_device = [input_grad]

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


@pytest.mark.parametrize(
    "grad_dtype, input_dtype",
    (
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.float32),
    ),
)
def test_bw_gelu_grad_dtype_must_match_input(grad_dtype, input_dtype, device, expect_error):
    shape = torch.Size([1, 1, 320, 384])
    torch.manual_seed(213919)

    in_data = (torch.rand(shape, dtype=torch.bfloat16) * 200 - 100).requires_grad_(True)
    grad_data = torch.rand(shape, dtype=torch.bfloat16) * 10 - 5

    input_tensor = ttnn.from_torch(in_data.detach(), input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, grad_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    if grad_dtype != input_dtype:
        with expect_error(RuntimeError, "grad_output and input data types to match"):
            ttnn.gelu_bw(grad_tensor, input_tensor)
        return

    result = ttnn.gelu_bw(grad_tensor, input_tensor)[0]
    assert result.dtype == input_dtype

    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)
    assert compare_pcc([result], golden_function(grad_data, in_data))


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
    """Program-cache regression guard for the gelu_bw Metal 2.0 program factory.

    The Metal 2.0 factory binds tensors as named TensorParameters instead of passing buffer
    addresses through runtime args, so on a program-cache HIT the framework re-resolves each
    TensorArgument into the cached program rather than the op re-emitting addresses. A stale
    binding would keep pointing at the first call's buffers and silently produce wrong results
    (or read freed memory) on every later call.

    Every call below allocates fresh inputs/outputs (distinct seeds -> different data AND
    different buffer addresses), asserts correctness, and asserts the cache reused the program
    instead of compiling a new one. Both compute kernels ("none" -> poly, "tanh") and the
    preallocated-output path are covered, since each binds a different set of tensors.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    shape = torch.Size([1, 1, 320, 384])  # multi-tile, spans several cores
    golden_function = ttnn.get_golden_function(ttnn.gelu_bw)

    def fresh_inputs(seed):
        in_data, input_tensor = gen_data(shape, -100, 100, device, True, seed=seed)
        grad_data, grad_tensor = gen_data(shape, -5, 5, device, seed=seed + 1000)
        return in_data, input_tensor, grad_data, grad_tensor

    def check(result, grad_data, in_data, label):
        assert compare_pcc([result], golden_function(grad_data, in_data)), f"{label}: output mismatch"

    # --- Cache misses: the two approximations must not share an entry ---
    in_data, input_tensor, grad_data, grad_tensor = fresh_inputs(seed=0)
    result = ttnn.gelu_bw(grad_tensor, input_tensor, approximate="none")[0]
    check(result, grad_data, in_data, "approximate='none' (first call)")
    assert device.num_program_cache_entries() == 1, "first gelu_bw(approximate='none') must create exactly one entry"

    in_data, input_tensor, grad_data, grad_tensor = fresh_inputs(seed=1)
    result = ttnn.gelu_bw(grad_tensor, input_tensor, approximate="tanh")[0]
    check(result, grad_data, in_data, "approximate='tanh' (first call)")
    assert device.num_program_cache_entries() == 2, (
        "gelu_bw(approximate='tanh') must create a SEPARATE cache entry from 'none' -- the "
        "approximation selects a different compute kernel, so it must be part of the program hash."
    )

    # --- Cache hits: re-run both modes with new buffers. This is the rebinding check. ---
    for seed, approximate in ((42, "none"), (99, "tanh"), (7, "none"), (13, "tanh")):
        in_data, input_tensor, grad_data, grad_tensor = fresh_inputs(seed=seed)
        result = ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate)[0]
        check(result, grad_data, in_data, f"approximate={approximate!r} cache hit (seed={seed})")
        assert device.num_program_cache_entries() == 2, (
            f"re-running approximate={approximate!r} with freshly allocated tensors must reuse the "
            "cached program (no new entry). A new entry means buffers were wrongly folded into the hash."
        )

    # --- Cache hits on the preallocated-output path (binds OUTPUT to a caller-owned tensor) ---
    for approximate in ("none", "tanh"):
        entries_before = None
        for seed in (100, 200):
            in_data, input_tensor, grad_data, grad_tensor = fresh_inputs(seed=seed)
            input_grad = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.gelu_bw(grad_tensor, input_tensor, approximate=approximate, input_grad=input_grad, queue_id=0)
            check(input_grad, grad_data, in_data, f"approximate={approximate!r} preallocated output (seed={seed})")

            if entries_before is None:
                entries_before = device.num_program_cache_entries()
            else:
                assert device.num_program_cache_entries() == entries_before, (
                    f"re-running approximate={approximate!r} with a fresh preallocated output must reuse "
                    "the cached program; the caller-owned output tensor must be rebound, not re-hashed."
                )

    device.disable_and_clear_program_cache()
