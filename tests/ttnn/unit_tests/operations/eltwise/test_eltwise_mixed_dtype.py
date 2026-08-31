# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device

# bfloat8_b has no torch equivalent; host tensors use bfloat16 (same mantissa width)
# so from_torch packing and ULP comparison share one resolution.
TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}

FLOAT_DTYPES = (ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32)
# ULP is measured in the output dtype. bfloat8_b is compared at bfloat16 resolution
# (see assert_with_ulp). Same-format bf16/fp32 neg is exact; block-float packing
# and host vs device bfp8 packers can differ.
ULP_THRESHOLD = {
    (ttnn.bfloat16, ttnn.bfloat16): 0,
    (ttnn.bfloat16, ttnn.float32): 0,
    (ttnn.float32, ttnn.bfloat16): 0,
    (ttnn.float32, ttnn.float32): 0,
    (ttnn.bfloat8_b, ttnn.bfloat8_b): 1,
    (ttnn.bfloat8_b, ttnn.bfloat16): 1,
    (ttnn.bfloat8_b, ttnn.float32): 1,
    (ttnn.bfloat16, ttnn.bfloat8_b): 1,
    (ttnn.float32, ttnn.bfloat8_b): 1,
}

SHAPE = (64, 64)


def _preallocate(device, torch_input, in_dtype, out_dtype):
    input_tensor = ttnn.from_torch(torch_input, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.from_torch(
        torch.zeros(torch_input.shape, dtype=TORCH_DTYPE[out_dtype]),
        dtype=out_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return input_tensor, output_tensor


def test_neg_mixed_dtype(device):
    """Preallocated output dtype can differ from the input. Each distinct (in, out)
    pair must miss the program cache once; a second pass over the same pairs must hit."""
    torch.manual_seed(0)
    fixed_input = torch.ones(SHAPE, dtype=torch.float32) * 1.22
    ttnn_op = ttnn.neg

    device.disable_and_clear_program_cache()
    device.enable_program_cache()

    pairs = []
    for in_dt in FLOAT_DTYPES:
        for out_dt in FLOAT_DTYPES:
            torch_input = fixed_input.to(TORCH_DTYPE[in_dt])
            input_tensor, output_tensor = _preallocate(device, torch_input, in_dt, out_dt)
            pairs.append((in_dt, out_dt, input_tensor, output_tensor))

    for in_dt, out_dt, input_tensor, output_tensor in pairs:
        entries_before = device.num_program_cache_entries()
        ttnn_op(input_tensor, output_tensor=output_tensor)
        entries_after = device.num_program_cache_entries()
        assert entries_after == entries_before + 1, (
            f"expected one cache miss for in={in_dt} out={out_dt}, " f"entries {entries_before} -> {entries_after}"
        )
        # Golden uses device-visible input so bfloat8_b packing is in the reference.
        # Pass the ttnn tensor into assert_with_ulp so bfloat8_b is compared at
        # bfloat16 resolution; ttnn.to_torch() would upcast it to float32.
        golden_fn = ttnn.get_golden_function(ttnn_op)
        golden = golden_fn(ttnn.to_torch(input_tensor)).to(TORCH_DTYPE[out_dt])
        assert output_tensor.dtype == out_dt
        assert_with_ulp(golden, output_tensor, ULP_THRESHOLD[(in_dt, out_dt)])

    entries_after_first_pass = device.num_program_cache_entries()
    # second pass over the same (in, out) pairs must be program-cache hits
    for _, _, input_tensor, output_tensor in pairs:
        ttnn_op(input_tensor, output_tensor=output_tensor)
    assert (
        device.num_program_cache_entries() == entries_after_first_pass
    ), "second pass over the same (in, out) pairs must be program-cache hits"


def test_neg_rejects_int_float_preallocated_output(device, expect_error):
    input_tensor = ttnn.from_torch(
        torch.ones(SHAPE, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.from_torch(
        torch.zeros(SHAPE, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(RuntimeError, "Integer and float dtypes cannot be mixed"):
        ttnn.neg(input_tensor, output_tensor=output_tensor)


def test_typecast_rejects_mismatched_preallocated_output(device, expect_error):
    input_tensor = ttnn.from_torch(
        torch.ones(SHAPE, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    output_tensor = ttnn.from_torch(
        torch.zeros(SHAPE, dtype=torch.float32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    typecast_to_bf16 = [
        ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, ttnn.DataType.FLOAT32.value, ttnn.DataType.BFLOAT16.value)
    ]
    with expect_error(RuntimeError, "does not match the typecast/bitcast target dtype"):
        ttnn.unary_chain(input_tensor, typecast_to_bf16, output_tensor=output_tensor)
    with expect_error(RuntimeError, "dtype should match"):
        ttnn.typecast(input_tensor, ttnn.bfloat16, output_tensor=output_tensor)
