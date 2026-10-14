# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import (
    assert_allclose,
    assert_with_ulp,
    flush_subnormal_values_to_zero,
    generate_all_bfloat16_bitpatterns,
)

GELU_VARIANTS = (
    (ttnn.GeluVariant.Accurate, "none"),
    (ttnn.GeluVariant.Tanh, "tanh"),
)
GELU_VARIANT_PARAMS = [pytest.param(variant, approximate, id=approximate) for variant, approximate in GELU_VARIANTS]
SHAPE_TEST_CASES = (
    pytest.param(torch.Size([32]), id="rank1-tile-aligned"),
    pytest.param(torch.Size([25, 34]), id="rank2-unaligned"),
    pytest.param(torch.Size([1, 32, 32]), id="rank3-tile-aligned"),
    pytest.param(torch.Size([1, 3, 323, 389]), id="rank4-unaligned"),
)
EXHAUSTIVE_TEST_CASES = (
    pytest.param(ttnn.GeluVariant.Accurate, "none", torch.bfloat16, ttnn.bfloat16, 2e-2, 9e-3, id="none-bf16"),
    # The tanh kernel has a 0.0134 absolute error at its BF16 zero crossing.
    pytest.param(ttnn.GeluVariant.Tanh, "tanh", torch.bfloat16, ttnn.bfloat16, 4.9e-2, 1.5e-2, id="tanh-bf16"),
    pytest.param(ttnn.GeluVariant.Accurate, "none", torch.float32, ttnn.float32, 1e-2, 9e-3, id="none-fp32"),
    pytest.param(ttnn.GeluVariant.Tanh, "tanh", torch.float32, ttnn.float32, 1e-4, 1e-4, id="tanh-fp32"),
)
SPECIAL_VALUE_DTYPES = (
    pytest.param(torch.bfloat16, ttnn.bfloat16, 4.9e-2, 9e-3, id="bf16"),
    pytest.param(torch.float32, ttnn.float32, 1e-4, 1e-4, id="fp32"),
)
SPECIAL_VALUE_CASES = (
    pytest.param(ttnn.GeluVariant.Accurate, "none", float("inf"), "one", id="none-pos-inf"),
    pytest.param(ttnn.GeluVariant.Accurate, "none", float("-inf"), "zero", id="none-neg-inf"),
    pytest.param(
        ttnn.GeluVariant.Accurate,
        "none",
        float("nan"),
        "nan",
        id="none-nan",
        marks=pytest.mark.xfail(
            reason="GELU polynomial backward currently treats NaN as a large positive value and returns 1.0",
            strict=True,
        ),
    ),
    pytest.param(ttnn.GeluVariant.Tanh, "tanh", float("inf"), "one", id="tanh-pos-inf"),
    pytest.param(ttnn.GeluVariant.Tanh, "tanh", float("-inf"), "zero", id="tanh-neg-inf"),
    pytest.param(ttnn.GeluVariant.Tanh, "tanh", float("nan"), "nan", id="tanh-nan"),
)


def _gelu_bw_reference(input_tensor, grad_tensor, approximate):
    input_tensor = input_tensor.to(torch.float32).detach().requires_grad_(True)
    output_tensor = torch.nn.functional.gelu(input_tensor, approximate=approximate)
    output_tensor.backward(grad_tensor.to(torch.float32))
    return input_tensor.grad


def _make_exhaustive_inputs(torch_dtype):
    # These are every BF16 bit pattern, promoted losslessly for the FP32 path.
    # Tenstorrent hardware flushes subnormal values, so flush them before both the
    # host reference and device execution. Non-finite encodings are covered by the
    # dedicated special-value test below.
    input_tensor = generate_all_bfloat16_bitpatterns(torch_dtype)
    input_tensor = flush_subnormal_values_to_zero(input_tensor)
    input_tensor[~torch.isfinite(input_tensor)] = 0.0

    # This exhaustive test isolates the derivative. The preallocated-output and
    # program-cache tests below use non-constant gradients to validate scaling.
    return input_tensor, torch.ones_like(input_tensor)


def _bf16_tolerance(approximate):
    return (4.9e-2, 1.5e-2) if approximate == "tanh" else (2e-2, 9e-3)


@pytest.mark.parametrize("variant,approximate,torch_dtype,ttnn_dtype,rtol,atol", EXHAUSTIVE_TEST_CASES)
def test_gelu_bw_exhaustive_allclose(device, variant, approximate, torch_dtype, ttnn_dtype, rtol, atol):
    input_data, grad_data = _make_exhaustive_inputs(torch_dtype)
    expected = _gelu_bw_reference(input_data, grad_data, approximate)

    # PyTorch's tanh-GELU backward can itself overflow for the largest finite
    # inputs. Those encodings are still sent to the device, but have no finite
    # PyTorch reference for an allclose comparison.
    finite_reference = torch.isfinite(expected)

    input_tensor = ttnn.from_torch(input_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant)[0])

    assert torch.isfinite(actual[finite_reference]).all(), "device output is non-finite for a finite reference"
    assert_allclose(expected[finite_reference], actual[finite_reference], rtol=rtol, atol=atol)


def test_gelu_bw_default_matches_none(device):
    input_data = torch.linspace(-5.0, 5.0, 32 * 32, dtype=torch.bfloat16).reshape(32, 32)
    grad_data = torch.linspace(-2.0, 2.0, 32 * 32, dtype=torch.bfloat16).reshape(32, 32)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    default_actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor)[0])
    none_actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor, variant=ttnn.GeluVariant.Accurate)[0])
    assert torch.equal(default_actual, none_actual)


def test_gelu_bw_rejects_fast_lut(device, expect_error):
    input_data = torch.zeros((32, 32), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(
        torch.ones_like(input_data), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    with expect_error(RuntimeError, "does not support GeluVariant::FAST_LUT"):
        ttnn.gelu_bw(grad_tensor, input_tensor, variant=ttnn.GeluVariant.FastLut)


@pytest.mark.parametrize("variant,approximate", GELU_VARIANT_PARAMS)
@pytest.mark.parametrize("shape", SHAPE_TEST_CASES)
def test_gelu_bw_shape_coverage(device, variant, approximate, shape):
    torch.manual_seed(shape.numel())
    input_data = torch.rand(shape, dtype=torch.bfloat16) * 4 - 2
    grad_data = torch.rand(shape, dtype=torch.bfloat16) * 4 - 2
    input_tensor = ttnn.from_torch(input_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant)[0])
    rtol, atol = _bf16_tolerance(approximate)
    assert_allclose(_gelu_bw_reference(input_data, grad_data, approximate), actual, rtol=rtol, atol=atol)


@pytest.mark.parametrize("torch_dtype,ttnn_dtype,rtol,atol", SPECIAL_VALUE_DTYPES)
@pytest.mark.parametrize("variant,approximate,input_value,expected", SPECIAL_VALUE_CASES)
def test_gelu_bw_special_values(
    device, variant, approximate, input_value, expected, torch_dtype, ttnn_dtype, rtol, atol
):
    if approximate == "tanh" and torch.isinf(torch.tensor(input_value)) and ttnn_dtype == ttnn.float32:
        pytest.xfail("FP32 tanh GELU backward overflows for infinite inputs and produces NaN")
    if approximate == "tanh" and torch.isnan(torch.tensor(input_value)) and ttnn_dtype == ttnn.bfloat16:
        pytest.xfail("BF16 tanh GELU backward treats NaN as a large positive value and returns 1.0")

    input_data = torch.tensor([input_value] + [0.0] * 31, dtype=torch_dtype)
    grad_data = torch.ones_like(input_data)
    input_tensor = ttnn.from_torch(
        input_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device, preserve_nan_values=True
    )
    grad_tensor = ttnn.from_torch(grad_data, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant)[0])[0]

    if expected == "nan":
        assert torch.isnan(actual), "GELU backward must propagate NaN inputs"
    else:
        expected_value = 1.0 if expected == "one" else 0.0
        assert torch.isclose(actual, torch.tensor(expected_value, dtype=torch_dtype), rtol=rtol, atol=atol)


@pytest.mark.parametrize("variant,approximate", GELU_VARIANT_PARAMS)
def test_bw_gelu_opt_output(variant, approximate, device):
    shape = torch.Size([1, 1, 320, 384])
    input_data = torch.linspace(-5.0, 5.0, shape.numel(), dtype=torch.bfloat16).reshape(shape)
    grad_data = torch.linspace(-2.0, 2.0, shape.numel(), dtype=torch.bfloat16).reshape(shape)
    input_tensor = ttnn.from_torch(input_data, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_grad = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages(device)
    ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant, input_grad=input_grad, queue_id=0)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages(device))
    rtol, atol = _bf16_tolerance(approximate)
    assert_allclose(
        _gelu_bw_reference(input_data, grad_data, approximate),
        input_grad.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch(),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("variant,approximate", GELU_VARIANT_PARAMS)
@pytest.mark.parametrize(
    "grad_dtype,input_dtype,ulp_threshold",
    (
        # float32 output: the mixed program and its widened twin are the same float32 computation.
        (ttnn.bfloat16, ttnn.float32, 0),
        # bfloat16 output: the device packs float32 DEST to bfloat16 while the twin's float32
        # result is rounded on the host, so allow the one ULP that final rounding can differ by.
        (ttnn.float32, ttnn.bfloat16, 1),
    ),
    ids=("bf16_grad-fp32_input", "fp32_grad-bf16_input"),
)
def test_bw_gelu_mixed_grad_and_input_dtypes(variant, approximate, grad_dtype, input_dtype, ulp_threshold, device):
    """grad_output and input need not share a dtype: each kernel switches the unpacker's format
    between the two operand buffers when they differ, including eltwise_bw_gelu_tanh_fp32.cpp,
    which reads grad_output partway through its chain after several input reads.

    Widening bfloat16 to float32 is exact, so a mixed-dtype call must reproduce the same-dtype
    float32 call on the widened operands -- the operand formats are the only difference between
    the two programs. That isolates the format switching from the kernels' own accuracy, which
    test_gelu_bw_exhaustive_allclose bounds against PyTorch for the same-dtype paths."""
    shape = torch.Size([1, 1, 32, 32])
    input_data = torch.linspace(-5.0, 5.0, shape.numel(), dtype=torch.float32).reshape(shape)
    grad_data = torch.linspace(-2.0, 2.0, shape.numel(), dtype=torch.float32).reshape(shape)
    input_tensor = ttnn.from_torch(input_data, input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    grad_tensor = ttnn.from_torch(grad_data, grad_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant)[0]
    assert output.dtype == input_dtype

    widened_input = ttnn.from_torch(
        ttnn.to_torch(input_tensor).float(), ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    widened_grad = ttnn.from_torch(
        ttnn.to_torch(grad_tensor).float(), ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    expected = ttnn.to_torch(ttnn.gelu_bw(widened_grad, widened_input, variant=variant)[0]).float()

    output_torch_dtype = torch.bfloat16 if input_dtype == ttnn.bfloat16 else torch.float32
    assert_with_ulp(
        expected_result=expected.to(output_torch_dtype),
        actual_result=ttnn.to_torch(output).to(output_torch_dtype),
        ulp_threshold=ulp_threshold,
    )


# Test gradients across program cache hits
def test_bw_gelu_program_cache_regression(device):
    device.enable_program_cache()
    device.clear_program_cache()
    shape = torch.Size([1, 1, 320, 384])

    def fresh_inputs(seed):
        torch.manual_seed(seed)
        input_data = (torch.rand(shape, dtype=torch.bfloat16) * 4 - 2).detach()
        grad_data = (torch.rand(shape, dtype=torch.bfloat16) * 4 - 2).detach()
        return (
            input_data,
            grad_data,
            ttnn.from_torch(input_data, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
            ttnn.from_torch(grad_data, ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        )

    try:
        for expected_entries, (variant, approximate) in enumerate(GELU_VARIANTS, start=1):
            input_data, grad_data, input_tensor, grad_tensor = fresh_inputs(expected_entries)
            actual = ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant)[0]
            rtol, atol = _bf16_tolerance(approximate)
            assert_allclose(
                _gelu_bw_reference(input_data, grad_data, approximate), ttnn.to_torch(actual), rtol=rtol, atol=atol
            )
            assert device.num_program_cache_entries() == expected_entries

        for seed, variant, approximate in (
            (42, ttnn.GeluVariant.Accurate, "none"),
            (99, ttnn.GeluVariant.Tanh, "tanh"),
        ):
            input_data, grad_data, input_tensor, grad_tensor = fresh_inputs(seed)
            actual = ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant)[0]
            rtol, atol = _bf16_tolerance(approximate)
            assert_allclose(
                _gelu_bw_reference(input_data, grad_data, approximate), ttnn.to_torch(actual), rtol=rtol, atol=atol
            )
            assert device.num_program_cache_entries() == 2

        for variant, approximate in GELU_VARIANTS:
            entries_before = None
            for seed in (100, 200):
                input_data, grad_data, input_tensor, grad_tensor = fresh_inputs(seed)
                rtol, atol = _bf16_tolerance(approximate)
                input_grad = ttnn.from_torch(
                    torch.zeros(shape, dtype=torch.bfloat16),
                    ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant, input_grad=input_grad, queue_id=0)
                assert_allclose(
                    _gelu_bw_reference(input_data, grad_data, approximate),
                    ttnn.to_torch(input_grad),
                    rtol=rtol,
                    atol=atol,
                )
                if entries_before is None:
                    entries_before = device.num_program_cache_entries()
                else:
                    assert device.num_program_cache_entries() == entries_before
    finally:
        device.disable_and_clear_program_cache()


# Layout, sharding and padding are data movement only: none of them may change a single value, so
# every configuration below must be bit-identical (0 ULP) to the plain interleaved TILE call on the
# same data. That checks the plumbing exactly, independent of the kernels' own accuracy, which the
# exhaustive and ULP tests bound for the interleaved path.
LAYOUT_DTYPES = (
    pytest.param(ttnn.bfloat16, torch.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, torch.float32, id="fp32"),
)


def _core_range(x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x1, y1))})


def _sharded(memory_layout, shard_shape, grid, buffer_type=ttnn.BufferType.L1):
    return ttnn.MemoryConfig(
        memory_layout, buffer_type, ttnn.ShardSpec(grid, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR)
    )


def _gelu_bw_as(input_data, grad_data, dtype, torch_dtype, variant, layout, memory_config, device):
    input_tensor = ttnn.from_torch(input_data, dtype, layout=layout, device=device, memory_config=memory_config)
    grad_tensor = ttnn.from_torch(grad_data, dtype, layout=layout, device=device, memory_config=memory_config)
    output = ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant, memory_config=memory_config)[0]
    assert output.layout == layout
    return ttnn.to_torch(output).to(torch_dtype)


def _layout_inputs(shape):
    numel = torch.Size(shape).numel()
    input_data = torch.linspace(-5.0, 5.0, numel, dtype=torch.float32).reshape(shape)
    grad_data = torch.linspace(-2.0, 2.0, numel, dtype=torch.float32).reshape(shape).flip(-1)
    return input_data, grad_data


@pytest.mark.parametrize("variant,approximate", GELU_VARIANT_PARAMS)
@pytest.mark.parametrize("dtype,torch_dtype", LAYOUT_DTYPES)
@pytest.mark.parametrize(
    "memory_config",
    (
        # Every operand L1-sharded on one spec: the circular buffers alias the shards.
        pytest.param(_sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (32, 128), _core_range(0, 7)), id="height"),
        pytest.param(_sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, (128, 64), _core_range(1, 1)), id="block"),
        # DRAM-sharded: not aliasable, so every operand is addressed by page instead.
        pytest.param(
            _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, (256, 64), _core_range(1, 0), ttnn.BufferType.DRAM),
            id="width_dram",
        ),
    ),
)
def test_bw_gelu_sharded_matches_interleaved(variant, approximate, dtype, torch_dtype, memory_config, device):
    input_data, grad_data = _layout_inputs((1, 1, 256, 128))
    expected = _gelu_bw_as(
        input_data, grad_data, dtype, torch_dtype, variant, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, device
    )
    actual = _gelu_bw_as(input_data, grad_data, dtype, torch_dtype, variant, ttnn.TILE_LAYOUT, memory_config, device)
    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=0)


@pytest.mark.parametrize("variant,approximate", GELU_VARIANT_PARAMS)
@pytest.mark.parametrize("dtype,torch_dtype", LAYOUT_DTYPES)
@pytest.mark.parametrize(
    "shape,memory_config",
    (
        pytest.param((1, 1, 256, 128), ttnn.DRAM_MEMORY_CONFIG, id="interleaved"),
        # Each 100-element row splits into two 50-element width-shard pages, which are not NoC-aligned.
        pytest.param(
            (1, 1, 37, 100),
            _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, (37, 50), _core_range(1, 0)),
            id="width_sharded_unaligned",
        ),
    ),
)
def test_bw_gelu_row_major_matches_tile(variant, approximate, dtype, torch_dtype, shape, memory_config, device):
    input_data, grad_data = _layout_inputs(shape)
    expected = _gelu_bw_as(
        input_data, grad_data, dtype, torch_dtype, variant, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, device
    )
    actual = _gelu_bw_as(
        input_data, grad_data, dtype, torch_dtype, variant, ttnn.ROW_MAJOR_LAYOUT, memory_config, device
    )
    assert_with_ulp(expected_result=expected, actual_result=actual, ulp_threshold=0)


@pytest.mark.parametrize("variant,approximate", GELU_VARIANT_PARAMS)
@pytest.mark.parametrize("dtype,torch_dtype", LAYOUT_DTYPES)
def test_bw_gelu_preserves_input_physical_padding(variant, approximate, dtype, torch_dtype, device):
    """A 40x40 logical input padded to 96x96 (9 tiles) must get a 96x96 output: the op writes one
    page per input tile, and sizing the output from the logical shape (64x64, 4 tiles) left it
    undersized. The logical region must match the ordinarily tile-padded call exactly."""
    input_data, grad_data = _layout_inputs((1, 1, 40, 40))
    padded_shape = [1, 1, 96, 96]
    input_tensor = ttnn.tilize_with_val_padding(
        ttnn.from_torch(input_data, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device), padded_shape, 0.0
    )
    grad_tensor = ttnn.tilize_with_val_padding(
        ttnn.from_torch(grad_data, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device), padded_shape, 0.0
    )

    output = ttnn.gelu_bw(grad_tensor, input_tensor, variant=variant)[0]

    assert list(output.padded_shape) == padded_shape
    expected = _gelu_bw_as(
        input_data, grad_data, dtype, torch_dtype, variant, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, device
    )
    assert_with_ulp(
        expected_result=expected, actual_result=ttnn.to_torch(output).to(torch_dtype)[..., :40, :40], ulp_threshold=0
    )
