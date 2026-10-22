# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import contextlib

import pytest
import torch

import ttnn


@contextlib.contextmanager
def comparison_mode(pcc=0.99, raise_on_failure=True):
    with (
        ttnn.manage_config("enable_comparison_mode", True),
        ttnn.manage_config("comparison_mode_pcc", pcc),
        ttnn.manage_config("comparison_mode_should_raise_exception", raise_on_failure),
    ):
        yield


SINGLE_TILE = (1, 1, 32, 32)
MOREH_SOFTMAX_OP = ttnn._ttnn.operations.moreh.MorehSoftmaxOp
MOREH_SOFTMAX_BACKWARD_OP = ttnn._ttnn.operations.moreh.MorehSoftmaxBackwardOp


@contextlib.contextmanager
def _global_comparison_mode(tmp_path):
    # Global goldens are retained only while a report path is configured.
    existing_tensor_ids = set(ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR)
    try:
        with comparison_mode(), ttnn.manage_config("root_report_path", tmp_path), ttnn.manage_config(
            "report_name", "global_golden_state"
        ):
            yield
    finally:
        for tensor_id in set(ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR) - existing_tensor_ids:
            ttnn.decorators.TENSOR_ID_TO_GLOBAL_LEVEL_GOLDEN_TENSOR.pop(tensor_id, None)


def _to_device(torch_tensor, device, dtype=None, layout=ttnn.TILE_LAYOUT, memory_config=None):
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _complex_input(torch_real, torch_imag, device):
    return ttnn.complex_tensor(_to_device(torch_real, device), _to_device(torch_imag, device))


def _block_float_sensitive_values(shape):
    # Each 16-value block shares one exponent, so BFLOAT8_B and BFLOAT4_B flush the ones beside 1024 to zero.
    values = torch.ones(shape)
    values[..., ::16] = 1024.0
    return values


def _bfloat16_sensitive_values(shape):
    return torch.full(shape, 1.0 + 2.0**-10)


def _bit_patterns(float_values, bits_dtype):
    return float_values.view(bits_dtype).to(torch.int32)


def _linspace_tile(low, high):
    return torch.linspace(low, high, 1024).reshape(SINGLE_TILE)


def _repeated_tile(values):
    return torch.tensor(values).repeat(1024 // len(values)).reshape(SINGLE_TILE)


def _small_integer_values(shape):
    return torch.randint(-4, 5, shape).to(torch.bfloat16)


def _positive_operands():
    return (
        torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1,
        torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1,
        ttnn.bfloat16,
    )


def _small_integer_operands():
    return (
        torch.randint(0, 3, SINGLE_TILE).to(torch.bfloat16),
        torch.randint(0, 3, SINGLE_TILE).to(torch.bfloat16),
        ttnn.bfloat16,
    )


def _ldexp_operands():
    return (
        torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1,
        torch.randint(-2, 3, SINGLE_TILE).to(torch.bfloat16),
        ttnn.bfloat16,
    )


def _int32_operands():
    return (
        torch.randint(1, 16, SINGLE_TILE, dtype=torch.int32),
        torch.randint(1, 16, SINGLE_TILE, dtype=torch.int32),
        ttnn.int32,
    )


def _registered_golden_output(operation, *args, **kwargs):
    preprocess = (
        operation.preprocess_golden_function_inputs or ttnn.decorators.default_preprocess_golden_function_inputs
    )
    golden_args, golden_kwargs = preprocess(args, kwargs)
    if operation.python_fully_qualified_name.endswith("_bw"):
        golden_args, golden_kwargs = ttnn.decorators.prepare_backward_golden_inputs((golden_args, golden_kwargs))
    return operation.golden_function(*golden_args, **golden_kwargs)


def _assert_golden_matches_output(golden, output):
    torch_output = ttnn.to_torch(output)
    assert (
        golden.shape == torch_output.shape
    ), f"golden shape {tuple(golden.shape)} != output {tuple(torch_output.shape)}"
    assert golden.dtype == torch_output.dtype, f"golden dtype {golden.dtype} != output dtype {torch_output.dtype}"
    assert torch.equal(golden, torch_output), "golden values differ from the stored output values"


def _assert_golden_dtype_and_close(golden, output, *, rtol, atol):
    torch_output = ttnn.to_torch(output)
    assert (
        golden.shape == torch_output.shape
    ), f"golden shape {tuple(golden.shape)} != output {tuple(torch_output.shape)}"
    assert golden.dtype == torch_output.dtype, f"golden dtype {golden.dtype} != output dtype {torch_output.dtype}"
    torch.testing.assert_close(golden.float(), torch_output.float(), rtol=rtol, atol=atol)


def _capture_local_comparison_records(monkeypatch):
    comparison_records = []
    record_tensor_comparison_data = ttnn.graph.record_tensor_comparison_data

    def record(**kwargs):
        comparison_records.extend(kwargs.get("local_tensor_comparison_records") or [])
        return record_tensor_comparison_data(**kwargs)

    monkeypatch.setattr(ttnn.graph, "record_tensor_comparison_data", record)
    return comparison_records


def _single_core_height_sharded_memory_config():
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _sparse_matmul_program_config():
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(6, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=1,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def _compare_torch_tensors(golden, output, *, fail_on_bad_comparison=True):
    ttnn.decorators.set_tensor_id(ttnn.decorators.get_all_tensors(golden), force=True)
    ttnn.decorators.set_tensor_id(ttnn.decorators.get_all_tensors(output), force=True)
    return ttnn.decorators.compare_tensors_using_pcc(
        "ttnn.test_operation",
        golden,
        output,
        desired_pcc=0.99,
        level="locally",
        fail_on_bad_comparison=fail_on_bad_comparison,
    )


@pytest.mark.requires_fast_runtime_mode_off
def test_leaky_relu_with_positional_negative_slope_in_comparison_mode(device):
    torch_input = torch.full((1, 1, 32, 32), -2.0, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # A positional slope used to be swallowed by the golden's *args, so it was compared against the default
    # slope of 0.01 instead of 0.5.
    with comparison_mode():
        output_tensor = ttnn.leaky_relu(input_tensor, 0.5)

    assert isinstance(output_tensor, ttnn.Tensor)


def _arange_tile():
    return torch.arange(1, 1025, dtype=torch.float32).to(torch.bfloat16).reshape(SINGLE_TILE)


def _scaled_columns_tile():
    columns = torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 32)
    row_scales = torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 32, 1)
    return (row_scales * columns).to(torch.bfloat16)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, torch_input, input_dtype, reduction_kwargs",
    [
        # The device multiplies the reduction by scalar; the golden used to ignore it and reported a false mismatch.
        pytest.param(ttnn.sum, torch.ones(SINGLE_TILE, dtype=torch.bfloat16), None, {"scalar": 0.5}, id="sum"),
        # For int32 input the scaled sum is computed in float32 and truncated back to int32, so a fractional scalar
        # (0.5) must not leave the golden as a float tensor.
        pytest.param(
            ttnn.sum,
            torch.ones(SINGLE_TILE, dtype=torch.int32),
            ttnn.int32,
            {"scalar": 0.5},
            id="sum_int32_fractional_scalar",
        ),
        # scalar=0.0 makes the device mean exactly zero; the golden used to ignore scalar and return the unscaled mean.
        pytest.param(ttnn.mean, _arange_tile(), None, {"scalar": 0.0}, id="mean"),
        # A negative scalar flips the ordering, so the device max is scalar * min(input) and the device min is
        # scalar * max(input); the golden must swap max and min before scaling.
        pytest.param(ttnn.max, _arange_tile(), None, {"scalar": -2.0}, id="max"),
        pytest.param(ttnn.min, _arange_tile(), None, {"scalar": -2.0}, id="min"),
        # Variance scales with scalar**2, so scalar=0.0 must give zero; the golden used to return the unscaled variance.
        pytest.param(ttnn.var, _scaled_columns_tile(), None, {"scalar": 0.0, "correction": False}, id="var"),
        # Standard deviation scales with |scalar|, so scalar=0.0 must give zero; the golden used to return it unscaled.
        pytest.param(ttnn.std, _scaled_columns_tile(), None, {"scalar": 0.0, "correction": False}, id="std"),
    ],
)
def test_reduction_with_scalar_in_comparison_mode(device, operation, torch_input, input_dtype, reduction_kwargs):
    input_tensor = _to_device(torch_input, device, dtype=input_dtype)

    with comparison_mode():
        output_tensor = operation(input_tensor, dim=-1, keepdim=True, **reduction_kwargs)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_selu_with_non_default_parameters_in_comparison_mode(device):
    torch_input = torch.full((1, 1, 32, 32), -1.0, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # The golden used to call torch's selu with its fixed constants, ignoring the scale and alpha arguments.
    with comparison_mode():
        output_tensor = ttnn.selu(input_tensor, scale=0.9, alpha=1.2)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_softmax_over_dimension_zero_in_comparison_mode(device):
    torch_input = torch.zeros((2, 32, 32), dtype=torch.bfloat16)
    torch_input[1] = 4.0
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # The golden used `dim or -1`, which turned dim=0 into the last axis and computed softmax over the wrong dimension.
    with comparison_mode():
        output_tensor = ttnn.softmax(input_tensor, dim=0)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_rms_norm_with_bias_in_comparison_mode(device):
    torch_input = torch.zeros((32, 32), dtype=torch.bfloat16)
    torch_weight = torch.ones((1, 32), dtype=torch.bfloat16)
    torch_bias = torch.arange(32, dtype=torch.float32).to(torch.bfloat16).reshape(1, 32)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    # The input is all zeros, so the output is exactly the bias; the golden used to drop bias and expect zeros.
    with comparison_mode():
        output_tensor = ttnn.rms_norm(input_tensor, weight=weight, bias=bias)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_rms_norm_with_residual_in_comparison_mode(device):
    torch_input = torch.zeros((32, 32), dtype=torch.bfloat16)
    torch_residual = torch.arange(32, dtype=torch.float32).to(torch.bfloat16).repeat(32, 1)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)

    # The input is all zeros, so the output depends only on the residual; the golden used to ignore
    # residual_input_tensor and normalize the input alone.
    with comparison_mode():
        output_tensor = ttnn.rms_norm(input_tensor, residual_input_tensor=residual)

    assert isinstance(output_tensor, ttnn.Tensor)


def _positive_complex_parts():
    return torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1


def _offset_imaginary_complex_parts():
    return torch.rand(SINGLE_TILE, dtype=torch.bfloat16), torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 2


def _polar_complex_parts():
    return torch.full(SINGLE_TILE, 2.0, dtype=torch.bfloat16), torch.zeros(SINGLE_TILE, dtype=torch.bfloat16)


def _partly_zero_complex_parts():
    # Zero real and zero imaginary parts never coincide, so every value of the ops below is finite.
    torch_real = torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1
    torch_real[..., ::2] = 0
    torch_imag = torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1
    torch_imag[..., 1::4] = 0
    return torch_real, torch_imag


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, make_parts, op_kwargs",
    [
        pytest.param(ttnn.abs, _positive_complex_parts, {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, id="abs"),
        # The real-valued output must be compared with both an inherited and an explicit memory config.
        pytest.param(ttnn.real, _offset_imaginary_complex_parts, {"memory_config": None}, id="real_inherited"),
        pytest.param(
            ttnn.real, _offset_imaginary_complex_parts, {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, id="real_dram"
        ),
        # polar takes one ComplexTensor holding (radius, angle) as (real, imag); the golden used to call torch.polar
        # with that single argument although torch.polar needs separate radius and angle tensors.
        pytest.param(ttnn.polar, _polar_complex_parts, {}, id="polar"),
        pytest.param(ttnn.imag, _partly_zero_complex_parts, {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, id="imag"),
        pytest.param(ttnn.angle, _partly_zero_complex_parts, {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, id="angle"),
        pytest.param(ttnn.conj, _partly_zero_complex_parts, {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, id="conj"),
        pytest.param(
            ttnn.is_real, _partly_zero_complex_parts, {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, id="is_real"
        ),
        pytest.param(
            ttnn.is_imag, _partly_zero_complex_parts, {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, id="is_imag"
        ),
        pytest.param(
            ttnn.reciprocal, _partly_zero_complex_parts, {"memory_config": ttnn.DRAM_MEMORY_CONFIG}, id="reciprocal"
        ),
    ],
)
def test_complex_unary_op_in_comparison_mode(device, operation, make_parts, op_kwargs):
    complex_input = _complex_input(*make_parts(), device)

    # A ComplexTensor is two real device tensors rather than a ttnn.Tensor, so default input preprocessing used to
    # pass the wrapper to torch unconverted; it must be rebuilt as a torch complex tensor.
    with comparison_mode():
        operation(complex_input, **op_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
def test_acos_bfloat8_b_out_of_domain_input_is_compared_in_comparison_mode(device, monkeypatch):
    torch_input = torch.linspace(-2.0, 2.0, 1024).reshape(SINGLE_TILE)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.bfloat8_b)
    comparison_records = _capture_local_comparison_records(monkeypatch)

    # A NaN from an out-of-domain lane makes the shared exponent of its whole BFLOAT8_B block non-finite. The golden
    # used to return None and skip the comparison; it now models the block and compares non-finite positions.
    with comparison_mode():
        ttnn.acos(input_tensor)

    assert comparison_records, "the out-of-domain BFLOAT8_B call produced no local comparison"


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, low, high",
    [(ttnn.exp, -4.0, 4.0), (ttnn.tanh, -3.0, 3.0)],
    ids=["exp", "tanh"],
)
def test_fast_and_approximate_mode_degenerate_output_in_comparison_mode(device, operation, low, high):
    # Fast exp is accurate to ~5% relative error and the approximate tanh LUT to ~0.03 absolute error, so the golden
    # must relax its tolerance in this mode. A constant tile makes PCC degenerate, so that tolerance decides the result.
    for value in torch.linspace(low, high, 9).tolist():
        input_tensor = _to_device(torch.full(SINGLE_TILE, value, dtype=torch.bfloat16), device)
        with comparison_mode():
            operation(input_tensor, fast_and_approximate_mode=True)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, torch_input, args, op_kwargs",
    [
        # fast_and_approximate_mode is a TTNN-only kwarg that the generic unary golden wrapper must discard instead of
        # forwarding it to the torch log function.
        pytest.param(ttnn.log, _linspace_tile(0.5, 1.5), (), {"fast_and_approximate_mode": True}, id="log"),
        pytest.param(ttnn.log10, _linspace_tile(0.5, 1.5), (), {"fast_and_approximate_mode": True}, id="log10"),
        pytest.param(ttnn.log1p, _linspace_tile(0.5, 1.5), (), {"fast_and_approximate_mode": True}, id="log1p"),
        pytest.param(ttnn.log2, _linspace_tile(0.5, 1.5), (), {"fast_and_approximate_mode": True}, id="log2"),
        # The fast erf stays within PCC of the exact torch reference, so the golden need not model the mode.
        pytest.param(ttnn.erf, _linspace_tile(-3.0, 3.0), (), {"fast_and_approximate_mode": True}, id="erf"),
        # The golden ignores the approximation mode, so the approximate result must stay within PCC of the exact one.
        pytest.param(ttnn.sqrt, _linspace_tile(0.01, 100.0), (), {"fast_and_approximate_mode": True}, id="sqrt"),
        pytest.param(
            ttnn.sigmoid,
            _linspace_tile(-8.0, 8.0),
            (),
            {"vector_mode": 4, "mode": ttnn.SigmoidMode.AccurateWithFastExp},
            id="sigmoid_accurate_with_fast_exp",
        ),
        pytest.param(
            ttnn.sigmoid,
            _linspace_tile(-8.0, 8.0),
            (),
            {"vector_mode": 4, "mode": ttnn.SigmoidMode.FastApproximate},
            id="sigmoid_fast_approximate",
        ),
        pytest.param(
            ttnn.sigmoid_accurate,
            _linspace_tile(-8.0, 8.0),
            (),
            {"fast_and_approximate_mode": True},
            id="sigmoid_accurate",
        ),
        # Half the inputs are out of domain; the golden must predict where the bfloat16 device output is non-finite.
        pytest.param(ttnn.acos, _linspace_tile(-2.0, 2.0), (), {}, id="acos_out_of_domain"),
        pytest.param(ttnn.asin, _linspace_tile(-2.0, 2.0), (), {}, id="asin_out_of_domain"),
        pytest.param(ttnn.acosh, _linspace_tile(0.0, 4.0), (), {}, id="acosh_out_of_domain"),
        # A tiny or zero bfloat16 scalar divisor is ill-conditioned; the golden relaxes the value comparison and only
        # requires the non-finite positions to agree.
        pytest.param(ttnn.remainder, _linspace_tile(-4.0, 4.0), (1e-3,), {}, id="remainder_tiny_scalar"),
        pytest.param(ttnn.remainder, _linspace_tile(-4.0, 4.0), (0.0,), {}, id="remainder_zero_scalar"),
        # Near-overflow, infinite and NaN inputs must produce the reciprocal torch predicts, including signed zero.
        pytest.param(
            ttnn.reciprocal,
            _repeated_tile([2.0**126, float("-inf"), float("nan"), 2.0]),
            (),
            {},
            id="reciprocal_special_values",
        ),
    ],
)
def test_unary_on_deterministic_tile_in_comparison_mode(device, operation, torch_input, args, op_kwargs):
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device)

    with comparison_mode():
        operation(input_tensor, *args, **op_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", [ttnn.tril, ttnn.triu], ids=["tril", "triu"])
def test_triangular_with_keyword_diagonal_in_comparison_mode(device, operation):
    torch_input = torch.rand((32, 32)) * 0.01
    torch_input += torch.diag(torch.full((32,), 100.0)) + torch.diag(torch.full((31,), 100.0), diagonal=1)
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device)

    # The generic unary wrapper filters out keyword arguments, which used to drop diagonal and compare against
    # diagonal=0; the input has two large diagonals so a wrong diagonal gives a clearly different result.
    with comparison_mode():
        operation(input_tensor, diagonal=1)


@pytest.mark.requires_fast_runtime_mode_off
def test_logit_with_eps_in_comparison_mode(device):
    torch_input = torch.linspace(0.0, 1.0, 1024).reshape(SINGLE_TILE)
    torch_input[..., ::2] = 0.0
    torch_input[..., 1::4] = 1.0
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device)

    # The input contains exact 0 and 1, where logit diverges unless it is clamped to [eps, 1 - eps], so eps must
    # reach the golden. The public binding makes eps keyword-only, so only the keyword form can be exercised.
    with comparison_mode():
        ttnn.logit(input_tensor, eps=0.1)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
def test_rdiv_with_rounding_mode_in_comparison_mode(device, rounding_mode):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 0.5, device)

    # "trunc" and "floor" round the quotient, so the golden must pass rounding_mode to torch.div instead of
    # computing a true division.
    with comparison_mode():
        ttnn.rdiv(input_tensor, 2.0, rounding_mode=rounding_mode)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", [ttnn.normalize_global, ttnn.normalize_hw], ids=["global", "hw"])
def test_normalize_uses_population_standard_deviation_in_comparison_mode(device, operation):
    torch_input = torch.rand(SINGLE_TILE, dtype=torch.float32)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.float32)
    dims = (0, 1, 2, 3) if operation is ttnn.normalize_global else (-2, -1)

    # The device divides by the population standard deviation, but torch.std defaults to the sample one; the golden
    # must use correction=0. The last lines pin it against an independent population-std reference.
    with comparison_mode():
        operation(input_tensor)

    population_reference = (torch_input - torch_input.mean(dims, keepdim=True)) / torch_input.std(
        dims, keepdim=True, correction=0
    )
    torch.testing.assert_close(_registered_golden_output(operation, input_tensor), population_reference)


@pytest.mark.requires_fast_runtime_mode_off
def test_logical_not_int32_min_in_comparison_mode(device):
    torch_input = torch.zeros(SINGLE_TILE, dtype=torch.int32)
    torch_input[..., ::2] = torch.iinfo(torch.int32).min
    input_tensor = _to_device(torch_input, device, dtype=ttnn.int32)

    # INT32_MIN is nonzero but is the one int32 whose negation overflows, so logical_not must still map it to
    # false; every other lane is zero and maps to true.
    with comparison_mode():
        ttnn.logical_not(input_tensor)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", [ttnn.bitwise_and, ttnn.bitwise_or], ids=["bitwise_and", "bitwise_or"])
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.uint32], ids=["uint16", "uint32"])
@pytest.mark.parametrize("operand_form", ["scalar", "tensor"])
def test_bitwise_with_highest_unsigned_values_in_comparison_mode(device, operation, dtype, operand_form):
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype)
    highest_value = 2**16 - 1 if dtype == ttnn.uint16 else 2**32 - 1
    torch_input = torch.full(SINGLE_TILE, highest_value, dtype=torch.int64)
    torch_input[..., ::2] = highest_value - 1
    input_tensor = _to_device(torch_input.to(torch_dtype), device, dtype=dtype)
    other = 0x5555
    if operand_form == "tensor":
        other = _to_device(torch.full(SINGLE_TILE, 0x5555, dtype=torch.int64).to(torch_dtype), device, dtype=dtype)

    # Torch has no uint16/uint32 bitwise kernels, so the golden widens to int64. Values at the top of the unsigned
    # range catch sign-extension errors when the result is narrowed back.
    with comparison_mode():
        operation(input_tensor, other)


@pytest.mark.requires_fast_runtime_mode_off
def test_logical_right_shift_with_invalid_counts_in_comparison_mode(device):
    input_tensor = _to_device(torch.full(SINGLE_TILE, -1, dtype=torch.int32), device, dtype=ttnn.int32)
    shift_counts = torch.tensor([31, 32, -1, 33], dtype=torch.int32).repeat(256).reshape(SINGLE_TILE)
    shift_tensor = _to_device(shift_counts, device, dtype=ttnn.int32)

    # Shift counts of 32, 33 and -1 are outside the valid 0..31 range and have no Torch-defined result, so the
    # golden must reproduce the device's values bit-for-bit (verified below).
    with comparison_mode():
        output = ttnn.logical_right_shift(input_tensor, shift_tensor)

    _assert_golden_matches_output(
        _registered_golden_output(ttnn.logical_right_shift, input_tensor, shift_tensor), output
    )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "input_dtype, ops_chain",
    [
        pytest.param(
            ttnn.bfloat16,
            [
                ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
                ttnn.UnaryWithParam(
                    ttnn.UnaryOpType.TYPECAST, ttnn.DataType.BFLOAT16.value, ttnn.DataType.FLOAT32.value
                ),
            ],
            id="relu_typecast",
        ),
        pytest.param(
            ttnn.uint16,
            [ttnn.UnaryWithParam(ttnn.UnaryOpType.BITCAST, ttnn.DataType.UINT16.value, ttnn.DataType.BFLOAT16.value)],
            id="bitcast",
        ),
    ],
)
def test_unary_chain_with_dtype_changing_op_in_comparison_mode(device, input_dtype, ops_chain):
    float_values = torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1
    torch_input = float_values if input_dtype == ttnn.bfloat16 else _bit_patterns(float_values, torch.int16)
    input_tensor = _to_device(torch_input, device, dtype=input_dtype)

    # TYPECAST and BITCAST chain ops carry their source and target dtypes as float params, which the golden used
    # to pass to a torch op as if they were scalar arguments instead of decoding them.
    with comparison_mode():
        ttnn.unary_chain(input_tensor, ops_chain)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "input_dtype, output_dtype, float_dtype, bits_dtype",
    [
        (ttnn.uint16, ttnn.bfloat16, torch.bfloat16, torch.int16),
        (ttnn.uint32, ttnn.float32, torch.float32, torch.int32),
    ],
    ids=["uint16_to_bfloat16", "uint32_to_float32"],
)
def test_bitcast_to_same_width_float_in_comparison_mode(device, input_dtype, output_dtype, float_dtype, bits_dtype):
    torch_input = _bit_patterns(torch.rand(SINGLE_TILE, dtype=float_dtype) + 1, bits_dtype)
    input_tensor = _to_device(torch_input, device, dtype=input_dtype)

    # bitcast returns the requested same-width dtype, so the golden must reinterpret the bits as that dtype.
    with comparison_mode():
        output = ttnn.bitcast(input_tensor, output_dtype)

    assert _registered_golden_output(ttnn.bitcast, input_tensor, output_dtype).dtype == ttnn.to_torch(output).dtype


CONCAT_BW_GRAD_SHAPE = (1, 1, 64, 32)
TENSOR_OPERANDS = ("input_tensor", "other_tensor")
TENSOR_AB_OPERANDS = ("input_tensor_a", "input_tensor_b")


def _binary_backward_inputs(device, grad_shape=SINGLE_TILE):
    # input_a < input_b and input_b > 0 keep xlogy, remainder, fmod, min and max away from their branch boundaries.
    grad = _to_device(torch.rand(grad_shape, dtype=torch.bfloat16), device)
    input_a = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_b = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, device)
    return grad, input_a, input_b


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, extra_args, grad_shape",
    [
        pytest.param(ttnn.addalpha_bw, (2.0,), SINGLE_TILE, id="addalpha_bw"),
        pytest.param(ttnn.subalpha_bw, (2.0,), SINGLE_TILE, id="subalpha_bw"),
        pytest.param(ttnn.rsub_bw, (), SINGLE_TILE, id="rsub_bw"),
        pytest.param(ttnn.assign_bw, (), SINGLE_TILE, id="assign_bw"),
        pytest.param(ttnn.concat_bw, (2,), CONCAT_BW_GRAD_SHAPE, id="concat_bw"),
    ],
)
def test_binary_backward_with_disabled_gradient_in_comparison_mode(device, operation, extra_args, grad_shape):
    grad, input_a, input_b = _binary_backward_inputs(device, grad_shape)

    # With are_required_outputs=[True, False] the device returns None for the disabled gradient, so the golden
    # must return the same list layout instead of computing both gradients.
    with comparison_mode():
        gradients = operation(grad, input_a, input_b, *extra_args, are_required_outputs=[True, False])

    assert gradients[1] is None


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, operand_names, extra_kwargs, grad_shape",
    [
        pytest.param(ttnn.add_bw, TENSOR_OPERANDS, {}, SINGLE_TILE, id="add_bw"),
        pytest.param(ttnn.sub_bw, TENSOR_OPERANDS, {}, SINGLE_TILE, id="sub_bw"),
        pytest.param(ttnn.mul_bw, TENSOR_OPERANDS, {}, SINGLE_TILE, id="mul_bw"),
        pytest.param(ttnn.div_bw, TENSOR_OPERANDS, {}, SINGLE_TILE, id="div_bw"),
        pytest.param(ttnn.addalpha_bw, TENSOR_AB_OPERANDS, {"alpha": 2.0}, SINGLE_TILE, id="addalpha_bw"),
        pytest.param(ttnn.subalpha_bw, TENSOR_AB_OPERANDS, {"alpha": 2.0}, SINGLE_TILE, id="subalpha_bw"),
        pytest.param(ttnn.squared_difference_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="squared_difference_bw"),
        pytest.param(ttnn.remainder_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="remainder_bw"),
        pytest.param(ttnn.fmod_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="fmod_bw"),
        pytest.param(ttnn.atan2_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="atan2_bw"),
        pytest.param(ttnn.xlogy_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="xlogy_bw"),
        pytest.param(ttnn.hypot_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="hypot_bw"),
        pytest.param(ttnn.ldexp_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="ldexp_bw"),
        pytest.param(ttnn.logaddexp_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="logaddexp_bw"),
        pytest.param(ttnn.logaddexp2_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="logaddexp2_bw"),
        pytest.param(ttnn.rsub_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="rsub_bw"),
        pytest.param(ttnn.min_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="min_bw"),
        pytest.param(ttnn.max_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="max_bw"),
        pytest.param(ttnn.assign_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="assign_bw"),
        pytest.param(ttnn.concat_bw, TENSOR_AB_OPERANDS, {"dim": 2}, CONCAT_BW_GRAD_SHAPE, id="concat_bw"),
        pytest.param(ttnn.bias_gelu_bw, TENSOR_AB_OPERANDS, {}, SINGLE_TILE, id="bias_gelu_bw"),
    ],
)
def test_binary_backward_with_keyword_operands_in_comparison_mode(
    device, operation, operand_names, extra_kwargs, grad_shape
):
    grad, input_a, input_b = _binary_backward_inputs(device, grad_shape)
    operands = dict(zip(operand_names, (input_a, input_b)))

    # The public keyword names for the operands differ per op (input_tensor/other_tensor vs input_tensor_a/b) and
    # the golden must accept them, instead of failing on a call that only works positionally.
    with comparison_mode():
        operation(grad_tensor=grad, **operands, **extra_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, operand_name, extra_kwargs",
    [
        pytest.param(ttnn.add_bw, "input_tensor_a", {"scalar": 2.0}, id="add_bw"),
        pytest.param(ttnn.sub_bw, "input_tensor_a", {"scalar": 2.0}, id="sub_bw"),
        pytest.param(ttnn.mul_bw, "input_tensor_a", {"scalar": 2.0}, id="mul_bw"),
        pytest.param(ttnn.div_bw, "input_tensor_a", {"scalar": 2.0}, id="div_bw"),
        pytest.param(ttnn.remainder_bw, "input_tensor_a", {"scalar": 2.0}, id="remainder_bw"),
        pytest.param(ttnn.fmod_bw, "input_tensor_a", {"scalar": 2.0}, id="fmod_bw"),
        pytest.param(ttnn.bias_gelu_bw, "input_tensor", {"bias": 0.5}, id="bias_gelu_bw"),
        pytest.param(ttnn.assign_bw, "input_tensor", {}, id="assign_bw"),
    ],
)
def test_binary_backward_with_keyword_scalar_in_comparison_mode(device, operation, operand_name, extra_kwargs):
    grad, input_a, _ = _binary_backward_inputs(device)

    # The scalar overloads name the tensor operand input_tensor_a (input_tensor for bias_gelu_bw) and the unary
    # assign_bw overload input_tensor, unlike the input_tensor/other_tensor names of the tensor overloads.
    with comparison_mode():
        operation(grad_tensor=grad, **{operand_name: input_a}, **extra_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, torch_input, op_kwargs",
    [
        # The golden named this parameter alpha, so the public lambd keyword fell into **kwargs and the default 0.5
        # was used instead.
        pytest.param(ttnn.hardshrink_bw, _linspace_tile(-4.0, 4.0), {"lambd": 2.2}, id="hardshrink_bw_lambd"),
        pytest.param(ttnn.softshrink_bw, _linspace_tile(-4.0, 4.0), {"lambd": 2.0}, id="softshrink_bw_lambd"),
        # The golden named the bounds min_val/max_val, so the public min/max keywords were ignored and the default
        # [-1, 1] range was used; the input straddles +-0.8 and +-1 to tell them apart.
        pytest.param(
            ttnn.hardtanh_bw,
            _repeated_tile([-0.9, 0.0, 0.9, 0.5]),
            {"min": -0.8, "max": 0.8},
            id="hardtanh_bw_bounds",
        ),
        # The golden named this parameter alpha, so the public negative_slope keyword was ignored and the default
        # slope of 0.01 was used instead of 0.5.
        pytest.param(
            ttnn.leaky_relu_bw, _linspace_tile(-2.0, 2.0), {"negative_slope": 0.5}, id="leaky_relu_bw_negative_slope"
        ),
        # Torch and the device both take the negative ELU branch at x = 0, so the gradient there is alpha * grad.
        pytest.param(ttnn.elu_bw, torch.zeros(SINGLE_TILE), {"alpha": 2.0}, id="elu_bw_at_zero"),
        # The device evaluates sqrt(pi) / 2 * exp(erfinv(x)**2) * grad, the exact erfinv derivative that autograd
        # gives.
        pytest.param(ttnn.erfinv_bw, _linspace_tile(-0.9, 0.9), {}, id="erfinv_bw"),
        # A negative eps disables clamping: torch and the device both give NaN outside [0, 1] and 1 / (x * (1 - x))
        # inside it.
        pytest.param(
            ttnn.logiteps_bw, _repeated_tile([-2.0, 0.25, 0.75, 2.0]), {"eps": -0.001}, id="logiteps_bw_negative_eps"
        ),
        # The device writes zero for NaN inputs; torch autograd must produce the same gradient at those positions.
        pytest.param(ttnn.relu6_bw, _repeated_tile([float("nan"), -1.0, 3.0, 7.0]), {}, id="relu6_bw_nan_input"),
    ],
)
def test_unary_backward_with_unit_grad_in_comparison_mode(device, operation, torch_input, op_kwargs):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device)

    with comparison_mode():
        operation(grad, input_tensor, **op_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
def test_softplus_bw_with_only_beta_in_comparison_mode(device):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch.linspace(-2.0, 2.0, 1024).reshape(SINGLE_TILE).to(torch.bfloat16), device)

    # The golden applied beta and threshold only when both were given, so a call with only beta silently used
    # the default beta of 1.
    with comparison_mode():
        output = ttnn.softplus_bw(grad, input_tensor, beta=4.0)

    golden = _registered_golden_output(ttnn.softplus_bw, grad, input_tensor, beta=4.0)[0]
    torch.testing.assert_close(golden.float(), ttnn.to_torch(output[0]).float(), rtol=5e-2, atol=2e-2)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_rdiv_bw_with_keyword_rounding_mode_in_comparison_mode(device, rounding_mode):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, device)

    # Trunc and floor rounding make the quotient piecewise constant, so TTNN defines the gradient as zero; the
    # golden used to return the gradient of plain division.
    with comparison_mode():
        ttnn.rdiv_bw(grad, input_tensor, 2.0, rounding_mode=rounding_mode)


@pytest.mark.requires_fast_runtime_mode_off
def test_rpow_bw_matches_device_power_derivative_in_comparison_mode(device):
    torch_grad = torch.ones(SINGLE_TILE, dtype=torch.bfloat16)
    torch_input = torch.linspace(0.5, 2.0, 1024).reshape(SINGLE_TILE).to(torch.bfloat16)
    grad = _to_device(torch_grad, device)
    input_tensor = _to_device(torch_input, device)

    # rpow_bw on device differentiates x ** exponent (not the forward rpow's exponent ** x), and PCC cannot tell
    # the two monotone curves apart, so both are also checked against grad * exponent * x ** (exponent - 1).
    with comparison_mode():
        output = ttnn.rpow_bw(grad, input_tensor, 3.0)

    ideal = torch_grad.float() * 3.0 * torch_input.float() ** 2
    golden = _registered_golden_output(ttnn.rpow_bw, grad, input_tensor, 3.0)[0]
    torch.testing.assert_close(golden.float(), ideal, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ttnn.to_torch(output[0]).float(), ideal, rtol=2e-2, atol=2e-2)


@pytest.mark.requires_fast_runtime_mode_off
def test_gelu_bw_tanh_variant_in_comparison_mode(device):
    torch_input = torch.linspace(-3.0, 3.0, 1024).reshape(SINGLE_TILE)
    torch_grad = torch.ones(SINGLE_TILE)
    grad = _to_device(torch_grad, device, dtype=ttnn.float32)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.float32)

    # The public API selects the tanh derivative with variant=GeluVariant.Tanh, but the golden only understood
    # approximate="tanh" and computed the exact-erf gradient instead.
    with comparison_mode():
        ttnn.gelu_bw(grad, input_tensor, variant=ttnn.GeluVariant.Tanh)

    reference_input = torch_input.clone().requires_grad_(True)
    torch.nn.functional.gelu(reference_input, approximate="tanh").backward(torch_grad)
    golden = _registered_golden_output(ttnn.gelu_bw, grad, input_tensor, variant=ttnn.GeluVariant.Tanh)[0]
    torch.testing.assert_close(golden, reference_input.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.requires_fast_runtime_mode_off
def test_where_bw_with_bfloat16_predicate_in_comparison_mode(device):
    torch_condition = torch.zeros(SINGLE_TILE, dtype=torch.bfloat16)
    torch_condition[..., ::2] = 1
    grad, input_a, input_b = (_to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device) for _ in range(3))
    condition = _to_device(torch_condition, device)

    # TTNN predicates are 0/1 floating-point tensors, but torch.where needs a boolean condition, so the golden
    # used to fail on a bfloat16 predicate.
    with comparison_mode():
        ttnn.where_bw(grad, condition, input_a, input_b)


@pytest.mark.requires_fast_runtime_mode_off
def test_lerp_with_public_keywords_in_comparison_mode(device):
    input_tensor, end_tensor = (_to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device) for _ in range(2))

    # The golden named its operands input_tensor_a/b/c, so a call with the public keywords input, end and weight
    # used to fail in the golden.
    with comparison_mode():
        ttnn.lerp(input=input_tensor, end=end_tensor, weight=0.5)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("weight_form", ["scalar", "tensor"])
def test_lerp_bw_with_public_keywords_in_comparison_mode(device, weight_form):
    grad, input_a, input_b, weight = (
        _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device) for _ in range(4)
    )
    weight_kwargs = {"scalar": 0.5} if weight_form == "scalar" else {"input_tensor_c": weight}

    # The public overloads pass the weight either as a tensor (input_tensor_c) or as scalar, while the golden
    # expected positional end_tensor and weight arguments.
    with comparison_mode():
        ttnn.lerp_bw(grad_tensor=grad, input_tensor_a=input_a, input_tensor_b=input_b, **weight_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("output_form", ["dtype", "output_tensor"])
@pytest.mark.parametrize(
    "operation, make_operands",
    [
        pytest.param(ttnn.subtract, _positive_operands, id="subtract"),
        pytest.param(ttnn.multiply, _positive_operands, id="multiply"),
        pytest.param(ttnn.divide, _positive_operands, id="divide"),
        pytest.param(ttnn.rsub, _positive_operands, id="rsub"),
        pytest.param(ttnn.maximum, _positive_operands, id="maximum"),
        pytest.param(ttnn.remainder, _positive_operands, id="remainder"),
        pytest.param(ttnn.squared_difference, _positive_operands, id="squared_difference"),
        pytest.param(ttnn.bias_gelu, _positive_operands, id="bias_gelu"),
        pytest.param(ttnn.xlogy, _positive_operands, id="xlogy"),
        pytest.param(ttnn.ldexp, _ldexp_operands, id="ldexp"),
        pytest.param(ttnn.logaddexp, _positive_operands, id="logaddexp"),
        pytest.param(ttnn.logaddexp2, _positive_operands, id="logaddexp2"),
        pytest.param(ttnn.pow, _positive_operands, id="pow"),
        pytest.param(ttnn.eq, _small_integer_operands, id="eq"),
        pytest.param(ttnn.ne, _small_integer_operands, id="ne"),
        pytest.param(ttnn.lt, _small_integer_operands, id="lt"),
        pytest.param(ttnn.le, _small_integer_operands, id="le"),
        pytest.param(ttnn.gt, _small_integer_operands, id="gt"),
        pytest.param(ttnn.ge, _small_integer_operands, id="ge"),
        pytest.param(ttnn.logical_and, _small_integer_operands, id="logical_and"),
        pytest.param(ttnn.logical_or, _small_integer_operands, id="logical_or"),
        pytest.param(ttnn.logical_xor, _small_integer_operands, id="logical_xor"),
        pytest.param(ttnn.gcd, _int32_operands, id="gcd"),
        pytest.param(ttnn.lcm, _int32_operands, id="lcm"),
    ],
)
def test_binary_with_float32_output_in_comparison_mode(device, operation, make_operands, output_form):
    torch_a, torch_b, input_dtype = make_operands()
    input_a = _to_device(torch_a, device, dtype=input_dtype)
    input_b = _to_device(torch_b, device, dtype=input_dtype)
    if output_form == "dtype":
        output_kwargs = {"dtype": ttnn.float32}
    else:
        output_kwargs = {"output_tensor": _to_device(torch.zeros(SINGLE_TILE), device, dtype=ttnn.float32)}

    # Every binary op takes dtype and a preallocated output_tensor, and the device stores its result in that dtype
    # (FLOAT32 here), so the golden must return FLOAT32 rather than the input dtype or a bool comparison result.
    with comparison_mode():
        output = operation(input_a, input_b, **output_kwargs)

    _assert_golden_dtype_and_close(
        _registered_golden_output(operation, input_a, input_b, **output_kwargs), output, rtol=1e-2, atol=1e-2
    )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("output_form", ["dtype", "output_tensor"])
@pytest.mark.parametrize(
    "operation, identity_value, block_float_operand",
    [
        pytest.param(ttnn.add, 0.0, "a", id="add"),
        pytest.param(ttnn.subtract, 0.0, "a", id="subtract"),
        pytest.param(ttnn.multiply, 1.0, "a", id="multiply"),
        pytest.param(ttnn.divide, 1.0, "a", id="divide"),
        pytest.param(ttnn.rsub, 0.0, "b", id="rsub"),
        pytest.param(ttnn.maximum, 0.0, "a", id="maximum"),
    ],
)
def test_binary_with_block_float_output_in_comparison_mode(
    device, operation, identity_value, block_float_operand, output_form
):
    block_float_values = _to_device(_block_float_sensitive_values(SINGLE_TILE), device, dtype=ttnn.float32)
    identity = _to_device(torch.full(SINGLE_TILE, identity_value), device, dtype=ttnn.float32)
    input_a, input_b = (block_float_values, identity) if block_float_operand == "a" else (identity, block_float_values)
    if output_form == "dtype":
        output_kwargs = {"dtype": ttnn.bfloat4_b}
    else:
        output_kwargs = {"output_tensor": _to_device(torch.zeros(SINGLE_TILE), device, dtype=ttnn.bfloat8_b)}

    # The identity operand makes the exact result the 1s-beside-1024 input. BFLOAT4_B/BFLOAT8_B outputs share an
    # exponent per 16 values, which flushes those 1s to zero, so the golden must quantize the same way.
    with comparison_mode():
        output = operation(input_a, input_b, **output_kwargs)

    _assert_golden_matches_output(_registered_golden_output(operation, input_a, input_b, **output_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_add_int32_with_integral_float_scalar_in_comparison_mode(device):
    input_tensor = _to_device(torch.full(SINGLE_TILE, 16777217, dtype=torch.int32), device, dtype=ttnn.int32)

    # Torch promotes int32 + 2.0 to float32, which changes the dtype and rounds 2**24 + 1 away; the device packs
    # an integral scalar as an integer and stays exact in int32.
    with comparison_mode():
        output = ttnn.add(input_tensor, 2.0)

    assert torch.equal(ttnn.to_torch(output), torch.full(SINGLE_TILE, 16777219, dtype=torch.int32))
    _assert_golden_matches_output(_registered_golden_output(ttnn.add, input_tensor, 2.0), output)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "other_form, minimum_kwargs",
    [
        # minimum takes the same activations and output dtype as the other binary ops, but its golden ignored both,
        # so a negating activation or a float32 output gave different values.
        pytest.param(
            "tensor",
            {"input_tensor_a_activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)]},
            id="tensor_activation",
        ),
        pytest.param("tensor", {"dtype": ttnn.float32}, id="tensor_dtype"),
        # The scalar overload runs on the unary path, which accepts activations but never applies them, so the
        # golden must ignore them too; a negating activation would otherwise change the minimum of values in
        # [1, 2) and 0.5.
        pytest.param(
            "scalar",
            {"input_tensor_a_activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)]},
            id="scalar_operand_activation",
        ),
        pytest.param(
            "scalar", {"activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)]}, id="scalar_result_activation"
        ),
    ],
)
def test_minimum_with_value_affecting_options_in_comparison_mode(device, other_form, minimum_kwargs):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, device)
    other = 0.5
    if other_form == "tensor":
        other = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, device)

    with comparison_mode():
        output = ttnn.minimum(input_tensor, other, **minimum_kwargs)

    golden = _registered_golden_output(ttnn.minimum, input_tensor, other, **minimum_kwargs)
    _assert_golden_matches_output(golden, output)


def test_situ_glu_golden_accepts_keyword_betas():
    gate = torch.randn((32, 64))
    up = torch.randn((32, 64))
    golden_function = ttnn.get_golden_function(ttnn.situ_glu)

    # The betas can be passed by keyword or by position; both spellings must reach the golden and agree.
    # The device operation is Blackhole-only, so the registered golden is called directly.
    torch.testing.assert_close(golden_function(gate, up, beta1=1.0, beta2=2.0), golden_function(gate, up, 1.0, 2.0))


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", [ttnn.l1_loss, ttnn.mse_loss], ids=["l1_loss", "mse_loss"])
@pytest.mark.parametrize(
    "reduction",
    [ttnn.LossReductionMode.NONE, ttnn.LossReductionMode.MEAN, ttnn.LossReductionMode.SUM],
    ids=["none", "mean", "sum"],
)
def test_loss_with_reduction_enum_in_comparison_mode(device, operation, reduction):
    reference, prediction = (_to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device) for _ in range(2))

    # The device takes a LossReductionMode enum while the torch losses need a string. Reduced losses are a single
    # value where PCC is undefined, so they are compared within 3 ULP of a float32 reference instead.
    with comparison_mode():
        operation(reference, prediction, reduction=reduction)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, None], ids=["bfloat16", "default"])
def test_full_with_dtype_in_comparison_mode(device, dtype):
    full_args = ([1, 1, 32, 32], 1.5)
    full_kwargs = {"layout": ttnn.TILE_LAYOUT, "device": device}
    if dtype is not None:
        full_kwargs["dtype"] = dtype

    # TTNN creates BFLOAT16 tensors when dtype is omitted while torch.full defaults to float32, and the golden
    # used to ignore an explicit dtype as well.
    with comparison_mode():
        output = ttnn.full(*full_args, **full_kwargs)

    _assert_golden_matches_output(_registered_golden_output(ttnn.full, *full_args, **full_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, make_input, input_dtype",
    [
        pytest.param(ttnn.ones_like, lambda: torch.zeros(SINGLE_TILE, dtype=torch.int32), ttnn.int32, id="ones_like"),
        pytest.param(ttnn.zeros_like, lambda: torch.rand(SINGLE_TILE, dtype=torch.bfloat16), None, id="zeros_like"),
        # clone can convert dtype while copying, but the golden was a plain identity.
        pytest.param(ttnn.clone, lambda: torch.rand(SINGLE_TILE, dtype=torch.bfloat16), None, id="clone"),
    ],
)
def test_dtype_override_in_comparison_mode(device, operation, make_input, input_dtype):
    input_tensor = _to_device(make_input(), device, dtype=input_dtype)

    # The golden used to ignore dtype and return a tensor in the input's dtype instead of float32.
    with comparison_mode():
        output = operation(input_tensor, dtype=ttnn.float32)

    _assert_golden_matches_output(_registered_golden_output(operation, input_tensor, dtype=ttnn.float32), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_zeros_like_with_tensor_keyword_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # The golden named its argument input_tensor, so a call with the public keyword tensor= used to fail in it.
    with comparison_mode():
        ttnn.zeros_like(tensor=input_tensor)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, fill_values",
    [pytest.param(ttnn.fill_rm, (1.0, 0.0), id="fill_rm"), pytest.param(ttnn.fill_ones_rm, (), id="fill_ones_rm")],
)
def test_fill_rm_dtype_follows_any_in_comparison_mode(device, operation, fill_values):
    any_tensor = _to_device(torch.zeros(SINGLE_TILE, dtype=torch.bfloat16), device)
    fill_args = (*SINGLE_TILE, 16, 16, any_tensor, *fill_values)

    # The `any` tensor only supplies the output dtype; the golden used to return float32 regardless, so the
    # dtype comparison against the bfloat16 device output failed.
    with comparison_mode():
        output = operation(*fill_args)

    _assert_golden_matches_output(_registered_golden_output(operation, *fill_args), output)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "dtype, make_values",
    [
        pytest.param(ttnn.bfloat16, _bfloat16_sensitive_values, id="bfloat16"),
        pytest.param(ttnn.bfloat8_b, _block_float_sensitive_values, id="bfloat8_b"),
    ],
)
def test_to_dtype_converts_values_in_comparison_mode(dtype, make_values):
    input_tensor = ttnn.from_torch(make_values(SINGLE_TILE), dtype=ttnn.float32)

    # The golden used to be an identity, so it kept full float32 precision. The input needs rounding to bfloat16
    # or block-float quantization (1s beside 1024) to be told apart from the device's stored values.
    with comparison_mode():
        output = ttnn.to_dtype(input_tensor, dtype)

    _assert_golden_matches_output(_registered_golden_output(ttnn.to_dtype, input_tensor, dtype), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_to_torch_with_dtype_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # The golden used to ignore dtype and return the bfloat16 tensor unchanged instead of converting it to float32.
    with comparison_mode():
        output = ttnn.to_torch(input_tensor, dtype=torch.float32)

    golden = _registered_golden_output(ttnn.to_torch, input_tensor, dtype=torch.float32)
    assert golden.dtype == output.dtype, f"golden dtype {golden.dtype} != output dtype {output.dtype}"
    assert torch.equal(golden, output)


@pytest.mark.requires_fast_runtime_mode_off
def test_from_torch_with_col_tilize_in_comparison_mode():
    torch_input = torch.randn((32, 64), dtype=torch.float32)

    # Column tilization stores the transposed matrix, so the result's last two dimensions are swapped; the golden
    # used to ignore col_tilize and returned the original (32, 64) shape.
    with comparison_mode():
        ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, col_tilize=True)


@pytest.mark.requires_fast_runtime_mode_off
def test_as_tensor_with_preprocess_in_comparison_mode(device):
    torch_input = torch.rand(SINGLE_TILE, dtype=torch.bfloat16)

    # as_tensor runs the caller's preprocess callback on the torch input before conversion; the golden used to
    # convert the raw input, so it missed the negation.
    with comparison_mode():
        ttnn.as_tensor(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            preprocess=torch.neg,
        )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, extra_args, dtype_keyword, output_dtype, make_values",
    [
        pytest.param(ttnn.tilize, (), "dtype", ttnn.bfloat16, _bfloat16_sensitive_values, id="tilize_bfloat16"),
        pytest.param(ttnn.tilize, (), "dtype", ttnn.bfloat8_b, _block_float_sensitive_values, id="tilize_bfloat8_b"),
        pytest.param(
            ttnn.tilize_with_val_padding,
            (list(SINGLE_TILE), 0.0),
            "dtype",
            ttnn.bfloat8_b,
            _block_float_sensitive_values,
            id="tilize_with_val_padding",
        ),
        # tilize_with_zero_padding names its dtype argument output_dtype.
        pytest.param(
            ttnn.tilize_with_zero_padding,
            (),
            "output_dtype",
            ttnn.bfloat8_b,
            _block_float_sensitive_values,
            id="tilize_with_zero_padding",
        ),
    ],
)
def test_tilize_with_output_dtype_in_comparison_mode(
    device, operation, extra_args, dtype_keyword, output_dtype, make_values
):
    input_tensor = _to_device(make_values(SINGLE_TILE), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tilize_args = (input_tensor, *extra_args)
    tilize_kwargs = {dtype_keyword: output_dtype}

    # The device stores the tiles in the output dtype, but the golden used to return the float32 input unchanged
    # instead of the rounded bfloat16 values or the BFLOAT8_B values with the 1s beside 1024 flushed to zero.
    with comparison_mode():
        output = operation(*tilize_args, **tilize_kwargs)

    _assert_golden_matches_output(_registered_golden_output(operation, *tilize_args, **tilize_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_embedding_with_block_float_dtype_in_comparison_mode(device):
    indices = _to_device(
        torch.randint(0, 64, (1, 32), dtype=torch.int32), device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    weights = _to_device(
        _block_float_sensitive_values((64, 32)), device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    embedding_kwargs = {"layout": ttnn.TILE_LAYOUT, "dtype": ttnn.bfloat8_b}

    # A tiled embedding output is typecast to dtype, and BFLOAT8_B flushes the 1s beside 1024 to zero, so the
    # golden must quantize the looked-up rows the same way.
    with comparison_mode():
        output = ttnn.embedding(indices, weights, **embedding_kwargs)

    _assert_golden_matches_output(
        _registered_golden_output(ttnn.embedding, indices, weights, **embedding_kwargs), output
    )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", ["interleaved_to_sharded", "interleaved_to_sharded_partial"])
def test_interleaved_to_sharded_with_output_dtype_in_comparison_mode(device, operation):
    input_tensor = _to_device(_block_float_sensitive_values(SINGLE_TILE), device, dtype=ttnn.float32)
    if operation == "interleaved_to_sharded":
        call_args = (input_tensor, _single_core_height_sharded_memory_config(), ttnn.bfloat8_b)
        call_kwargs = {}
    else:
        call_args = (
            input_tensor,
            (1, 1),
            [32, 32],
            1,
            0,
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        call_kwargs = {"output_dtype": ttnn.bfloat8_b}
    operation = getattr(ttnn, operation)

    # The output dtype is an optional trailing argument (positional or output_dtype=) that the golden used to
    # ignore, so it kept float32 values instead of the bfloat8_b-quantized ones stored on device.
    with comparison_mode():
        output = operation(*call_args, **call_kwargs)

    _assert_golden_matches_output(_registered_golden_output(operation, *call_args, **call_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_sharded_to_interleaved_with_output_dtype_in_comparison_mode(device):
    input_tensor = _to_device(
        _block_float_sensitive_values(SINGLE_TILE),
        device,
        dtype=ttnn.float32,
        memory_config=_single_core_height_sharded_memory_config(),
    )

    # The trailing positional output dtype used to be ignored by the golden, which kept float32 values instead of
    # the bfloat8_b-quantized ones stored on device.
    with comparison_mode():
        output = ttnn.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat8_b)

    _assert_golden_matches_output(
        _registered_golden_output(ttnn.sharded_to_interleaved, input_tensor, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat8_b),
        output,
    )


@pytest.mark.requires_fast_runtime_mode_off
def test_sharded_to_interleaved_partial_with_output_dtype_in_comparison_mode(device):
    source = _to_device(_block_float_sensitive_values(SINGLE_TILE), device, dtype=ttnn.float32)
    cache_tensor = _to_device(
        torch.zeros(SINGLE_TILE), device, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    sharded_slice = ttnn.interleaved_to_sharded_partial(
        source, (1, 1), [32, 32], 1, 0, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ShardOrientation.ROW_MAJOR
    )
    partial_args = (sharded_slice, cache_tensor, 1, 0)
    partial_kwargs = {"memory_config": ttnn.L1_MEMORY_CONFIG, "output_dtype": ttnn.bfloat8_b}

    # The partial write copies a slice into the bfloat8_b cache tensor; the golden used to write the slice at full
    # float32 precision instead of quantizing it to the output dtype first.
    with comparison_mode():
        ttnn.sharded_to_interleaved_partial(*partial_args, **partial_kwargs)

    _assert_golden_matches_output(
        _registered_golden_output(ttnn.sharded_to_interleaved_partial, *partial_args, **partial_kwargs), cache_tensor
    )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("dtype", [ttnn.int8, ttnn.uint8], ids=["int8", "uint8"])
def test_quantize_saturates_narrow_output_in_comparison_mode(device, dtype):
    # Integer inputs avoid rounding ties. UINT8 covers only the documented upper saturation, since
    # the device behavior for negative UINT8 inputs is not specified.
    low = -512 if dtype == ttnn.int8 else 0
    torch_input = torch.arange(low, low + 1024, dtype=torch.float32).reshape(SINGLE_TILE).to(torch.bfloat16)
    input_tensor = _to_device(torch_input, device)

    # The device saturates narrow outputs, while a direct torch cast to int8/uint8 wraps around, so the golden
    # must clamp before casting.
    with comparison_mode():
        output = ttnn.quantize(input_tensor, 1.0, 0, dtype=dtype)

    _assert_golden_matches_output(_registered_golden_output(ttnn.quantize, input_tensor, 1.0, 0, dtype=dtype), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_requantize_saturates_int8_output_in_comparison_mode(device):
    torch_input = torch.tensor([0, 300, -10, -300], dtype=torch.int32).repeat(256).reshape(SINGLE_TILE)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.int32)
    requantize_args = (input_tensor, 1.0, 0, 1.0, 0)

    # The int8 requantize path saturates on device (300 -> 127, -300 -> -128), while a direct torch cast wraps
    # around, so the golden must clamp before casting.
    with comparison_mode():
        output = ttnn.requantize(*requantize_args, dtype=ttnn.int8)

    _assert_golden_matches_output(_registered_golden_output(ttnn.requantize, *requantize_args, dtype=ttnn.int8), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_pad_fallback_matches_comparison_mode_output(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)
    pad_kwargs = {"padding": ((0, 0), (0, 0), (0, 1), (0, 1)), "value": 0.0}

    with comparison_mode():
        output = ttnn.pad(input_tensor, **pad_kwargs)

    # The fallback's postprocessing used to reshape the result to the tile-aligned padded shape, so its logical
    # shape differed from the device output.
    fallback_output = ttnn.get_fallback_function(ttnn.pad)(input_tensor, **pad_kwargs)
    assert fallback_output.dtype == output.dtype
    assert tuple(fallback_output.shape) == tuple(output.shape)
    assert torch.equal(ttnn.to_torch(fallback_output), ttnn.to_torch(output))


@pytest.mark.requires_fast_runtime_mode_off
def test_fold_legacy_transpose_with_asymmetric_padding_in_comparison_mode(device):
    if device.core_grid.y < 8:
        pytest.skip("the sharded legacy fold path needs an 8x8 core grid")
    torch_input = torch.rand((16, 3, 224, 224), dtype=torch.bfloat16)
    sharded_memory_config = ttnn.create_sharded_memory_config(
        torch_input.shape,
        core_grid=ttnn.CoreGrid(y=8, x=6),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_tensor = _to_device(torch_input, device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=sharded_memory_config)
    grid_size = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})

    # The legacy transpose path honors only pad_top, pad_left and pad_c_back, and a sharded input pads both sides of
    # H and W by them; the golden used to apply all six pad values on their own sides.
    with comparison_mode():
        ttnn.fold(input_tensor, 2, 2, use_transpose_as_fold=True, padding=[2, 4, 2, 4, 0, 1], grid_size=grid_size)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
@pytest.mark.parametrize(
    "operation, extra_args",
    [
        pytest.param(ttnn.avg_pool2d, (), id="avg_pool2d"),
        pytest.param(ttnn.max_pool2d, ([1, 1],), id="max_pool2d"),
    ],
)
def test_pool2d_with_block_float_dtype_in_comparison_mode(device, operation, extra_args):
    batch_size, input_h, input_w, channels = 1, 8, 8, 32
    torch_input = _block_float_sensitive_values((1, 1, batch_size * input_h * input_w, channels))
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    pool_args = (input_tensor, batch_size, input_h, input_w, channels, [2, 2], [2, 2], [0, 0], *extra_args)
    pool_kwargs = {"dtype": ttnn.bfloat8_b, "output_layout": ttnn.TILE_LAYOUT}

    # The golden ignored the output dtype and kept full precision, while the device output is quantized to
    # BFLOAT8_B (1s beside 1024 flush to zero).
    with comparison_mode():
        output = operation(*pool_args, **pool_kwargs)

    _assert_golden_matches_output(_registered_golden_output(operation, *pool_args, **pool_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_without_bias_in_comparison_mode(device):
    batch_size, in_channels, out_channels, input_height, input_width = 1, 32, 32, 8, 8
    input_tensor = ttnn.from_torch(
        torch.randn((batch_size, input_height, input_width, in_channels), dtype=torch.bfloat16), dtype=ttnn.bfloat16
    )
    weight_tensor = ttnn.from_torch(
        torch.randn((out_channels, in_channels, 1, 1), dtype=torch.bfloat16), dtype=ttnn.bfloat16
    )

    # bias_tensor is optional, but the golden unconditionally reshaped it and raised on None; stride and padding
    # also lacked defaults although this call omits them.
    with comparison_mode():
        ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=weight_tensor,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv2d_with_block_float_dtype_in_comparison_mode(device):
    batch_size, in_channels, out_channels, input_height, input_width = 1, 32, 32, 8, 8
    input_tensor = ttnn.from_torch(
        _block_float_sensitive_values((batch_size, input_height, input_width, in_channels)), dtype=ttnn.bfloat16
    )
    weight_tensor = ttnn.from_torch(
        torch.eye(out_channels, in_channels).reshape(out_channels, in_channels, 1, 1), dtype=ttnn.bfloat16
    )
    conv_kwargs = {
        "input_tensor": input_tensor,
        "weight_tensor": weight_tensor,
        "device": device,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "batch_size": batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "kernel_size": (1, 1),
        "stride": (1, 1),
        "padding": (0, 0),
        "dtype": ttnn.bfloat8_b,
    }

    # A 1x1 identity convolution reproduces the 1s-beside-1024 channels, and the BFLOAT8_B output flushes those 1s
    # to zero, so the golden must quantize its float32 result to dtype.
    with comparison_mode():
        output = ttnn.conv2d(**conv_kwargs)

    _assert_golden_matches_output(_registered_golden_output(ttnn.conv2d, **conv_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_grid_sample_with_precomputed_grid_in_comparison_mode(device):
    batch_size, channels, height, width = 1, 256, 12, 40
    grid_h, grid_w = 1, 1408
    input_shape_nhwc = [batch_size, height, width, channels]
    input_tensor = _to_device(
        torch.randn(input_shape_nhwc, dtype=torch.bfloat16),
        device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    theta = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    torch_grid = torch.nn.functional.affine_grid(theta, (batch_size, 1, grid_h, grid_w), align_corners=False)
    prepared_grid = ttnn.prepare_grid_sample_grid(
        ttnn.from_torch(torch_grid, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32),
        input_shape_nhwc,
        mode="bilinear",
        align_corners=False,
        padding_mode="zeros",
        output_dtype=ttnn.bfloat16,
    )
    prepared_grid = ttnn.to_device(prepared_grid, device)

    # A prepared grid holds pixel indices and bilinear weights rather than normalized coordinates, but the golden
    # always treated the grid as coordinates for torch's grid_sample.
    with comparison_mode():
        ttnn.grid_sample(input_tensor, prepared_grid, use_precomputed_grid=True)


@pytest.mark.requires_fast_runtime_mode_off
def test_group_norm_without_optional_arguments_in_comparison_mode(device):
    if device.core_grid.y == 7:
        pytest.skip("the interleaved group_norm grid is not supported on this device")
    input_tensor = _to_device(
        torch.rand((1, 1, 256, 1024), dtype=torch.bfloat16), device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # weight, bias, memory_config and core_grid are optional, but the golden unpacked them unconditionally and
    # raised on None; an omitted weight or bias must leave the output unscaled or unshifted.
    with comparison_mode():
        ttnn.group_norm(input_tensor, num_groups=32, inplace=False)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation", [ttnn.scale_mask_softmax, ttnn.scale_mask_softmax_in_place], ids=["out_of_place", "in_place"]
)
def test_scale_mask_softmax_without_scale_in_comparison_mode(device, operation):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # The public API makes scale and mask optional and names them scale/mask, but the golden required a positional
    # scalar, so this call with neither used to fail in the golden.
    with comparison_mode():
        operation(input_tensor)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, op_kwargs, stat_columns",
    [
        # The golden ignored dtype and returned float32 statistics, while the device stores them in the requested
        # dtype.
        pytest.param(ttnn.rms_norm_pre_all_gather, {"dtype": ttnn.bfloat16}, (0,), id="rms_norm"),
        # Without dtype the device stores the statistics as BFLOAT16, as rms_norm_pre_all_gather does, so the golden
        # must return BFLOAT16 statistics rather than float32.
        pytest.param(ttnn.layer_norm_pre_all_gather, {}, (0, 32), id="layer_norm_default_dtype"),
    ],
)
def test_norm_pre_all_gather_dtype_in_comparison_mode(device, operation, op_kwargs, stat_columns):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    with comparison_mode():
        output = operation(input_tensor, **op_kwargs)

    golden = _registered_golden_output(operation, input_tensor, **op_kwargs)
    torch_output = ttnn.to_torch(output)
    assert golden.dtype == torch_output.dtype, f"golden dtype {golden.dtype} != output dtype {torch_output.dtype}"
    for stat_column in stat_columns:
        torch.testing.assert_close(
            golden[..., stat_column].float(), torch_output[..., stat_column].float(), rtol=1e-2, atol=1e-2
        )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "pre_operation, post_operation",
    [
        pytest.param(ttnn.rms_norm_pre_all_gather, ttnn.rms_norm_post_all_gather, id="rms_norm"),
        # The 1s normalize to about -0.258 beside 1024's 3.87, and the BFLOAT8_B output stores them as -0.25.
        pytest.param(ttnn.layer_norm_pre_all_gather, ttnn.layer_norm_post_all_gather, id="layer_norm"),
    ],
)
def test_norm_post_all_gather_with_block_float_dtype_in_comparison_mode(device, pre_operation, post_operation):
    input_tensor = _to_device(_block_float_sensitive_values(SINGLE_TILE).to(torch.bfloat16), device)
    stats = pre_operation(input_tensor)

    # The golden ignored the output dtype and kept full precision, while the BFLOAT8_B device output shares one
    # exponent per 16 values, so the golden must quantize the normalized result to dtype.
    with comparison_mode():
        output = post_operation(input_tensor, stats, dtype=ttnn.bfloat8_b)

    golden = _registered_golden_output(post_operation, input_tensor, stats, dtype=ttnn.bfloat8_b)
    torch.testing.assert_close(golden.float(), ttnn.to_torch(output).float(), rtol=1e-2, atol=1e-3)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, input_dtype, output_dtype, make_values",
    [
        # The device rounds the result to bfloat16 / BFLOAT8_B; multiplying by the identity exposes exactly that
        # rounding.
        pytest.param(
            ttnn.matmul, ttnn.float32, ttnn.bfloat16, _bfloat16_sensitive_values, id="matmul_float32_to_bfloat16"
        ),
        pytest.param(
            ttnn.matmul, ttnn.bfloat16, ttnn.bfloat8_b, _block_float_sensitive_values, id="matmul_bfloat16_to_bfloat8_b"
        ),
        # The golden returned the bfloat16 input dtype instead of the requested float32 output.
        pytest.param(ttnn.linear, ttnn.bfloat16, ttnn.float32, _small_integer_values, id="linear_bfloat16_to_float32"),
    ],
)
def test_matmul_with_output_dtype_in_comparison_mode(device, operation, input_dtype, output_dtype, make_values):
    input_a = _to_device(make_values((32, 32)), device, dtype=input_dtype)
    input_b = _to_device(torch.eye(32), device, dtype=input_dtype)

    # The golden ignored the output dtype and kept the input precision and dtype.
    with comparison_mode():
        output = operation(input_a, input_b, dtype=output_dtype)

    _assert_golden_matches_output(_registered_golden_output(operation, input_a, input_b, dtype=output_dtype), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_addmm_with_block_float_dtype_in_comparison_mode(device):
    addend = _to_device(torch.zeros((32, 32), dtype=torch.bfloat16), device)
    mat1 = _to_device(_block_float_sensitive_values((32, 32)).to(torch.bfloat16), device)
    mat2 = _to_device(torch.eye(32, dtype=torch.bfloat16), device)

    # The golden ignored the output dtype and kept full precision, while the device output is quantized to
    # BFLOAT8_B (1s beside 1024 flush to zero).
    with comparison_mode():
        output = ttnn.addmm(addend, mat1, mat2, dtype=ttnn.bfloat8_b)

    _assert_golden_matches_output(
        _registered_golden_output(ttnn.addmm, addend, mat1, mat2, dtype=ttnn.bfloat8_b), output
    )


@pytest.mark.requires_fast_runtime_mode_off
def test_matmul_batched_weights_with_output_dtype_in_comparison_mode(device):
    grid = device.compute_with_storage_grid_size()
    if grid.y < 2:
        pytest.skip("the global circular buffer needs two core rows")
    input_a = _to_device(_block_float_sensitive_values(SINGLE_TILE).to(torch.bfloat16), device)
    weights = [_to_device(torch.eye(32, dtype=torch.bfloat16).reshape(SINGLE_TILE), device)]
    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(1, 1),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=1,
        transpose_mcast=False,
        fuse_batch=True,
    )
    global_cb = ttnn.create_global_circular_buffer(
        device,
        [(ttnn.CoreCoord(0, 0), ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, 1))}))],
        3200,
    )
    worker_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sub_device_manager = device.create_sub_device_manager([ttnn.SubDevice([worker_cores])], 0)
    device.load_sub_device_manager(sub_device_manager)
    matmul_kwargs = {
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "program_config": program_config,
        "global_cb": global_cb,
        "sub_device_id": ttnn.SubDeviceId(0),
        "dtype": ttnn.bfloat8_b,
    }
    try:
        # The golden ignored the output dtype and kept full precision, while the device output is quantized to
        # BFLOAT8_B (1s beside 1024 flush to zero).
        # TTNN rejects transpose_a, transpose_b, and activation, so only dtype can change values.
        with comparison_mode():
            outputs = ttnn.matmul_batched_weights(input_a, weights, **matmul_kwargs)

        goldens = _registered_golden_output(ttnn.matmul_batched_weights, input_a, weights, **matmul_kwargs)
        for golden, output in zip(goldens, outputs):
            _assert_golden_matches_output(golden, output)
    finally:
        device.reset_sub_device_stall_group()
        device.clear_loaded_sub_device_manager()
        device.remove_sub_device_manager(sub_device_manager)


def _indexed_sparse_matmul_kwargs(device, num_experts):
    active_ids = [5, 1]
    sparsity_values = torch.zeros((1, 1, 1, num_experts), dtype=torch.bfloat16)
    sparsity_values[..., active_ids] = 1
    return {
        "sparsity": _to_device(sparsity_values, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT),
        "indices": _to_device(
            torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, -1),
            device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "is_input_a_sparse": False,
        "is_input_b_sparse": True,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "program_config": _sparse_matmul_program_config(),
    }


@pytest.mark.requires_fast_runtime_mode_off
def test_sparse_matmul_indexed_output_in_comparison_mode(device):
    m, k, n, num_experts = 32, 128, 192, 8
    input_a = _to_device(torch.randn((1, 1, m, k), dtype=torch.bfloat16), device)
    input_b = _to_device(torch.randn((1, num_experts, k, n), dtype=torch.bfloat16), device)
    sparse_matmul_kwargs = _indexed_sparse_matmul_kwargs(device, num_experts)

    # Indexed mode gathers the listed groups of B into a compact group axis in index order and never reads
    # sparsity; the golden used to raise NotImplementedError for any call with indices.
    with comparison_mode():
        ttnn.sparse_matmul(input_a, input_b, **sparse_matmul_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
def test_sparse_matmul_with_block_float_dtype_in_comparison_mode(device):
    m, k, n, num_experts = 32, 128, 192, 8
    input_a = _to_device(_block_float_sensitive_values((1, 1, m, k)).to(torch.bfloat16), device)
    input_b = _to_device(torch.eye(k, n, dtype=torch.bfloat16).expand(1, num_experts, k, n).contiguous(), device)
    sparse_matmul_kwargs = {**_indexed_sparse_matmul_kwargs(device, num_experts), "dtype": ttnn.bfloat8_b}

    # Every expert of B is an identity, so each gathered result repeats A's 1s-beside-1024 rows; the BFLOAT8_B
    # output flushes those 1s to zero, so the golden must quantize to dtype.
    with comparison_mode():
        output = ttnn.sparse_matmul(input_a, input_b, **sparse_matmul_kwargs)

    _assert_golden_matches_output(
        _registered_golden_output(ttnn.sparse_matmul, input_a, input_b, **sparse_matmul_kwargs), output
    )


@pytest.mark.requires_fast_runtime_mode_off
def test_sparse_matmul_compact_output_in_comparison_mode(device):
    num_blocks, num_experts = 4, 8
    m, k, n = 32, 128, 192
    sparsity_values = torch.zeros((1, 1, num_blocks, num_experts), dtype=torch.bfloat16)
    for block, expert in enumerate([3, 1, 7, 2]):
        sparsity_values[0, 0, block, expert] = 1
    input_a = _to_device(torch.randn((1, num_blocks, m, k), dtype=torch.bfloat16), device)
    input_b = _to_device(torch.randn((1, num_experts, k, n), dtype=torch.bfloat16), device)
    sparsity = _to_device(sparsity_values, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    compact_output = _to_device(torch.zeros((1, num_blocks, m, n), dtype=torch.bfloat16), device)

    # A [1, nnz, M, N] output packs only the active results in sparsity scan order (here experts 3, 1, 7, 2);
    # the golden used to raise NotImplementedError when the output shape differed from the expanded one.
    with comparison_mode():
        ttnn.sparse_matmul(
            input_a,
            input_b,
            sparsity=sparsity,
            nnz=num_blocks,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_sparse_matmul_program_config(),
            dtype=ttnn.bfloat16,
            optional_output_tensor=compact_output,
        )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, op",
    [
        pytest.param(ttnn.moreh_softmax, MOREH_SOFTMAX_OP.SOFTMIN, id="softmax_as_softmin"),
        pytest.param(ttnn.moreh_softmax, MOREH_SOFTMAX_OP.LOGSOFTMAX, id="softmax_as_logsoftmax"),
        pytest.param(ttnn.moreh_softmin, MOREH_SOFTMAX_OP.SOFTMAX, id="softmin_as_softmax"),
        pytest.param(ttnn.moreh_softmin, MOREH_SOFTMAX_OP.LOGSOFTMAX, id="softmin_as_logsoftmax"),
    ],
)
def test_moreh_softmax_with_op_in_comparison_mode(device, operation, op):
    input_tensor = _to_device(torch.randn((32, 32), dtype=torch.bfloat16), device)

    # moreh_softmax and moreh_softmin share one kernel family and `op` overrides each one's default function; the
    # golden always computed the default, so a cross-selected op (e.g. softmax as softmin) mismatched.
    with comparison_mode():
        operation(input_tensor, 1, op=op)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, op",
    [
        pytest.param(ttnn.moreh_softmax_backward, MOREH_SOFTMAX_BACKWARD_OP.SOFTMIN, id="softmax_as_softmin"),
        pytest.param(ttnn.moreh_softmax_backward, MOREH_SOFTMAX_BACKWARD_OP.LOGSOFTMAX, id="softmax_as_logsoftmax"),
        pytest.param(ttnn.moreh_softmin_backward, MOREH_SOFTMAX_BACKWARD_OP.SOFTMAX, id="softmin_as_softmax"),
        pytest.param(ttnn.moreh_softmin_backward, MOREH_SOFTMAX_BACKWARD_OP.LOGSOFTMAX, id="softmin_as_logsoftmax"),
    ],
)
def test_moreh_softmax_backward_with_op_in_comparison_mode(device, operation, op):
    output_tensor = _to_device(torch.softmax(torch.randn((32, 32)), dim=1).to(torch.bfloat16), device)
    output_grad_tensor = _to_device(torch.randn((32, 32), dtype=torch.bfloat16), device)

    # The softmax and softmin backward ops share one kernel family and `op` overrides each one's default
    # derivative; the golden always used the default, so a cross-selected op (or logsoftmax) mismatched.
    with comparison_mode():
        operation(output_tensor, output_grad_tensor, 1, op=op)


@pytest.mark.requires_fast_runtime_mode_off
def test_moreh_mean_over_all_dims_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand((2, 3, 32, 64), dtype=torch.bfloat16), device)

    # moreh keeps a reduced tile dimension (one of the last two) as size 1 even with keepdim=False, so its output
    # shape differs from torch's rank-reduced result and the shape check used to fail. The single bfloat16 value
    # must then match within the device's accumulation error rather than a sub-ULP tolerance.
    with comparison_mode():
        ttnn.moreh_mean(input_tensor, dim=None, keepdim=False)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, args, op_kwargs",
    [
        pytest.param(ttnn.moreh_norm, (2.0,), {"dim": 0}, id="norm"),
        # The single bfloat16 sum must match within the device's accumulation error rather than a sub-ULP tolerance.
        pytest.param(ttnn.moreh_sum, (None,), {}, id="sum_all_dims"),
        pytest.param(ttnn.moreh_sum, (0,), {}, id="sum_dim_0"),
    ],
)
def test_moreh_reduction_of_rank_1_input_in_comparison_mode(device, operation, args, op_kwargs):
    input_tensor = _to_device(torch.empty([5]).uniform_(-1, 1).to(torch.bfloat16), device)

    # For a rank-1 input the reduced dimension is a tile dimension that moreh keeps as size 1, so the output is
    # [1] while torch returns a 0-d scalar, which used to fail the shape check.
    with comparison_mode():
        operation(input_tensor, *args, keepdim=False, **op_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
def test_topk_default_dim_in_comparison_mode(device):
    input_tensor = _to_device(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), device)

    # topk defaults to dim=-1 (and k=32), but the golden defaulted dim to None and passed it to torch.topk, which
    # failed for a call that omits dim.
    with comparison_mode():
        ttnn.topk(input_tensor, 4)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("variant", ["indices_tensor", "stable"])
def test_topk_labels_and_stable_ties_in_comparison_mode(device, variant):
    shape = (1, 1, 32, 64)
    if variant == "indices_tensor":
        # Distinct small integers are exact in bfloat16, so no ties can make the returned labels ambiguous.
        torch_input = torch.stack([torch.randperm(shape[-1]) for _ in range(shape[-2])]).reshape(shape)
        torch_input = torch_input.to(torch.bfloat16)
        labels = torch.arange(shape[-1] - 1, -1, -1, dtype=torch.int32).expand(shape).contiguous()
        topk_kwargs = {"indices_tensor": _to_device(labels, device, dtype=ttnn.uint16)}
    else:
        torch_input = torch.zeros(shape, dtype=torch.bfloat16)
        topk_kwargs = {"stable": True}
    input_tensor = _to_device(torch_input, device)

    # indices_tensor supplies the label returned for each position, and stable=True keeps the lowest index first
    # among ties; torch.topk guarantees neither, so the golden used to return different indices.
    with comparison_mode():
        _, indices = ttnn.topk(input_tensor, 4, dim=-1, **topk_kwargs)

    _, golden_indices = _registered_golden_output(ttnn.topk, input_tensor, 4, dim=-1, **topk_kwargs)
    assert torch.equal(golden_indices.to(torch.int64), ttnn.to_torch(indices).to(torch.int64))


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation",
    [ttnn.transformer.attention_softmax, ttnn.transformer.attention_softmax_],
    ids=["out_of_place", "in_place"],
)
def test_attention_softmax_with_causal_mask_in_comparison_mode(device, operation):
    input_tensor = _to_device(torch.randn(SINGLE_TILE, dtype=torch.bfloat16), device)
    attention_mask = _to_device(torch.zeros(SINGLE_TILE, dtype=torch.bfloat16), device)

    # head_size is optional in the public API (None here), but the golden declared head_size and attention_mask
    # as required keyword-only arguments.
    with comparison_mode():
        operation(input_tensor, head_size=None, attention_mask=attention_mask, causal_mask=True)


@pytest.mark.requires_fast_runtime_mode_off
def test_scaled_dot_product_attention_decode_non_causal_ignores_cur_pos_in_comparison_mode(device):
    batch, num_heads, num_kv_heads, sequence_length, head_dim = 1, 8, 1, 128, 64
    query = _to_device(torch.randn((1, batch, num_heads, head_dim)), device, dtype=ttnn.bfloat16)
    key = _to_device(torch.randn((batch, num_kv_heads, sequence_length, head_dim)), device, dtype=ttnn.bfloat16)
    value = _to_device(torch.randn((batch, num_kv_heads, sequence_length, head_dim)), device, dtype=ttnn.bfloat16)
    attention_mask = _to_device(torch.zeros((batch, 1, num_heads, sequence_length)), device, dtype=ttnn.bfloat16)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(8, 1),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # The non-causal decode reader attends the whole KV cache and cur_pos only bounds causal decode; the golden
    # used to truncate the keys at cur_pos (0 here), attending to a single position.
    with comparison_mode():
        ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            key,
            value,
            is_causal=False,
            attn_mask=attention_mask,
            cur_pos=[0],
            scale=head_dim**-0.5,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )


def _gated_delta_rule_decode_inputs(device, B=2, T=3, H=2, HV=4, K=128, V=128):
    q, k = (torch.nn.functional.normalize(torch.randn(B, T, H, K), dim=-1) for _ in range(2))
    v = torch.randn(B, T, HV, V)
    g = -torch.rand(B, T, HV) * 2
    beta = torch.rand(B, T, HV)
    initial_state = 0.05 * torch.randn(B, HV, K, V)
    inputs = [_to_device(t, device, dtype=ttnn.float32) for t in (q, k, v, g, beta)]
    return inputs, _to_device(initial_state, device, dtype=ttnn.float32)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "op_kwargs",
    [
        pytest.param({}, id="no_state"),
        pytest.param({"output_final_state": True}, id="final_state"),
        pytest.param({"output_per_token_state": True}, id="per_token_state"),
    ],
)
def test_fused_recurrent_gated_delta_rule_in_comparison_mode(device, op_kwargs):
    # GQA (HV = 2 H); the golden must take the op's (g, beta) order, not FLA's (beta, g).
    inputs, initial_state = _gated_delta_rule_decode_inputs(device)
    with comparison_mode():
        ttnn.transformer.fused_recurrent_gated_delta_rule(
            *inputs, initial_state=initial_state, memory_config=ttnn.DRAM_MEMORY_CONFIG, **op_kwargs
        )


@pytest.mark.requires_fast_runtime_mode_off
def test_fused_recurrent_gated_delta_rule_scale_in_comparison_mode(device):
    # o is linear in scale, so PCC cannot detect an ignored scale; compare values.
    inputs, initial_state = _gated_delta_rule_decode_inputs(device)
    operation = ttnn.transformer.fused_recurrent_gated_delta_rule
    output, _ = operation(*inputs, initial_state=initial_state, scale=0.25)
    golden, _ = _registered_golden_output(operation, *inputs, initial_state=initial_state, scale=0.25)
    _assert_golden_dtype_and_close(golden, output, rtol=1e-2, atol=1e-3)


def _signed_operands():
    return tuple(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) * 4 - 2 for _ in range(2))


def _logical_mask(step):
    mask = torch.zeros(SINGLE_TILE, dtype=torch.bfloat16)
    mask[..., ::step] = 1
    return mask


def _channels_with_different_ranges():
    # Channels with different ranges keep the normalized values from correlating with the original input.
    return torch.cat(
        [torch.rand(SINGLE_TILE, dtype=torch.bfloat16), torch.rand(SINGLE_TILE, dtype=torch.bfloat16) * 10 + 50], dim=1
    )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, make_inputs",
    [
        # These in-place ops overwrite input_a, but their goldens were the out-of-place ones, so the stored global
        # golden of input_a kept the pre-op value and the later to_torch comparison used it.
        pytest.param(ttnn.ldexp_, lambda: _ldexp_operands()[:2], id="ldexp_"),
        pytest.param(ttnn.logaddexp_, _signed_operands, id="logaddexp_"),
        pytest.param(ttnn.logaddexp2_, _signed_operands, id="logaddexp2_"),
        # These in-place ops overwrite input_a; its stored global golden must become the logical result.
        pytest.param(ttnn.logical_and_, lambda: (_logical_mask(2), _logical_mask(3)), id="logical_and_"),
        pytest.param(ttnn.logical_or_, lambda: (_logical_mask(2), _logical_mask(3)), id="logical_or_"),
        pytest.param(ttnn.logical_xor_, lambda: (_logical_mask(2), _logical_mask(3)), id="logical_xor_"),
        # logical_not_ overwrites its input; the stored global golden must become the negated result.
        pytest.param(ttnn.logical_not_, lambda: (_logical_mask(2),), id="logical_not_"),
        # bias_gelu_ overwrites input_a with gelu(input_a + input_b); its stored global golden must become that result.
        pytest.param(ttnn.bias_gelu_, _signed_operands, id="bias_gelu_"),
        # normalize_hw is out-of-place, but its golden used to write the normalized values into its input tensor,
        # which is the stored global golden of the device input, corrupting it.
        pytest.param(ttnn.normalize_hw, lambda: (_channels_with_different_ranges(),), id="normalize_hw"),
    ],
)
def test_global_golden_tracks_op_writes(device, tmp_path, operation, make_inputs):
    torch_inputs = make_inputs()

    # The later to_torch comparison of the first input runs against its stored global golden, which must hold what
    # the op left in that tensor rather than a stale or corrupted value.
    with _global_comparison_mode(tmp_path):
        inputs = [_to_device(torch_input, device) for torch_input in torch_inputs]
        operation(*inputs)
        ttnn.to_torch(inputs[0])


@pytest.mark.requires_fast_runtime_mode_off
def test_group_norm_default_inplace_updates_global_golden(device, tmp_path):
    batch_size, channels, height, width, num_groups = 1, 64, 32, 1, 2
    torch_input = torch.rand((batch_size, 1, height * width, channels), dtype=torch.bfloat16)
    sharded_memory_config, grid_size = ttnn.determine_expected_group_norm_sharded_config_and_grid_size(
        device=device,
        num_channels=channels,
        num_groups=num_groups,
        input_nhw=batch_size * height * width,
        is_height_sharded=True,
        is_row_major=True,
    )
    num_cores_across_channel = ttnn.get_group_norm_cores_across_channel(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, grid_size, ttnn.ShardOrientation.ROW_MAJOR
    )
    input_mask = ttnn.to_device(
        ttnn.create_group_norm_input_mask(
            num_channel=channels,
            num_groups=num_groups,
            num_cores_across_channel=num_cores_across_channel,
            data_type=ttnn.bfloat8_b,
        ),
        device,
    )
    weight, bias = (
        _to_device(
            ttnn.create_group_norm_weight_bias_rm(
                input_tensor=torch_parameter, num_channels=channels, num_cores_x=num_cores_across_channel
            ),
            device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for torch_parameter in (
            torch.ones((channels,), dtype=torch.bfloat16),
            torch.zeros((channels,), dtype=torch.bfloat16),
        )
    )

    # group_norm writes its result back into the input by default; the stored global golden of the input must be
    # replaced with the normalized output, otherwise the to_torch comparison uses the pre-norm values.
    with _global_comparison_mode(tmp_path):
        input_tensor = _to_device(
            torch_input,
            device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=sharded_memory_config,
        )
        ttnn.group_norm(
            input_tensor,
            num_groups=num_groups,
            input_mask=input_mask,
            weight=weight,
            bias=bias,
            memory_config=sharded_memory_config,
            core_grid=grid_size,
        )
        ttnn.to_torch(input_tensor)


def test_ring_joint_golden_with_circular_kv_cache():
    batch, heads, head_dim, chunk, window = 1, 2, 32, 32, 32
    num_slabs, chunk_index = 2, 2
    written = (chunk_index + 1) * chunk
    natural_k = torch.randn((batch, heads, written, head_dim))
    natural_v = torch.randn((batch, heads, written, head_dim))
    query = torch.randn((batch, heads, chunk, head_dim))
    slab_groups = [slab + num_slabs * ((chunk_index - slab) // num_slabs) for slab in range(num_slabs)]
    physical_k = torch.cat([natural_k[..., g * chunk : (g + 1) * chunk, :] for g in slab_groups], dim=-2)
    physical_v = torch.cat([natural_v[..., g * chunk : (g + 1) * chunk, :] for g in slab_groups], dim=-2)
    golden_function = ttnn.get_golden_function(ttnn.transformer.ring_joint_scaled_dot_product_attention)
    attention_kwargs = {
        "is_causal": True,
        "sliding_window_size": window,
        "kv_actual_isl": written - chunk,
        "logical_n": written,
    }

    # A circular cache keeps only the last num_slabs chunks, chunk group g in slab g % num_slabs. With a window no
    # wider than a chunk every attended key is resident, so the result must equal attention over the ordered keys.
    expected, _, _ = golden_function(query, natural_k, natural_v, **attention_kwargs)
    actual, _, _ = golden_function(query, physical_k, physical_v, circular_kv_cache=True, **attention_kwargs)

    torch.testing.assert_close(actual, expected)


@pytest.mark.requires_fast_runtime_mode_off
def test_moe_expert_token_remap_in_comparison_mode(device):
    # 32 bfloat16 experts make each topk row 64 bytes; narrower rows land misaligned in the double-buffered
    # topk circular buffer, whose pages are only L1-aligned, and read corrupted data from DRAM.
    batch, seq, experts, reduction_size = 16, 1, 32, 16
    topk = torch.rand((1, batch, seq, experts), dtype=torch.bfloat16) + 0.5
    mapping = torch.ones((1, 1, experts, 1), dtype=torch.int32)
    metadata = torch.arange(experts, dtype=torch.int32).expand(1, batch, seq, experts).contiguous()
    topk_tensor = _to_device(topk, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    mapping_tensor = _to_device(mapping, device, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
    metadata_tensor = _to_device(metadata, device, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)

    # Every expert is local and selected by every token, so the remapped weights equal topk and every reduced flag
    # is set; freshly allocated outputs must match that golden.
    with comparison_mode():
        ttnn.moe_expert_token_remap(topk_tensor, mapping_tensor, metadata_tensor, reduction_size=reduction_size)


@pytest.mark.requires_fast_runtime_mode_off
def test_moe_expert_token_remap_preallocated_outputs_update_global_golden(device, tmp_path):
    # 32 experts keep each topk row DRAM-aligned, as in test_moe_expert_token_remap_in_comparison_mode.
    batch, seq, experts, reduction_size = 16, 1, 32, 16
    topk = torch.rand((1, batch, seq, experts), dtype=torch.bfloat16) + 0.5
    mapping = torch.ones((1, 1, experts, 1), dtype=torch.int32)
    metadata = torch.arange(experts, dtype=torch.int32).expand(1, batch, seq, experts).contiguous()

    # Both preallocated outputs are overwritten by the op; their stored global goldens must become the remapped
    # weights and the reduced activity flags, otherwise the later to_torch comparisons run against the zeros.
    with _global_comparison_mode(tmp_path):
        topk_tensor = _to_device(topk, device, layout=ttnn.ROW_MAJOR_LAYOUT)
        mapping_tensor = _to_device(mapping, device, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
        metadata_tensor = _to_device(metadata, device, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_mapping = _to_device(torch.zeros_like(topk), device, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_reduced = _to_device(
            torch.zeros((1, 1, batch * seq // reduction_size, experts), dtype=torch.int32),
            device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.moe_expert_token_remap(
            topk_tensor,
            mapping_tensor,
            metadata_tensor,
            optional_output_mapping_tensor=output_mapping,
            optional_output_reduced_tensor=output_reduced,
            reduction_size=reduction_size,
        )
        ttnn.to_torch(output_mapping)
        ttnn.to_torch(output_reduced)


def test_ring_joint_golden_stats_match_device_scratch_layout():
    batch, heads, head_dim, sequence, joint_sequence = 1, 2, 32, 40, 8
    query, key, value = (torch.randn((batch, heads, sequence, head_dim)) for _ in range(3))
    joint_query, joint_key, joint_value = (torch.randn((batch, heads, joint_sequence, head_dim)) for _ in range(3))
    golden_function = ttnn.get_golden_function(ttnn.transformer.ring_joint_scaled_dot_product_attention)

    # The third output is the device's running-max and running-sum scratch over the tile-padded main and joint
    # lengths (2 * (64 + 32) rows here), not a final log-sum-exp, so the golden returns that shape and skips values.
    _, _, stats = golden_function(query, key, value, joint_query, joint_key, joint_value, logical_n=sequence)

    assert stats.shape == (batch, heads, 2 * (64 + 32), 1)
    assert stats._ttnn_comparison_config.method == "skip"


@pytest.mark.requires_fast_runtime_mode_off
def test_std_hw_on_non_tile_aligned_input_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand((1, 1, 33, 33), dtype=torch.bfloat16), device)

    # The device sums squared deviations over the 33x33 logical elements but divides by the padded 64x64 area,
    # which the golden must reproduce.
    with comparison_mode():
        ttnn.std_hw(input_tensor)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", [ttnn.rand, ttnn.randn], ids=["rand", "randn"])
def test_random_creation_in_comparison_mode(device, operation):
    # Host and device draw different samples, so comparison mode checks only the output structure, not the values.
    with comparison_mode():
        operation(SINGLE_TILE, device=device)


@pytest.mark.requires_fast_runtime_mode_off
def test_gelu_fast_lut_skips_comparison(device, monkeypatch):
    input_tensor = _to_device(torch.linspace(-3.0, 3.0, 1024).reshape(SINGLE_TILE).to(torch.bfloat16), device)
    comparison_records = _capture_local_comparison_records(monkeypatch)

    # The FastLut variant is outside the exact GELU contract, so its golden intentionally produces no comparison.
    with comparison_mode():
        ttnn.gelu(input_tensor, variant=ttnn.GeluVariant.FastLut)

    assert not comparison_records


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16, torch.uint32], ids=["uint8", "uint16", "uint32"])
def test_leaky_relu_golden_keeps_unsigned_values(dtype):
    values = torch.tensor([0, 1, torch.iinfo(dtype).max], dtype=dtype)

    # Unsigned inputs have no negative lanes, so leaky_relu is the identity for them whatever the slope.
    assert torch.equal(ttnn.get_golden_function(ttnn.leaky_relu)(values, 0.5), values)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, source_on_device",
    [
        pytest.param(ttnn.copy_host_to_device_tensor, False, id="host_to_device"),
        pytest.param(ttnn.copy_device_to_host_tensor, True, id="device_to_host"),
    ],
)
def test_copy_tensor_updates_global_golden(device, tmp_path, operation, source_on_device):
    # A None device keeps the tensor on host.
    source_device, destination_device = (device, None) if source_on_device else (None, device)

    # The copy overwrites the destination tensor, so its stored global golden must become the source values.
    with _global_comparison_mode(tmp_path):
        source = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), source_device)
        destination = _to_device(torch.zeros(SINGLE_TILE, dtype=torch.bfloat16), destination_device)
        operation(source, destination)
        ttnn.to_torch(destination)


@pytest.mark.requires_fast_runtime_mode_off
def test_moreh_clip_grad_norm_with_infinite_gradient_in_comparison_mode(device):
    torch_grad = torch.rand(SINGLE_TILE, dtype=torch.bfloat16)
    torch_grad[0, 0, 0, 0] = float("inf")
    inputs = [_to_device(torch_grad, device), _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)]

    # An infinite gradient makes the total norm non-finite; comparison accepts matching non-finite totals.
    with comparison_mode():
        ttnn.moreh_clip_grad_norm(inputs, 1.0, 2.0, False)


@pytest.mark.requires_fast_runtime_mode_off
def test_load_tensor_in_comparison_mode(device, tmp_path):
    file_name = str(tmp_path / "tensor.tensorbin")
    ttnn.dump_tensor(file_name, _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device))

    # Loading onto a device must compare the loaded device tensor against the host tensor read from the file.
    with comparison_mode():
        ttnn.load_tensor(file_name, device=device)


@pytest.mark.requires_fast_runtime_mode_off
def test_atanh_boundary_and_out_of_domain_matches_golden(device):
    torch_input = torch.tensor([-1.0, 1.0, -2.0, 2.0]).repeat(256).reshape(SINGLE_TILE)
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device)

    # atanh is -inf at -1, +inf at 1 and NaN beyond; the golden's non-finite values must match the device in sign
    # and position. The message lists both value sets so a mismatch shows which side to change.
    output = ttnn.atanh(input_tensor)

    device_values = ttnn.to_torch(output).reshape(-1)[:4].tolist()
    golden_values = ttnn.get_golden_function(ttnn.atanh)(torch_input.to(torch.bfloat16)).reshape(-1)[:4].tolist()
    assert str(device_values) == str(golden_values), f"device {device_values} != golden {golden_values}"


@pytest.mark.requires_fast_runtime_mode_off
def test_nextafter_toward_subnormal_in_comparison_mode(device):
    input_a = _to_device(torch.zeros(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_b = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)

    # The step from 0 toward 1 is the smallest subnormal, which the device may flush to zero.
    with comparison_mode():
        ttnn.nextafter(input_a, input_b)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("dtype", [ttnn.uint16, ttnn.uint32], ids=["uint16", "uint32"])
@pytest.mark.parametrize("operand_form", ["scalar", "tensor"])
def test_ne_with_highest_unsigned_values_in_comparison_mode(device, dtype, operand_form):
    torch_dtype = ttnn.ttnn_dtype_to_torch_dtype(dtype)
    highest_value = 2**16 - 1 if dtype == ttnn.uint16 else 2**32 - 1
    torch_input = torch.zeros(SINGLE_TILE, dtype=torch.int64)
    torch_input[..., ::2] = highest_value
    input_tensor = _to_device(torch_input.to(torch_dtype), device, dtype=dtype)
    other = highest_value
    if operand_form == "tensor":
        other = _to_device(
            torch.full(SINGLE_TILE, highest_value, dtype=torch.int64).to(torch_dtype), device, dtype=dtype
        )

    # Zero and the top of the unsigned range must compare as unsigned values, not as sign-extended ones.
    with comparison_mode():
        ttnn.ne(input_tensor, other)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "math_op",
    [ttnn.BcastOpMath.ADD, ttnn.BcastOpMath.SUB, ttnn.BcastOpMath.MUL],
    ids=["add", "sub", "mul"],
)
def test_bcast_width_with_tile_wide_operand_in_comparison_mode(device, math_op):
    input_a = _to_device(torch.rand((1, 1, 32, 64), dtype=torch.bfloat16), device)
    input_b = _to_device(torch.rand((1, 1, 32, 32), dtype=torch.bfloat16), device)

    # A width broadcast reads only the first column of input_b's tile, which the golden must reproduce.
    with comparison_mode():
        ttnn.bcast(input_a, input_b, math_op, ttnn.BcastOpDim.W)


@pytest.mark.requires_fast_runtime_mode_off
def test_narrow_before_partially_padded_tile_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand((1, 1, 33, 32), dtype=torch.bfloat16), device)

    # TILE narrowing needs tile-aligned start and length, so the rows [0, 32) end at the tile boundary right
    # before the partially padded second tile; the view must hold exactly those logical rows.
    with comparison_mode():
        ttnn.narrow(input_tensor, -2, 0, 32)


@pytest.mark.requires_fast_runtime_mode_off
def test_xlogy_bw_at_boundaries_in_comparison_mode(device):
    pairs = torch.cartesian_prod(
        torch.tensor([0.0, 1.0, 2.0, float("nan")]), torch.tensor([-1.0, 0.0, 2.0, float("nan")])
    )
    input_a = _to_device(pairs[:, 0].repeat(64).reshape(SINGLE_TILE).to(torch.bfloat16), device)
    input_b = _to_device(pairs[:, 1].repeat(64).reshape(SINGLE_TILE).to(torch.bfloat16), device)
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)

    # Zero, negative and NaN operands hit every boundary of x * log(y); both gradients must match torch autograd.
    with comparison_mode():
        ttnn.xlogy_bw(grad, input_a, input_b)


@pytest.mark.requires_fast_runtime_mode_off
def test_sort_wide_float32_descending_in_comparison_mode(device):
    width = 8192
    input_tensor = _to_device(torch.randn((1, 1, 32, width)), device, dtype=ttnn.float32)

    # A row this wide takes the multi-core DRAM sort; no returned index may point into the padded storage.
    with comparison_mode():
        _, indices = ttnn.sort(input_tensor, dim=-1, descending=True)

    assert int(ttnn.to_torch(indices).to(torch.int64).max()) < width
