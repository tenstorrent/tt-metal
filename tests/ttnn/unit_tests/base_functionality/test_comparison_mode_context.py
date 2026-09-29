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

    # generated/ttnn.leaky_relu.md: leaky-relu-golden-ignores-positional-slope
    with comparison_mode():
        output_tensor = ttnn.leaky_relu(input_tensor, 0.5)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_sum_with_scalar_in_comparison_mode(device):
    torch_input = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.sum.md: sum-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.sum(input_tensor, dim=-1, keepdim=True, scalar=0.5)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_sum_int32_with_fractional_scalar_in_comparison_mode(device):
    torch_input = torch.ones((1, 1, 32, 32), dtype=torch.int32)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # generated/ttnn.sum.md: sum-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.sum(input_tensor, dim=-1, keepdim=True, scalar=0.5)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_mean_with_zero_scalar_in_comparison_mode(device):
    torch_input = torch.arange(1, 1025, dtype=torch.float32).to(torch.bfloat16).reshape(1, 1, 32, 32)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.mean.md: mean-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.mean(input_tensor, dim=-1, keepdim=True, scalar=0.0)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_max_with_negative_scalar_in_comparison_mode(device):
    torch_input = torch.arange(1, 1025, dtype=torch.float32).to(torch.bfloat16).reshape(1, 1, 32, 32)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.max.md: max-scalar-not-modeled
    with comparison_mode():
        output_tensor = ttnn.max(input_tensor, dim=-1, keepdim=True, scalar=-2.0)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_min_with_negative_scalar_in_comparison_mode(device):
    torch_input = torch.arange(1, 1025, dtype=torch.float32).to(torch.bfloat16).reshape(1, 1, 32, 32)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.min.md: inherits mean-scalar-ignored from ttnn.mean
    with comparison_mode():
        output_tensor = ttnn.min(input_tensor, dim=-1, keepdim=True, scalar=-2.0)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_var_with_zero_scalar_in_comparison_mode(device):
    columns = torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 32)
    row_scales = torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 32, 1)
    torch_input = (row_scales * columns).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.var.md: var-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.var(input_tensor, dim=-1, keepdim=True, scalar=0.0, correction=False)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_std_with_zero_scalar_in_comparison_mode(device):
    columns = torch.arange(32, dtype=torch.float32).reshape(1, 1, 1, 32)
    row_scales = torch.arange(1, 33, dtype=torch.float32).reshape(1, 1, 32, 1)
    torch_input = (row_scales * columns).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.std.md: std-scalar-ignored
    with comparison_mode():
        output_tensor = ttnn.std(input_tensor, dim=-1, keepdim=True, scalar=0.0, correction=False)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_selu_with_non_default_parameters_in_comparison_mode(device):
    torch_input = torch.full((1, 1, 32, 32), -1.0, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.selu.md: selu-golden-ignores-parameters
    with comparison_mode():
        output_tensor = ttnn.selu(input_tensor, scale=0.9, alpha=1.2)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_softmax_over_dimension_zero_in_comparison_mode(device):
    torch_input = torch.zeros((2, 32, 32), dtype=torch.bfloat16)
    torch_input[1] = 4.0
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.softmax.md: softmax-zero-dim
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

    # generated/ttnn.rms_norm.md: rms-norm-missing-bias
    with comparison_mode():
        output_tensor = ttnn.rms_norm(input_tensor, weight=weight, bias=bias)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_rms_norm_with_residual_in_comparison_mode(device):
    torch_input = torch.zeros((32, 32), dtype=torch.bfloat16)
    torch_residual = torch.arange(32, dtype=torch.float32).to(torch.bfloat16).repeat(32, 1)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    residual = ttnn.from_torch(torch_residual, layout=ttnn.TILE_LAYOUT, device=device)

    # generated/ttnn.rms_norm.md: rms-norm-missing-residual
    with comparison_mode():
        output_tensor = ttnn.rms_norm(input_tensor, residual_input_tensor=residual)

    assert isinstance(output_tensor, ttnn.Tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_abs_of_complex_tensor_in_comparison_mode(device):
    complex_input = _complex_input(
        torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, device
    )

    # generated/ttnn.abs.md: abs-complex-preprocessing
    with comparison_mode():
        ttnn.abs(complex_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("memory_config", [None, ttnn.DRAM_MEMORY_CONFIG], ids=["inherited", "dram"])
def test_real_of_complex_tensor_in_comparison_mode(device, memory_config):
    complex_input = _complex_input(
        torch.rand(SINGLE_TILE, dtype=torch.bfloat16), torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 2, device
    )

    # generated/ttnn.real.md: complex-input-not-preprocessed
    with comparison_mode():
        ttnn.real(complex_input, memory_config=memory_config)


@pytest.mark.requires_fast_runtime_mode_off
def test_polar_of_complex_tensor_in_comparison_mode(device):
    complex_input = _complex_input(
        torch.full(SINGLE_TILE, 2.0, dtype=torch.bfloat16), torch.zeros(SINGLE_TILE, dtype=torch.bfloat16), device
    )

    # generated/ttnn.polar.md: single-argument-polar-golden
    with comparison_mode():
        ttnn.polar(complex_input)


@pytest.mark.requires_fast_runtime_mode_off
def test_acos_bfloat8_b_out_of_domain_input_is_compared_in_comparison_mode(device, monkeypatch):
    torch_input = torch.linspace(-2.0, 2.0, 1024).reshape(SINGLE_TILE)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.bfloat8_b)
    comparison_records = _capture_local_comparison_records(monkeypatch)

    # generated/ttnn.acos.md: acos-invalid-bfloat8-skips-comparison
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
    # generated/ttnn.exp.md: exp-fast-mode-reference
    # generated/ttnn.tanh.md: tanh-fast-mode-ignored
    # A constant tile makes PCC degenerate, so the golden's comparison policy decides the result.
    for value in torch.linspace(low, high, 9).tolist():
        input_tensor = _to_device(torch.full(SINGLE_TILE, value, dtype=torch.bfloat16), device)
        with comparison_mode():
            operation(input_tensor, fast_and_approximate_mode=True)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", [ttnn.log, ttnn.log10, ttnn.log1p, ttnn.log2], ids=lambda op: op.__name__)
def test_log_family_with_fast_and_approximate_mode_in_comparison_mode(device, operation):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 0.5, device)

    # generated/ttnn.log.md: unary-fast-flag-forwarding
    # generated/ttnn.log10.md, ttnn.log1p.md, ttnn.log2.md: inherit unary-fast-flag-forwarding from ttnn.log
    with comparison_mode():
        operation(input_tensor, fast_and_approximate_mode=True)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", [ttnn.tril, ttnn.triu], ids=["tril", "triu"])
def test_triangular_with_keyword_diagonal_in_comparison_mode(device, operation):
    torch_input = torch.rand((32, 32)) * 0.01
    torch_input += torch.diag(torch.full((32,), 100.0)) + torch.diag(torch.full((31,), 100.0), diagonal=1)
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device)

    # generated/ttnn.tril.md: tril-keyword-diagonal-dropped
    # generated/ttnn.triu.md: inherits tril-keyword-diagonal-dropped from ttnn.tril
    with comparison_mode():
        operation(input_tensor, diagonal=1)


@pytest.mark.requires_fast_runtime_mode_off
def test_logit_with_eps_in_comparison_mode(device):
    torch_input = torch.linspace(0.0, 1.0, 1024).reshape(SINGLE_TILE)
    torch_input[..., ::2] = 0.0
    torch_input[..., 1::4] = 1.0
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device)

    # generated/ttnn.logit.md: logit-positional-eps-dropped
    # The public binding makes eps keyword-only, so only the keyword form can be exercised.
    with comparison_mode():
        ttnn.logit(input_tensor, eps=0.1)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("rounding_mode", ["trunc", "floor"])
def test_rdiv_with_rounding_mode_in_comparison_mode(device, rounding_mode):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 0.5, device)

    # generated/ttnn.rdiv.md: rounding-mode-ignored
    with comparison_mode():
        ttnn.rdiv(input_tensor, 2.0, rounding_mode=rounding_mode)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", [ttnn.normalize_global, ttnn.normalize_hw], ids=["global", "hw"])
def test_normalize_uses_population_standard_deviation_in_comparison_mode(device, operation):
    torch_input = torch.rand(SINGLE_TILE, dtype=torch.float32)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.float32)
    dims = (0, 1, 2, 3) if operation is ttnn.normalize_global else (-2, -1)

    # generated/ttnn.normalize_global.md: normalize-global-sample-standard-deviation
    # generated/ttnn.normalize_hw.md: normalize-hw-sample-standard-deviation
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

    # generated/ttnn.logical_not.md: logical-not-int32-min-overflow
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

    # generated/ttnn.bitwise_and.md: bitwise-and-unsigned-reference
    # generated/ttnn.bitwise_or.md: inherits bitwise-and-unsigned-reference from ttnn.bitwise_and
    with comparison_mode():
        operation(input_tensor, other)


@pytest.mark.requires_fast_runtime_mode_off
def test_logical_right_shift_with_invalid_counts_in_comparison_mode(device):
    input_tensor = _to_device(torch.full(SINGLE_TILE, -1, dtype=torch.int32), device, dtype=ttnn.int32)
    shift_counts = torch.tensor([31, 32, -1, 33], dtype=torch.int32).repeat(256).reshape(SINGLE_TILE)
    shift_tensor = _to_device(shift_counts, device, dtype=ttnn.int32)

    # generated/ttnn.logical_right_shift.md: logical-right-shift-invalid-counts
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

    # generated/ttnn.unary_chain.md: dtype-changing-chain-ops-unsupported
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
def test_bitcast_fallback_keeps_requested_dtype(device, input_dtype, output_dtype, float_dtype, bits_dtype):
    torch_input = _bit_patterns(torch.rand(SINGLE_TILE, dtype=float_dtype) + 1, bits_dtype)
    input_tensor = _to_device(torch_input, device, dtype=input_dtype)

    with comparison_mode():
        output = ttnn.bitcast(input_tensor, output_dtype)

    # generated/ttnn.bitcast.md: bitcast-postprocess-dtype
    # The postprocessor runs only on the golden fallback path, not in comparison mode.
    fallback_output = ttnn.get_fallback_function(ttnn.bitcast)(input_tensor, output_dtype)
    assert fallback_output.dtype == output.dtype
    # Compare with the reinterpreted input bits: the device FLOAT32 bitcast result is checked separately.
    assert torch.equal(ttnn.to_torch(fallback_output), ttnn.to_torch(input_tensor).view(float_dtype))


@pytest.mark.requires_fast_runtime_mode_off
def test_addalpha_bw_with_disabled_gradient_in_comparison_mode(device):
    grad, input_a, input_b = (_to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device) for _ in range(3))

    # generated/ttnn.addalpha_bw.md: addalpha-required-outputs-ignored
    with comparison_mode():
        gradients = ttnn.addalpha_bw(grad, input_a, input_b, 2.0, are_required_outputs=[True, False])

    assert gradients[1] is None


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, operand_names, extra_kwargs",
    [
        pytest.param(ttnn.sub_bw, ("input_tensor", "other_tensor"), {}, id="sub_bw"),
        pytest.param(ttnn.subalpha_bw, ("input_tensor_a", "input_tensor_b"), {"alpha": 2.0}, id="subalpha_bw"),
        pytest.param(ttnn.squared_difference_bw, ("input_tensor_a", "input_tensor_b"), {}, id="squared_difference_bw"),
    ],
)
def test_binary_backward_with_keyword_operands_in_comparison_mode(device, operation, operand_names, extra_kwargs):
    grad, input_a, input_b = (_to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device) for _ in range(3))
    operands = dict(zip(operand_names, (input_a, input_b)))

    # generated/ttnn.sub_bw.md: backward-keyword-arguments
    # generated/ttnn.subalpha_bw.md, ttnn.squared_difference_bw.md: inherit backward-keyword-arguments from ttnn.sub_bw
    with comparison_mode():
        operation(grad_tensor=grad, **operands, **extra_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
def test_hardshrink_bw_with_keyword_lambd_in_comparison_mode(device):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch.linspace(-4.0, 4.0, 1024).reshape(SINGLE_TILE).to(torch.bfloat16), device)

    # generated/ttnn.hardshrink_bw.md: hardshrink-bw-lambd-keyword
    with comparison_mode():
        ttnn.hardshrink_bw(grad, input_tensor, lambd=2.2)


@pytest.mark.requires_fast_runtime_mode_off
def test_softshrink_bw_with_keyword_lambd_in_comparison_mode(device):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch.linspace(-4.0, 4.0, 1024).reshape(SINGLE_TILE).to(torch.bfloat16), device)

    # generated/ttnn.softshrink_bw.md: softshrink-bw-keyword-lambd
    with comparison_mode():
        ttnn.softshrink_bw(grad, input_tensor, lambd=2.0)


@pytest.mark.requires_fast_runtime_mode_off
def test_hardtanh_bw_with_keyword_bounds_in_comparison_mode(device):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    torch_input = torch.tensor([-0.9, 0.0, 0.9, 0.5]).repeat(256).reshape(SINGLE_TILE)
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device)

    # generated/ttnn.hardtanh_bw.md: hardtanh-bw-keyword-bounds
    with comparison_mode():
        ttnn.hardtanh_bw(grad, input_tensor, min=-0.8, max=0.8)


@pytest.mark.requires_fast_runtime_mode_off
def test_leaky_relu_bw_with_keyword_slope_in_comparison_mode(device):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch.linspace(-2.0, 2.0, 1024).reshape(SINGLE_TILE).to(torch.bfloat16), device)

    # generated/ttnn.leaky_relu_bw.md: leaky-relu-bw-keyword-slope-ignored
    with comparison_mode():
        ttnn.leaky_relu_bw(grad, input_tensor, negative_slope=0.5)


@pytest.mark.requires_fast_runtime_mode_off
def test_softplus_bw_with_only_beta_in_comparison_mode(device):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch.linspace(-2.0, 2.0, 1024).reshape(SINGLE_TILE).to(torch.bfloat16), device)

    # generated/ttnn.softplus_bw.md: softplus-bw-partial-kwargs
    with comparison_mode():
        output = ttnn.softplus_bw(grad, input_tensor, beta=4.0)

    golden = _registered_golden_output(ttnn.softplus_bw, grad, input_tensor, beta=4.0)[0]
    torch.testing.assert_close(golden.float(), ttnn.to_torch(output[0]).float(), rtol=5e-2, atol=2e-2)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
def test_rdiv_bw_with_keyword_rounding_mode_in_comparison_mode(device, rounding_mode):
    grad = _to_device(torch.ones(SINGLE_TILE, dtype=torch.bfloat16), device)
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, device)

    # generated/ttnn.rdiv_bw.md: keyword-rounding-mode-ignored
    with comparison_mode():
        ttnn.rdiv_bw(grad, input_tensor, 2.0, rounding_mode=rounding_mode)


@pytest.mark.requires_fast_runtime_mode_off
def test_rpow_bw_matches_reverse_power_derivative_in_comparison_mode(device):
    torch_grad = torch.ones(SINGLE_TILE, dtype=torch.bfloat16)
    torch_input = torch.linspace(0.5, 2.0, 1024).reshape(SINGLE_TILE).to(torch.bfloat16)
    grad = _to_device(torch_grad, device)
    input_tensor = _to_device(torch_input, device)

    # generated/ttnn.rpow_bw.md: rpow-bw-uses-forward-power
    # TTNN and the golden share the reversed derivative, so both are checked against the ideal reference.
    with comparison_mode():
        output = ttnn.rpow_bw(grad, input_tensor, 3.0)

    ideal = torch_grad.float() * torch.log(torch.tensor(3.0)) * torch.pow(3.0, torch_input.float())
    golden = _registered_golden_output(ttnn.rpow_bw, grad, input_tensor, 3.0)[0]
    torch.testing.assert_close(golden.float(), ideal, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(ttnn.to_torch(output[0]).float(), ideal, rtol=2e-2, atol=2e-2)


@pytest.mark.requires_fast_runtime_mode_off
def test_gelu_bw_tanh_variant_in_comparison_mode(device):
    torch_input = torch.linspace(-3.0, 3.0, 1024).reshape(SINGLE_TILE)
    torch_grad = torch.ones(SINGLE_TILE)
    grad = _to_device(torch_grad, device, dtype=ttnn.float32)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.float32)

    # generated/ttnn.gelu_bw.md: gelu-bw-tanh-variant-ignored
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

    # generated/ttnn.where_bw.md: nonboolean-predicate-golden
    with comparison_mode():
        ttnn.where_bw(grad, condition, input_a, input_b)


@pytest.mark.requires_fast_runtime_mode_off
def test_lerp_with_public_keywords_in_comparison_mode(device):
    input_tensor, end_tensor = (_to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device) for _ in range(2))

    # generated/ttnn.lerp.md: lerp-golden-public-keywords-unsupported
    with comparison_mode():
        ttnn.lerp(input=input_tensor, end=end_tensor, weight=0.5)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("weight_form", ["scalar", "tensor"])
def test_lerp_bw_with_public_keywords_in_comparison_mode(device, weight_form):
    grad, input_a, input_b, weight = (
        _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device) for _ in range(4)
    )
    weight_kwargs = {"scalar": 0.5} if weight_form == "scalar" else {"input_tensor_c": weight}

    # generated/ttnn.lerp_bw.md: lerp-bw-public-keywords-unsupported
    with comparison_mode():
        ttnn.lerp_bw(grad_tensor=grad, input_tensor_a=input_a, input_tensor_b=input_b, **weight_kwargs)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("output_form", ["dtype", "output_tensor"])
def test_add_with_block_float_output_in_comparison_mode(device, output_form):
    input_a = _to_device(_block_float_sensitive_values(SINGLE_TILE), device, dtype=ttnn.float32)
    input_b = _to_device(torch.zeros(SINGLE_TILE), device, dtype=ttnn.float32)
    if output_form == "dtype":
        output_kwargs = {"dtype": ttnn.bfloat4_b}
    else:
        output_kwargs = {"output_tensor": _to_device(torch.zeros(SINGLE_TILE), device, dtype=ttnn.bfloat8_b)}

    # generated/ttnn.add.md: add-output-dtype-ignored
    with comparison_mode():
        output = ttnn.add(input_a, input_b, **output_kwargs)

    _assert_golden_matches_output(_registered_golden_output(ttnn.add, input_a, input_b, **output_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_add_int32_with_integral_float_scalar_in_comparison_mode(device):
    input_tensor = _to_device(torch.full(SINGLE_TILE, 16777217, dtype=torch.int32), device, dtype=ttnn.int32)

    # generated/ttnn.add.md: add-integral-float-scalar
    with comparison_mode():
        output = ttnn.add(input_tensor, 2.0)

    assert torch.equal(ttnn.to_torch(output), torch.full(SINGLE_TILE, 16777219, dtype=torch.int32))
    _assert_golden_matches_output(_registered_golden_output(ttnn.add, input_tensor, 2.0), output)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "minimum_kwargs",
    [
        pytest.param({"input_tensor_a_activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)]}, id="activation"),
        pytest.param({"dtype": ttnn.float32}, id="dtype"),
    ],
)
def test_minimum_with_value_affecting_options_in_comparison_mode(device, minimum_kwargs):
    input_a, input_b = (_to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, device) for _ in range(2))

    # generated/ttnn.minimum.md: minimum-options-not-modeled
    with comparison_mode():
        output = ttnn.minimum(input_a, input_b, **minimum_kwargs)

    _assert_golden_matches_output(_registered_golden_output(ttnn.minimum, input_a, input_b, **minimum_kwargs), output)


def test_situ_glu_golden_accepts_keyword_betas():
    gate = torch.randn((32, 64))
    up = torch.randn((32, 64))
    golden_function = ttnn.get_golden_function(ttnn.situ_glu)

    # generated/ttnn.situ_glu.md: situ-glu-golden-keyword-betas
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

    # generated/ttnn.l1_loss.md: reduction-enum-not-mapped
    # generated/ttnn.mse_loss.md: enum-reduction-not-adapted
    with comparison_mode():
        operation(reference, prediction, reduction=reduction)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, None], ids=["bfloat16", "default"])
def test_full_with_dtype_in_comparison_mode(device, dtype):
    full_args = ([1, 1, 32, 32], 1.5)
    full_kwargs = {"layout": ttnn.TILE_LAYOUT, "device": device}
    if dtype is not None:
        full_kwargs["dtype"] = dtype

    # generated/ttnn.full.md: full-dtype-ignored
    with comparison_mode():
        output = ttnn.full(*full_args, **full_kwargs)

    _assert_golden_matches_output(_registered_golden_output(ttnn.full, *full_args, **full_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_ones_like_with_dtype_override_in_comparison_mode(device):
    input_tensor = _to_device(torch.zeros(SINGLE_TILE, dtype=torch.int32), device, dtype=ttnn.int32)

    # generated/ttnn.ones_like.md: ones-like-dtype-override
    with comparison_mode():
        output = ttnn.ones_like(input_tensor, dtype=ttnn.float32)

    _assert_golden_matches_output(_registered_golden_output(ttnn.ones_like, input_tensor, dtype=ttnn.float32), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_zeros_like_with_tensor_keyword_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # generated/ttnn.zeros_like.md: zeros-like-keyword-arguments
    with comparison_mode():
        ttnn.zeros_like(tensor=input_tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_zeros_like_with_dtype_override_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # generated/ttnn.zeros_like.md: zeros-like-dtype-override-ignored
    with comparison_mode():
        output = ttnn.zeros_like(input_tensor, dtype=ttnn.float32)

    _assert_golden_matches_output(_registered_golden_output(ttnn.zeros_like, input_tensor, dtype=ttnn.float32), output)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, fill_values",
    [pytest.param(ttnn.fill_rm, (1.0, 0.0), id="fill_rm"), pytest.param(ttnn.fill_ones_rm, (), id="fill_ones_rm")],
)
def test_fill_rm_dtype_follows_any_in_comparison_mode(device, operation, fill_values):
    any_tensor = _to_device(torch.zeros(SINGLE_TILE, dtype=torch.bfloat16), device)
    fill_args = (*SINGLE_TILE, 16, 16, any_tensor, *fill_values)

    # generated/ttnn.fill_rm.md: fill-rm-golden-drops-any-dtype
    # generated/ttnn.fill_ones_rm.md: inherits fill-rm-golden-drops-any-dtype from ttnn.fill_rm
    with comparison_mode():
        output = operation(*fill_args)

    _assert_golden_matches_output(_registered_golden_output(operation, *fill_args), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_clone_with_dtype_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # generated/ttnn.clone.md: clone-golden-ignores-dtype
    with comparison_mode():
        output = ttnn.clone(input_tensor, dtype=ttnn.float32)

    _assert_golden_matches_output(_registered_golden_output(ttnn.clone, input_tensor, dtype=ttnn.float32), output)


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

    # generated/ttnn.to_dtype.md: identity-does-not-cast-values
    with comparison_mode():
        output = ttnn.to_dtype(input_tensor, dtype)

    _assert_golden_matches_output(_registered_golden_output(ttnn.to_dtype, input_tensor, dtype), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_to_torch_with_dtype_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # generated/ttnn.to_torch.md: to-torch-golden-ignores-conversion-options
    with comparison_mode():
        output = ttnn.to_torch(input_tensor, dtype=torch.float32)

    golden = _registered_golden_output(ttnn.to_torch, input_tensor, dtype=torch.float32)
    assert golden.dtype == output.dtype, f"golden dtype {golden.dtype} != output dtype {output.dtype}"
    assert torch.equal(golden, output)


@pytest.mark.requires_fast_runtime_mode_off
def test_from_torch_with_col_tilize_in_comparison_mode():
    torch_input = torch.randn((32, 64), dtype=torch.float32)

    # generated/ttnn.from_torch.md: col-tilize-not-modeled
    with comparison_mode():
        ttnn.from_torch(torch_input, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, col_tilize=True)


@pytest.mark.requires_fast_runtime_mode_off
def test_as_tensor_with_preprocess_in_comparison_mode(device):
    torch_input = torch.rand(SINGLE_TILE, dtype=torch.bfloat16)

    # generated/ttnn.as_tensor.md: as-tensor-preprocess-not-modeled
    with comparison_mode():
        ttnn.as_tensor(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            preprocess=torch.neg,
        )


@pytest.mark.parametrize(
    "operation",
    [ttnn.allocate_tensor_on_device, ttnn.allocate_tensor_on_host],
    ids=["allocate_tensor_on_device", "allocate_tensor_on_host"],
)
def test_allocator_skip_policy_still_rejects_shape_mismatch(operation):
    golden = ttnn.get_golden_function(operation)((2, 3), ttnn.bfloat16, ttnn.TILE_LAYOUT, None, None)

    # generated/ttnn.allocate_tensor_on_device.md: allocator-skip-bypasses-shape-check
    # generated/ttnn.allocate_tensor_on_host.md: inherits allocator-skip-bypasses-shape-check
    comparison_records = _compare_torch_tensors(
        golden, torch.zeros((2, 4), dtype=torch.bfloat16), fail_on_bad_comparison=False
    )

    assert comparison_records, "the skip policy returned no comparison record for a shape mismatch"
    assert not comparison_records[0]["matches"]


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "output_dtype, make_values",
    [
        pytest.param(ttnn.bfloat16, _bfloat16_sensitive_values, id="bfloat16"),
        pytest.param(ttnn.bfloat8_b, _block_float_sensitive_values, id="bfloat8_b"),
    ],
)
def test_tilize_with_output_dtype_in_comparison_mode(device, output_dtype, make_values):
    input_tensor = _to_device(make_values(SINGLE_TILE), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)

    # generated/ttnn.tilize.md: tilize-output-dtype-not-modeled
    with comparison_mode():
        output = ttnn.tilize(input_tensor, dtype=output_dtype)

    _assert_golden_matches_output(_registered_golden_output(ttnn.tilize, input_tensor, dtype=output_dtype), output)


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

    # generated/ttnn.interleaved_to_sharded.md: interleaved-to-sharded-output-dtype-not-modeled
    # generated/ttnn.interleaved_to_sharded_partial.md: inherits interleaved-to-sharded-output-dtype-not-modeled
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

    # generated/ttnn.sharded_to_interleaved.md: sharded-to-interleaved-output-dtype-not-modeled
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

    # generated/ttnn.sharded_to_interleaved_partial.md: sharded-to-interleaved-partial-output-dtype-not-modeled
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

    # generated/ttnn.quantize.md: narrow-output-saturation-missing
    with comparison_mode():
        output = ttnn.quantize(input_tensor, 1.0, 0, dtype=dtype)

    _assert_golden_matches_output(_registered_golden_output(ttnn.quantize, input_tensor, 1.0, 0, dtype=dtype), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_requantize_saturates_int8_output_in_comparison_mode(device):
    torch_input = torch.tensor([0, 300, -10, -300], dtype=torch.int32).repeat(256).reshape(SINGLE_TILE)
    input_tensor = _to_device(torch_input, device, dtype=ttnn.int32)
    requantize_args = (input_tensor, 1.0, 0, 1.0, 0)

    # generated/ttnn.requantize.md: requantize-narrow-output-does-not-saturate
    with comparison_mode():
        output = ttnn.requantize(*requantize_args, dtype=ttnn.int8)

    _assert_golden_matches_output(_registered_golden_output(ttnn.requantize, *requantize_args, dtype=ttnn.int8), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_dequantize_fallback_defaults_to_bfloat16(device):
    input_tensor = _to_device(torch.full(SINGLE_TILE, 3, dtype=torch.int32), device, dtype=ttnn.int32)

    with comparison_mode():
        output = ttnn.dequantize(input_tensor, 0.5, 2)

    # generated/ttnn.dequantize.md: dequantize-input-dtype-postprocess
    # The postprocessor runs only on the golden fallback path, not in comparison mode.
    fallback_output = ttnn.get_fallback_function(ttnn.dequantize)(input_tensor, 0.5, 2)
    assert fallback_output.dtype == ttnn.bfloat16
    assert torch.equal(ttnn.to_torch(fallback_output), ttnn.to_torch(output))


@pytest.mark.requires_fast_runtime_mode_off
def test_pad_fallback_keeps_logical_shape(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)
    padding = ((0, 0), (0, 0), (0, 1), (0, 1))

    with comparison_mode():
        output = ttnn.pad(input_tensor, padding=padding, value=0.0)

    # generated/ttnn.pad.md: pad-fallback-postprocess-shape
    # The postprocessor runs only on the golden fallback path, not in comparison mode.
    fallback_output = ttnn.get_fallback_function(ttnn.pad)(input_tensor, padding=padding, value=0.0)
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

    # generated/ttnn.fold.md: fold-legacy-asymmetric-padding
    with comparison_mode():
        ttnn.fold(input_tensor, 2, 2, use_transpose_as_fold=True, padding=[2, 4, 2, 4, 0, 1], grid_size=grid_size)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_avg_pool2d_with_block_float_dtype_in_comparison_mode(device):
    batch_size, input_h, input_w, channels = 1, 8, 8, 32
    torch_input = _block_float_sensitive_values((1, 1, batch_size * input_h * input_w, channels))
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    pool_args = (input_tensor, batch_size, input_h, input_w, channels, [2, 2], [2, 2], [0, 0])
    pool_kwargs = {"dtype": ttnn.bfloat8_b, "output_layout": ttnn.TILE_LAYOUT}

    # generated/ttnn.avg_pool2d.md: avg-pool-output-dtype-ignored
    with comparison_mode():
        output = ttnn.avg_pool2d(*pool_args, **pool_kwargs)

    _assert_golden_matches_output(_registered_golden_output(ttnn.avg_pool2d, *pool_args, **pool_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_max_pool2d_with_block_float_dtype_in_comparison_mode(device):
    batch_size, input_h, input_w, channels = 1, 8, 8, 32
    torch_input = _block_float_sensitive_values((1, 1, batch_size * input_h * input_w, channels))
    input_tensor = _to_device(torch_input.to(torch.bfloat16), device, layout=ttnn.ROW_MAJOR_LAYOUT)
    pool_args = (input_tensor, batch_size, input_h, input_w, channels, [2, 2], [2, 2], [0, 0], [1, 1])
    pool_kwargs = {"dtype": ttnn.bfloat8_b, "output_layout": ttnn.TILE_LAYOUT}

    # generated/ttnn.max_pool2d.md: max-pool-dtype
    with comparison_mode():
        output = ttnn.max_pool2d(*pool_args, **pool_kwargs)

    _assert_golden_matches_output(_registered_golden_output(ttnn.max_pool2d, *pool_args, **pool_kwargs), output)


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

    # generated/ttnn.conv2d.md: missing-bias-none-handling
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

    # generated/ttnn.grid_sample.md: precomputed-grid-not-modeled
    with comparison_mode():
        ttnn.grid_sample(input_tensor, prepared_grid, use_precomputed_grid=True)


@pytest.mark.requires_fast_runtime_mode_off
def test_group_norm_without_optional_arguments_in_comparison_mode(device):
    if device.core_grid.y == 7:
        pytest.skip("the interleaved group_norm grid is not supported on this device")
    input_tensor = _to_device(
        torch.rand((1, 1, 256, 1024), dtype=torch.bfloat16), device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # generated/ttnn.group_norm.md: optional-group-norm-inputs-rejected
    with comparison_mode():
        ttnn.group_norm(input_tensor, num_groups=32, inplace=False)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation", [ttnn.scale_mask_softmax, ttnn.scale_mask_softmax_in_place], ids=["out_of_place", "in_place"]
)
def test_scale_mask_softmax_without_scale_in_comparison_mode(device, operation):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # generated/ttnn.scale_mask_softmax.md: scale-mask-optional-scale
    # generated/ttnn.scale_mask_softmax_in_place.md: inherits scale-mask-optional-scale
    with comparison_mode():
        operation(input_tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_rms_norm_pre_all_gather_with_dtype_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # generated/ttnn.rms_norm_pre_all_gather.md: rms-pre-stats-dtype-not-modeled
    with comparison_mode():
        output = ttnn.rms_norm_pre_all_gather(input_tensor, dtype=ttnn.bfloat16)

    golden = _registered_golden_output(ttnn.rms_norm_pre_all_gather, input_tensor, dtype=ttnn.bfloat16)
    torch_output = ttnn.to_torch(output)
    assert golden.dtype == torch_output.dtype, f"golden dtype {golden.dtype} != output dtype {torch_output.dtype}"
    torch.testing.assert_close(golden[..., 0].float(), torch_output[..., 0].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.requires_fast_runtime_mode_off
def test_rms_norm_post_all_gather_with_block_float_dtype_in_comparison_mode(device):
    input_tensor = _to_device(_block_float_sensitive_values(SINGLE_TILE).to(torch.bfloat16), device)
    stats = ttnn.rms_norm_pre_all_gather(input_tensor)

    # generated/ttnn.rms_norm_post_all_gather.md: rms-post-output-dtype-not-modeled
    with comparison_mode():
        output = ttnn.rms_norm_post_all_gather(input_tensor, stats, dtype=ttnn.bfloat8_b)

    golden = _registered_golden_output(ttnn.rms_norm_post_all_gather, input_tensor, stats, dtype=ttnn.bfloat8_b)
    torch.testing.assert_close(golden.float(), ttnn.to_torch(output).float(), rtol=1e-2, atol=1e-3)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "input_dtype, output_dtype, make_values",
    [
        pytest.param(ttnn.float32, ttnn.bfloat16, _bfloat16_sensitive_values, id="float32_to_bfloat16"),
        pytest.param(ttnn.bfloat16, ttnn.bfloat8_b, _block_float_sensitive_values, id="bfloat16_to_bfloat8_b"),
    ],
)
def test_matmul_with_output_dtype_in_comparison_mode(device, input_dtype, output_dtype, make_values):
    input_a = _to_device(make_values((32, 32)), device, dtype=input_dtype)
    input_b = _to_device(torch.eye(32), device, dtype=input_dtype)

    # generated/ttnn.matmul.md: matmul-output-dtype-not-modeled
    with comparison_mode():
        output = ttnn.matmul(input_a, input_b, dtype=output_dtype)

    _assert_golden_matches_output(_registered_golden_output(ttnn.matmul, input_a, input_b, dtype=output_dtype), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_linear_with_float32_dtype_in_comparison_mode(device):
    input_a = _to_device(torch.randint(-4, 5, (32, 32)).to(torch.bfloat16), device)
    input_b = _to_device(torch.eye(32, dtype=torch.bfloat16), device)

    # generated/ttnn.linear.md: linear-output-dtype-ignored
    with comparison_mode():
        output = ttnn.linear(input_a, input_b, dtype=ttnn.float32)

    _assert_golden_matches_output(_registered_golden_output(ttnn.linear, input_a, input_b, dtype=ttnn.float32), output)


@pytest.mark.requires_fast_runtime_mode_off
def test_addmm_with_block_float_dtype_in_comparison_mode(device):
    addend = _to_device(torch.zeros((32, 32), dtype=torch.bfloat16), device)
    mat1 = _to_device(_block_float_sensitive_values((32, 32)).to(torch.bfloat16), device)
    mat2 = _to_device(torch.eye(32, dtype=torch.bfloat16), device)

    # generated/ttnn.addmm.md: addmm-output-dtype-ignored
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
        # generated/ttnn.matmul_batched_weights.md: batched-weights-arguments-not-modeled
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


@pytest.mark.requires_fast_runtime_mode_off
def test_sparse_matmul_indexed_output_in_comparison_mode(device):
    m, k, n, num_experts = 32, 128, 192, 8
    active_ids = [5, 1]
    sparsity_values = torch.zeros((1, 1, 1, num_experts), dtype=torch.bfloat16)
    sparsity_values[..., active_ids] = 1
    input_a = _to_device(torch.randn((1, 1, m, k), dtype=torch.bfloat16), device)
    input_b = _to_device(torch.randn((1, num_experts, k, n), dtype=torch.bfloat16), device)
    sparsity = _to_device(sparsity_values, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    indices = _to_device(
        torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, -1),
        device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # generated/ttnn.sparse_matmul.md: sparse-matmul-indexed-output-unsupported
    with comparison_mode():
        ttnn.sparse_matmul(
            input_a,
            input_b,
            sparsity=sparsity,
            indices=indices,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=_sparse_matmul_program_config(),
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

    # generated/ttnn.sparse_matmul.md: sparse-matmul-compact-output-unsupported
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

    # generated/ttnn.moreh_softmax.md: forward-op-dispatch-ignored
    # generated/ttnn.moreh_softmin.md: inherits forward-op-dispatch-ignored from ttnn.moreh_softmax
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

    # generated/ttnn.moreh_softmax_backward.md: backward-op-dispatch-ignored
    # generated/ttnn.moreh_softmin_backward.md: inherits backward-op-dispatch-ignored
    with comparison_mode():
        operation(output_tensor, output_grad_tensor, 1, op=op)


@pytest.mark.requires_fast_runtime_mode_off
def test_moreh_mean_over_all_dims_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand((2, 3, 32, 64), dtype=torch.bfloat16), device)

    # generated/ttnn.moreh_mean.md: mean-reduction-shape-comparison
    with comparison_mode():
        ttnn.moreh_mean(input_tensor, dim=None, keepdim=False)


@pytest.mark.requires_fast_runtime_mode_off
def test_moreh_norm_of_rank_1_input_in_comparison_mode(device):
    input_tensor = _to_device(torch.empty([5]).uniform_(-1, 1).to(torch.bfloat16), device)

    # generated/ttnn.moreh_norm.md: rank1-reduction-shape
    with comparison_mode():
        ttnn.moreh_norm(input_tensor, 2.0, dim=0, keepdim=False)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("dim", [None, 0], ids=["all_dims", "dim_0"])
def test_moreh_sum_of_rank_1_input_in_comparison_mode(device, dim):
    input_tensor = _to_device(torch.empty([5]).uniform_(-1, 1).to(torch.bfloat16), device)

    # generated/ttnn.moreh_sum.md: rank1-reduction-shape
    with comparison_mode():
        ttnn.moreh_sum(input_tensor, dim, keepdim=False)


@pytest.mark.requires_fast_runtime_mode_off
def test_topk_default_dim_in_comparison_mode(device):
    input_tensor = _to_device(torch.randn((1, 1, 32, 64), dtype=torch.bfloat16), device)

    # generated/ttnn.topk.md: topk-default-dim
    with comparison_mode():
        ttnn.topk(input_tensor, 4)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("variant", ["indices_tensor", "stable"])
def test_topk_labels_and_stable_ties_in_comparison_mode(device, variant):
    shape = (1, 1, 32, 64)
    if variant == "indices_tensor":
        torch_input = torch.randn(shape, dtype=torch.bfloat16)
        labels = torch.arange(shape[-1] - 1, -1, -1, dtype=torch.int32).expand(shape).contiguous()
        topk_kwargs = {"indices_tensor": _to_device(labels, device, dtype=ttnn.uint16)}
    else:
        torch_input = torch.zeros(shape, dtype=torch.bfloat16)
        topk_kwargs = {"stable": True}
    input_tensor = _to_device(torch_input, device)

    # generated/ttnn.topk.md: topk-labels-and-stability
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

    # generated/ttnn.transformer.attention_softmax.md: attention-softmax-causal-mask
    # generated/ttnn.transformer.attention_softmax_.md: inherits attention-softmax-causal-mask
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

    # generated/ttnn.transformer.scaled_dot_product_attention_decode.md: decode-noncausal-cur-pos-truncation
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


@pytest.mark.requires_fast_runtime_mode_off
def test_ldexp_inplace_updates_global_golden(device, tmp_path):
    # generated/ttnn.ldexp_.md: inplace-mutation-not-modeled
    with _global_comparison_mode(tmp_path):
        input_a = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, device)
        input_b = _to_device(torch.randint(-2, 3, SINGLE_TILE).to(torch.bfloat16), device)
        ttnn.ldexp_(input_a, input_b)
        ttnn.to_torch(input_a)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("operation", [ttnn.logaddexp_, ttnn.logaddexp2_], ids=["logaddexp_", "logaddexp2_"])
def test_logaddexp_inplace_updates_global_golden(device, tmp_path, operation):
    # generated/ttnn.logaddexp.md: logaddexp-family-inplace-global-golden
    # generated/ttnn.logaddexp_.md, ttnn.logaddexp2_.md: inherit logaddexp-family-inplace-global-golden
    with _global_comparison_mode(tmp_path):
        input_a = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) * 4 - 2, device)
        input_b = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) * 4 - 2, device)
        operation(input_a, input_b)
        ttnn.to_torch(input_a)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation", [ttnn.logical_and_, ttnn.logical_or_, ttnn.logical_xor_], ids=["and", "or", "xor"]
)
def test_logical_binary_inplace_updates_global_golden(device, tmp_path, operation):
    torch_a = torch.zeros(SINGLE_TILE, dtype=torch.bfloat16)
    torch_a[..., ::2] = 1
    torch_b = torch.zeros(SINGLE_TILE, dtype=torch.bfloat16)
    torch_b[..., ::3] = 1

    # generated/ttnn.logical_and_.md: logical-inplace-global-state
    # generated/ttnn.logical_or_.md, ttnn.logical_xor_.md: inherit logical-inplace-global-state
    with _global_comparison_mode(tmp_path):
        input_a = _to_device(torch_a, device)
        input_b = _to_device(torch_b, device)
        operation(input_a, input_b)
        ttnn.to_torch(input_a)


@pytest.mark.requires_fast_runtime_mode_off
def test_logical_not_inplace_updates_global_golden(device, tmp_path):
    torch_input = torch.zeros(SINGLE_TILE, dtype=torch.bfloat16)
    torch_input[..., ::2] = 1

    # generated/ttnn.logical_not_.md: logical-not-inplace-global-state
    with _global_comparison_mode(tmp_path):
        input_tensor = _to_device(torch_input, device)
        ttnn.logical_not_(input_tensor)
        ttnn.to_torch(input_tensor)


@pytest.mark.requires_fast_runtime_mode_off
def test_normalize_hw_keeps_global_golden_input(device, tmp_path):
    # Channels with different ranges keep the normalized values from correlating with the original input.
    torch_input = torch.cat(
        [torch.rand(SINGLE_TILE, dtype=torch.bfloat16), torch.rand(SINGLE_TILE, dtype=torch.bfloat16) * 10 + 50], dim=1
    )

    # generated/ttnn.normalize_hw.md: normalize-hw-mutates-global-golden-input
    with _global_comparison_mode(tmp_path):
        input_tensor = _to_device(torch_input, device)
        ttnn.normalize_hw(input_tensor)
        ttnn.to_torch(input_tensor)


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

    # generated/ttnn.group_norm.md: group-norm-inplace-state-not-preserved
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
