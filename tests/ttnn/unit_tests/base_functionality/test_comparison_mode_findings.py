# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.base_functionality.test_comparison_mode_context import (
    SINGLE_TILE,
    _assert_golden_matches_output,
    _block_float_sensitive_values,
    _complex_input,
    _registered_golden_output,
    _sparse_matmul_program_config,
    _to_device,
    comparison_mode,
)


def _assert_golden_dtype_and_close(golden, output, *, rtol, atol):
    torch_output = ttnn.to_torch(output)
    assert (
        golden.shape == torch_output.shape
    ), f"golden shape {tuple(golden.shape)} != output {tuple(torch_output.shape)}"
    assert golden.dtype == torch_output.dtype, f"golden dtype {golden.dtype} != output dtype {torch_output.dtype}"
    torch.testing.assert_close(golden.float(), torch_output.float(), rtol=rtol, atol=atol)


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
def test_tilize_with_val_padding_with_block_float_dtype_in_comparison_mode(device):
    input_tensor = _to_device(
        _block_float_sensitive_values(SINGLE_TILE), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    tilize_args = (input_tensor, list(SINGLE_TILE), 0.0)

    # The device stores the tiles in dtype, and BFLOAT8_B flushes the 1s beside 1024 to zero, so the golden must
    # quantize instead of returning the float32 input unchanged.
    with comparison_mode():
        output = ttnn.tilize_with_val_padding(*tilize_args, dtype=ttnn.bfloat8_b)

    _assert_golden_matches_output(
        _registered_golden_output(ttnn.tilize_with_val_padding, *tilize_args, dtype=ttnn.bfloat8_b), output
    )


@pytest.mark.requires_fast_runtime_mode_off
def test_tilize_with_zero_padding_with_block_float_output_dtype_in_comparison_mode(device):
    input_tensor = _to_device(
        _block_float_sensitive_values(SINGLE_TILE), device, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    # tilize_with_zero_padding names its dtype argument output_dtype; BFLOAT8_B flushes the 1s beside 1024 to zero,
    # so the golden must quantize instead of returning the float32 input unchanged.
    with comparison_mode():
        output = ttnn.tilize_with_zero_padding(input_tensor, output_dtype=ttnn.bfloat8_b)

    _assert_golden_matches_output(
        _registered_golden_output(ttnn.tilize_with_zero_padding, input_tensor, output_dtype=ttnn.bfloat8_b), output
    )


@pytest.mark.requires_fast_runtime_mode_off
def test_layer_norm_pre_all_gather_default_dtype_in_comparison_mode(device):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16), device)

    # Without dtype the device stores the statistics as BFLOAT16, as rms_norm_pre_all_gather does, so the golden
    # must return BFLOAT16 statistics rather than float32.
    with comparison_mode():
        output = ttnn.layer_norm_pre_all_gather(input_tensor)

    golden = _registered_golden_output(ttnn.layer_norm_pre_all_gather, input_tensor)
    torch_output = ttnn.to_torch(output)
    assert golden.dtype == torch_output.dtype, f"golden dtype {golden.dtype} != output dtype {torch_output.dtype}"
    for stat_column in (0, 32):
        torch.testing.assert_close(
            golden[..., stat_column].float(), torch_output[..., stat_column].float(), rtol=1e-2, atol=1e-2
        )


@pytest.mark.requires_fast_runtime_mode_off
def test_layer_norm_post_all_gather_with_block_float_dtype_in_comparison_mode(device):
    input_tensor = _to_device(_block_float_sensitive_values(SINGLE_TILE).to(torch.bfloat16), device)
    stats = ttnn.layer_norm_pre_all_gather(input_tensor)

    # The 1s normalize to about -0.258 beside 1024's 3.87; the BFLOAT8_B output shares one exponent per 16 values
    # and stores them as -0.25, so the golden must quantize the normalized result to dtype.
    with comparison_mode():
        output = ttnn.layer_norm_post_all_gather(input_tensor, stats, dtype=ttnn.bfloat8_b)

    golden = _registered_golden_output(ttnn.layer_norm_post_all_gather, input_tensor, stats, dtype=ttnn.bfloat8_b)
    torch.testing.assert_close(golden.float(), ttnn.to_torch(output).float(), rtol=1e-2, atol=1e-3)


@pytest.mark.requires_fast_runtime_mode_off
def test_sparse_matmul_with_block_float_dtype_in_comparison_mode(device):
    m, k, n, num_experts = 32, 128, 192, 8
    active_ids = [5, 1]
    sparsity_values = torch.zeros((1, 1, 1, num_experts), dtype=torch.bfloat16)
    sparsity_values[..., active_ids] = 1
    input_a = _to_device(_block_float_sensitive_values((1, 1, m, k)).to(torch.bfloat16), device)
    input_b = _to_device(torch.eye(k, n, dtype=torch.bfloat16).expand(1, num_experts, k, n).contiguous(), device)
    sparsity = _to_device(sparsity_values, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    indices = _to_device(
        torch.tensor(active_ids, dtype=torch.int32).reshape(1, 1, 1, -1),
        device,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    sparse_matmul_kwargs = {
        "sparsity": sparsity,
        "indices": indices,
        "is_input_a_sparse": False,
        "is_input_b_sparse": True,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "program_config": _sparse_matmul_program_config(),
        "dtype": ttnn.bfloat8_b,
    }

    # Every expert of B is an identity, so each gathered result repeats A's 1s-beside-1024 rows; the BFLOAT8_B
    # output flushes those 1s to zero, so the golden must quantize to dtype.
    with comparison_mode():
        output = ttnn.sparse_matmul(input_a, input_b, **sparse_matmul_kwargs)

    _assert_golden_matches_output(
        _registered_golden_output(ttnn.sparse_matmul, input_a, input_b, **sparse_matmul_kwargs), output
    )


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "minimum_kwargs",
    [
        pytest.param({"input_tensor_a_activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)]}, id="operand"),
        pytest.param({"activations": [ttnn.UnaryWithParam(ttnn.UnaryOpType.NEG)]}, id="result"),
    ],
)
def test_minimum_with_scalar_and_activations_in_comparison_mode(device, minimum_kwargs):
    input_tensor = _to_device(torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1, device)

    # The scalar overload runs on the unary path, which accepts activations but never applies them, so the golden
    # must ignore them too; a negating activation would otherwise change the minimum of values in [1, 2) and 0.5.
    with comparison_mode():
        output = ttnn.minimum(input_tensor, 0.5, **minimum_kwargs)

    _assert_golden_matches_output(_registered_golden_output(ttnn.minimum, input_tensor, 0.5, **minimum_kwargs), output)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation",
    [ttnn.imag, ttnn.angle, ttnn.conj, ttnn.is_real, ttnn.is_imag, ttnn.reciprocal],
    ids=["imag", "angle", "conj", "is_real", "is_imag", "reciprocal"],
)
def test_complex_unary_op_in_comparison_mode(device, operation):
    torch_real = torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1
    torch_real[..., ::2] = 0
    torch_imag = torch.rand(SINGLE_TILE, dtype=torch.bfloat16) + 1
    torch_imag[..., 1::4] = 0
    complex_input = _complex_input(torch_real, torch_imag, device)

    # A ComplexTensor is two real device tensors rather than a ttnn.Tensor, so the golden must receive it rebuilt as
    # a torch complex tensor. Zero real and zero imaginary parts never coincide, so every value is finite.
    with comparison_mode():
        operation(complex_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize(
    "operation, takes_identity_operand",
    [
        pytest.param(ttnn.matmul, True, id="matmul"),
        pytest.param(ttnn.linear, True, id="linear"),
        pytest.param(ttnn.clone, False, id="clone"),
    ],
)
def test_fallback_keeps_requested_dtype(device, operation, takes_identity_operand):
    input_tensor = _to_device(torch.randint(-4, 5, (32, 32)).to(torch.bfloat16), device)
    operands = (input_tensor,)
    if takes_identity_operand:
        operands += (_to_device(torch.eye(32, dtype=torch.bfloat16), device),)

    with comparison_mode():
        output = operation(*operands, dtype=ttnn.float32)

    # The golden returns the requested FLOAT32, but the fallback's output postprocessing casts results back to the
    # first input's dtype. That postprocessor runs only on the golden fallback path, not in comparison mode.
    fallback_output = ttnn.get_fallback_function(operation)(*operands, dtype=ttnn.float32)
    assert fallback_output.dtype == output.dtype
    assert torch.equal(ttnn.to_torch(fallback_output), ttnn.to_torch(output))
