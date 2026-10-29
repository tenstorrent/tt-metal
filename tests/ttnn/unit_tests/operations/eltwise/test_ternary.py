# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from math import isnan
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, assert_equal


def test_where_golden_treats_negative_predicates_as_true():
    predicate = torch.tensor([-2, 0, 3, -1], dtype=torch.int32)
    true_values = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    false_values = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    golden_function = ttnn.get_golden_function(ttnn.where)

    output = golden_function(predicate, true_values, false_values)

    assert torch.equal(output, torch.tensor([10, 2, 30, 40], dtype=torch.int32))


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("scalar1", [5.5])
@pytest.mark.parametrize("scalar2", [-13.25])
def test_mac_tensor_with_2_scalaras(device, h, w, scalar1, scalar2):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = scalar1
    torch_input_tensor2 = scalar2

    golden_fn = ttnn.get_golden_function(ttnn.mac)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.mac(input_tensor, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=2)


def assert_where_exact(torch_input_tensor, torch_input1, torch_input2, device):
    """Run ``ttnn.where`` and assert bit-exact equality with the torch golden.

    ``where`` is a per-element selection between the two input branches with no arithmetic, so the
    output must match torch exactly.
    """

    def from_torch_if_tensor(x):
        if not isinstance(x, torch.Tensor):
            return x

        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor, input1, input2 = (
        from_torch_if_tensor(arg)
        for arg in ((torch_input_tensor > 0).to(torch_input_tensor.dtype), torch_input1, torch_input2)
    )
    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(torch_input_tensor > 0, torch_input1, torch_input2)
    output_tensor = ttnn.where(input_tensor, input1, input2)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "hc, ht, hf, wc, wt, wf",
    [
        [64, 64, 64, 128, 128, 1],
        [64, 64, 64, 128, 1, 128],
        [64, 64, 64, 1, 128, 128],
        [64, 64, 1, 128, 128, 128],
        [64, 1, 64, 128, 128, 128],
        [1, 64, 64, 128, 128, 128],
        [64, 64, 1, 128, 128, 1],
        [64, 1, 64, 128, 1, 128],
        [64, 1, 64, 128, 128, 1],
        [64, 64, 1, 128, 1, 128],
        [1, 1, 64, 128, 128, 1],
        [64, 1, 64, 1, 1, 128],
    ],
)
def test_where_bcast(device, dtype, hc, ht, hf, wc, wt, wf):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hc, wc), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor1 = torch.rand((ht, wt), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor2 = torch.rand((hf, wf), dtype=dtype).uniform_(-100, 100)

    assert_where_exact(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, device)


@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("value", [15.5, 0, 1.0, -5.0])
@pytest.mark.parametrize(
    "hc, ht, hf, wc, wt, wf",
    [
        [64, 64, 64, 128, 128, 128],  # no bcast
        # Row / Col bcast cases
        [64, 64, 64, 128, 128, 1],
        [64, 64, 64, 128, 1, 128],
        [64, 64, 64, 1, 128, 128],
        [64, 64, 1, 128, 128, 128],
        [64, 1, 64, 128, 128, 128],
        [1, 64, 64, 128, 128, 128],
        [64, 64, 1, 128, 128, 1],
        [64, 1, 64, 128, 1, 128],
        [1, 64, 64, 1, 128, 128],
        [64, 1, 1, 128, 1, 1],  # scalar bcast case
    ],
)
def test_addcmul_with_bcast(device, tor_dtype, ttnn_dtype, hc, ht, hf, wc, wt, wf, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hc, wc), dtype=tor_dtype).uniform_(-100, 500)
    torch_input_tensor1 = torch.rand((ht, wt), dtype=tor_dtype).uniform_(-200, 200)
    torch_input_tensor2 = torch.rand((hf, wf), dtype=tor_dtype).uniform_(-300, 400)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn.addcmul(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.0,
        -0.5,
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    [
        ((1, 2, 1088, 1024), (1, 2, 1, 1024), (1, 2, 1088, 1024)),  # Composite
        ((1, 2, 1088, 1024), (1, 2, 1, 1), (1, 2, 1088, 1024)),  # Composite
        ((4, 2, 1088, 1024), (1, 2, 1088, 1024), (1, 1, 1088, 1024)),  # HLK
    ],
)
def test_addcmul_with_bcast_bf8b(device, torch_dtype, ttnn_dtype, a_shape, b_shape, c_shape, value):
    """
    Test addcmul: Block format datatype inputs with subtile broadcast use composite Addcmul implementation.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(a_shape, dtype=torch_dtype)
    torch_input_tensor1 = torch.randn(b_shape, dtype=torch_dtype)
    torch_input_tensor2 = torch.randn(c_shape, dtype=torch_dtype)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("value", [0.5, 3])
@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)),
        ((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1)),
        ((1, 1, 1, 1), (1, 1, 1024, 1024), (1, 1, 1, 1)),
    ],
)
def test_addcmul(device, torch_dtype, ttnn_dtype, value, in_data1_shape, in_data2_shape, in_data3_shape):
    in_data1 = torch.full(in_data1_shape, 0.0031, dtype=torch_dtype)
    in_data2 = torch.full(in_data2_shape, 508.0, dtype=torch_dtype)
    in_data3 = torch.full(in_data3_shape, 748.0, dtype=torch_dtype)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    assert_with_ulp(expected_result=golden_tensor, actual_result=output_tensor)


@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)),
        ((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1)),
        ((1, 1, 1, 1), (1, 1, 1024, 1024), (1, 1, 1, 1)),
    ],
)
def test_addcmul_with_int32_inputs(device, in_data1_shape, in_data2_shape, in_data3_shape):
    in_data1 = torch.randint(-100, 100, in_data1_shape, dtype=torch.int32)
    in_data2 = torch.randint(-100, 100, in_data2_shape, dtype=torch.int32)
    in_data3 = torch.randint(-100, 100, in_data3_shape, dtype=torch.int32)
    value = 6
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    assert_equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1), (1, 1, 32, 32), (1, 1, 1, 1)),
    ],
)
@pytest.mark.parametrize("value", [6, -6.0, float(2**31 + 7), 2**32 - 1])
def test_addcmul_with_uint32_inputs(device, in_data1_shape, in_data2_shape, in_data3_shape, value):
    # Use values spanning the full uint32 range. Store them as int32 bit-patterns;
    # TTNN interprets those bit-patterns as uint32, and the golden is compared via 32-bit wraparound.
    in_data1 = torch.randint(0, 2**32, in_data1_shape, dtype=torch.int64).to(torch.int32)
    in_data2 = torch.randint(0, 2**16, in_data2_shape, dtype=torch.int64).to(torch.int32)
    in_data3 = torch.randint(0, 2**16, in_data3_shape, dtype=torch.int64).to(torch.int32)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    # Float values are narrowed to float32 by the C++ binding, so apply the same rounding
    # to the golden to keep both sides in sync (e.g. float(2**31+7) → 2147483648.0 in float32).
    golden_value = float(torch.tensor(value, dtype=torch.float32).item()) if isinstance(value, float) else value

    # Promote to int64 before calling the golden function to prevent intermediate overflow,
    # then truncate back to int32 to match the 32-bit wraparound semantics of uint32 hardware.
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(
        in_data1.to(torch.int64), in_data2.to(torch.int64), in_data3.to(torch.int64), value=golden_value
    ).to(torch.int32)

    assert_equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize("value", [0.5, 1.0, 7.7, 22.42, 107.6])
@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)),
        ((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1)),
        ((1, 1, 1, 1), (1, 1, 1024, 1024), (1, 1, 1, 1)),
    ],
)
def test_addcdiv(device, torch_dtype, ttnn_dtype, value, in_data1_shape, in_data2_shape, in_data3_shape):
    torch.manual_seed(10)
    in_data1 = torch.empty(in_data1_shape, dtype=torch_dtype).uniform_(-100, 100)
    in_data2 = torch.empty(in_data2_shape, dtype=torch_dtype).uniform_(-100, 100)
    in_data3 = torch.empty(in_data3_shape, dtype=torch_dtype).uniform_(-100, 100)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcdiv(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.addcdiv)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    # Normalize: if input3 is zero and golden is nan and ttnn output is inf, change ttnn output to nan
    zero_mask = in_data3 == 0
    golden_nan_mask = torch.isnan(golden_tensor)
    output_inf_mask = torch.isinf(output_tensor)
    normalize_mask = zero_mask & golden_nan_mask & output_inf_mask
    if normalize_mask.any():
        output_tensor = torch.where(
            normalize_mask,
            torch.tensor(float("nan"), dtype=output_tensor.dtype, device=output_tensor.device),
            output_tensor,
        )

    assert_with_ulp(expected_result=golden_tensor, actual_result=output_tensor, ulp_threshold=1, allow_nonfinite=True)


_MIXED_INPUT_CASES = [
    (ttnn.bfloat16, ttnn.float32, None),
    (ttnn.float32, ttnn.bfloat16, None),
]
_MIXED_OUTPUT_CASES = [
    (ttnn.float32, ttnn.float32, ttnn.bfloat16),
    (ttnn.bfloat16, ttnn.bfloat16, ttnn.float32),
]
_TO_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}


def _mixed_dtype_case_id(a_dtype, bc_dtype, out_dtype):
    name = {ttnn.bfloat16: "bf16", ttnn.float32: "fp32"}
    return f"a_{name[a_dtype]}-bc_{name[bc_dtype]}-out_{name[out_dtype or a_dtype]}"


def _assert_ternary_close(expected, result, out_dtype):
    if out_dtype == ttnn.float32:
        assert torch.allclose(result, expected, rtol=1e-5, atol=1e-5)
    else:
        assert_with_ulp(expected_result=expected.to(torch.bfloat16), actual_result=result, ulp_threshold=1)


@pytest.mark.parametrize(
    "op, a_dtype, bc_dtype, out_dtype",
    [
        pytest.param(op, *case, id=f"{op}-{_mixed_dtype_case_id(*case)}")
        for op in ["lerp", "addcdiv", "mac", "addcmul", "snake_beta"]
        # addcmul and snake_beta require all inputs to share a dtype.
        for case in (
            _MIXED_OUTPUT_CASES if op in ("addcmul", "snake_beta") else _MIXED_INPUT_CASES + _MIXED_OUTPUT_CASES
        )
    ],
)
@pytest.mark.parametrize("bc_shape", [(1, 1, 64, 128), (1, 1, 1, 128)], ids=["no_bcast", "row_bcast"])
def test_ternary_mixed_dtypes(device, op, a_dtype, bc_dtype, out_dtype, bc_shape):
    # Regression for #60223: fp32 operands or output require an fp32 DEST and per-CB unpack formats.
    torch.manual_seed(0)
    a = torch.randn((1, 1, 64, 128), dtype=_TO_TORCH_DTYPE[a_dtype])
    b = torch.randn(bc_shape, dtype=_TO_TORCH_DTYPE[bc_dtype])
    c = (torch.rand(bc_shape) + 0.5).to(_TO_TORCH_DTYPE[bc_dtype])
    tt_a, tt_b, tt_c = (
        ttnn.from_torch(x, dtype=d, layout=ttnn.TILE_LAYOUT, device=device)
        for x, d in ((a, a_dtype), (b, bc_dtype), (c, bc_dtype))
    )
    output_tensor = None
    if out_dtype is not None:
        output_tensor = ttnn.from_torch(torch.zeros(a.shape), dtype=out_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    A, B, C = a.float(), b.float(), c.float()
    if op == "lerp":
        output = ttnn.lerp(tt_a, tt_b, tt_c, output_tensor=output_tensor)
        expected = A + C * (B - A)
    elif op == "addcdiv":
        output = ttnn.addcdiv(tt_a, tt_b, tt_c, value=0.5, output_tensor=output_tensor)
        expected = A + 0.5 * B / C
    elif op == "addcmul":
        output = ttnn.addcmul(tt_a, tt_b, tt_c, value=0.5, output_tensor=output_tensor)
        expected = A + 0.5 * B * C
    elif op == "snake_beta":
        output = ttnn.snake_beta(tt_a, tt_b, tt_c, output_tensor=output_tensor)
        expected = A + torch.sin(B * A) ** 2 / C
    else:
        output = ttnn.mac(tt_a, tt_b, tt_c, output_tensor=output_tensor)
        expected = A * B + C

    out_dtype = out_dtype or a_dtype
    assert output.dtype == out_dtype
    _assert_ternary_close(expected, ttnn.to_torch(output), out_dtype)


@pytest.mark.parametrize(
    "op",
    ["lerp_tts", "mac_tts", "mac_tst"],
)
@pytest.mark.parametrize(
    "a_dtype, t_dtype",
    [(ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.bfloat16)],
    ids=["a_bf16-t_fp32", "a_fp32-t_bf16"],
)
@pytest.mark.parametrize(
    "t_shape",
    [(1, 1, 64, 128), (1, 1, 64, 1), (1, 1, 1, 1), (1, 1, 1, 128)],
    ids=["no_bcast", "col_bcast", "scalar_bcast", "row_bcast"],
)
def test_ternary_scalar_mixed_dtypes(device, op, a_dtype, t_dtype, t_shape):
    # Tensor-scalar variants share the predicate/tensor CB pair, which can hold different dtypes (#60223).
    torch.manual_seed(0)
    a = torch.randn((1, 1, 64, 128), dtype=_TO_TORCH_DTYPE[a_dtype])
    t = torch.randn(t_shape, dtype=_TO_TORCH_DTYPE[t_dtype])
    tt_a = ttnn.from_torch(a, dtype=a_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_t = ttnn.from_torch(t, dtype=t_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    A, T = a.float(), t.float()
    if op == "lerp_tts":
        output = ttnn.lerp(tt_a, tt_t, 0.25)
        expected = A + 0.25 * (T - A)
    elif op == "mac_tts":
        output = ttnn.mac(tt_a, tt_t, 1.5)
        expected = A * T + 1.5
    else:
        output = ttnn.mac(tt_a, 1.5, tt_t)
        expected = A * 1.5 + T

    assert output.dtype == a_dtype
    _assert_ternary_close(expected, ttnn.to_torch(output), a_dtype)


def test_ternary_scalar_distinguishes_cache_entries(device):
    """Regression guard for the ternary scalar static/dynamic contract.

    addcmul(a, b, c, value) packs `value` (scalar_input_a) into the compute-kernel runtime args.
    The scalar is intentionally EXCLUDED from to_hash()/compute_program_hash (so calls differing
    only in the scalar cache-hit instead of recompiling) but must be re-applied to the cached
    program on every dispatch via TernaryDeviceOperation::get_dynamic_runtime_args(). Pins both
    halves of the contract:
      * a different scalar must NOT grow the cache  -> guards against re-adding the scalar to the
        hash (the recompile-per-scalar hack).
      * a different scalar must change the output   -> guards against the frozen-runtime-arg bug,
        where the cache hit reuses the first call's baked-in scalar (only correctly re-patched now
        that the factory registers buffer bindings and declares the scalar dynamic).
      * the same scalar must reproduce the output   -> guards against wrongly perturbing the scalar
        on the fast path.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    shape = (256, 256)
    torch.manual_seed(0)
    a = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    b = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    c = ttnn.from_torch(
        torch.rand(shape, dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    out_v1 = ttnn.to_torch(ttnn.addcmul(a, b, c, value=2.0)).float().clone()
    entries_after_first = device.num_program_cache_entries()
    assert entries_after_first == 1, f"Expected 1 cache entry after first addcmul, got {entries_after_first}"

    # Same scalar: must reuse the cached program AND reproduce the exact values.
    out_v1_again = ttnn.to_torch(ttnn.addcmul(a, b, c, value=2.0)).float().clone()
    assert device.num_program_cache_entries() == 1, "same scalar must reuse the cached program (no new cache entry)"
    assert torch.equal(out_v1, out_v1_again), "same scalar must reproduce identical output"

    # Different scalar: scalar is excluded from the hash, so it must STILL be a cache hit (no new
    # entry) and yet produce different values (scalar re-patched on the cache hit).
    out_v2 = ttnn.to_torch(ttnn.addcmul(a, b, c, value=7.0)).float().clone()
    assert device.num_program_cache_entries() == 1, (
        "a different scalar must NOT create a new cache entry -- the scalar is dynamic, not part of "
        "the program hash. A new entry here means the scalar was (re-)added to compute_program_hash."
    )
    assert not torch.equal(out_v1, out_v2), (
        "a different scalar must change the output (scalar re-patched on the fast path). Identical "
        "output here means the cache hit reused the first call's frozen scalar runtime arg."
    )

    device.disable_and_clear_program_cache()


def test_ternary_addcmul_cache_hit_refreshes_operand_addresses(device):
    """Regression guard for the ternary buffer-address static/dynamic contract (cache-hit bug).

    Two same-shaped addcmul calls reuse the same cached program (correct), but the cache-hit path
    must re-patch the runtime args holding the operand/output buffer ADDRESSES. Before the fix, the
    second same-shaped call whose operand lived in a DIFFERENT buffer ran against the first call's
    stale addresses and returned silent garbage (PCC ~0). Here the operands are distinct
    ttnn.chunk() slices of a single parent tensor, so they are the same shape but different buffers.
    TernaryDeviceOperation::get_dynamic_runtime_args() must refresh reader slots 0/1/2 and writer
    slot 0 on every dispatch.
    """
    device.enable_program_cache()
    device.clear_program_cache()

    torch.manual_seed(0)
    N, H = 32, 64

    base_t = torch.rand((1, 1, N, H), dtype=torch.bfloat16)
    chunks_t = torch.rand((1, 1, 1, 4 * H), dtype=torch.bfloat16)

    base = ttnn.from_torch(base_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    chunks = ttnn.from_torch(chunks_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    chunk = ttnn.chunk(chunks, 4, -1)  # 4 shape-identical operands in different buffers

    def reference(i):
        return base_t + base_t * chunks_t[..., i * H : (i + 1) * H]

    # Pre-allocate BOTH outputs and keep them alive across both calls, passing each via
    # output_tensor. This forces the two dispatches to write to DISTINCT, simultaneously-live output
    # buffers, so the second call's writer slot 0 must be re-patched to out1's address -- otherwise it
    # writes into out0 and out1 stays zero. Without two live outputs the allocator could hand the
    # second call out0's just-freed address, masking a missing writer-address patch.
    out0 = ttnn.from_torch(
        torch.zeros((1, 1, N, H), dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    out1 = ttnn.from_torch(
        torch.zeros((1, 1, N, H), dtype=torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    # First call compiles the addcmul program.
    ttnn.addcmul(base, base, chunk[0], value=1.0, output_tensor=out0)
    entries_after_first = device.num_program_cache_entries()

    # Cache hit with a different-buffer operand AND a different output buffer of identical shape: the
    # addcmul program must be reused (no new cache entry), yet the stale-address regression made this
    # return ~0 PCC.
    ttnn.addcmul(base, base, chunk[2], value=1.0, output_tensor=out1)
    assert (
        device.num_program_cache_entries() == entries_after_first
    ), "same-shaped addcmul must reuse the cached program (no new cache entry)"

    assert_with_pcc(reference(0), ttnn.to_torch(out0).float(), 0.99)
    assert_with_pcc(reference(2), ttnn.to_torch(out1).float(), 0.99)

    device.disable_and_clear_program_cache()


def test_ternary_cache_miss_different_alignment(device):
    """Same logical shape, over-padded TILE vs default padding, must be two cache entries."""
    device.enable_program_cache()

    logical = (1, 1, 32, 32)
    padded = [1, 1, 64, 32]
    torch.manual_seed(0)

    def make(overpad):
        tensors = []
        for _ in range(3):
            host = torch.rand(logical, dtype=torch.bfloat16)
            if overpad:
                tt = ttnn.tilize_with_val_padding(
                    ttnn.from_torch(host, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16),
                    padded,
                    0.0,
                )
            else:
                tt = ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tensors.append(tt)
        return tensors

    over = make(True)
    normal = make(False)
    assert list(over[0].shape) == list(normal[0].shape)
    assert list(over[0].padded_shape) != list(normal[0].padded_shape)

    device.clear_program_cache()
    ttnn.addcmul(*over, value=1.0)
    assert device.num_program_cache_entries() == 1
    ttnn.addcmul(*normal, value=1.0)
    assert device.num_program_cache_entries() == 2

    device.disable_and_clear_program_cache()


def _height_sharded(device, shard_shape=(32, 64)):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR),
    )


ROW_MAJOR_REJECTION_CASES = [
    ("rm sharded predicate, rm interleaved operands", ttnn.ROW_MAJOR_LAYOUT, "mixed", None, None),
    ("tile operands, rm sharded output", ttnn.TILE_LAYOUT, "interleaved", ttnn.ROW_MAJOR_LAYOUT, None),
    ("rm sharded operands, tile sharded output", ttnn.ROW_MAJOR_LAYOUT, "sharded", ttnn.TILE_LAYOUT, None),
    ("rm sharded operands, interleaved memory_config", ttnn.ROW_MAJOR_LAYOUT, "sharded", None, "dram"),
    ("rm sharded operands, computed output", ttnn.ROW_MAJOR_LAYOUT, "sharded", None, None),
]


@pytest.mark.parametrize(
    "label, operand_layout, operand_memory, out_layout, memory_config_kwarg",
    ROW_MAJOR_REJECTION_CASES,
    ids=[c[0] for c in ROW_MAJOR_REJECTION_CASES],
)
def test_ternary_rejects_row_major(
    device, expect_error, label, operand_layout, operand_memory, out_layout, memory_config_kwarg
):
    """ttnn.where must reject every row-major configuration in any slot."""
    shape = (1, 1, 64, 64)
    torch.manual_seed(0)
    sharded = _height_sharded(device)

    def make(host, layout, memory_config):
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=memory_config)

    pred = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    operands = [
        pred.to(torch.bfloat16),
        torch.rand(shape, dtype=torch.bfloat16),
        torch.rand(shape, dtype=torch.bfloat16),
    ]

    if operand_memory == "mixed":
        memories = [sharded, ttnn.L1_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]
    elif operand_memory == "sharded":
        memories = [sharded] * 3
    else:
        memories = [ttnn.DRAM_MEMORY_CONFIG] * 3

    args = [make(host, operand_layout, memory) for host, memory in zip(operands, memories)]

    kwargs = {}
    if out_layout is not None:
        kwargs["output_tensor"] = make(torch.zeros(shape, dtype=torch.bfloat16), out_layout, sharded)
    if memory_config_kwarg == "dram":
        kwargs["memory_config"] = ttnn.DRAM_MEMORY_CONFIG

    with expect_error(RuntimeError, r"Tile layout"):
        ttnn.where(*args, **kwargs)


@pytest.mark.parametrize(
    "call, slot",
    [("where_true", "input_b"), ("where_false", "input_c"), ("lerp_tts", "input_b"), ("mac_tst", "input_c")],
)
def test_ternary_rejects_row_major_operand(device, expect_error, call, slot):
    """A row-major operand must be rejected even when input_a is TILE."""
    shape = (1, 1, 64, 64)
    torch.manual_seed(0)

    def make(layout):
        return ttnn.from_torch(
            torch.rand(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device
        )

    tile, rm = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
    with expect_error(RuntimeError, rf"requires {slot} to be in Tile layout"):
        if call == "where_true":
            ttnn.where(make(tile), make(rm), make(tile))
        elif call == "where_false":
            ttnn.where(make(tile), make(tile), make(rm))
        elif call == "lerp_tts":
            ttnn.lerp(make(tile), make(rm), 0.5)
        else:
            ttnn.mac(make(tile), 2.0, make(rm))


def test_ternary_accepts_tile_sharded(device):
    """The TILE counterpart of the rejected cases must still work."""
    shape = (1, 1, 64, 64)
    torch.manual_seed(0)
    sharded = _height_sharded(device)

    pred = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    a = torch.rand(shape, dtype=torch.bfloat16)
    b = torch.rand(shape, dtype=torch.bfloat16)

    def make(host, memory_config):
        return ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    for out_memory in (sharded, ttnn.DRAM_MEMORY_CONFIG):
        out = make(torch.zeros(shape, dtype=torch.bfloat16), out_memory)
        result = ttnn.where(
            make(pred.to(torch.bfloat16), sharded), make(a, sharded), make(b, sharded), output_tensor=out
        )
        assert_equal(ttnn.to_torch(result), torch.where(pred, a, b))


def test_ternary_cache_miss_different_output_alignment(device):
    """Same inputs, preallocated output over-padded vs default, must be two cache entries."""
    device.enable_program_cache()

    logical = (1, 1, 32, 32)
    padded = [1, 1, 64, 32]
    torch.manual_seed(0)

    pred = torch.rand(logical, dtype=torch.bfloat16) > 0.5
    a = torch.rand(logical, dtype=torch.bfloat16)
    b = torch.rand(logical, dtype=torch.bfloat16)
    expected = torch.where(pred, a, b)

    def tile(host):
        return ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def over_padded(host):
        return ttnn.tilize_with_val_padding(
            ttnn.from_torch(host, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=ttnn.bfloat16), padded, 0.0
        )

    # Build every tensor before clearing so that tilize's own entries do not count.
    inputs = [tile(pred.to(torch.bfloat16)), tile(a), tile(b)]
    out_default = tile(torch.zeros(logical, dtype=torch.bfloat16))
    out_over = over_padded(torch.zeros(logical, dtype=torch.bfloat16))
    assert list(out_default.shape) == list(out_over.shape)
    assert list(out_default.padded_shape) != list(out_over.padded_shape)

    device.clear_program_cache()
    for entries, out in enumerate((out_default, out_over), start=1):
        assert_equal(ttnn.to_torch(ttnn.where(*inputs, output_tensor=out)), expected)
        assert device.num_program_cache_entries() == entries

    device.disable_and_clear_program_cache()


@pytest.mark.parametrize("pass_memory_config", [True, False])
def test_ternary_preallocated_output_memory_distinguishes_cache_entries(device, pass_memory_config):
    """Regression test for cache key containing the memory of the preallocated output."""
    device.enable_program_cache()
    device.clear_program_cache()

    shape = (1, 1, 64, 64)
    torch.manual_seed(0)
    pred = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    t_true = torch.rand(shape, dtype=torch.bfloat16)
    t_false = torch.rand(shape, dtype=torch.bfloat16)
    expected = torch.where(pred, t_true, t_false)

    def to_dev(host, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )

    for entries, out_memory_config in enumerate((ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG), start=1):
        out = to_dev(torch.zeros(shape, dtype=torch.bfloat16), out_memory_config)
        kwargs = {"memory_config": ttnn.DRAM_MEMORY_CONFIG} if pass_memory_config else {}
        result = ttnn.where(
            to_dev(pred.to(torch.bfloat16)), to_dev(t_true), to_dev(t_false), output_tensor=out, **kwargs
        )

        assert device.num_program_cache_entries() == entries, (
            f"an output in {out_memory_config} must get its own cache entry -- sharing one "
            "means the key does not carry the preallocated output's memory, so the cached program "
            "addresses the wrong address space"
        )
        assert_equal(ttnn.to_torch(result), expected)

    device.disable_and_clear_program_cache()
