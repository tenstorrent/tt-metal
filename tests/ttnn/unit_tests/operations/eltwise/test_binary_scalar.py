# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import random

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn",
    (
        (ttnn.gt),
        (ttnn.lt),
        (ttnn.ne),
        (ttnn.ge),
        (ttnn.le),
        (ttnn.eq),
    ),
)
def test_binary_scalar_ops(input_shapes, device, ttnn_fn):
    torch.manual_seed(0)
    torch_input = torch.randn(input_shapes, dtype=torch.bfloat16) * 100
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.zeros_like(input_tensor)
    scalar = random.randint(-80, 80)
    ttnn_fn(input_tensor, scalar, output_tensor=output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    golden_tensor = golden_fn(torch_input, scalar)

    out = ttnn.to_torch(output_tensor).to(torch.bool)

    assert torch.equal(out, golden_tensor)


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
@pytest.mark.parametrize(
    "scalar",
    [
        7,
        -13,
        0,
        -1,
        2,
        10000,
    ],
)
def test_binary_scalar_int32_arithmetic(device, op_name, scalar):
    """Verify int32 tensor + int scalar passes the scalar as int32 (not float)."""
    ttnn_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)
    torch_input = torch.tensor(
        [
            1,
            -1,
            0,
            2147483640,
            2147483647,
            -2147483647,
            -2147483648,
            1000,
            -1000,
            42,
            123456789,
            -123456789,
            500,
            -500,
            999,
            -999,
            77,
            -77,
            2,
            -2,
            10,
            -10,
            100,
            -100,
            7,
            9,
            11,
            15,
        ],
        dtype=torch.int32,
    )
    expected = torch_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn_fn(tt_input, scalar)
    result = ttnn.to_torch(tt_output)

    assert torch.equal(expected, result)


@pytest.mark.parametrize("op_name", ["add", "sub"])
@pytest.mark.parametrize("scalar", [0, 1, 100, 65535])
def test_binary_scalar_uint32_arithmetic(device, op_name, scalar):
    """Verify uint32 tensor + int scalar near and at uint32 max boundary."""
    ttnn_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)
    torch_input = torch.tensor(
        [
            0,
            1,
            2,
            255,
            65535,
            100000,
            2147483647,
            2147483648,
            3000000000,
            4000000000,
            4294967290,
            4294967291,
            4294967294,
            4294967295,
            16777215,
            16777216,
            16777217,
            500,
            1000,
            10000,
            1000000,
            1000000000,
            2500000000,
            3500000000,
            3999999999,
            4294000000,
            4294900000,
            4294960000,
            4294967000,
            4294967200,
        ],
    )
    expected = torch_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_output = ttnn_fn(tt_input, scalar)

    expected_tt = ttnn.from_torch(
        expected,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    comparison = ttnn.eq(tt_output, expected_tt)
    comparison_torch = ttnn.to_torch(comparison)
    assert torch.all(comparison_torch), "Mismatch in uint32 scalar arithmetic"


@pytest.mark.parametrize("op_name", ["add", "sub", "mul"])
@pytest.mark.parametrize("scalar", [1.5, -2.25, 0.0, 100.0])
def test_binary_scalar_float32_arithmetic(device, op_name, scalar):
    """Verify float32 tensor + float scalar still works correctly."""
    ttnn_fn = getattr(ttnn, op_name)
    torch_fn = getattr(torch, op_name)
    torch.manual_seed(42)
    torch_input = torch.randn([1, 1, 32, 32], dtype=torch.float32)
    expected = torch_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn_fn(tt_input, scalar)
    result = ttnn.to_torch(tt_output)

    assert torch.allclose(expected, result, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "ttnn_fn",
    [ttnn.eq, ttnn.ne, ttnn.gt, ttnn.lt, ttnn.ge, ttnn.le],
)
@pytest.mark.parametrize("scalar", [0, 1, -1, 42, -100])
def test_binary_scalar_int32_relational(device, ttnn_fn, scalar):
    """Verify relational ops with int32 tensor and int scalar."""
    torch_input = torch.tensor(
        [
            -100,
            42,
            -1,
            0,
            1,
            200,
            -200,
            -50,
            -10,
            2147483640,
            2147483647,
            -2147483647,
            -2147483648,
            300,
            -300,
            -150,
            -5,
        ],
        dtype=torch.int32,
    )

    golden_fn = ttnn.get_golden_function(ttnn_fn)
    expected = golden_fn(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn_fn(tt_input, scalar)
    result = ttnn.to_torch(tt_output).to(torch.bool)

    assert torch.equal(expected, result)


@pytest.mark.parametrize(
    "scalar",
    [
        16777217,
        16777366,
        2147483640,
        2147483647,
        -2147483647,
        -2147483540,
        -2147483648,
    ],
)
def test_binary_scalar_int32_large_values(scalar, device):
    """Verify that large int32 scalars are not corrupted by float conversion.

    Values > 2^24 cannot be represented exactly in float32.  With ScalarVariant
    they should be packed as int32 directly and arrive on the device unchanged.
    """

    torch_input = torch.ones([1, 1, 32, 32], dtype=torch.int32)
    expected = torch.add(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.add(tt_input, scalar)
    result = ttnn.to_torch(tt_output)

    assert torch.equal(expected, result), (
        f"Large scalar {scalar} was likely truncated to float. "
        f"Expected {expected.flatten()[0].item()}, got {result.flatten()[0].item()}"
    )


@pytest.mark.parametrize(
    "scalar",
    [
        16777217,
        16777366,
        2147483640,
        2147483647,
        4294967200,
        4294967294,
    ],
)
def test_binary_scalar_uint32_large_values(scalar, device):
    """Verify that large uint32 scalars are not corrupted by float conversion.

    Values > 2^24 cannot be represented exactly in float32.  With ScalarVariant
    they should be packed as uint32 directly and arrive on the device unchanged.
    """

    torch_input = torch.ones([1, 1, 32, 32], dtype=torch.int64)
    expected = torch.add(torch_input, scalar)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.add(tt_input, scalar)
    result = ttnn.to_torch(tt_output, dtype=torch.int64)

    assert torch.equal(expected, result), (
        f"Large scalar {scalar} was likely truncated to float. "
        f"Expected {expected.flatten()[0].item()}, got {result.flatten()[0].item()}"
    )


# Scalar as the mathematical left operand. The tensor still occupies operand slot a on the
# device; the compute kernel is told to read the scalar as the left-hand side, so these run
# the same LLKs as the tensor-scalar form.


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 4, 4])),
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_fn, torch_fn",
    (
        (ttnn.add, lambda s, t: s + t),
        (ttnn.subtract, lambda s, t: s - t),
        (ttnn.multiply, lambda s, t: s * t),
        (ttnn.div, lambda s, t: s / t),
    ),
)
@pytest.mark.parametrize("dtype", (ttnn.bfloat16, ttnn.float32))
@pytest.mark.parametrize("layout", (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT))
def test_scalar_tensor_arithmetic(input_shapes, device, ttnn_fn, torch_fn, dtype, layout):
    # Row-major is covered explicitly: an all-row-major call dispatches on a separate host
    # path from the tiled one, and a mirrored scalar counts as row-major there.
    torch.manual_seed(0)
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    # away from zero: div by ~0 is not what this test is about
    torch_input = torch.rand(input_shapes, dtype=torch_dtype) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)

    scalar = 3.14
    output = ttnn.to_torch(ttnn_fn(scalar, input_tensor))

    assert_with_pcc(torch_fn(scalar, torch_input), output, 0.999)


@pytest.mark.parametrize("scalar", (7, -13, 2, 100))
@pytest.mark.parametrize(
    "ttnn_fn, torch_fn",
    (
        (ttnn.add, lambda s, t: s + t),
        (ttnn.subtract, lambda s, t: s - t),
        (ttnn.multiply, lambda s, t: s * t),
    ),
)
def test_scalar_tensor_int32(device, scalar, ttnn_fn, torch_fn):
    torch.manual_seed(0)
    torch_input = torch.randint(-500, 500, (1, 1, 320, 384), dtype=torch.int32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn_fn(scalar, input_tensor))

    assert torch.equal(torch_fn(scalar, torch_input), output)


@pytest.mark.parametrize("rounding_mode", (None, "floor", "trunc"))
def test_scalar_tensor_div_int32_rounding(device, rounding_mode):
    """int32 division with rounding is why this rides the real DIV LLKs rather than a
    reciprocal-multiply rewrite, which cannot express either."""
    torch.manual_seed(0)
    torch_input = torch.randint(-1000, 1000, (1, 1, 320, 384), dtype=torch.int32)
    torch_input[torch_input == 0] = 1
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 100
    output = ttnn.to_torch(ttnn.div(scalar, input_tensor, rounding_mode=rounding_mode))
    numerator = torch.full_like(torch_input, scalar)

    if rounding_mode is None:
        assert_with_pcc(numerator.float() / torch_input.float(), output, 0.999)
    else:
        assert torch.equal(torch.div(numerator, torch_input, rounding_mode=rounding_mode).float(), output)


def test_scalar_tensor_activations_follow_math_operands(device):
    """Under a mirrored scalar the caller's operand-b activations must still land on the
    tensor, even though the tensor is physically operand a."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    reciprocated = ttnn.div(scalar, input_tensor, input_tensor_b_activations=[ttnn.UnaryOpType.RECIP])

    # s / (1/t) == s * t
    assert_with_pcc(scalar * torch_input, ttnn.to_torch(reciprocated), 0.999)


@pytest.mark.parametrize("ttnn_fn", (ttnn.add, ttnn.subtract, ttnn.multiply, ttnn.div))
def test_scalar_tensor_keyword_form(device, ttnn_fn):
    """The scalar-first overload must be reachable by the operand names the docs use, not
    just positionally."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    by_keyword = ttnn_fn(input_tensor_a=scalar, input_tensor_b=input_tensor)
    positional = ttnn_fn(scalar, input_tensor)

    assert_with_pcc(ttnn.to_torch(positional), ttnn.to_torch(by_keyword), 0.9999)


@pytest.mark.parametrize("fast_and_approximate_mode", (True, False))
@pytest.mark.parametrize(
    "ttnn_fn, torch_fn",
    (
        (ttnn.subtract, lambda s, t: s - t),
        (ttnn.div, lambda s, t: s / t),
    ),
)
def test_scalar_tensor_fpu_and_sfpu_paths(device, ttnn_fn, torch_fn, fast_and_approximate_mode):
    """bf16 with fast_and_approximate_mode selects the FPU kernel, a different code path from
    the SFPU one. subtract runs the mirrored SUB LLK there; div is the case that lowers to a
    preprocess plus a commutative op (RECIP on the mathematical right operand, then MUL), and
    that preprocess has to invert along with the caller's activation spans."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    output = ttnn.to_torch(ttnn_fn(scalar, input_tensor, fast_and_approximate_mode=fast_and_approximate_mode))

    assert_with_pcc(torch_fn(scalar, torch_input), output, 0.999)


@pytest.mark.parametrize(
    "ttnn_fn, torch_fn",
    (
        (ttnn.add, lambda s, t: s + t),
        (ttnn.subtract, lambda s, t: s - t),
        (ttnn.multiply, lambda s, t: s * t),
        (ttnn.div, lambda s, t: s / t),
    ),
)
def test_scalar_tensor_row_major_sharded(device, ttnn_fn, torch_fn):
    """A sharded row-major tensor skips the interleaved row-major fast path and falls through
    to the tiled dispatch, which converts the output back to row major -- a third host branch
    beyond interleaved row-major and tiled."""
    torch.manual_seed(0)
    # 224 rows / 32-row shard = 7 shards, one per core in the 7-core range, so the shard grid
    # divides the tensor exactly and no core is left partially filled.
    shape = (1, 1, 224, 128)
    torch_input = torch.rand(shape, dtype=torch.bfloat16) + 0.5

    sharded_memory_config = ttnn.create_sharded_memory_config(
        [32, 128],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_memory_config,
    )
    assert input_tensor.memory_config().is_sharded()

    scalar = 3.14
    output = ttnn.to_torch(ttnn_fn(scalar, input_tensor))

    assert_with_pcc(torch_fn(scalar, torch_input), output, 0.999)


def test_rsub_operator_is_reachable(device):
    """`s - t` binds to Tensor.__rsub__ -> ttnn.rsub -> BinaryOpType::RSUB, a distinct op from
    the mirrored SUB that subtract(scalar, tensor) runs, so this covers the operator hook only.
    The mirrored path is covered by test_scalar_tensor_arithmetic."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    assert_with_pcc(3.0 - torch_input, ttnn.to_torch(3.0 - input_tensor), 0.999)


@pytest.mark.parametrize("ttnn_fn, torch_fn", ((ttnn.add, lambda s, t: s + t), (ttnn.subtract, lambda s, t: s - t)))
@pytest.mark.parametrize("ttnn_dtype, torch_dtype", ((ttnn.uint32, torch.int64), (ttnn.uint16, torch.int32)))
def test_scalar_tensor_unsigned(device, ttnn_fn, torch_fn, ttnn_dtype, torch_dtype):
    """add and subtract document UINT32/UINT16, which take a distinct SFPU path from the signed
    and floating dtypes. Subtract discriminates operand order, so this asserts exact equality."""
    scalar = 900
    torch_input = torch.tensor([[[[1, 2, 3, 4, 100, 500, 899, 900]]]], dtype=torch_dtype).repeat(1, 1, 32, 4)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn_fn(scalar, input_tensor)).to(torch_dtype)

    assert torch.equal(torch_fn(scalar, torch_input), output)


@pytest.mark.parametrize("rounding_mode", ("floor", "trunc"))
@pytest.mark.parametrize("ttnn_dtype, torch_dtype", ((ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)))
def test_scalar_tensor_div_float_rounding(device, rounding_mode, ttnn_dtype, torch_dtype):
    """The float rounding-mode branch of div is a separate code path from the int32 one: it
    divides and then applies ttnn.floor/trunc, forwarding the mirror flag on its own call.

    Operand order is checked against torch on the unrounded quotient. The rounding is then
    checked against that same device quotient rather than torch's, because rounding collapses
    a quotient to a small integer and a bf16 value landing either side of an integer boundary
    flips it by a whole step -- a precision artifact, not a dispatch error."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch_dtype) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    quotient = ttnn.to_torch(ttnn.div(scalar, input_tensor))
    assert_with_pcc(scalar / torch_input, quotient, 0.999)

    rounded = ttnn.to_torch(ttnn.div(scalar, input_tensor, rounding_mode=rounding_mode))
    expected = torch.floor(quotient) if rounding_mode == "floor" else torch.trunc(quotient)

    assert torch.equal(expected, rounded)


def test_scalar_tensor_scalar_side_activations(device):
    """The mirrored scalar is the mathematical first operand, so input_tensor_a_activations must
    land on the scalar even though the scalar physically occupies operand slot b."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    negated = ttnn.subtract(scalar, input_tensor, input_tensor_a_activations=[ttnn.UnaryOpType.NEG])

    assert_with_pcc(-scalar - torch_input, ttnn.to_torch(negated), 0.999)


def test_scalar_tensor_tile_height_sharded(device):
    """A sharded TILE tensor is a third dataflow configuration alongside interleaved TILE and
    sharded row major. Subtract discriminates operand order."""
    torch.manual_seed(0)
    # 224 rows / 32-row shard = 7 shards, one per core in the 7-core range, so the shard grid
    # divides the tensor exactly and no core is left partially filled.
    shape = (1, 1, 224, 128)
    torch_input = torch.rand(shape, dtype=torch.bfloat16) + 0.5

    sharded_memory_config = ttnn.create_sharded_memory_config(
        [32, 128],
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((0, 0), (0, 6))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_memory_config,
    )
    assert input_tensor.memory_config().is_sharded()

    output = ttnn.to_torch(ttnn.subtract(3.14, input_tensor))

    assert_with_pcc(3.14 - torch_input, output, 0.999)


@pytest.mark.parametrize("ttnn_fn, torch_fn", ((ttnn.subtract, lambda s, t: s - t), (ttnn.div, lambda s, t: s / t)))
def test_scalar_tensor_explicit_memory_config(device, ttnn_fn, torch_fn):
    """memory_config reaches the prim independently of the operand rewrite, so a mirrored scalar
    with a non-default output config exercises a combination the defaulted calls do not."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    output = ttnn_fn(scalar, input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)

    assert output.memory_config().buffer_type == ttnn.BufferType.L1
    assert_with_pcc(torch_fn(scalar, torch_input), ttnn.to_torch(output), 0.999)


@pytest.mark.parametrize("ttnn_fn, torch_fn", ((ttnn.subtract, lambda s, t: s - t), (ttnn.div, lambda s, t: s / t)))
def test_scalar_tensor_preallocated_output(device, ttnn_fn, torch_fn):
    """A preallocated output takes a different route through the host dispatch -- output_preallocated
    gates the row-major branch and compute_output_specs returns the supplied spec verbatim."""
    torch.manual_seed(0)
    shape = (1, 1, 320, 384)
    torch_input = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    preallocated = ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    scalar = 3.14
    returned = ttnn_fn(scalar, input_tensor, output_tensor=preallocated)

    # The result must land in the caller's buffer, not just in the returned handle.
    assert_with_pcc(torch_fn(scalar, torch_input), ttnn.to_torch(preallocated), 0.999)
    assert_with_pcc(torch_fn(scalar, torch_input), ttnn.to_torch(returned), 0.999)
