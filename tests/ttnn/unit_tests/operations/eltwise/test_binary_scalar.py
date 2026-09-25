# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import random

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc, assert_with_ulp

pytestmark = pytest.mark.use_module_device

# Scalar-first arithmetic against the device measures at most 2 ULP, on bf16 add and subtract.
_SCALAR_FIRST_ULP_THRESHOLD = 3


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


@pytest.mark.parametrize("s", [3, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sub_scalar(device, s, h, w):
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor - s
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = input_tensor - s
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(expected_result=torch_output_tensor, actual_result=output_tensor, ulp_threshold=1)


# Scalar as the mathematical left operand. The tensor still occupies operand slot a on the
# device; the compute kernel is told to read the scalar as the left-hand side, so these run
# the same LLKs as the tensor-scalar form.
#
# Ops are parametrized by name and resolved with getattr, matching the tensor-scalar tests above.
# A function object as a parameter value would put its repr -- and so its memory address -- into
# the test id, which differs per pytest-xdist worker and breaks collection.


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 4, 4])),
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("op_name", ("add", "subtract", "multiply", "div"))
@pytest.mark.parametrize("dtype", (ttnn.bfloat16, ttnn.float32))
@pytest.mark.parametrize("layout", (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT))
def test_scalar_tensor_arithmetic(input_shapes, device, op_name, dtype, layout):
    ttnn_fn, torch_fn = getattr(ttnn, op_name), getattr(torch, op_name)
    # Row-major is covered explicitly: an all-row-major call dispatches on a separate host
    # path from the tiled one, and a scalar counts as row-major there.
    torch.manual_seed(0)
    torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}[dtype]
    # away from zero: div by ~0 is not what this test is about
    torch_input = torch.rand(input_shapes, dtype=torch_dtype) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device)

    scalar = 3.14
    output = ttnn.to_torch(ttnn_fn(scalar, input_tensor))

    assert_with_pcc(torch_fn(scalar, torch_input), output, 0.999)


@pytest.mark.parametrize("scalar", (7, -13, 2, 100))
@pytest.mark.parametrize("op_name", ("add", "subtract", "multiply"))
def test_scalar_tensor_int32(device, scalar, op_name):
    ttnn_fn, torch_fn = getattr(ttnn, op_name), getattr(torch, op_name)
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


@pytest.mark.parametrize("rounding_mode", (None, "floor", "trunc"))
def test_scalar_tensor_div_int32_promoted_by_float_scalar(device, rounding_mode):
    """A float scalar promotes an INT32 tensor to FLOAT32 before dividing, which recurses through
    div's scalar impl. That recursion has to carry scalar_is_lhs: the tensor-first overload pins
    it to false, so a promoted scalar numerator would otherwise compute tensor / scalar. Dispatch is
    on the scalar's type, not its value, so 3.5 and an integral-valued 4.0 both promote."""
    torch.manual_seed(0)
    torch_input = torch.randint(-1000, 1000, (1, 1, 320, 384), dtype=torch.int32)
    torch_input[torch_input == 0] = 1
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    for scalar in (3.5, 4.0):
        output = ttnn.to_torch(ttnn.div(scalar, input_tensor, rounding_mode=rounding_mode))
        expected = torch.div(
            torch.full_like(torch_input, scalar, dtype=torch.float32),
            torch_input.float(),
            rounding_mode=rounding_mode,
        )
        assert_with_pcc(expected, output, 0.999)


def test_scalar_tensor_activations_follow_math_operands(device):
    """With a scalar first operand the caller's operand-b activations must still land on the
    tensor, even though the tensor is physically operand a."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    reciprocated = ttnn.div(scalar, input_tensor, input_tensor_b_activations=[ttnn.UnaryOpType.RECIP])

    # s / (1/t) == s * t
    assert_with_pcc(scalar * torch_input, ttnn.to_torch(reciprocated), 0.999)


def test_scalar_tensor_reflected_sub_operator(device):
    """`s - t` binds to Tensor.__rsub__, which routes to subtract's scalar-first overload, so the
    operator runs SUB with swapped operands rather than BinaryOpType::RSUB.

    ttnn.rsub stays available as a named op and is asserted equivalent here, which is what makes
    the routing choice safe: the two agree bit-for-bit, while RSUB carries restrictions the
    swapped-operand path does not (no SFPU kernel on Quasar, non-bfloat16 output rejected under
    fast_and_approximate_mode=false)."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.0
    via_operator = ttnn.to_torch(scalar - input_tensor)
    assert_with_pcc(scalar - torch_input, via_operator, 0.999)

    assert torch.equal(ttnn.to_torch(ttnn.rsub(input_tensor, scalar)), via_operator)


@pytest.mark.parametrize("op_name", ("add", "subtract"))
@pytest.mark.parametrize("ttnn_dtype, torch_dtype", ((ttnn.uint32, torch.int64), (ttnn.uint16, torch.int32)))
def test_scalar_tensor_unsigned(device, op_name, ttnn_dtype, torch_dtype):
    """UINT32/UINT16 select the integer SFPU LLKs (add_int_tile / sub_int_tile) under their own
    DataFormat specialization -- a different compiled kernel from the float LLK, and from INT32's
    specialization of the same LLK. Subtract discriminates operand order, so this asserts exact
    equality."""
    ttnn_fn, torch_fn = getattr(ttnn, op_name), getattr(torch, op_name)
    scalar = 900
    torch_input = torch.tensor([[[[1, 2, 3, 4, 100, 500, 899, 900]]]], dtype=torch_dtype).repeat(1, 1, 32, 4)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.to_torch(ttnn_fn(scalar, input_tensor)).to(torch_dtype)

    assert torch.equal(torch_fn(scalar, torch_input), output)


@pytest.mark.parametrize("rounding_mode", ("floor", "trunc"))
@pytest.mark.parametrize("ttnn_dtype, torch_dtype", ((ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)))
def test_scalar_tensor_div_float_rounding(device, rounding_mode, ttnn_dtype, torch_dtype):
    """The float rounding-mode branch of div is a separate code path from the int32 one: it
    divides and then applies ttnn.floor/trunc, forwarding scalar_is_lhs on its own call.

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
    """The scalar is the mathematical first operand, so input_tensor_a_activations must
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


@pytest.mark.parametrize("op_name", ("subtract", "div"))
def test_scalar_tensor_explicit_memory_config(device, op_name):
    """memory_config reaches the prim independently of the operand rewrite, so a scalar first operand
    with a non-default output config exercises a combination the defaulted calls do not."""
    ttnn_fn, torch_fn = getattr(ttnn, op_name), getattr(torch, op_name)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    output = ttnn_fn(scalar, input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)

    assert output.memory_config().buffer_type == ttnn.BufferType.L1
    assert_with_pcc(torch_fn(scalar, torch_input), ttnn.to_torch(output), 0.999)


@pytest.mark.parametrize("op_name", ("subtract", "div"))
def test_scalar_tensor_preallocated_output(device, op_name):
    """A preallocated output takes a different route through the host dispatch -- output_preallocated
    gates the row-major branch and compute_output_specs returns the supplied spec verbatim."""
    ttnn_fn, torch_fn = getattr(ttnn, op_name), getattr(torch, op_name)
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


@pytest.mark.parametrize("op_name", ("add", "subtract", "multiply", "div"))
def test_scalar_tensor_keyword_form(device, op_name):
    """The scalar-first overload must be reachable by the operand names the docs use, not
    just positionally."""
    ttnn_fn = getattr(ttnn, op_name)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    by_keyword = ttnn_fn(input_tensor_a=scalar, input_tensor_b=input_tensor)
    positional = ttnn_fn(scalar, input_tensor)

    # Same program, same inputs, so this is bit-exact: any divergence means the keyword form
    # resolved to a different overload.
    assert torch.equal(ttnn.to_torch(positional), ttnn.to_torch(by_keyword))


@pytest.mark.parametrize("fast_and_approximate_mode", (True, False))
@pytest.mark.parametrize("op_name", ("subtract", "div"))
def test_scalar_tensor_fpu_and_sfpu_paths(device, op_name, fast_and_approximate_mode):
    """bf16 with fast_and_approximate_mode selects the FPU kernel, a different code path from
    the SFPU one. subtract runs the SUB LLK with swapped operands there; div is the case that lowers to a
    preprocess plus a commutative op (RECIP on the mathematical right operand, then MUL), and
    that preprocess has to swap along with the caller's per-operand activations."""
    ttnn_fn, torch_fn = getattr(ttnn, op_name), getattr(torch, op_name)
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    scalar = 3.14
    output = ttnn.to_torch(ttnn_fn(scalar, input_tensor, fast_and_approximate_mode=fast_and_approximate_mode))

    assert_with_pcc(torch_fn(scalar, torch_input), output, 0.999)


@pytest.mark.parametrize("op_name", ("add", "subtract", "multiply", "div"))
def test_scalar_tensor_row_major_sharded(device, op_name):
    """A sharded row-major tensor skips the interleaved row-major fast path and falls through
    to the tiled dispatch, which converts the output back to row major -- a third host branch
    beyond interleaved row-major and tiled."""
    ttnn_fn, torch_fn = getattr(ttnn, op_name), getattr(torch, op_name)
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


@pytest.mark.parametrize("tensor_dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("scalar", [2.5, 0.5, -1.5])
@pytest.mark.parametrize("ttnn_op", [ttnn.multiply, ttnn.div])
def test_int_tensor_float_scalar_promotes(device, ttnn_op, tensor_dtype, scalar):
    # The scalar is packed using the tensor's dtype, so without promotion 2.5 arrives as 2 and 0.5
    # as 0 -- div(int32, 0.5) used to return inf instead of 14. mul/div promote, matching both torch
    # and what the tensor-tensor path already does for a mixed int/float pair.
    torch_input = torch.tensor([[7, 6, 12, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_op(a, scalar)
    torch_golden = (torch_input.float() * scalar) if ttnn_op is ttnn.multiply else (torch_input.float() / scalar)

    assert output.dtype == ttnn.float32
    assert_with_ulp(expected_result=torch_golden, actual_result=output, ulp_threshold=1)


@pytest.mark.parametrize("tensor_dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract])
def test_int_tensor_fractional_scalar_rejected(device, ttnn_op, tensor_dtype, expect_error):
    # add/subtract reject a mixed int/float tensor pair rather than promoting, so a scalar they
    # cannot represent is rejected too instead of being silently truncated.
    torch_input = torch.tensor([[7, 6, 12, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "cannot represent the scalar"):
        ttnn_op(a, 2.5)


@pytest.mark.parametrize("tensor_dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract, ttnn.multiply])
def test_int_tensor_integer_scalar_stays_exact_above_2_24(device, ttnn_op, tensor_dtype):
    # float32 carries a 24-bit mantissa, so a promoted tensor cannot represent integers past 2^24:
    # 16777217 * 2 would come back as 33554432 rather than 33554434. An integer scalar reaches the
    # kernel intact on the integer path, so it stays there and keeps these exact -- this is how a
    # caller asks for exactness now that an integral *float* like 2.0 promotes instead.
    torch_input = torch.tensor([[2**24 - 1, 2**24, 2**24 + 1, 2**24 + 3]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_op(a, 2)
    torch_golden = {
        ttnn.add: torch_input + 2,
        ttnn.subtract: torch_input - 2,
        ttnn.multiply: torch_input * 2,
    }[ttnn_op]

    # compared as integers: routing these through float32 would silently round them
    assert output.dtype == tensor_dtype
    assert ttnn.to_torch(output).flatten().tolist() == torch_golden.flatten().tolist()


@pytest.mark.parametrize("tensor_dtype", [ttnn.int32, ttnn.uint32])
def test_int_tensor_integral_float_scalar_promotes_like_torch(device, tensor_dtype):
    # Promotion keys off the scalar's type, not its value, so 2.0 promotes exactly as 2.5 does.
    # torch agrees -- int32_tensor * 2.0 is float32 there too -- and deciding by value instead would
    # make the output dtype depend on a runtime number. add/subtract are not here because they
    # reject a mixed int/float pair rather than promoting, which is separate policy (#55685).
    torch_input = torch.tensor([[7, 6, 12, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    for ttnn_op, torch_op in ((ttnn.multiply, torch.mul), (ttnn.div, torch.div)):
        output = ttnn_op(a, 2.0)
        expected = torch_op(torch_input.float(), 2.0)

        assert output.dtype == ttnn.float32, f"{ttnn_op} with 2.0 should promote like torch"
        assert_with_ulp(expected_result=expected, actual_result=output, ulp_threshold=1)


@pytest.mark.parametrize("tensor_dtype", [ttnn.int32, ttnn.uint32])
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract, ttnn.multiply])
def test_int_tensor_integer_scalar_unchanged(device, ttnn_op, tensor_dtype):
    # An integer scalar loses nothing in the pack, so it must keep the integer dtype and value.
    torch_input = torch.tensor([[7, 6, 12, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn_op(a, 2)
    torch_golden = {
        ttnn.add: torch_input + 2,
        ttnn.subtract: torch_input - 2,
        ttnn.multiply: torch_input * 2,
    }[ttnn_op]

    assert output.dtype == tensor_dtype
    assert torch.equal(ttnn.to_torch(output).float(), torch_golden.float())


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("scalar", [2.5, 0.5, -1.5, 3e9])
def test_int_tensor_float_scalar_rounded_division(device, rounding_mode, scalar):
    # The DIV_FLOOR/DIV_TRUNC kernels are int32-only and take the divisor through the int32 scalar
    # packing, so a divisor they cannot carry has to be divided in floating point and rounded after.
    # These used to be rejected outright even though div(rounding_mode=None) accepted them. 3e9 is
    # the integral-but-out-of-range case, which the fractional check alone would have missed.
    torch_input = torch.tensor([[-13, -7, 6, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.div(a, scalar, rounding_mode=rounding_mode)
    quotient = torch_input.float() / scalar
    torch_golden = torch.floor(quotient) if rounding_mode == "floor" else torch.trunc(quotient)

    # Negative numerators are in the input on purpose: the quotient has to be rounded while it is
    # still floating point, or floor(-13/2.5) comes back as -5 instead of -6.
    assert output.dtype == ttnn.float32
    assert_with_ulp(expected_result=torch_golden, actual_result=output, ulp_threshold=1)


@pytest.mark.parametrize("rounding_mode", ["floor", "trunc"])
@pytest.mark.parametrize("scalar, expected_dtype", [(2, ttnn.int32), (2.0, ttnn.float32)])
def test_int_tensor_rounded_division_dtype_follows_scalar_type(device, rounding_mode, scalar, expected_dtype):
    # ttnn.div decides from the divisor's type rather than its value: an integer 2 keeps exact int32
    # division, while 2.0 promotes the tensor and divides in floating point. Worth pinning because it
    # is the opposite of what multiply does -- multiply(int32, 2.0) stays int32, since an integral
    # scalar reaches the kernel intact and promoting would cap exact integers at 2**24.
    torch_input = torch.tensor([[-13, -7, 6, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.div(a, scalar, rounding_mode=rounding_mode)
    quotient = torch_input.float() / 2
    torch_golden = torch.floor(quotient) if rounding_mode == "floor" else torch.trunc(quotient)

    assert output.dtype == expected_dtype
    assert ttnn.to_torch(output).float().flatten().tolist() == torch_golden.flatten().tolist()


@pytest.mark.parametrize(
    "tensor_dtype, scalar",
    [
        # Integral, so the old fractional-only check let these through to an undefined
        # float-to-integer cast: inf arrived as INT32_MIN and 2**32 as 0.
        (ttnn.int32, 2.0**31),  # one past INT32_MAX
        (ttnn.int32, -(2.0**31) - 2048),  # one representable step below INT32_MIN
        (ttnn.int32, float("inf")),
        (ttnn.int32, float("-inf")),
        (ttnn.uint32, -3.0),  # negative against an unsigned tensor
        (ttnn.uint32, 2.0**32),  # one past UINT32_MAX
        (ttnn.uint32, float("inf")),
    ],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.subtract])
def test_int_tensor_unrepresentable_scalar_rejected(device, ttnn_op, tensor_dtype, scalar, expect_error):
    # Being integral is not enough: the value also has to be finite and inside the tensor dtype's
    # range, or the cast in pack_scalar_runtime_arg is undefined and silently changes it.
    torch_input = torch.tensor([[7, 6, 12, 100]], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "cannot represent the scalar"):
        ttnn_op(a, scalar)


@pytest.mark.parametrize(
    "tensor_dtype, scalar",
    [
        (ttnn.int32, -(2.0**31)),  # INT32_MIN, must survive the lower bound
        (ttnn.int32, 2147483520.0),  # largest float32 below INT32_MAX
        (ttnn.uint32, 0.0),  # lower bound for an unsigned tensor
        (ttnn.uint32, 4294965248.0),  # largest float32 below UINT32_MAX
    ],
)
def test_int_tensor_boundary_scalar_accepted(device, tensor_dtype, scalar):
    # The limits themselves have to stay on the integer path, since rejecting them would be as wrong
    # as accepting the values past them. INT32_MAX and UINT32_MAX are not float32 values, so the
    # check compares against powers of two and these are the largest floats below each limit.
    # Added to zero so the sum itself cannot overflow the dtype and confuse the result.
    torch_input = torch.zeros([1, 32], dtype=torch.int32)
    a = ttnn.from_torch(torch_input, dtype=tensor_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output = ttnn.add(a, scalar)

    assert output.dtype == tensor_dtype
    assert ttnn.to_torch(output, dtype=torch.int64).flatten().tolist() == [int(scalar)] * torch_input.numel()


@pytest.mark.parametrize("op_name", ("add", "subtract", "multiply", "div"))
@pytest.mark.parametrize("ttnn_dtype, torch_dtype", ((ttnn.bfloat16, torch.bfloat16), (ttnn.int32, torch.int32)))
def test_scalar_tensor_golden_matches_device(device, op_name, ttnn_dtype, torch_dtype):
    """The goldens key dtype-dependent branches off the tensor operand rather than the argument
    position, so they stay callable when a scalar occupies operand a. Reading the dtype off
    operand a instead raises AttributeError, which comparison mode swallows into a skip."""
    torch.manual_seed(0)
    if torch_dtype == torch.int32:
        torch_input = torch.randint(1, 100, (1, 1, 320, 384), dtype=torch_dtype)
        scalar = 7
    else:
        torch_input = torch.rand((1, 1, 320, 384), dtype=torch_dtype) + 0.5
        scalar = 3.14
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    golden = ttnn.get_golden_function(getattr(ttnn, op_name))
    expected = golden(scalar, torch_input)
    output = ttnn.to_torch(getattr(ttnn, op_name)(scalar, input_tensor))

    # The golden is the reference the device is graded against, so it has to carry operand order
    # too: for subtract and div a reversed golden would still be a valid tensor of the right shape.
    # Comparing through .float() would let a golden of the wrong dtype pass, which is the class of
    # bug this keys off the tensor operand to avoid.
    assert expected.dtype == output.dtype
    if torch_dtype == torch.int32 and op_name != "div":
        # Integer add/subtract/multiply run the integer LLKs and are bit-exact on device.
        assert_equal(expected, output)
    else:
        # Measured worst case is 2 ULP, on bf16 add and subtract.
        assert_with_ulp(expected_result=expected, actual_result=output, ulp_threshold=_SCALAR_FIRST_ULP_THRESHOLD)


@pytest.mark.parametrize(
    "op_name",
    (
        "add",
        "subtract",
        "multiply",
        # Comparison mode never grades scalar-first div: no threshold, however impossible, fails
        # it. That is not this PR's doing -- tensor-tensor add and div skip grading on main too --
        # so it is marked strict, to fail the day div starts being compared and this can go.
        pytest.param("div", marks=pytest.mark.xfail(strict=True, reason="scalar-first div is not graded")),
    ),
)
def test_scalar_tensor_comparison_mode_validates(device, op_name, expect_error):
    """Comparison mode must actually grade a scalar-first call, not skip it. A golden that raises
    is caught at decorators.py and downgraded to a warning, so the op looks validated while it is
    not; comparison_mode_should_raise_exception turns that silence into a failure.

    The threshold is relaxed from the 0.9999 default because bf16 add and subtract miss it against
    their goldens for tensor-first calls too -- a pre-existing tolerance gap this test is not
    about. What is asserted here is that the comparison runs at all, which only means something
    once a threshold the call cannot meet is shown to fail it."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with ttnn.manage_config("enable_comparison_mode", True), ttnn.manage_config(
        "comparison_mode_should_raise_exception", True
    ):
        # An unreachable threshold first: if this does not raise, the comparison never ran and the
        # 0.99 call below would pass for the same reason, proving nothing.
        with ttnn.manage_config("comparison_mode_pcc", 1.0):
            with expect_error(RuntimeError, "Comparing output tensor 0 against CPU locally failed"):
                getattr(ttnn, op_name)(3.14, input_tensor)

        with ttnn.manage_config("comparison_mode_pcc", 0.99):
            getattr(ttnn, op_name)(3.14, input_tensor)


@pytest.mark.parametrize("op_name", ("add", "subtract", "multiply"))
@pytest.mark.parametrize("torch_dtype, bit_width", ((torch.uint16, 16), (torch.uint32, 32)))
@pytest.mark.parametrize("scalar", (70000, -2.5, 2**32))
def test_unsigned_golden_wraps_scalar_outside_operand_range(op_name, torch_dtype, bit_width, scalar):
    """A scalar wider than the operand wraps to its storage width, so the golden masks it rather
    than rejecting it. Materializing the scalar in the unsigned dtype instead fails Torch's
    conversion range check, turning the wrap into an error in both operand orders."""
    # A scalar the integer arms of the variant cannot hold arrives as a float, and multiply then
    # promotes the operand to float32 instead of wrapping it.
    binds_as_float = not float(scalar).is_integer() or not -(1 << 31) <= scalar < (1 << 32)
    if op_name == "multiply" and bit_width == 32 and binds_as_float:
        pytest.skip("multiply promotes a 32-bit integer operand against a float-bound scalar")

    torch.manual_seed(0)
    torch_fn = getattr(torch, op_name)
    mask = (1 << bit_width) - 1
    torch_input = torch.randint(0, 60000, (1, 1, 32, 32)).to(torch_dtype)
    wide, wrapped = torch_input.to(torch.int64), int(scalar) & mask

    golden = ttnn.get_golden_function(getattr(ttnn, op_name))

    assert torch.equal(golden(torch_input, scalar), (torch_fn(wide, wrapped) & mask).to(torch_dtype))
    assert torch.equal(golden(scalar, torch_input), (torch_fn(wrapped, wide) & mask).to(torch_dtype))


@pytest.mark.parametrize("rounding_mode", ("trunc", "floor"))
@pytest.mark.parametrize("scalar", (7, 2.5))
@pytest.mark.parametrize("scalar_first", (True, False))
def test_scalar_tensor_div_rounding_golden_matches_device(device, rounding_mode, scalar, scalar_first):
    """The integer division path requires both operands integral, not just the tensor one: a float
    scalar promotes an INT32 tensor before dividing. Only the device decides which is right, so
    this grades the golden against it for both scalar types, in both operand orders -- the
    promotion applies to the tensor-first form that predates the scalar-first overloads too."""
    torch.manual_seed(0)
    torch_input = torch.randint(1, 100, (1, 1, 320, 384), dtype=torch.int32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    golden_args = (scalar, torch_input) if scalar_first else (torch_input, scalar)
    device_args = (scalar, input_tensor) if scalar_first else (input_tensor, scalar)
    expected = ttnn.get_golden_function(ttnn.div)(*golden_args, rounding_mode=rounding_mode)
    output = ttnn.to_torch(ttnn.div(*device_args, rounding_mode=rounding_mode))

    assert expected.dtype == output.dtype
    if expected.dtype.is_floating_point:
        assert_with_ulp(expected_result=expected, actual_result=output, ulp_threshold=_SCALAR_FIRST_ULP_THRESHOLD)
    else:
        # An integer scalar keeps exact integer division on both sides.
        assert_equal(expected, output)


@pytest.mark.parametrize("scalar_first", (True, False))
@pytest.mark.parametrize("op_name", ("multiply", "div"))
@pytest.mark.parametrize(
    "scalar, promotes",
    (
        (2.5, True),
        (-1.5, True),
        # An integral float still promotes: the device keys the decision off the scalar's type so
        # the output dtype cannot depend on a runtime value.
        (2.0, True),
        # Too wide for either integer arm of the scalar variant, so it arrives as a float.
        ((1 << 32) + 512, True),
        # The integer arms, which stay on the integer path and keep the operand's dtype.
        (2, False),
        (-2, False),
    ),
)
@pytest.mark.parametrize("ttnn_dtype, torch_dtype", ((ttnn.int32, torch.int32), (ttnn.uint32, torch.uint32)))
def test_float_scalar_promotes_integer_tensor(device, ttnn_dtype, torch_dtype, scalar, promotes, op_name, scalar_first):
    """multiply and div promote a 32-bit integer operand against a float scalar rather than
    truncating it; every other op rejects the call. The golden has to follow the same rule, or it
    reports an integer where the device returns float32 and comparison mode fails on dtype alone."""
    if op_name == "div" and not promotes:
        pytest.skip("div is true division, so an integer scalar has no integer device path here")

    torch_input = torch.arange(1, 1025, dtype=torch.int64).reshape(1, 1, 32, 32).to(torch_dtype)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    golden_args = (scalar, torch_input) if scalar_first else (torch_input, scalar)
    device_args = (scalar, input_tensor) if scalar_first else (input_tensor, scalar)
    expected = ttnn.get_golden_function(getattr(ttnn, op_name))(*golden_args)
    output = ttnn.to_torch(getattr(ttnn, op_name)(*device_args))

    assert expected.dtype == output.dtype, f"golden {expected.dtype} != device {output.dtype}"
    assert (expected.dtype == torch.float32) == promotes, f"{scalar!r} promoted to {expected.dtype}"
    if promotes:
        assert_with_ulp(expected_result=expected, actual_result=output, ulp_threshold=_SCALAR_FIRST_ULP_THRESHOLD)
    else:
        assert_equal(expected, output)


@pytest.mark.parametrize("op_name", ("add", "subtract", "rsub"))
@pytest.mark.parametrize("scalar", (-(2**31) - 1, -(2**31) - 128))
def test_scalar_float32_rounds_into_range_before_the_exactness_check(device, op_name, scalar):
    """A scalar too wide for either integer arm is bound as a float, and float32 is spaced 256
    apart at 2**31 -- so the 128 integers in [-2**31 - 128, -2**31) round to exactly -2**31, clear
    the range check that would otherwise reject them, and reach the kernel as -2**31. Taking the
    unrounded value here does not cost a unit at the near edge: at the far one it wraps to the
    opposite end of the range. One integer below the band, float32 lands out of range and the
    device rejects the call instead."""
    torch_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32).repeat(32, 8)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.get_golden_function(getattr(ttnn, op_name))(torch_input, scalar)
    output = ttnn.to_torch(getattr(ttnn, op_name)(input_tensor, scalar))

    assert_equal(expected, output)


@pytest.mark.parametrize("op_name", ("add", "subtract", "rsub"))
def test_scalar_below_the_rounding_band_is_rejected(device, op_name, expect_error):
    """One integer below the band the rounding lands outside the operand's range, and the integer
    path refuses the call rather than packing a value it would change."""
    torch_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32).repeat(32, 8)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "cannot represent the scalar"):
        getattr(ttnn, op_name)(input_tensor, -(2**31) - 129)


@pytest.mark.parametrize("op_name", ("add", "subtract"))
@pytest.mark.parametrize(
    "scalar",
    (
        # Above the operand's range, and fractional: the integer path refuses both rather than
        # letting the pack change the value silently. The golden still masks, which stays
        # unreachable through the device because the op rejects the call before a golden runs.
        (1 << 32) + 512,
        -2.5,
    ),
)
def test_uint32_scalar_the_integer_path_cannot_represent_is_rejected(device, op_name, scalar, expect_error):
    """A UINT32 operand takes a scalar only where packing it as an integer is exact. Only add and
    subtract refuse: multiply promotes the operand to float32 and keeps the scalar. UINT16 wraps
    all of these, so the guard is keyed to the 32-bit path rather than to unsignedness."""
    torch_input = torch.tensor([[[[0, 1, 2, 3, 100, 500, 4464, 65535]]]], dtype=torch.int64).repeat(1, 1, 32, 4)
    input_tensor = ttnn.from_torch(
        torch_input.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=device
    )

    with expect_error(RuntimeError, "cannot represent the scalar"):
        getattr(ttnn, op_name)(scalar, input_tensor)


@pytest.mark.parametrize(
    "op_name, rounding_mode", (("div", "trunc"), ("div", "floor"), ("div", None), ("multiply", None))
)
@pytest.mark.parametrize("scalar_first", (True, False))
def test_zero_dim_float_tensor_promotes_like_a_python_float(device, op_name, rounding_mode, scalar_first):
    """_has_float_scalar reads a 0-d tensor's dtype to decide whether to promote its INT32 partner,
    which is only correct if the device promotes for a 0-d operand the same way it does for a
    Python float. On device that is a tensor-tensor call, so nothing about the scalar overloads
    guarantees it. Both ops sharing the helper are covered: promotion moves the operand to float32
    and so decides which branch each of them takes afterwards."""
    torch.manual_seed(0)
    torch_input = torch.randint(1, 100, (1, 1, 320, 384), dtype=torch.int32)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    torch_scalar = torch.tensor(2.5, dtype=torch.float32)
    scalar_tensor = ttnn.from_torch(torch_scalar, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    golden_args = (torch_scalar, torch_input) if scalar_first else (torch_input, torch_scalar)
    device_args = (scalar_tensor, input_tensor) if scalar_first else (input_tensor, scalar_tensor)
    # multiply has no rounding_mode, and promotion is what decides its output dtype just the same.
    kwargs = {"rounding_mode": rounding_mode} if op_name == "div" else {}
    op = getattr(ttnn, op_name)
    expected = ttnn.get_golden_function(op)(*golden_args, **kwargs)
    output = ttnn.to_torch(op(*device_args, **kwargs))

    assert expected.dtype == output.dtype, f"golden {expected.dtype} != device {output.dtype}"
    assert expected.dtype == torch.float32, f"promotion did not happen: {expected.dtype}"
    if rounding_mode is None:
        assert_with_ulp(expected_result=expected, actual_result=output, ulp_threshold=_SCALAR_FIRST_ULP_THRESHOLD)
    else:
        assert_equal(expected, output)


def test_tensor_operand_keeps_a_dtype_when_neither_operand_is_shaped():
    """A 0-d tensor against a Python number leaves no shaped operand, and the dtype-dependent
    branches still dereference whichever one is returned."""
    golden = ttnn.get_golden_function(ttnn.add)

    assert_equal(golden(torch.tensor(3, dtype=torch.int32), 2), torch.tensor(5, dtype=torch.int32))


@pytest.mark.parametrize("op_name", ("add", "subtract", "multiply"))
@pytest.mark.parametrize(
    "ttnn_dtype, torch_dtype, bit_width, scalar",
    (
        (ttnn.uint16, torch.uint16, 16, (1 << 16) + 4464),
        # A negative scalar reaches the operand width through a float-to-unsigned conversion whose
        # truncated value is out of range, which C++ leaves undefined rather than defining as
        # modulo. These are measured to wrap, so the golden matches what the hardware does; it is
        # not a guarantee the standard makes.
        (ttnn.uint16, torch.uint16, 16, -2.5),
        (ttnn.uint16, torch.uint16, 16, -2),
        (ttnn.uint32, torch.uint32, 32, -2),
    ),
)
def test_scalar_tensor_unsigned_out_of_range_scalar(device, op_name, ttnn_dtype, torch_dtype, bit_width, scalar):
    """A scalar wider than the operand reaches the kernel through an out-of-range cast in
    pack_scalar_runtime_arg, so the width it lands on is the device's to define, not the golden's.
    The wraparound the golden models is only a reference if it agrees here."""
    torch_input = torch.tensor([[[[0, 1, 2, 3, 100, 500, 4464, 65535]]]], dtype=torch.int64).repeat(1, 1, 32, 4)
    torch_input = (torch_input & ((1 << bit_width) - 1)).to(torch_dtype)
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    expected = ttnn.get_golden_function(getattr(ttnn, op_name))(scalar, torch_input)
    output = ttnn.to_torch(getattr(ttnn, op_name)(scalar, input_tensor)).to(torch_dtype)

    assert_equal(expected, output)
