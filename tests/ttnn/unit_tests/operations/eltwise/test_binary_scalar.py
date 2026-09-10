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
    assert_with_pcc(expected.float(), output.float(), 0.999)


@pytest.mark.parametrize("op_name", ("add", "subtract", "multiply", "div"))
def test_scalar_tensor_comparison_mode_validates(device, op_name):
    """Comparison mode must actually grade a scalar-first call, not skip it. A golden that raises
    is caught at decorators.py and downgraded to a warning, so the op looks validated while it is
    not; comparison_mode_should_raise_exception turns that silence into a failure.

    The threshold is relaxed from the 0.9999 default because bf16 add and subtract miss it against
    their goldens for tensor-first calls too -- a pre-existing tolerance gap this test is not
    about. What is asserted here is that the comparison runs at all."""
    torch.manual_seed(0)
    torch_input = torch.rand((1, 1, 320, 384), dtype=torch.bfloat16) + 0.5
    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with ttnn.manage_config("enable_comparison_mode", True), ttnn.manage_config("comparison_mode_pcc", 0.99):
        with ttnn.manage_config("comparison_mode_should_raise_exception", True):
            getattr(ttnn, op_name)(3.14, input_tensor)
