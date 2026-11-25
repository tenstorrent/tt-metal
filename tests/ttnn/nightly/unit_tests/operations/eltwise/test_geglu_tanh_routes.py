# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The TT-NN calls that compute geglu_tanh's product and run its one program."""

import pytest
import torch
import ttnn

OP_TYPE = "GEGLU_TANH"
ROUTE_BOARDS = ("blackhole",)


def _board(device):
    return "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"


def _skip_unless_routed(device):
    if _board(device) not in ROUTE_BOARDS:
        pytest.skip("this board keeps the composite")


def _device_operations(call):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        result = call()
    finally:
        graph = ttnn.graph.end_graph_capture()
    operations = [name for name in ttnn.graph.extract_calltrace(graph) if "DeviceOperation" in name]
    arguments = " ".join(str(argument) for node in graph for argument in node.get("arguments", []))
    return result, operations, arguments


def _operands(shape):
    generator = torch.Generator().manual_seed(0)
    return (32 * (2 * torch.rand(shape, generator=generator) - 1)).to(torch.bfloat16)


def _on_device(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _golden(gate, up):
    """The exact product of the leaves, in float64."""
    a, b = gate.to(torch.float64), up.to(torch.float64)
    return (a / (1 + torch.exp(-(1.5957691216057308 * a + 0.07135481627260025 * a**3)))) * (b)


def _routed(gate, up, **kwargs):
    """ttnn.multiply with the gate's GELU_TANH: the call that runs the GEGLU_TANH program."""
    return ttnn.multiply(
        gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)], **kwargs
    )


def _assert_close(actual, golden, rtol=2.0**-7, atol=2.0**-100):
    """Within 2 BF16 steps of the golden; a result the kernel flushes to 0 is tiny in the golden too."""
    actual = actual.to(torch.float64)
    error = (actual - golden).abs()
    bad = error > atol + rtol * golden.abs()
    assert not bool(
        bad.any()
    ), f"{int(bad.sum())} of {bad.numel()} lanes off the golden; worst error {error.max().item():.3e}"


def _assert_pcc(actual, golden, bound):
    a, b = actual.to(torch.float64).flatten(), golden.to(torch.float64).flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pcc >= bound, f"PCC {pcc:.6f} against the golden, below {bound}"


@pytest.mark.parametrize("order", ["gate_first", "up_first"])
def test_multiply_with_the_leaves_activations_runs_geglu_tanh(device, order):
    """ttnn.multiply of BF16 operands carrying exactly the leaves' pre-activations, in either order, is the multiply with the gate's GELU_TANH, as
    one program."""
    _skip_unless_routed(device)
    gate, up = _operands((1, 1, 64, 64)), _operands((1, 1, 64, 64)).flip(-1)
    expected = ttnn.to_torch(_routed(_on_device(gate, device), _on_device(up, device)))
    g, u = _on_device(gate, device), _on_device(up, device)
    if order == "gate_first":
        call = lambda: ttnn.multiply(
            g,
            u,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)],
            input_tensor_b_activations=[],
        )
    else:
        call = lambda: ttnn.multiply(
            u,
            g,
            input_tensor_a_activations=[],
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)],
        )
    out, operations, arguments = _device_operations(call)
    assert OP_TYPE in arguments and len(operations) == 1, f"launched {operations}, not one {OP_TYPE} program"
    out = ttnn.to_torch(out)
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    _assert_close(out, _golden(gate, up))


@pytest.mark.parametrize("output", ["l1_width_sharded", "unset"])
@pytest.mark.parametrize("order", ["gate_first", "up_first"])
def test_multiply_with_width_sharded_operands_runs_geglu_tanh(device, order, output):
    """L1 width-sharded operands, and an L1 width-sharded output as deepseek_v3's decode MLP asks for one (or the
    operands' placement, unset): the routed multiply is one program, in that placement, with the multiply with the gate's GELU_TANH's bits.
    """
    _skip_unless_routed(device)
    gate, up = _operands((1, 1, 32, 1024)), _operands((1, 1, 32, 1024)).flip(-1)
    expected = ttnn.to_torch(_routed(_on_device(gate, device), _on_device(up, device)))
    sharded = ttnn.create_sharded_memory_config(
        (32, 128),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    g = ttnn.from_torch(gate, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded)
    u = ttnn.from_torch(up, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded)
    kwargs = dict(memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG) if output == "l1_width_sharded" else {}
    if order == "gate_first":
        call = lambda: ttnn.multiply(
            g,
            u,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)],
            input_tensor_b_activations=[],
            **kwargs,
        )
    else:
        call = lambda: ttnn.multiply(
            u,
            g,
            input_tensor_a_activations=[],
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)],
            **kwargs,
        )
    out, operations, arguments = _device_operations(call)
    assert OP_TYPE in arguments and len(operations) == 1, f"launched {operations}, not one {OP_TYPE} program"
    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert out.memory_config().buffer_type == ttnn.BufferType.L1
    assert torch.equal(ttnn.to_torch(out).view(torch.int16), expected.view(torch.int16))


def _assert_within_bfloat8_b_blocks(actual, golden):
    """Within one BFLOAT8_B step of its block, and half a BF16 step, of the golden: a block is 16 consecutive values
    of a row, sharing the exponent of its largest, so its step is 2^(e - 6) for that largest value's exponent e."""
    actual, golden = actual.to(torch.float64), golden.to(torch.float64)
    largest = torch.maximum(actual.abs(), golden.abs()).reshape(*golden.shape[:-1], -1, 16).amax(-1, keepdim=True)
    step = torch.exp2(torch.floor(torch.log2(largest.clamp(min=2.0**-126))) - 6)
    bound = (1.5 * step).expand(*step.shape[:-1], 16).reshape(golden.shape)
    error = (actual - golden).abs()
    bad = error > bound
    assert not bool(bad.any()), f"{int(bad.sum())} of {bad.numel()} lanes off the golden by more than a block step"


@pytest.mark.parametrize("case", ["bfloat8_b_output", "bfloat8_b_operands", "bfloat8_b_operands_and_output"])
def test_multiply_with_bfloat8_b_runs_geglu_tanh(device, case):
    """BFLOAT8_B operands unpack to BF16 in DEST, and the BF16 result packs to a BFLOAT8_B output: the routed multiply
    (as an MLP calls it, with dtype=ttnn.bfloat8_b) is one program, the golden of the operands as the device holds
    them, to the output's precision."""
    _skip_unless_routed(device)
    gate, up = _operands((1, 1, 64, 64)), _operands((1, 1, 64, 64)).flip(-1)
    dtype = ttnn.bfloat16 if case == "bfloat8_b_output" else ttnn.bfloat8_b
    g, u = _on_device(gate, device, dtype), _on_device(up, device, dtype)
    output = ttnn.bfloat16 if case == "bfloat8_b_operands" else ttnn.bfloat8_b
    call = lambda: ttnn.multiply(
        g,
        u,
        dtype=output,
        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)],
        input_tensor_b_activations=[],
    )
    out, operations, arguments = _device_operations(call)
    assert OP_TYPE in arguments and len(operations) == 1, f"launched {operations}, not one {OP_TYPE} program"
    assert out.dtype == output
    golden = _golden(ttnn.to_torch(g), ttnn.to_torch(u))
    if output == ttnn.bfloat16:
        _assert_close(ttnn.to_torch(out), golden)
    else:
        _assert_within_bfloat8_b_blocks(ttnn.to_torch(out), golden)


@pytest.mark.parametrize("case", ["mixed_operands", "float32_operands", "post_activation"])
def test_multiply_keeps_binary_ng_elsewhere(device, case):
    """A multiply outside the route's dtypes or activations launches no GEGLU_TANH program."""
    gate, up = _operands((1, 1, 64, 64)), _operands((1, 1, 64, 64))
    kwargs = dict(
        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)], input_tensor_b_activations=[]
    )
    dtype = ttnn.float32 if case == "float32_operands" else ttnn.bfloat16
    g, u = _on_device(gate, device, dtype), _on_device(up, device, dtype)
    if case == "mixed_operands":
        u = _on_device(up, device, ttnn.bfloat8_b)
    if case == "post_activation":
        kwargs["activations"] = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
    out, _, arguments = _device_operations(lambda: ttnn.multiply(g, u, **kwargs))
    assert OP_TYPE not in arguments
    # The stock path computes the same product: a BFLOAT8_B operand keeps about 7 bits per block.
    golden = _golden(gate, up)
    if case == "post_activation":
        golden = golden.clamp(min=0)
    _assert_pcc(ttnn.to_torch(out), golden, 0.99 if case == "mixed_operands" else 0.999)


@pytest.mark.parametrize("parameter", [0.0, 1.0])
def test_multiply_with_a_mode_parameter_runs_geglu_tanh(device, parameter):
    """GELU_TANH's parameter selects a mode of its one function, which the kernel computes as for the plain call: the
    multiply is the same one GEGLU_TANH program, with the same bits."""
    _skip_unless_routed(device)
    gate, up = _operands((1, 1, 64, 64)), _operands((1, 1, 64, 64)).flip(-1)
    g, u = _on_device(gate, device), _on_device(up, device)
    expected = ttnn.to_torch(
        ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH)])
    )
    call = lambda: ttnn.multiply(
        g, u, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH, parameter)]
    )
    out, operations, arguments = _device_operations(call)
    assert OP_TYPE in arguments and len(operations) == 1, f"launched {operations}, not one {OP_TYPE} program"
    assert torch.equal(ttnn.to_torch(out).view(torch.int16), expected.view(torch.int16))


def test_multiply_with_another_parameter_keeps_binary_ng(device):
    """A parameter value outside the modes the route lists launches no GEGLU_TANH program."""
    gate, up = _operands((1, 1, 64, 64)), _operands((1, 1, 64, 64))
    g, u = _on_device(gate, device), _on_device(up, device)
    activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH, 0.5)
    _, _, arguments = _device_operations(lambda: ttnn.multiply(g, u, input_tensor_a_activations=[activation]))
    assert OP_TYPE not in arguments
