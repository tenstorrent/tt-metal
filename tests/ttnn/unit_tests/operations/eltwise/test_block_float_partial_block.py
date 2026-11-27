# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

# A BFLOAT8_B / BFLOAT4_B tile shares one exponent per 16-lane block of a row. At these widths the last block of each
# row holds real lanes and padding, and an op whose result at the zero padding is inf must not zero the real lanes.
# W=48 and W=64 are block aligned and are the controls.
WIDTHS = [1, 33, 40, 50, 63, 48, 64]

UNARY_OPS = {
    "reciprocal": (ttnn.reciprocal, torch.reciprocal),
    "rsqrt": (ttnn.rsqrt, torch.rsqrt),
    "log": (ttnn.log, torch.log),
    "log2": (ttnn.log2, torch.log2),
    "log10": (ttnn.log10, torch.log10),
}

# (atol, rtol) against the exact result of the stored inputs; the bfloat4_b bound is the error of the aligned controls.
TOLERANCE = {ttnn.bfloat8_b: (0.04, 0.04), ttnn.bfloat4_b: (0.2, 0.3)}


def _to_device(torch_tensor, dtype, device):
    return ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _assert_real_lanes_kept(result, expected, dtype):
    assert (result != 0).all(), f"{(result == 0).sum()} of {result.numel()} real lanes came back as 0"
    atol, rtol = TOLERANCE[dtype]
    torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfloat8_b", "bfloat4_b"])
@pytest.mark.parametrize("width", WIDTHS, ids=lambda w: f"W={w}")
@pytest.mark.parametrize("op_name", list(UNARY_OPS))
def test_unary_block_float_partial_block_ignores_padding(device, op_name, width, dtype):
    ttnn_op, torch_op = UNARY_OPS[op_name]
    torch.manual_seed(0)
    torch_input = torch.empty(2, 32, width).uniform_(2.0, 4.0)
    input_tensor = _to_device(torch_input, dtype, device)

    result = ttnn.to_torch(ttnn_op(input_tensor)).float()
    expected = torch_op(ttnn.to_torch(input_tensor).float())

    _assert_real_lanes_kept(result, expected, dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfloat8_b", "bfloat4_b"])
@pytest.mark.parametrize("width", WIDTHS, ids=lambda w: f"W={w}")
def test_div_block_float_partial_block_ignores_padding(device, width, dtype):
    torch.manual_seed(0)
    torch_b = torch.empty(2, 32, width).uniform_(2.0, 4.0)
    a = _to_device(torch.ones(2, 32, width), dtype, device)
    b = _to_device(torch_b, dtype, device)

    result = ttnn.to_torch(ttnn.div(a, b)).float()
    expected = 1.0 / ttnn.to_torch(b).float()

    _assert_real_lanes_kept(result, expected, dtype)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfloat8_b", "bfloat4_b"])
@pytest.mark.parametrize("width", [33, 50, 64], ids=lambda w: f"W={w}")
@pytest.mark.parametrize("op_name", ["remainder", "fmod"])
def test_mod_block_float_partial_block_ignores_padding(device, op_name, width, dtype):
    torch.manual_seed(0)
    a = _to_device(torch.empty(2, 32, width).uniform_(5.0, 9.0), dtype, device)
    b = _to_device(torch.empty(2, 32, width).uniform_(2.0, 4.0), dtype, device)

    result = ttnn.to_torch(getattr(ttnn, op_name)(a, b)).float()
    expected = getattr(torch, op_name)(ttnn.to_torch(a).float(), ttnn.to_torch(b).float())

    # A result can be exactly 0 here, so only the tolerance is checked.
    atol, rtol = TOLERANCE[dtype]
    torch.testing.assert_close(result, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("width", [33, 50], ids=lambda w: f"W={w}")
def test_unary_block_float_partial_block_preallocated_output(device, width):
    torch.manual_seed(0)
    input_tensor = _to_device(torch.empty(32, width).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)
    output_tensor = _to_device(torch.zeros(32, width), ttnn.bfloat8_b, device)

    ttnn.reciprocal(input_tensor, output_tensor=output_tensor)

    result = ttnn.to_torch(output_tensor).float()
    expected = torch.reciprocal(ttnn.to_torch(input_tensor).float())
    _assert_real_lanes_kept(result, expected, ttnn.bfloat8_b)


@pytest.mark.parametrize("exponent", [-1, -2, -1.0], ids=["int-1", "int-2", "float-1"])
@pytest.mark.parametrize("width", [33, 50], ids=lambda w: f"W={w}")
def test_pow_negative_exponent_block_float_partial_block(device, width, exponent):
    torch.manual_seed(0)
    input_tensor = _to_device(torch.empty(2, 32, width).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)

    result = ttnn.to_torch(ttnn.pow(input_tensor, exponent)).float()
    expected = torch.pow(ttnn.to_torch(input_tensor).float(), exponent)

    _assert_real_lanes_kept(result, expected, ttnn.bfloat8_b)


@pytest.mark.parametrize("eps", [None, 0.0], ids=["eps=None", "eps=0"])
@pytest.mark.parametrize("width", [33, 50, 64], ids=lambda w: f"W={w}")
def test_logit_block_float_partial_block(device, width, eps):
    torch.manual_seed(0)
    # Away from 0.5, where logit is 0 and would read as a dropped lane.
    input_tensor = _to_device(torch.empty(2, 32, width).uniform_(0.6, 0.9), ttnn.bfloat8_b, device)

    result = ttnn.to_torch(ttnn.logit(input_tensor, eps=eps)).float()
    expected = torch.logit(ttnn.to_torch(input_tensor).float())

    _assert_real_lanes_kept(result, expected, ttnn.bfloat8_b)


@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["out_bfloat8_b", "out_bfloat16"])
@pytest.mark.parametrize("width", [1, 33, 50, 64], ids=lambda w: f"W={w}")
def test_fast_div_block_float_partial_block_input(device, width, output_dtype):
    """Fast DIV multiplies by a reciprocal of b packed in b's format, so b's padding must not reach that pack."""
    torch.manual_seed(0)
    a = _to_device(torch.empty(2, 32, width).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)
    b = _to_device(torch.empty(2, 32, width).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)

    result = ttnn.to_torch(ttnn.div(a, b, fast_and_approximate_mode=True, dtype=output_dtype)).float()
    expected = ttnn.to_torch(a).float() / ttnn.to_torch(b).float()

    _assert_real_lanes_kept(result, expected, ttnn.bfloat8_b)


def _traced_ops(fn):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    result = fn()
    calltrace = ttnn.graph.extract_calltrace(ttnn.graph.end_graph_capture())
    return result, calltrace


@pytest.mark.parametrize("width", [33, 50], ids=lambda w: f"W={w}")
def test_accurate_div_block_float_partial_block_input_not_widened(device, width):
    """Only fast DIV packs a reciprocal of b, so an explicit fast_and_approximate_mode=False must not widen b."""
    torch.manual_seed(0)
    a = _to_device(torch.empty(2, 32, width).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)
    b = _to_device(torch.empty(2, 32, width).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)

    unset, unset_ops = _traced_ops(lambda: ttnn.div(a, b))
    accurate, accurate_ops = _traced_ops(lambda: ttnn.div(a, b, fast_and_approximate_mode=False))
    _, fast_ops = _traced_ops(lambda: ttnn.div(a, b, fast_and_approximate_mode=True))

    assert accurate_ops.count("TypecastDeviceOperation") == unset_ops.count("TypecastDeviceOperation")
    assert fast_ops.count("TypecastDeviceOperation") > accurate_ops.count("TypecastDeviceOperation")
    expected = ttnn.to_torch(a).float() / ttnn.to_torch(b).float()
    _assert_real_lanes_kept(ttnn.to_torch(accurate).float(), expected, ttnn.bfloat8_b)
    _assert_real_lanes_kept(ttnn.to_torch(unset).float(), expected, ttnn.bfloat8_b)


@pytest.mark.parametrize("side", ["lhs", "rhs"])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["out_bfloat8_b", "out_bfloat16"])
@pytest.mark.parametrize("width", [33, 50, 64], ids=lambda w: f"W={w}")
def test_binary_input_activation_block_float_partial_block(device, width, output_dtype, side):
    torch.manual_seed(0)
    a = _to_device(torch.empty(2, 32, width).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)
    b = _to_device(torch.empty(2, 32, width).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)
    activation = {f"input_tensor_{'a' if side == 'lhs' else 'b'}_activations": [ttnn.UnaryOpType.RECIP]}

    result = ttnn.to_torch(ttnn.multiply(a, b, dtype=output_dtype, **activation)).float()
    torch_a, torch_b = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
    expected = (1.0 / torch_a) * torch_b if side == "lhs" else torch_a / torch_b

    _assert_real_lanes_kept(result, expected, ttnn.bfloat8_b)


def test_div_block_float_partial_block_sharded_preallocated_output(device):
    """Interleaved inputs into a preallocated sharded output: the BFLOAT16 intermediate takes the output's layout."""
    torch.manual_seed(0)
    shape = (1, 1, 64, 50)
    sharded = ttnn.create_sharded_memory_config(
        shape=(32, 64),
        core_grid=ttnn.CoreGrid(y=1, x=2),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    a = _to_device(torch.ones(shape), ttnn.bfloat8_b, device)
    b = _to_device(torch.empty(shape).uniform_(2.0, 4.0), ttnn.bfloat8_b, device)
    output_tensor = ttnn.from_torch(
        torch.zeros(shape), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded
    )

    ttnn.div(a, b, output_tensor=output_tensor)

    assert output_tensor.memory_config() == sharded
    result = ttnn.to_torch(output_tensor).float()
    _assert_real_lanes_kept(result, 1.0 / ttnn.to_torch(b).float(), ttnn.bfloat8_b)
