# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive BF16 accuracy of the unary ops that run in the packer's ReLU stage.

The packer serves sharded tensors only, so the inputs are height-sharded. The reference is torch
on the operand as BF16 DEST holds it, stored as a BF16 tile holds the result. The ops are exact,
so every output must equal it bit for bit. Each op logs one ULP line: the largest ULP error
against the reference and the outputs of another class, for the packer and for the SFPU kernel it
replaces, run on the same input as a chain of the op and IDENTITY (see the device-perf test).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import generate_all_bfloat16_bitpatterns

SMALLEST_NORMAL = 2.0**-126
# The 256 x 256 input, one row of tiles per core.
SHARDED = ttnn.create_sharded_memory_config(
    shape=(32, 256),
    core_grid=ttnn.CoreGrid(y=1, x=8),
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)

CLASSES = {
    "pos_nan": lambda t: torch.isnan(t) & ~torch.signbit(t),
    "neg_nan": lambda t: torch.isnan(t) & torch.signbit(t),
    "neg_zero": lambda t: (t == 0) & torch.signbit(t),
    "pos_subnormal": lambda t: (t > 0) & (t < SMALLEST_NORMAL),
    "neg_subnormal": lambda t: (t < 0) & (t > -SMALLEST_NORMAL),
}
VALUES = {
    "pos_inf": float("inf"),
    "neg_inf": float("-inf"),
    "pos_zero": 0.0,
}

# (raw class, class it is held as) pairs: the operand DEST holds and the result a tile stores.
DEST_INPUT = (
    ("neg_nan", "neg_inf"),
    ("neg_subnormal", "pos_zero"),
    ("neg_zero", "pos_zero"),
    ("pos_nan", "pos_inf"),
    ("pos_subnormal", "pos_zero"),
)
STORED_RESULT = (("neg_zero", "pos_zero"),)

# op: (TT-NN call, torch reference, operand transport, result transport)
OPS = {
    "relu": (
        lambda x: ttnn.relu(x),
        lambda x: torch.nn.functional.relu(x),
        DEST_INPUT,
        STORED_RESULT,
    ),
    "relu_min": (
        lambda x: ttnn.relu_min(x, 0.0),
        lambda x: torch.clamp_min(x, min=0.0),
        DEST_INPUT,
        STORED_RESULT,
    ),
    "threshold": (
        lambda x: ttnn.threshold(x, 0.0, 0.0),
        lambda x: torch.nn.functional.threshold(x, 0.0, 0.0),
        DEST_INPUT,
        STORED_RESULT,
    ),
    "relu6": (
        lambda x: ttnn.relu6(x),
        lambda x: torch.nn.functional.relu6(x),
        DEST_INPUT,
        STORED_RESULT,
    ),
    "relu_max": (
        lambda x: ttnn.relu_max(x, 6.0),
        lambda x: torch.clamp(x, min=0.0, max=6.0),
        DEST_INPUT,
        STORED_RESULT,
    ),
}

# op: TT-NN's SFPU kernel for the op, which the packer replaces.
OLD = {
    "relu": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
    "relu_min": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 0.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
    "threshold": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.THRESHOLD, 0.0, 0.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
    "relu6": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
    "relu_max": lambda x: ttnn.unary_chain(
        x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MAX, 6.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
    ),
}


def _transport(t, source, pairs):
    for raw, held in pairs:
        t = torch.where(CLASSES[raw](source), torch.full_like(t, VALUES[held]), t)
    return t


def _stored_classes(t):
    """0 +inf (or NaN), 1 -inf, 2 zero of either sign, 3 finite nonzero."""
    t = t.to(torch.float64)
    classes = torch.full(t.shape, 3, dtype=torch.int8)
    classes[t == 0] = 2
    classes[(t == float("inf")) | torch.isnan(t)] = 0
    classes[t == float("-inf")] = 1
    return classes


def _versus(expected, output):
    """The largest ULP error against ``expected`` over outputs of its class, and the outputs of another class."""
    e, o = expected.to(torch.float64), output.to(torch.float64)
    same = _stored_classes(e) == _stored_classes(o)
    finite = same & torch.isfinite(e) & (e != 0)
    exponent = torch.floor(torch.log2(torch.where(finite, e.abs(), torch.ones_like(e))))
    ulp = torch.where(finite, (e - o).abs() / 2.0 ** (exponent.clamp(min=-126) - 7), torch.zeros_like(e))
    return ulp.max().item(), int((~same).sum())


@pytest.mark.parametrize("op", list(OPS))
def test_pack_relu_exhaustive_bfloat16(op, device):
    run, reference, operand, result = OPS[op]
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16).reshape(256, 256)
    expected = reference(_transport(x.clone(), x, operand))
    expected = _transport(expected, expected, result)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=SHARDED)
    actual = ttnn.to_torch(run(tt_x))
    stock = ttnn.to_torch(OLD[op](tt_x))

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    (ours, ours_classes), (old, old_classes) = _versus(expected, actual), _versus(expected, stock)
    print(
        f"ULP {op} {board} ours={ours:.3f} stock={old:.3f} "
        f"ours_class_mismatches={ours_classes} stock_class_mismatches={old_classes}"
    )
    mismatch = actual.view(torch.int16) != expected.view(torch.int16)
    assert not mismatch.any(), (
        f"{op}: {mismatch.sum().item()} mismatches, first at x={x[mismatch][0].item()}: "
        f"expected {expected[mismatch][0].item()}, got {actual[mismatch][0].item()}"
    )
