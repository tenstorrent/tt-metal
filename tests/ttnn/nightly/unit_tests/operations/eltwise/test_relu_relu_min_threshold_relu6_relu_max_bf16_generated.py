# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 relu, relu_min, threshold, relu6, relu_max cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import generate_all_bfloat16_bitpatterns


# Exhaustive BF16 accuracy of the unary ops that run in the packer's ReLU stage.
#
# The packer serves sharded tensors only, so the inputs are height-sharded. The reference is torch
# on the operand as BF16 DEST holds it, stored as a BF16 tile holds the result. The ops are exact,
# so every output must equal it bit for bit.
PACK_RELU_SMALLEST_NORMAL = 2.0**-126


# The 256 x 256 input, one row of tiles per core.
PACK_RELU_SHARDED = ttnn.create_sharded_memory_config(
    shape=(32, 256),
    core_grid=ttnn.CoreGrid(y=1, x=8),
    strategy=ttnn.ShardStrategy.HEIGHT,
    use_height_and_width_as_shard_shape=True,
)


PACK_RELU_CLASSES = {
    "pos_nan": lambda t: torch.isnan(t) & ~torch.signbit(t),
    "neg_nan": lambda t: torch.isnan(t) & torch.signbit(t),
    "neg_zero": lambda t: (t == 0) & torch.signbit(t),
    "pos_subnormal": lambda t: (t > 0) & (t < PACK_RELU_SMALLEST_NORMAL),
    "neg_subnormal": lambda t: (t < 0) & (t > -PACK_RELU_SMALLEST_NORMAL),
}


PACK_RELU_VALUES = {
    "pos_inf": float("inf"),
    "neg_inf": float("-inf"),
    "pos_zero": 0.0,
}


# (raw class, class it is held as) pairs: the operand DEST holds and the result a tile stores.
PACK_RELU_DEST_INPUT = (
    ("neg_nan", "neg_inf"),
    ("neg_subnormal", "pos_zero"),
    ("neg_zero", "pos_zero"),
    ("pos_nan", "pos_inf"),
    ("pos_subnormal", "pos_zero"),
)


PACK_RELU_STORED_RESULT = (("neg_zero", "pos_zero"),)


# op: (TT-NN call, torch reference, operand transport, result transport)
PACK_RELU_OPS = {
    "relu": (
        lambda x: ttnn.relu(x),
        lambda x: torch.nn.functional.relu(x),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
    "relu_min": (
        lambda x: ttnn.relu_min(x, 0.0),
        lambda x: torch.clamp_min(x, min=0.0),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
    "threshold": (
        lambda x: ttnn.threshold(x, 0.0, 0.0),
        lambda x: torch.nn.functional.threshold(x, 0.0, 0.0),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
    "relu6": (
        lambda x: ttnn.relu6(x),
        lambda x: torch.nn.functional.relu6(x),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
    "relu_max": (
        lambda x: ttnn.relu_max(x, 6.0),
        lambda x: torch.clamp(x, min=0.0, max=6.0),
        PACK_RELU_DEST_INPUT,
        PACK_RELU_STORED_RESULT,
    ),
}


def _pack_relu_transport(t, source, pairs):
    for raw, held in pairs:
        t = torch.where(PACK_RELU_CLASSES[raw](source), torch.full_like(t, PACK_RELU_VALUES[held]), t)
    return t


@pytest.mark.parametrize("op", list(PACK_RELU_OPS))
def test_pack_relu_exhaustive_bfloat16(op, device):
    run, reference, operand, result = PACK_RELU_OPS[op]
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16).reshape(256, 256)
    expected = reference(_pack_relu_transport(x.clone(), x, operand))
    expected = _pack_relu_transport(expected, expected, result)

    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=PACK_RELU_SHARDED
    )
    actual = ttnn.to_torch(run(tt_x))
    mismatch = actual.view(torch.int16) != expected.view(torch.int16)
    assert not mismatch.any(), (
        f"{op}: {mismatch.sum().item()} mismatches, first at x={x[mismatch][0].item()}: "
        f"expected {expected[mismatch][0].item()}, got {actual[mismatch][0].item()}"
    )
