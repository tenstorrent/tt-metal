# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive BF16 accuracy of the unary ops that run in the packer's ReLU stage.

The reference is torch on the operand as BF16 DEST holds it, stored as a BF16 tile holds the
result. The ops are exact, so every output must equal it bit for bit.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import generate_all_bfloat16_bitpatterns

SMALLEST_NORMAL = 2.0**-126

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


def _transport(t, source, pairs):
    for raw, held in pairs:
        t = torch.where(CLASSES[raw](source), torch.full_like(t, VALUES[held]), t)
    return t


@pytest.mark.parametrize("op", list(OPS))
def test_pack_relu_exhaustive_bfloat16(op, device):
    run, reference, operand, result = OPS[op]
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16).reshape(256, 256)
    expected = reference(_transport(x.clone(), x, operand))
    expected = _transport(expected, expected, result)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(run(tt_x))

    mismatch = actual.view(torch.int16) != expected.view(torch.int16)
    assert not mismatch.any(), (
        f"{op}: {mismatch.sum().item()} mismatches, first at x={x[mismatch][0].item()}: "
        f"expected {expected[mismatch][0].item()}, got {actual[mismatch][0].item()}"
    )
