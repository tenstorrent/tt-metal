# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Structural checks of the FlashKDA precision emulation. CPU only."""

import torch

from models.demos.deepseek_v3_d_p.reference.kda.ops import kda_recurrent_reference
from models.demos.deepseek_v3_d_p.tests.kda.accuracy_metric import slice_nrmse
from models.demos.deepseek_v3_d_p.tests.kda.edge_inputs import recurrence_edge_case
from models.demos.deepseek_v3_d_p.tests.kda.flashkda_emulation import flashkda_recurrence

SHAPE = {"heads": 2, "key_dim": 32, "value_dim": 16}
KEYS = ("q", "k", "v", "gate", "beta", "initial_state")


def test_state_is_stored_in_bf16_after_every_chunk() -> None:
    case = recurrence_edge_case("bounded", 48, **SHAPE)
    _, state = flashkda_recurrence(*(case[key] for key in KEYS))
    assert torch.equal(state, state.bfloat16().float())


def test_partial_last_chunk_matches_the_prefix_of_a_longer_sequence() -> None:
    long = recurrence_edge_case("bounded", 32, **SHAPE)
    output_long, _ = flashkda_recurrence(*(long[key] for key in KEYS))
    output_short, _ = flashkda_recurrence(*(long[key][:, :17] if key != "initial_state" else long[key] for key in KEYS))
    torch.testing.assert_close(output_short, output_long[:, :17], rtol=0, atol=1e-6)


def test_bf16_state_is_the_dominant_error_and_grows_with_length() -> None:
    errors = []
    for sequence in (64, 512):
        case = recurrence_edge_case("long_memory", sequence, **SHAPE)
        golden, _ = kda_recurrent_reference(*(case[key] for key in KEYS), dtype=torch.float64)
        emulated, _ = flashkda_recurrence(*(case[key] for key in KEYS))
        errors.append(float(slice_nrmse(emulated, golden, keep_dims=())))
    # A BF16 state carries ~3 significant digits; FP32 arithmetic alone would sit near 1e-6.
    assert 1e-4 < errors[0] < errors[1] < 1e-1, errors
