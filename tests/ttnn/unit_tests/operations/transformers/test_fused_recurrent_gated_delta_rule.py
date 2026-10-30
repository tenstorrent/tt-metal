# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Sanity subset of the device tests of ttnn.transformer.fused_recurrent_gated_delta_rule: four accuracy cases against
the registered golden, one multi-token versus chained single-token identity case, and two compatibility digests.

The ttnn sanity job ("ttnn misc ops group" in tests/pipeline_reorg/ttnn_sanity_tests.yaml) runs this directory on every
SKU, the ttsim simulators included, which are roughly 100x slower than silicon; this file is sized for that, and the
digests are hardware-only (the simulator's bits are not the silicon's). The full accuracy matrix, the identity matrix,
the determinism, trace and drift checks and every digest run on hardware in the L2 nightly:
tests/ttnn/nightly/unit_tests/operations/transformers/test_fused_recurrent_gated_delta_rule.py. The helpers the two
files share are in gdn_decode_test_utils.py."""

import pytest

from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import (
    assert_compatibility_golden,
    assert_decode_matches_golden,
    assert_multi_token_equals_chained,
    case_id,
    golden_case_id,
    hw_only,
)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="fused_recurrent_gated_delta_rule is Blackhole-only")

# (shape, B, T, with_state): single-token decode and a 3-draft verify at the 27B TP 4 head count, decode at TP 8, and
# the single-chip 9B head count from a zero state.
SANITY_CASES = [
    ("qwen27b_tp4", 1, 1, True),
    ("qwen27b_tp4", 1, 4, True),
    ("qwen27b_tp8", 1, 1, True),
    ("qwen9b_tp1", 1, 1, False),
]

# Decode and an 8-token verify at the 27B TP 4 head count, from a random state.
SANITY_GOLDEN_CASES = [
    ("qwen27b_tp4", 1, 1, True),
    ("qwen27b_tp4", 1, 8, True),
]


@pytest.mark.parametrize("shape, B, T, with_state", [pytest.param(*c, id=case_id(*c)) for c in SANITY_CASES])
def test_decode_matches_golden(device, shape, B, T, with_state):
    """o and the state after every token (T > 1) or the final state (T = 1) against the golden at the accuracy gates."""
    assert_decode_matches_golden(device, shape, B, T, with_state)


def test_multi_token_equals_chained_single_token(device):
    """One call with 4 tokens and per-token states is bit-identical to 4 chained single-token calls at B 1 (the
    speculative decoder's verify path equals its decode path); the K-draft and batch matrix is in the nightly file."""
    assert_multi_token_equals_chained(device, B=1, K_draft=3)


@hw_only
@pytest.mark.parametrize(
    "shape, B, T, with_state", [pytest.param(*c, id=golden_case_id(*c)) for c in SANITY_GOLDEN_CASES]
)
def test_compatibility_goldens(device, shape, B, T, with_state):
    """The op's bits on fixed inputs against the digests stored in fused_recurrent_gdn_goldens.json: a change of the
    arithmetic anywhere in the call shows here even when it stays within the accuracy gates."""
    assert_compatibility_golden(device, shape, B, T, with_state)
