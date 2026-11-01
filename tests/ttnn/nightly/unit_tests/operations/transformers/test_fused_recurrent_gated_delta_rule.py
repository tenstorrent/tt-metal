# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests of ttnn.transformer.fused_recurrent_gated_delta_rule against its registered golden over the full
accuracy matrix: the served per-device head counts (HV 6, 8, 12, 24, 32 with their key-head counts), the batch widths
within the grid, the token counts of decode and of speculative verify with 3, 7 and 11 draft tokens, with and without
an initial state.

Runs on hardware in the L2 nightly ("ttnn nightly transformers tests" in tests/pipeline_reorg/ops_unit_tests.yaml). The
sanity subset that gates every PR, on the simulators too, is
tests/ttnn/unit_tests/operations/transformers/test_fused_recurrent_gated_delta_rule.py; the helpers the two files share
are in tests/ttnn/unit_tests/operations/transformers/gdn_decode_test_utils.py."""

import pytest

from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import (
    accuracy_cases,
    assert_decode_matches_golden,
    case_id,
)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="fused_recurrent_gated_delta_rule is Blackhole-only")


@pytest.mark.parametrize("shape, B, T, with_state", [pytest.param(*c, id=case_id(*c)) for c in accuracy_cases()])
def test_decode_matches_golden(device, shape, B, T, with_state):
    """o and the state after every token (T > 1) or the final state (T = 1) against the golden at the accuracy gates."""
    assert_decode_matches_golden(device, shape, B, T, with_state)
