# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_golden_ops import (
    golden_combined_drafter_stages,
)
from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_test_modes import (
    USE_GOLDEN_PARAMS,
    require_golden_or_not_implemented,
)
from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_test_utils import (
    assert_stage_sequence_outputs_match,
    expected_combined_drafter_outputs,
    load_validated_dflash_reference,
)


def _run_golden_combined_drafter(reference: dict) -> list[dict]:
    return golden_combined_drafter_stages(reference)


def _run_device_combined_drafter(reference: dict) -> list[dict]:
    del reference
    require_golden_or_not_implemented(False, "DFlash combined drafter device pipeline")
    raise AssertionError("unreachable")


def _run_combined_drafter(reference: dict, *, use_golden: bool) -> list[dict]:
    if use_golden:
        return _run_golden_combined_drafter(reference)
    return _run_device_combined_drafter(reference)


@pytest.mark.parametrize("use_golden", USE_GOLDEN_PARAMS)
def test_dflash_combined_drafter_matches_golden_reference(use_golden: bool) -> None:
    reference = load_validated_dflash_reference()
    expected = expected_combined_drafter_outputs(reference)

    actual = _run_combined_drafter(reference, use_golden=use_golden)

    assert_stage_sequence_outputs_match(
        actual,
        expected,
        tensor_fields=("final_hidden", "draft_logits", "draft_token_ids"),
        exact_fields=("host_packet",),
    )
