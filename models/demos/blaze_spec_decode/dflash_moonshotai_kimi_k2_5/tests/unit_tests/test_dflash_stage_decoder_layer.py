# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_golden_ops import (
    golden_decoder_layer_stages,
)
from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_test_modes import (
    USE_GOLDEN_PARAMS,
    require_golden_or_not_implemented,
)
from models.demos.blaze_spec_decode.dflash_moonshotai_kimi_k2_5.tests.unit_tests.dflash_test_utils import (
    assert_stage_sequence_outputs_match,
    expected_decoder_layer_outputs,
    load_validated_dflash_reference,
)


def _run_golden_decoder_layer_stage(reference: dict) -> list[dict]:
    return golden_decoder_layer_stages(reference)


def _run_device_decoder_layer_stage(reference: dict) -> list[dict]:
    del reference
    require_golden_or_not_implemented(False, "DFlash decoder layer device stages")
    raise AssertionError("unreachable")


def _run_decoder_layer_stage(reference: dict, *, use_golden: bool) -> list[dict]:
    if use_golden:
        return _run_golden_decoder_layer_stage(reference)
    return _run_device_decoder_layer_stage(reference)


@pytest.mark.parametrize("use_golden", USE_GOLDEN_PARAMS)
def test_dflash_stage_decoder_layer_matches_golden_reference(use_golden: bool) -> None:
    reference = load_validated_dflash_reference()
    expected = expected_decoder_layer_outputs(reference)

    actual = _run_decoder_layer_stage(reference, use_golden=use_golden)

    assert_stage_sequence_outputs_match(actual, expected, tensor_fields=("hidden_states",))
