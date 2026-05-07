# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from models.demos.blaze_spec_decode.dflash_block_diffusion_moonshotai_kimi_k2_5.tests.unit_tests.dflash_golden_ops import (
    assert_close,
    golden_pre_decoder_fused_stage,
    load_stage_fixture,
)
from models.demos.blaze_spec_decode.dflash_block_diffusion_moonshotai_kimi_k2_5.tests.unit_tests.dflash_test_modes import (
    USE_GOLDEN_PARAMS,
    require_golden_or_not_implemented,
)


FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "dflash_block_diffusion"
    / "stage_pre_decoder_fused.json"
)


@pytest.mark.parametrize("use_golden", USE_GOLDEN_PARAMS)
def test_dflash_block_diffusion_stage_pre_decoder_fused_matches_golden_fixture(use_golden: bool) -> None:
    fixture = load_stage_fixture(FIXTURE_PATH, "pre_decoder_fused")
    require_golden_or_not_implemented(use_golden, "DFlash pre-decoder fused device stage")
    actual = golden_pre_decoder_fused_stage(fixture)
    expected = fixture["expected"]

    assert fixture["mapping"] == (
        "base-model norm/lmhead/sampling plus DFlash target-hidden projection, noise block, and RoPE prep"
    )
    assert_close(actual["base_logits"], expected["base_logits"])
    assert actual["base_token_id"] == expected["base_token_id"]
    assert_close(actual["target_context"], expected["target_context"])
    assert_close(actual["position_cos"], expected["position_cos"])
    assert_close(actual["position_sin"], expected["position_sin"])
    assert_close(actual["decoder_input"], expected["decoder_input"])
