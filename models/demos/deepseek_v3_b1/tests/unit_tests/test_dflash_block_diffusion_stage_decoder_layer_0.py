# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from models.demos.deepseek_v3_b1.tests.unit_tests.dflash_stage_fake_ops import (
    assert_close,
    fake_decoder_layer_stage,
    load_stage_fixture,
)


FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "dflash_block_diffusion"
    / "stage_decoder_layer_0.json"
)


def test_dflash_block_diffusion_stage_decoder_layer_0_matches_golden_fixture() -> None:
    fixture = load_stage_fixture(FIXTURE_PATH, "decoder_layer_0")
    actual = fake_decoder_layer_stage(fixture)

    assert fixture["mapping"] == "one DFlash drafter decoder layer stage"
    assert_close(actual, fixture["expected"]["hidden_states"])
