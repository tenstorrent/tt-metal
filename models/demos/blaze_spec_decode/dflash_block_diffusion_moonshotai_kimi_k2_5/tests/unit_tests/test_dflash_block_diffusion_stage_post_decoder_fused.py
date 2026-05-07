# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from models.demos.blaze_spec_decode.dflash_block_diffusion_moonshotai_kimi_k2_5.tests.unit_tests.dflash_stage_fake_ops import (
    assert_close,
    fake_post_decoder_fused_stage,
    load_stage_fixture,
)


FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "dflash_block_diffusion"
    / "stage_post_decoder_fused.json"
)


def test_dflash_block_diffusion_stage_post_decoder_fused_matches_golden_fixture() -> None:
    fixture = load_stage_fixture(FIXTURE_PATH, "post_decoder_fused")
    actual = fake_post_decoder_fused_stage(fixture)
    expected = fixture["expected"]

    assert fixture["mapping"] == "DFlash final norm, drafter lmhead/sampling, and host packet construction"
    assert_close(actual["final_hidden"], expected["final_hidden"])
    assert_close(actual["draft_logits"], expected["draft_logits"])
    assert actual["draft_token_ids"].tolist() == expected["draft_token_ids"]
    assert actual["host_packet"] == expected["host_packet"]
