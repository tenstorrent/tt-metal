# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from models.demos.blaze_spec_decode.dflash_block_diffusion_moonshotai_kimi_k2_5.tests.unit_tests.dflash_golden_ops import (
    assert_close,
    golden_combined_drafter,
    load_stage_fixture,
)


FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "dflash_block_diffusion"
FIXTURE_PATH = FIXTURE_DIR / "stage_combined_drafter.json"


def test_dflash_block_diffusion_combined_drafter_matches_golden_fixture() -> None:
    fixture = load_stage_fixture(FIXTURE_PATH, "combined_drafter")
    actual = golden_combined_drafter(fixture, FIXTURE_DIR)
    expected = fixture["expected"]

    assert fixture["mapping"] == (
        "full DFlash drafter pipeline: pre-decoder, decoder layers, and post-decoder packet emit"
    )
    assert_close(actual["final_hidden"], expected["final_hidden"])
    assert_close(actual["draft_logits"], expected["draft_logits"])
    assert actual["draft_token_ids"].tolist() == expected["draft_token_ids"]
    assert actual["host_packet"] == expected["host_packet"]


def test_dflash_block_diffusion_combined_drafter_fits_one_blackhole_galaxy() -> None:
    fixture = load_stage_fixture(FIXTURE_PATH, "combined_drafter")

    assert fixture["device_size_choice"] == {
        "smallest_viable": "one_blackhole_galaxy",
        "stage_slots": 4,
        "stage_shape": "4x2",
        "mapped_stage_count": 4,
        "fits_one_galaxy": True,
        "reason": "pre-decoder fused + two drafter decoder layers + post-decoder fused equals four stages",
    }
