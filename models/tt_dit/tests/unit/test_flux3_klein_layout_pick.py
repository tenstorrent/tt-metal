# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only: the per-request DiT layout choice of the FLUX 3 Image Klein pipeline (no device)."""
import pytest

from models.tt_dit.pipelines.flux3_klein.pipeline_flux3_klein import LARGE_LAYOUT_MIN_TOKENS, layout_name, pick_layout


def test_layout_names_on_the_4x8_mesh():
    assert layout_name(0, 1) == "sp4tp8" and layout_name(1, 0) == "sp8tp4"
    assert layout_name(0, 1, (8, 4)) == "sp8tp4"


@pytest.mark.parametrize(
    "tokens, expected",
    [  # measured crossover (docs/STAGE_TIMING.md 2026-10-21): equal at 12.4K tokens, SP8xTP4 wins from 16.4K up
        (768 * 768 // 256 + 80, "sp4tp8"),
        (4096 + 80, "sp4tp8"),
        (4096 + 2 * 4096 + 80, "sp4tp8"),
        (LARGE_LAYOUT_MIN_TOKENS - 1, "sp4tp8"),
        (LARGE_LAYOUT_MIN_TOKENS, "sp8tp4"),
        (16384 + 80, "sp8tp4"),
        (65536 + 10 * 4096 + 320, "sp8tp4"),
    ],
)
def test_large_layout_from_16k_tokens(tokens, expected):
    assert pick_layout(tokens, primary="sp4tp8", large="sp8tp4") == expected


def test_single_layout_pipelines_never_switch():
    assert pick_layout(1 << 20, primary="sp4tp8", large=None) == "sp4tp8"
    assert pick_layout(10, primary="sp8tp4", large=None) == "sp8tp4"
    assert pick_layout(10, primary="sp4tp8", large="sp8tp4", min_tokens=0) == "sp8tp4"
