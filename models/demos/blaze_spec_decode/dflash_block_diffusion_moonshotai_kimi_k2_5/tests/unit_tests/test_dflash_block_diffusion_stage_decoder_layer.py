# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from models.demos.blaze_spec_decode.dflash_block_diffusion_moonshotai_kimi_k2_5.tests.unit_tests.dflash_golden_ops import (
    assert_close,
    golden_decoder_layer_stage,
    load_stage_fixture,
)
from models.demos.blaze_spec_decode.dflash_block_diffusion_moonshotai_kimi_k2_5.tests.unit_tests.dflash_test_modes import (
    USE_GOLDEN_PARAMS,
    require_golden_or_not_implemented,
)


FIXTURE_DIR = (
    Path(__file__).parents[1]
    / "fixtures"
    / "dflash_block_diffusion"
)
DECODER_LAYER_FIXTURES = tuple(sorted(FIXTURE_DIR.glob("stage_decoder_layer_*.json")))
assert DECODER_LAYER_FIXTURES, f"No DFlash decoder layer fixtures found in {FIXTURE_DIR}"


def _layer_index_from_fixture_path(path: Path) -> int:
    prefix = "stage_decoder_layer_"
    suffix = ".json"
    assert path.name.startswith(prefix) and path.name.endswith(suffix)
    return int(path.name[len(prefix) : -len(suffix)])


@pytest.mark.parametrize("fixture_path", DECODER_LAYER_FIXTURES, ids=lambda path: path.stem)
@pytest.mark.parametrize("use_golden", USE_GOLDEN_PARAMS)
def test_dflash_block_diffusion_stage_decoder_layer_matches_golden_fixture(
    fixture_path: Path, use_golden: bool
) -> None:
    layer_idx = _layer_index_from_fixture_path(fixture_path)
    fixture = load_stage_fixture(fixture_path, f"decoder_layer_{layer_idx}")
    require_golden_or_not_implemented(use_golden, f"DFlash decoder layer {layer_idx} device stage")
    actual = golden_decoder_layer_stage(fixture)

    num_layers = int(fixture["config"]["num_hidden_layers"])
    assert 0 <= layer_idx < num_layers
    assert fixture["mapping"] == "one DFlash drafter decoder layer stage"
    assert_close(actual, fixture["expected"]["hidden_states"])
