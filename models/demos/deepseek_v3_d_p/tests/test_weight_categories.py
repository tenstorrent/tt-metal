# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests of the prefill runner's weight-category selection (no device, no checkpoint)."""

import pytest

from models.demos.deepseek_v3_d_p.tt.runners import weight_categories as wc


def test_parse_selects_named_categories():
    assert wc.parse_weight_categories("moe_gate, lm_head") == {wc.MOE_GATE, wc.LM_HEAD}
    assert wc.parse_weight_categories("moe_gate,lm_head,") == {wc.MOE_GATE, wc.LM_HEAD}


def test_parse_empty_selects_nothing():
    assert wc.parse_weight_categories("") == frozenset()
    assert wc.parse_weight_categories(None) == frozenset()


def test_parse_all_selects_everything():
    assert wc.parse_weight_categories("ALL") == frozenset(wc.CATEGORIES)


def test_parse_rejects_unknown_category():
    with pytest.raises(ValueError, match="unknown categories"):
        wc.parse_weight_categories("moe_gate,gate")


def test_check_selected_passes_when_everything_missing_is_selected():
    missing = {wc.MOE_GATE: [3, 4], wc.LM_HEAD: [-1]}
    assert wc.check_selected(missing, frozenset({wc.MOE_GATE, wc.LM_HEAD, wc.MTP}), "/cache") == missing


def test_check_selected_raises_naming_the_uncovered_category_and_the_fix():
    missing = {wc.MOE_GATE: [3, 4, 5], wc.ROUTED_EXPERTS: [3]}
    with pytest.raises(RuntimeError) as err:
        wc.check_selected(missing, frozenset({wc.MOE_GATE}), "/cache")
    message = str(err.value)
    assert "routed_experts (layers 3..3, 1 total)" in message
    assert "moe_gate (layers" not in message.split("These were not selected")[0]  # selected ones are not blamed
    assert f"{wc.WEIGHT_CATEGORIES_ENV}=moe_gate,routed_experts" in message


def test_format_missing_summarises_layer_ranges():
    assert (
        wc.format_missing({wc.EMBEDDING: [-1], wc.MOE_GATE: [3, 4, 5]}) == "embedding, moe_gate (layers 3..5, 3 total)"
    )


def test_chunk_size_env_selects_chunk_and_default_is_one_place(monkeypatch):
    from models.demos.deepseek_v3_d_p.utils import chunk_config

    monkeypatch.delenv("PREFILL_CHUNK_SIZE", raising=False)
    assert chunk_config.prefill_chunk_tokens() == chunk_config.DEFAULT_PREFILL_CHUNK_TOKENS
    monkeypatch.setenv("PREFILL_CHUNK_SIZE", "2560")
    assert chunk_config.prefill_chunk_tokens() == 2560
    monkeypatch.setenv("PREFILL_CHUNK_SIZE", "")
    assert chunk_config.prefill_chunk_tokens() == chunk_config.DEFAULT_PREFILL_CHUNK_TOKENS
