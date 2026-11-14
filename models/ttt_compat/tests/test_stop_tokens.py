# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from models.ttt_compat.tt.stop_tokens import build_stop_tokens


def test_gemma_style_eos_list_adds_end_of_turn():
    assert build_stop_tokens(1, None, [1, 106]) == [1, 106]


def test_single_int_matches_tokenizer_eos():
    assert build_stop_tokens(2, 2, 2) == [2]
    assert build_stop_tokens(2) == [2]


def test_config_and_generation_lists_merge_without_duplicates():
    assert build_stop_tokens(1, [1, 107], [107, 1, 106]) == [1, 107, 106]


def test_phi3_extra_id_is_appended():
    assert build_stop_tokens(2, None, 2, extra=[32007]) == [2, 32007]


def test_none_tokenizer_eos_is_skipped():
    assert build_stop_tokens(None, None, [5]) == [5]
