# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests of the Qwen3-ASR decoder stop logic (no device, no checkpoint weights)."""
import json

from models.demos.audio.qwen3_asr.tt.qwen3_asr_decoder import DEFAULT_EOS_IDS, decode_until_eos, load_eos_ids

EOT, IM_END = 151643, 151645


def _script(tokens):
    """step() that replays `tokens` after the first one."""
    it = iter(tokens)
    return lambda tok, offset: next(it)


def test_load_eos_ids_reads_list(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [EOT, IM_END]}))
    assert load_eos_ids(tmp_path) == (EOT, IM_END)


def test_load_eos_ids_scalar_and_fallbacks(tmp_path):
    assert load_eos_ids(None, tmp_path) == DEFAULT_EOS_IDS  # no file
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": IM_END}))
    assert load_eos_ids(tmp_path) == (IM_END,)
    (tmp_path / "generation_config.json").write_text("{}")
    assert load_eos_ids(tmp_path) == DEFAULT_EOS_IDS  # no eos key
    (tmp_path / "generation_config.json").write_text("not json")
    assert load_eos_ids(tmp_path) == DEFAULT_EOS_IDS  # unreadable


def test_stops_on_im_end_like_before():
    assert decode_until_eos(1, _script([2, IM_END, 9, 9]), 64, (EOT, IM_END)) == [1, 2]


def test_stops_on_endoftext():
    assert decode_until_eos(1, _script([2, EOT, 9, 9]), 64, (EOT, IM_END)) == [1, 2]


def test_endoftext_ignored_when_not_a_stop_id():
    assert decode_until_eos(1, _script([EOT, 3, IM_END]), 64, (IM_END,)) == [1, EOT, 3]


def test_first_token_is_stop():
    assert decode_until_eos(EOT, _script([]), 64, (EOT, IM_END)) == []


def test_max_new_tokens_keeps_non_stop_tail():
    assert decode_until_eos(1, _script([2, 3, 4, 5]), 3, (EOT, IM_END)) == [1, 2, 3]
