# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks that requests past the KV cache and page-table capacity are rejected."""

from types import SimpleNamespace

import pytest
import torch

from models.ttt_compat.tt import generator as generator_module
from models.ttt_compat.tt.common import preprocess_inputs_prefill
from models.ttt_compat.tt.generator import Generator


class _Args:
    max_seq_len = 4096
    max_context_len = 131072

    def encode_prompt(self, prompt, instruct=False):
        return list(range(prompt))


def _preprocess(lens, gen=200):
    return preprocess_inputs_prefill(
        lens, None, [_Args()], instruct=False, max_generated_tokens=gen, max_prefill_len=4096
    )


def test_uniform_long_batch_is_clipped_as_before():
    _, encoded, decoding_pos, _ = _preprocess([3900, 3900])
    assert [len(e) for e in encoded] == [3896, 3896]


def test_mixed_batch_that_fits_is_unchanged():
    _, encoded, _, _ = _preprocess([100, 3800])
    assert [len(e) for e in encoded] == [100, 3800]


def test_mixed_batch_longest_prompt_without_room_for_generation_is_rejected(expect_error):
    # The shortest prompt fits, so nothing is clipped; the 3900-token prompt plus 200 new tokens needs 4100 > 4096.
    with expect_error(AssertionError, "max_generated_tokens"):
        _preprocess([100, 3900])


def _generator(max_seq_len=512):
    gen = object.__new__(Generator)
    gen.data_parallel = 1  # Generator.__del__ reads it; object.__new__ skips __init__
    gen.model = [SimpleNamespace(configuration=SimpleNamespace(max_seq_len=max_seq_len))]
    return gen


def test_vision_generate_rejects_prompt_plus_generation_past_max_seq_len(expect_error):
    # 400-token prompt + 200 new tokens at max_seq_len 512 decodes up to position 598.
    with expect_error(ValueError, "exceeds max_seq_len"):
        next(_generator().generate(None, None, list(range(400)), max_gen_len=200))


def test_vision_max_gen_len_is_capped_to_the_remaining_room():
    gen = _generator()
    assert gen._cap_max_gen_len(200, 100) == 200
    assert gen._cap_max_gen_len(511, 400) == 112
    assert gen._cap_max_gen_len(112, 400) == 112


def test_vision_prompt_filling_the_cache_is_rejected(expect_error):
    with expect_error(ValueError, "no room"):
        _generator()._cap_max_gen_len(10, 512)


def _kv(block_size=64):
    return [[SimpleNamespace(shape=(128, 1, block_size, 32))]]


@pytest.mark.parametrize("prompt_len", [1500, 3000])
def test_page_table_row_narrower_than_the_prompt_is_rejected(prompt_len, expect_error):
    pages = torch.arange(100, 116, dtype=torch.int32).reshape(1, -1)  # 16 blocks of 64 = 1024 tokens
    with expect_error(ValueError, "K/V"):
        Generator._get_prefill_user_page_table(None, pages, _kv(), prompt_len, prefill_seq_len=2048)


def test_page_table_row_wide_enough_is_unchanged():
    pages = torch.arange(100, 116, dtype=torch.int32).reshape(1, -1)
    prepared = Generator._get_prefill_user_page_table(None, pages, _kv(), 1000, prefill_seq_len=1024)
    torch.testing.assert_close(prepared, pages)


def test_page_table_row_sized_for_the_real_prompt_is_accepted_when_prefill_is_padded():
    # A 100-token prompt owns 2 blocks of 64; its padded prefill runs 256 tokens (4 blocks). The row only holds the
    # blocks vLLM allocated for the real prompt, so the check must use the real length and -1 pad the tail.
    pages = torch.arange(100, 102, dtype=torch.int32).reshape(1, -1)
    prepared = Generator._get_prefill_user_page_table(None, pages, _kv(), 100, prefill_seq_len=256)
    assert prepared.tolist() == [[100, 101, -1, -1]]


def test_prompt_row_handles_nested_and_flat_input_ids():
    assert Generator._prompt_row([[1, 2, 3]]) == [1, 2, 3]
    assert Generator._prompt_row([1, 2, 3]) == [1, 2, 3]
    assert Generator._prompt_row(torch.tensor([[1, 2, 3]])).tolist() == [1, 2, 3]
    assert Generator._prompt_row(torch.tensor([1, 2, 3])).tolist() == [1, 2, 3]


class _Processor:
    image_token = "<img>"
    image_token_id = 7

    def __call__(self, **kwargs):
        return {"input_ids": [list(range(400))]}

    def decode(self, tokens, skip_special_tokens=True):
        return ""


def test_text_completion_caps_by_the_row_length_of_nested_input_ids(monkeypatch):
    monkeypatch.setattr(generator_module, "encode_content", lambda content, images, token: content)
    gen = _generator(max_seq_len=512)
    gen.processor = _Processor()
    gen.tokenizer = None
    seen = {}

    def fake_generate(**kwargs):
        seen.update(kwargs)
        return iter(())

    gen.generate = fake_generate
    gen.text_completion("hello", max_gen_len=500)
    assert len(seen["prompt_tokens"]) == 400
    assert seen["max_gen_len"] == 112
