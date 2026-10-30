# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace

import pytest
import torch

from models.tt_transformers.tt.generator import Generator

START, MESSAGE, END, CHANNEL, USER, ASSISTANT = 200006, 200008, 200007, 200005, 1428, 173781
CHAT = [START, USER, MESSAGE, 62316, 40617, 13, END, START, ASSISTANT]


def _generator(row_sharded=True):
    generator = object.__new__(Generator)
    generator.model = [SimpleNamespace(users_row_sharded=row_sharded)]
    return generator


def _batch(*prompts, pad=0):
    width = max(len(p) for p in prompts)
    tokens = torch.full((len(prompts), width), pad, dtype=torch.int64)
    for i, p in enumerate(prompts):
        tokens[i, : len(p)] = torch.tensor(p)
    return tokens, [len(p) for p in prompts]


def test_chat_prompts_ending_in_the_assistant_header_take_the_row_sharded_path():
    tokens, lens = _batch(CHAT, CHAT[:3] + [1, 2, 3] + CHAT[6:])
    assert _generator()._will_row_shard_prefill(tokens, object(), lens)


@pytest.mark.parametrize(
    "second",
    [
        CHAT + [CHANNEL],
        CHAT[:-1],
        [1, 2, 3, START, ASSISTANT],
    ],
    ids=["ends_in_channel", "no_assistant_role", "not_harmony"],
)
def test_any_prompt_without_the_assistant_header_keeps_the_ordinary_path(second):
    tokens, lens = _batch(CHAT, second)
    assert not _generator()._will_row_shard_prefill(tokens, object(), lens)


def test_padding_after_a_short_prompt_is_not_read_as_its_suffix():
    tokens, lens = _batch(CHAT, CHAT[:-1], pad=ASSISTANT)
    assert not _generator()._will_row_shard_prefill(tokens, object(), lens)
    tokens, lens = _batch(CHAT, [START, ASSISTANT], pad=CHANNEL)
    assert _generator()._will_row_shard_prefill(tokens, object(), lens)


def test_host_sampling_single_request_and_unsharded_models_are_unchanged():
    tokens, lens = _batch(CHAT, CHAT)
    assert not _generator()._will_row_shard_prefill(tokens, None, lens)
    assert not _generator(row_sharded=False)._will_row_shard_prefill(tokens, object(), lens)
    tokens, lens = _batch(CHAT)
    assert not _generator()._will_row_shard_prefill(tokens, object(), lens)


def test_first_row_sharded_prefill_prepares_decode_before_capture():
    events = []
    model = SimpleNamespace(
        users_row_sharded=True,
        _supports_on_device_sampling=True,
        sampling=SimpleNamespace(_trace_states={}),
        switch_mode=lambda mode: None,
    )
    args = SimpleNamespace(max_batch_size=128, max_local_batch_size=32, vocab_size=16)
    generator = Generator([model], [args], None)

    def prepare(**kwargs):
        generator._trace_io().require_preparation()
        events.append("prepare decode")
        generator._pending_decode_trace = {"on_device_sampling": True}

    def prefill(*args, **kwargs):
        assert generator._pending_decode_trace is not None
        events.append("capture prefill")
        generator.trace_id_prefill["row_sharded"] = 11
        return torch.zeros((128, 1), dtype=torch.int64)

    def record(prepared):
        assert generator._trace_io()._frozen
        assert prepared["on_device_sampling"]
        events.append("capture decode")
        return {0: 12}, [object()], (object(),)

    generator._prepare_decode_trace_once = prepare
    generator._row_sharded_batched_prefill = prefill
    generator._record_decode_trace_text = record
    tokens, lens = _batch(*([CHAT] * 128))
    generator.prefill_forward_text(
        tokens,
        page_table=torch.zeros((128, 4), dtype=torch.int32),
        prompt_lens=lens,
        sampling_params=object(),
        enable_trace=True,
        warmup_prefill=False,
    )
    assert events == ["prepare decode", "capture prefill", "capture decode"]
    assert generator.trace_ids_decode[True] == {0: 12}
    assert not generator._defer_trace_recording
