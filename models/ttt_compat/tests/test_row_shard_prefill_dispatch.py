# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace

import pytest
import torch

from models.ttt_compat.tt.generator import Generator

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
