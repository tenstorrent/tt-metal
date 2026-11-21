# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for encode_prompt_instruct (Llama-3 chat encoding).

The function is pure host code that only needs a tokenizer, so it is loaded from source
instead of importing models.ttt_compat.tt.common, which pulls in ttnn at module scope.
"""

import ast
from pathlib import Path

import pytest

_COMMON_PY = Path(__file__).resolve().parents[1] / "tt" / "common.py"

BEGIN_OF_TEXT = 128000
START_HEADER = 128006
END_HEADER = 128007
END_TURN = 128009


def _load_encode_prompt_instruct():
    tree = ast.parse(_COMMON_PY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "encode_prompt_instruct":
            namespace = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(_COMMON_PY), "exec"), namespace)
            return namespace["encode_prompt_instruct"]
    raise AssertionError(f"encode_prompt_instruct not found in {_COMMON_PY}")


class _FakeTokenizer:
    special_tokens = {
        "<|begin_of_text|>": BEGIN_OF_TEXT,
        "<|start_header_id|>": START_HEADER,
        "<|end_header_id|>": END_HEADER,
        "<|eot_id|>": END_TURN,
    }

    def encode(self, text, bos=False, eos=False):
        ids = [ord(c) % 1000 + 1 for c in text]
        if bos:
            ids = [1] + ids
        if eos:
            ids = ids + [2]
        return ids


@pytest.fixture(scope="module")
def encode_prompt_instruct():
    return _load_encode_prompt_instruct()


def _system_turn_ids(system_text):
    ids = [START_HEADER] + [ord(c) % 1000 + 1 for c in "system"] + [END_HEADER]
    ids += [ord(c) % 1000 + 1 for c in system_text]
    return ids + [END_TURN]


def test_system_prompt_is_tokenized_into_the_system_turn(encode_prompt_instruct):
    system_text = "You are a helpful assistant."
    encoded = encode_prompt_instruct(_FakeTokenizer(), "hello", system_text)

    assert all(isinstance(token, int) for token in encoded), f"non-int tokens in {encoded!r}"
    expected_prefix = [BEGIN_OF_TEXT] + _system_turn_ids(system_text)
    assert encoded[: len(expected_prefix)] == expected_prefix


def test_without_system_prompt_the_system_turn_is_omitted(encode_prompt_instruct):
    encoded = encode_prompt_instruct(_FakeTokenizer(), "hello")

    assert all(isinstance(token, int) for token in encoded), f"non-int tokens in {encoded!r}"
    assert encoded[0] == BEGIN_OF_TEXT
    assert encoded[1] == START_HEADER

    user_ids = [ord(c) % 1000 + 1 for c in "user"] + [END_HEADER]
    user_ids += [ord(c) % 1000 + 1 for c in "hello"] + [END_TURN]
    assistant_ids = [START_HEADER] + [ord(c) % 1000 + 1 for c in "assistant"] + [END_HEADER]
    assert encoded[1:] == [START_HEADER] + user_ids + assistant_ids


def test_system_prompt_turn_sits_before_the_user_turn(encode_prompt_instruct):
    encoded = encode_prompt_instruct(_FakeTokenizer(), "hello", "sys")
    user_start = len(_system_turn_ids("sys")) + 1

    assert encoded[user_start] == START_HEADER
    assert encoded[user_start + 1 : user_start + 1 + 4] == [ord(c) % 1000 + 1 for c in "user"]
