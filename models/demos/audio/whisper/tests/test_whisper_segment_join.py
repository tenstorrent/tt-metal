# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for joining Whisper timestamp-mode segment texts into plain text."""

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "tt" / "whisper_generator.py"


def _load_join_segment_texts():
    # whisper_generator imports ttnn at module level; load only the pure-Python helpers so the test needs no device stack.
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    wanted = {"_is_unspaced_script_char", "join_segment_texts"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SRC), "exec"), namespace)
    return namespace["join_segment_texts"]


join_segment_texts = _load_join_segment_texts()


@pytest.mark.parametrize(
    "segments, expected",
    [
        (["Hello there.", "How are you?"], "Hello there. How are you?"),
        (["你好，今天天气很好。", "我们去公园玩吧。"], "你好，今天天气很好。我们去公园玩吧。"),
        (["こんにちは", "世界"], "こんにちは世界"),
        (["Hello", "世界"], "Hello世界"),
        (["a", "", "  ", "b"], "a b"),
        ([], ""),
    ],
)
def test_join_segment_texts(segments, expected):
    assert join_segment_texts(iter(segments)) == expected
