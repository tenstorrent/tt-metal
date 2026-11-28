# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the Qwen3-ASR output parser (no device, no checkpoint)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt"))
from asr_output import parse_asr  # noqa: E402


def test_parse_strips_language_prefix():
    assert parse_asr("language English<asr_text>What's going on?") == ("English", "What's going on?")


def test_parse_keeps_multiline_text():
    assert parse_asr("language Chinese<asr_text>line one\nline two ") == ("Chinese", "line one\nline two")


def test_parse_without_prefix_returns_stripped_text():
    assert parse_asr("  plain text ") == ("", "plain text")


def test_parse_empty_transcript():
    assert parse_asr("language None<asr_text>") == ("None", "")
