# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for IncrementalDetokenizer (no device, no tokenizer download)."""

from models.common.incremental_detokenizer import DEFAULT_MAX_HOLD_TOKENS, IncrementalDetokenizer


class ByteTokenizer:
    """One token per UTF-8 byte, like a byte-fallback BPE splitting a multi-byte character."""

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return bytes(token_ids).decode("utf-8", errors="replace")


def _stream(text):
    detok = IncrementalDetokenizer(ByteTokenizer())
    pieces = [detok.push(b) for b in text.encode("utf-8")]
    pieces.append(detok.flush())
    return pieces


def test_multi_byte_text_streams_without_replacement_characters():
    text = "Hello \u9f98\u9f50 \U0001f92f\U0001f9d1\u200d\U0001f680 \u20a9"
    pieces = _stream(text)
    assert all("\ufffd" not in piece for piece in pieces)
    assert "".join(pieces) == text


def test_per_token_decode_is_what_the_helper_fixes():
    text = "\u9f98"
    per_token = "".join(ByteTokenizer().decode([b]) for b in text.encode("utf-8"))
    assert "\ufffd" in per_token
    assert "".join(_stream(text)) == text


def test_incomplete_tail_is_flushed():
    detok = IncrementalDetokenizer(ByteTokenizer())
    assert detok.push(0xE9) == ""
    assert detok.flush() == "\ufffd"
    assert detok.flush() == ""


def test_undecodable_byte_does_not_stall_the_stream():
    detok = IncrementalDetokenizer(ByteTokenizer())
    out = [detok.push(0x80)] + [detok.push(ord("a")) for _ in range(DEFAULT_MAX_HOLD_TOKENS)]
    assert "".join(out + [detok.flush()]).replace("\ufffd", "") == "a" * DEFAULT_MAX_HOLD_TOKENS
    assert out[-1] != "" or detok.flush() == ""
