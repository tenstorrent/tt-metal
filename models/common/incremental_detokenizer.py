# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Incremental detokenization for streaming console output.

Byte-level BPE tokenizers can split one multi-byte UTF-8 character across several tokens, so
decoding each token on its own prints U+FFFD for rare CJK characters, emoji sequences and any
byte-fallback token. This helper holds the unresolved trailing tokens and emits text only once the
decoded bytes form complete characters. It has no device dependency.
"""

from __future__ import annotations

# A genuinely undecodable byte would be held forever; flush it as-is after this many tokens.
DEFAULT_MAX_HOLD_TOKENS = 8


class IncrementalDetokenizer:
    def __init__(self, tokenizer, skip_special_tokens: bool = True, max_hold_tokens: int = DEFAULT_MAX_HOLD_TOKENS):
        self._tokenizer = tokenizer
        self._skip_special_tokens = skip_special_tokens
        self._max_hold_tokens = max_hold_tokens
        self._pending: list[int] = []

    def push(self, token_id: int) -> str:
        """Add one token; return the newly completed text ("" while a character is incomplete)."""
        self._pending.append(int(token_id))
        text = self._tokenizer.decode(self._pending, skip_special_tokens=self._skip_special_tokens)
        if text.endswith("\ufffd") and len(self._pending) <= self._max_hold_tokens:
            return ""
        self._pending.clear()
        return text

    def flush(self) -> str:
        """Return whatever is still held (an incomplete tail), e.g. at the end of generation."""
        if not self._pending:
            return ""
        text = self._tokenizer.decode(self._pending, skip_special_tokens=self._skip_special_tokens)
        self._pending.clear()
        return text
