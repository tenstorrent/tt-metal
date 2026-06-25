# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""KV cache wrapper for the Qwen3 GRPO completer.

The implementation now lives in :mod:`ttml.models.qwen3.kv_cache` (shared with
the standalone ``examples/qwen3`` scripts). ``Qwen3KVCache`` is kept as an alias
so existing GRPO call sites keep working; the shared ``KVCache`` already exposes
the ``update`` / ``get_seq_length`` / ``reset`` / ``clear`` interface the
completer consumes (mask construction lives in the completer).
"""

from __future__ import annotations

from ttml.models.qwen3.kv_cache import KVCache as Qwen3KVCache  # noqa: F401

__all__ = ["Qwen3KVCache"]
