# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""KV cache for autoregressive inference.

The implementation now lives in the shared :mod:`ttml.models.qwen3.kv_cache`
module (single source of truth, shared with the GRPO example). ``KVCache`` is
re-exported here so existing ``from utils.kv_cache import KVCache`` call sites
keep working.
"""

from ttml.models.qwen3.kv_cache import KVCache  # noqa: F401

__all__ = ["KVCache"]
