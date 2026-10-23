# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single source of truth for the prefill chunk width and its per-chip split.
"""

PREFILL_SP_FACTOR = 8  # Production configuration on a single galaxy stage
PREFILL_CHUNK_TOKENS = 5 * 1024  # Production configuration across prefill models
PREFILL_CHUNK_TOKENS_PER_CHIP = PREFILL_CHUNK_TOKENS // PREFILL_SP_FACTOR
