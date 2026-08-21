# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single source of truth for the prefill chunk width and its per-chip split.
"""

PRODUCTION_SP_FACTOR = 8  # Production configuration on a single galaxy stage
PREFILL_CHUNK_OUTPUT_TOKENS = 5 * 1024  # Production configuration across prefill models
ISL_TOKENS_PER_CHIP = PREFILL_CHUNK_OUTPUT_TOKENS // PRODUCTION_SP_FACTOR
