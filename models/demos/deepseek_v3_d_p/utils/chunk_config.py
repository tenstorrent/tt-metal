# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single source of truth for the prefill chunk width and its per-chip split.
"""

PREFILL_SP_FACTOR = 8  # Production configuration on a single galaxy stage
# Experiment: GLM-5.3 SC1/SC4 prefill at 2048-token chunks (256 rows per chip on 8x4) instead of the
# production 5 * 1024. Half of 5120 (2560) does not run: GLM-5.3's indexer splits each chip's rows
# over TP=4 and needs whole 32-row tiles, so the chunk must be a multiple of 8 * 4 * 32 = 1024.
# The KV chunk-address table asserts the runner's chunk size equals this constant, so it must match
# PREFILL_CHUNK_SIZE in run_multirank_pcc.sh.
PREFILL_CHUNK_TOKENS = 2 * 1024
PREFILL_CHUNK_TOKENS_PER_CHIP = PREFILL_CHUNK_TOKENS // PREFILL_SP_FACTOR
