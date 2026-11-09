# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single source of truth for the prefill chunk width and its per-chip split.
"""

PREFILL_SP_FACTOR = 8  # Production configuration on a single galaxy stage
# Experiment: half the production chunk (5 * 1024) to measure GLM-5.3 SC1/SC4 prefill at 2560-token
# chunks (320 rows per chip on 8x4). The KV chunk-address table asserts the runner's chunk size equals
# this constant, so it must match PREFILL_CHUNK_SIZE in run_multirank_pcc.sh.
PREFILL_CHUNK_TOKENS = 5 * 1024 // 2
PREFILL_CHUNK_TOKENS_PER_CHIP = PREFILL_CHUNK_TOKENS // PREFILL_SP_FACTOR
