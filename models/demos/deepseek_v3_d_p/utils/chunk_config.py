# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Single source of truth for the prefill chunk width and its per-chip split.
"""

import os

PREFILL_SP_FACTOR = 8  # Production configuration on a single galaxy stage
_PRODUCTION_CHUNK_TOKENS = 5 * 1024  # Production configuration across prefill models
# EXPERIMENT (k3-4x4): PREFILL_KV_PERIOD_TOKENS overrides the block-cyclic KV period for chunk-size
# sweeps; the per-chip constant below (which keys tuned configs) stays at the production value.
PREFILL_CHUNK_TOKENS = int(os.environ.get("PREFILL_KV_PERIOD_TOKENS", _PRODUCTION_CHUNK_TOKENS))
PREFILL_CHUNK_TOKENS_PER_CHIP = _PRODUCTION_CHUNK_TOKENS // PREFILL_SP_FACTOR
