# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Per-user sampling mix shared by Plan A (decode_harness --per-user-mix) and Plan B (batch 32).

user u (0-based) gets setting u % 4 and seed 1234 + u:
    0: greedy (temperature 0, top_k 0, top_p 1.0)
    1: T 0.7, top_k 50, top_p 0.9
    2: T 1.0, top_k 0 (K_MAX cap), top_p 1.0
    3: T 0.6, top_k 20, top_p 0.95
"""

SETTINGS = [(0.0, 0, 1.0), (0.7, 50, 0.9), (1.0, 0, 1.0), (0.6, 20, 0.95)]
SEED_BASE = 1234


def user_params(batch):
    """-> list of (temperature, top_k, top_p, seed), one per user."""
    return [(*SETTINGS[u % 4], SEED_BASE + u) for u in range(batch)]
