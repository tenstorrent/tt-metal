# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``paged_sparse_sdpa``: tt-metal's sparse_sdpa with the K rows gathered straight out of the paged bf16 TILE latent
cache (no whole-prefix ROW_MAJOR copy). See README.md and ``op.py``."""
