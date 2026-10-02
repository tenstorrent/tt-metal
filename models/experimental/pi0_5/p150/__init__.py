# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 (pi0_5) Model Implementation for Tenstorrent.

PI0.5 builds on PI0 with two main architectural changes:
    1. Suffix: no separate state token (state is part of language tokens).
       The flow-matching timestep is encoded with sincos + MLP and used as
       an adaRMSNorm conditioning signal (`adarms_cond`) rather than fused
       with action tokens.
    2. Action expert blocks: adaptive RMSNorm with per-layer (scale, shift,
       gate) modulation driven by `adarms_cond`, plus gated residuals.

Most code is reused from `models.experimental.pi0` via subclassing; this
package contains only the deltas.
"""

__version__ = "0.1.0"
