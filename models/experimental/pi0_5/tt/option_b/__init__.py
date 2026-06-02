# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
pi0.5 Option B — 4-stage pipeline on 32 Blackhole chips (TP=8 within stage).

See `docs/PI0_5_GALAXY_DEPLOYMENT_PLAN.md` §3 for the design rationale, and
`README.md` in this dir for the file map.
"""

from .stages import StageLayout, build_default_layout

__all__ = ["StageLayout", "build_default_layout"]
