# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: rms_norm's cross-core stat COMBINE, flat root vs two-level tree."""

from .tree_combine import (  # noqa: F401
    VARIANTS,
    Geometry,
    build_layout,
    combine,
    create_stat_memory_config,
    reference_rms_recip,
)
