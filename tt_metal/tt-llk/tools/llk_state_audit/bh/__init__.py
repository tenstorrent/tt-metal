# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Blackhole state-hazard audit (``--arch blackhole``): scanner, model, effect DB.

Entry points: ``effects_db.build(repo_root)`` -> :class:`effects_db.EffectDB`;
CLI ``python3 -m tools.llk_state_audit bh-effects|bh-hazards``. See SCHEMA.md.
"""

from . import model, resources, scan
from .effects_db import EffectDB, build, dumps, load_families
from .model import (
    CallEdge,
    CallGraph,
    Effect,
    FunctionModel,
    Pattern,
    PatternMatch,
    Resource,
)

__all__ = [
    "CallEdge",
    "CallGraph",
    "Effect",
    "EffectDB",
    "FunctionModel",
    "Pattern",
    "PatternMatch",
    "Resource",
    "build",
    "dumps",
    "load_families",
    "model",
    "resources",
    "scan",
]
