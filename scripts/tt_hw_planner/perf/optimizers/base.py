# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Common types for the optimizer-block catalog.

Every block subclasses `OptimizerBlock` and implements three methods:

  diagnose(joined, source, box) -> List[Finding]
      Read-only inspection of the joined Tracy⋈tracer data + the model
      source tree. Returns one Finding per actionable opportunity.

  propose(findings, source)     -> List[Patch]
      For each Finding, emit a structured Patch describing the change
      to make. Patches are NOT applied here; that's the runner's job.

  verify(before, after, source) -> VerificationResult
      After a patch was applied and a new perf run was collected, decide
      whether the change was a win. Used by the runner to auto-revert
      regressions.

Patches are always reversible: the runner stores them as text files
under perf-data/<run>/patches/<block>__<cluster>.patch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


class Severity(Enum):
    INFO = "info"
    WARN = "warn"
    BLOCKER = "blocker"


class PatchKind(Enum):
    KWARG_REPLACE = "kwarg_replace"
    TUNING_TABLE_ENTRY = "tuning_table_entry"
    SOURCE_REWRITE = "source_rewrite"


@dataclass(frozen=True)
class SourceLocation:
    """Where a Patch (or a Finding's evidence) lives in the tree."""

    path: str  # repo-relative
    line: Optional[int] = None
    column: Optional[int] = None
    func: Optional[str] = None
    variable: Optional[str] = None


@dataclass
class Finding:
    """One actionable opportunity for an optimizer block."""

    block_name: str
    cluster_id: str
    block_path: str  # signposted scope, e.g. decoder.layer_3.mlp
    severity: Severity
    cost_ms: float  # total time this issue contributes
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggestion: str = ""
    source_locations: List[SourceLocation] = field(default_factory=list)
    expected_gain_ms: float = 0.0


@dataclass
class Patch:
    """Structured representation of a code change. Reversible."""

    kind: PatchKind
    target: SourceLocation
    new_kwargs: Dict[str, Any] = field(default_factory=dict)
    diff_text: str = ""
    rationale: str = ""
    # Free-form fields the specific block needs for revert.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Did the apply move things in the right direction?"""

    ok: bool
    delta_ms: float
    accuracy_held: bool
    reason: str = ""


class OptimizerBlock(Protocol):
    """Every block exposes the same three methods + metadata."""

    name: str
    level: int
    requires: List[str]

    def diagnose(self, joined: List, source: "ModelSource", box: Any) -> List[Finding]:
        ...

    def propose(self, findings: List[Finding], source: "ModelSource") -> List[Patch]:
        ...

    def verify(self, before: List, after: List, source: "ModelSource") -> VerificationResult:
        ...


@dataclass
class ModelSource:
    """A pointer to the model source tree the block may mutate.

    Concrete implementations resolve paths under `models/tt_transformers/`
    or external demos. Each block reads what it needs lazily.
    """

    repo_root: Path
    demo_test_path: str
    base_model_name: str
    mesh_device: str
    model_id: str
    notes: List[str] = field(default_factory=list)
