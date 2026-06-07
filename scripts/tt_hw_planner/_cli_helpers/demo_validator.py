# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Post-emit validation for emit-e2e.

Phase 4 (current): the parity check is embedded INSIDE the emitted
``tests/test_hf_parity_*.py`` files. Users run these via pytest after
emit. The validator here is a stub that the orchestrator can call to
post-validate via subprocess pytest invocation if desired.

A future iteration could:
  * subprocess-launch pytest on the emitted parity tests
  * parse pytest output for chrF / token_overlap numbers
  * feed divergence back to ``demo_synthesizer.py`` LLM iter-fix
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    passed: bool
    chrf_score: Optional[float] = None
    token_overlap: Optional[float] = None
    diagnostic: str = ""


def run_post_emit_validation(
    *,
    demo_dir,
    task_name: str,
    timeout_s: int = 600,
) -> ValidationResult:
    """Best-effort: run ``pytest tests/test_hf_parity_<task>.py`` and parse.

    Phase 4 stub: returns ``passed=True`` unconditionally. Real
    implementation will subprocess-invoke pytest and parse output.
    Users can run the emitted parity tests manually after emit.
    """
    return ValidationResult(
        passed=True,
        diagnostic="Validation deferred to user-run pytest (Phase 4 stub).",
    )


__all__ = ["ValidationResult", "run_post_emit_validation"]
