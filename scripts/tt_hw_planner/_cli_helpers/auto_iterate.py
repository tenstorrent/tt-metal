from __future__ import annotations
from ..discovery import safe_relative_to_root, BRINGUP_ROOT

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def add_iter_loop_cli_args(parser: argparse.ArgumentParser) -> None:
    """Single source of truth for CLI flags that configure the auto-iterate
    loop. Both `pup` (auto-up) and `pprom` (promote) MUST call this so they
    expose the SAME iter-loop knobs.

    Why this exists: both `up` and `promote` eventually call
    `_run_auto_iterate_loop`. Before this helper, each command's parser
    declared its own copy of the iter-loop flags — and they silently
    drifted. `--parallel-agents` was added to `pup` on 2026-05-27, never
    mirrored to `pprom`; same story for `--auto-only-component`,
    `--auto-model-super-heavy`, `--strict-pcc`, `--escalate-on-pcc-fail`,
    `--pcc-engine`, and others. Result: `promote --auto` was silently
    running with serial agents + truncated tier ladder while `up --auto`
    used the configured concurrency / full ladder.

    This helper closes the gap. Adding a NEW iter-loop knob = edit this
    function ONCE; both commands inherit it. test_invariants.py asserts
    both parsers are called through this helper.

    NOTE: flags that are stage-specific (pre-iter-loop kernel-sweep,
    op-synth, decomposition planning under `up`; manual hand-off mode
    under `promote`) MUST NOT be added here — they stay on the
    individual parsers.
    """
    parser.add_argument(
        "--auto-model-super-heavy",
        default=None,
        help=(
            "Tiered mode: model alias for the THIRD tier — fires when the "
            "heavy tier (sonnet) has plateaued (attempts ≥ 5 OR consecutive "
            "same-class failures ≥ 3). Default for claude under "
            "--auto-model-tiered is 'opus'. Set this explicitly to override "
            "or to enable super_heavy when not using --auto-model-tiered."
        ),
    )
    parser.add_argument(
        "--strict-pcc",
        dest="strict_pcc",
        action="store_true",
        default=True,
        help=(
            "[default ON under --auto] After the fast-path demo "
            "pytest exits 0, run the same prompt through HF on CPU "
            "greedy and compare the first N tokens against what the "
            "TT demo decoded. If the outputs diverge beyond the "
            "tolerance, the run is demoted to FAIL (rc=17) instead "
            "of false-greenly reporting SUCCESS. With --auto, the "
            "mismatch is also fed back into the LLM repair loop so "
            "the planner can attempt to converge to a real working "
            "model."
        ),
    )
    parser.add_argument(
        "--no-strict-pcc",
        dest="strict_pcc",
        action="store_false",
        help=(
            "Disable the PCC gate. Useful for very large models "
            "(70 B+) where HF CPU reference inference takes longer "
            "than the user is willing to wait, or for models without "
            "a usable HF mirror (e.g. local fine-tunes). When "
            "disabled, the planner trusts pytest's exit code alone, "
            "which restores the pre-2026-05-23 'false green' "
            "behaviour."
        ),
    )
    parser.add_argument(
        "--escalate-on-pcc-fail",
        dest="escalate_on_pcc_fail",
        action="store_true",
        default=True,
        help=(
            "[default ON under --auto] When the ALREADY-SUPPORTED fast "
            "path passes pytest but fails the PCC gate (i.e. wrong "
            "routing produced a false-green before this guard fired), "
            "automatically invoke `auto-onboard --accept` to draft a "
            "new FamilyBackend for the model and then re-invoke `up` "
            "so the scaffold + per-component PCC>=0.99 iterate loop "
            "engages on hardware. Provides the iteration-to-bringup "
            "behaviour that the cold-start path already supports."
        ),
    )
    parser.add_argument(
        "--no-escalate-on-pcc-fail",
        dest="escalate_on_pcc_fail",
        action="store_false",
        help=(
            "Disable PCC-fail escalation. The PCC gate still fires, "
            "but a fast-path mismatch just exits with rc=17 instead "
            "of automatically attempting auto-onboard + scaffold + "
            "iterate. Useful when you want to inspect the failure "
            "interactively before letting the agent draft a new "
            "backend."
        ),
    )
    parser.add_argument(
        "--strict-pcc-tokens",
        type=int,
        default=None,
        help=(
            "Number of tokens to compare in the PCC gate (default: "
            "32; uses output_validation.DEFAULT_COMPARE_TOKENS). "
            "Lower values make the gate faster but more sensitive to "
            "noise; higher values are more reliable but cost CPU "
            "wall-clock per added token (~0.1-2s on a 4-13B model)."
        ),
    )
    parser.add_argument(
        "--pcc-engine",
        choices=("legacy", "evidence", "agentic"),
        default="agentic",
        help=(
            "Which correctness-gate engine to use when --strict-pcc "
            "fires (default: agentic). 'agentic' is the strongest "
            "engine and the new default as of 2026-05-24: it runs "
            "the evidence-engine 256-token wide-scan + mid-sequence "
            "collapse detector (catches medgemma-style 'first N "
            "tokens fine, then garbage' false-greens that legacy "
            "32-token window misses) AND drives the per-layer "
            "agentic probe (HF-vs-TT divergence localization, "
            "symptom-aware mechanical actions, edit-took-effect "
            "verification, convergence detection). 'evidence' is "
            "the same gate but without the agentic repair loop -- "
            "useful for CI where you want detection but not LLM "
            "spend. 'legacy' is the pre-2026-05-24 inline 32-token "
            "gate, preserved for byte-for-byte reproduction of "
            "older runs but NOT RECOMMENDED for new bring-ups."
        ),
    )


def _is_eligible_for_graduation(stub_path: Path) -> bool:
    """Pure check: is this stub's code real enough to count as graduated?

    Returns True iff:
      1. The stub file exists on disk
      2. The stub does NOT delegate to torch (i.e. NOT a Phase-1
         torch-wrapper or `_get_torch_submodule` fallback)

    Caller is responsible for verifying pytest passed; this only
    validates the stub-code-is-real half of the graduation criterion
    (the other half is pytest-passed, which the caller decides from
    the pytest report).

    Why this function exists (2026-06-03): the auto-iterate loop has
    5+ sites that decide "should this component be appended to
    graduated_this_run?" Each rolled its own inline check, some with
    the torch-wrapper guard and some without. Future graduation sites
    should call this helper to inherit the canonical criterion
    automatically, preventing the drift that historically allowed
    torch-wrapper trivial-PCC-passes to be miscounted as real
    graduations (the seamless-m4t bug case).
    """
    from ..cli import _stub_uses_torch_wrapper

    if not stub_path.is_file():
        return False
    if _stub_uses_torch_wrapper(stub_path):
        return False
    return True


def _should_snapshot_best_native(
    *,
    snap_exists: bool,
    prior_pcc: Optional[float],
    new_pcc: Optional[float],
) -> bool:
    """Pure decision: should we (over)write the ``.best_native`` snapshot?

    Mirrors the docstring rules of the nested
    ``_snapshot_best_native_stub`` inside ``_run_auto_iterate_loop``.
    Extracted to module level so the rule table is unit-testable
    without spinning up the whole iter loop.

    Rules (in order):
      * WRITE if no prior snapshot exists (any native body > none).
      * SKIP if new PCC is None (no quality signal — preserve prior).
      * WRITE if prior PCC is None (replace unmeasured with measurable).
      * WRITE if new PCC strictly improves over prior.
      * SKIP otherwise.

    The "SKIP if new PCC is None" rule existed in the docstring but
    NOT in the code from before 2026-06-03 — a TT_FATAL iter that
    crashed pre-PCC measurement would silently overwrite a prior
    measurable snapshot.
    """
    if not snap_exists:
        return True
    if new_pcc is None:
        return False
    if prior_pcc is None:
        return True
    return new_pcc > prior_pcc
