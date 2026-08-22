from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ValidationSweepCache:
    """Per-bring-up cache of stub-hash -> PCC verdict.

    Skips re-validating components whose `_stubs/<comp>.py` file
    content is unchanged since the last sweep that already validated
    them. Validation sweeps in the auto-iterate loop are expensive
    (3-5 min per round on TT hardware) and the LLM only touches one
    component per iter, so re-running PCC on the other 20+ untouched
    components is wasted work.

    Generic by design: keyed on (component_name, stub_file_path).
    Works for any model family — no SAM2-specific assumptions.

    Lifetime: one instance per bring-up. Caller manages instantiation
    (typically in the auto-iterate loop entry point).
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[str, str]] = {}

    @staticmethod
    def _hash(path: Path) -> str:
        if not path.is_file():
            return ""
        return hashlib.sha1(path.read_bytes()).hexdigest()

    def split_sweep_candidates(
        self,
        candidates: List[str],
        stub_paths: Dict[str, Path],
    ) -> Tuple[List[str], List[str]]:
        """Split a sweep candidate list into (must_run, cached_passing).

        ``must_run``: components whose stub hash changed since the last
        recorded PASS (or that have never been validated). These need
        the actual PCC test.

        ``cached_passing``: components whose stub hash matches the
        last-recorded PASSING hash. Safe to skip — the test would
        deterministically pass again.
        """
        must_run: List[str] = []
        cached_pass: List[str] = []
        for c in candidates:
            sp = stub_paths.get(c)
            if sp is None:
                must_run.append(c)
                continue
            cur = self._hash(sp)
            prev = self._cache.get(c)
            if prev is None:
                must_run.append(c)
                continue
            prev_hash, prev_verdict = prev
            if prev_verdict == "PASS" and prev_hash == cur and cur != "":
                cached_pass.append(c)
            else:
                must_run.append(c)
        return must_run, cached_pass

    def record(
        self,
        component: str,
        stub_path: Optional[Path],
        verdict: str,
    ) -> None:
        """Record the verdict for a component after a real sweep run.

        ``verdict``: "PASS" / "FAIL" / "SKIP" (callers can use other
        strings too; only "PASS" enables caching).
        """
        if stub_path is None:
            return
        self._cache[component] = (self._hash(stub_path), verdict)

    def __len__(self) -> int:
        return len(self._cache)


def should_skip_validation_sweep(focused_rc: int) -> bool:
    """Return True if the validation sweep should be skipped this iter.

    Skip when the focused (target-component) pytest didn't even produce
    a clean run: timeouts, signal kills, hard crashes. In those cases
    the agent gets the traceback and retries; running the validation
    sweep on the other 20+ components wastes 3-5 min of TT time and
    yields no new info (the broken target component would dominate the
    failure signal anyway).

    Generic: model-agnostic check on the focused pytest exit code.
    """
    if focused_rc == 0:
        return False
    if focused_rc == 124:
        return True
    if focused_rc < 0:
        return True
    return False
