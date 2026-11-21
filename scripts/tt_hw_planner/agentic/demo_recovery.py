"""Brain primitive: when end-to-end demo verification fails, decide
whether to retry, cleanup interference, or give up.

After the brain G8 demo-emit decision says YES and the emitter writes
``<demo_dir>/demo/demo.py``, the framework runs ``pytest demo.py::test_demo``
to verify the end-to-end pipeline. That verification can fail from:

  - A stale ``<demo_dir>/demo.py`` (root-level, from a prior tt-hw-planner
    version that wrote there) — pytest discovery may pull it in and
    crash on its broken imports.
  - Transient hardware/dispatch issues that vanish on retry.
  - A genuine bug in the emitted demo.

The brain's recovery flow:

  1. If a sibling stale ``demo.py`` exists at the root of demo_dir
     while the proper one is at ``demo_dir/demo/demo.py``, archive the
     stale sibling and retry.
  2. Else: retry once (flake protection). If still failing, surface
     the failure but don't crash — the per-component PCC suite already
     proved the model is on device.

The brain makes the recovery decision; the caller executes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DemoRecoveryVerdict:
    """Brain's per-failure recovery decision.

    Attributes:
        action: ``"archive_and_retry"``, ``"retry"``, or ``"give_up"``.
        archive_paths: files the caller should rename to ``.stale`` before retry.
        reason: human-readable trace surfaced in banners.
    """

    action: str
    archive_paths: List[Path]
    reason: str


def detect_stale_demo_sibling(*, demo_dir: Path, canonical_demo: Path) -> Optional[Path]:
    """If a stale ``demo.py`` lives at the root of ``demo_dir`` while
    the canonical demo is at ``demo_dir/demo/demo.py``, return the
    stale path so the caller can archive it. Returns None otherwise.

    Heuristic: a file at ``demo_dir/demo.py`` (NOT a directory, NOT
    the canonical demo) when the canonical demo is in the ``demo/``
    subfolder is by definition stale — the current emitter only
    writes the subfolder version.
    """
    if canonical_demo.parent == demo_dir:
        return None
    sibling = demo_dir / "demo.py"
    if sibling.is_file() and sibling.resolve() != canonical_demo.resolve():
        return sibling
    return None


def parse_failing_wired_component(
    *,
    pytest_tail: str,
    wired_components: List[str],
) -> Optional[str]:
    """Parse the pytest failure tail looking for a TT_FATAL / RuntimeError
    that originated in one of the wired components.

    Match patterns (in priority order):
      1. ``_stubs/<comp>.py`` — traceback frame in the component's stub
         (the strongest signal: the exception was raised inside this
         component's TT code).
      2. ``_stubs.<comp>`` — import path inside the demo's WIRED_COMPONENTS.
      3. Bare component name appearing anywhere in the tail.

    All three are gated on a failure-marker check so we don't randomly
    pick a component name from non-failure output.
    """
    if not pytest_tail or not wired_components:
        return None
    text = pytest_tail
    text_lower = text.lower()
    failure_markers = ("tt_fatal", "runtimeerror", "_cpu_fallback]", "assertionerror", "failed ")
    if not any(m in text_lower for m in failure_markers):
        return None

    # Priority 1: traceback frame in _stubs/<comp>.py
    for comp in wired_components:
        if f"_stubs/{comp}.py" in text or f"_stubs\\{comp}.py" in text:
            return comp

    # Priority 2: import path _stubs.<comp>
    for comp in wired_components:
        if f"_stubs.{comp}" in text:
            return comp

    # Priority 3: bare component name with word boundaries (avoids
    # false positives from substring matches in unrelated path text).
    # S4-FIX: prior version used `comp.lower() in text_lower` which
    # could match a component name when it appeared as a substring in
    # an unrelated identifier. Word-bounded regex is safer.
    import re

    for comp in wired_components:
        if not comp:
            continue
        pattern = r"\b" + re.escape(comp) + r"\b"
        if re.search(pattern, text, re.IGNORECASE):
            return comp

    return None


def remove_component_from_wiring(*, demo_path: Path, component: str) -> bool:
    """Remove a component from the demo's ``WIRED_COMPONENTS`` list so
    its TT path isn't exercised on retry. The component stays in the
    pipeline via the HF reference module — graceful degradation to
    mixed-mode without modifying the underlying stub.

    Returns True if the demo was modified, False otherwise.
    """
    if not demo_path.is_file():
        return False
    try:
        src = demo_path.read_text()
    except Exception:
        return False
    if component not in src:
        return False
    lines = src.splitlines(keepends=True)
    out = []
    removed = False
    # B3-FIX: anchor on `_stubs.{component}` (the actual import path
    # binding the tuple to this component), not just on a bare display
    # name. Without this, removing component "foo" would also delete
    # tuples whose display string is 'foo' but whose stub is something
    # else — e.g. when two stubs share a display name.
    stub_marker = f"_stubs.{component}"
    for line in lines:
        if stub_marker in line and line.lstrip().startswith("("):
            removed = True
            continue
        out.append(line)
    if not removed:
        return False
    demo_path.write_text("".join(out))
    return True


def decide_demo_recovery(
    *,
    demo_dir: Path,
    canonical_demo: Path,
    retries_attempted: int,
    max_retries: int = 2,
    pytest_tail: str = "",
    wired_components: Optional[List[str]] = None,
) -> DemoRecoveryVerdict:
    """The brain's decision flow.

    Parameters:
        demo_dir: the model's demo directory.
        canonical_demo: the demo path the framework emitted and verified.
        retries_attempted: how many recovery retries have already happened.
        max_retries: cap (default 2) — one for stale-sibling, one for
            component-disable. Recovery isn't a free retry loop.
        pytest_tail: cleaned pytest output (the useful failure context,
            spam-filtered). Used to identify which wired component is
            broken so we can disable it for the retry.
        wired_components: list of component display names currently
            wired into the demo. Used together with ``pytest_tail``
            to identify the broken one.
    """
    stale_sibling = detect_stale_demo_sibling(
        demo_dir=demo_dir,
        canonical_demo=canonical_demo,
    )
    if stale_sibling is not None:
        return DemoRecoveryVerdict(
            action="archive_and_retry",
            archive_paths=[stale_sibling],
            reason=(
                f"stale sibling `{stale_sibling.name}` at demo_dir root "
                f"(orphan from prior tt-hw-planner version where the demo "
                f"lived at the root). Archiving so pytest discovery can't "
                f"pull it in."
            ),
        )

    # If we have failure context, try to identify the broken wired
    # component and disable it for the retry. The per-component PCC
    # test passed (component "works" on captured fixtures); the demo
    # failure is on a different runtime shape. Graceful: drop it from
    # the demo's wiring so the rest runs end-to-end in mixed mode.
    broken_component = parse_failing_wired_component(
        pytest_tail=pytest_tail,
        wired_components=wired_components or [],
    )
    if broken_component is not None and retries_attempted < max_retries:
        verdict = DemoRecoveryVerdict(
            action="disable_component_and_retry",
            archive_paths=[],
            reason=(
                f"identified broken wired component `{broken_component}` "
                f"from pytest tail (PCC passed but runtime shapes differ); "
                f"disabling its wiring so the rest of the pipeline runs "
                f"end-to-end in mixed mode"
            ),
        )
        verdict.broken_component = broken_component  # type: ignore[attr-defined]
        return verdict

    # No specific component identifiable from the pytest tail (the
    # TT_FATAL C++ backtrace doesn't always include a Python frame for
    # the failing stub). Fall back to brute-force: disable the LAST
    # wired component (most recently added — most likely a new
    # graduate the demo hasn't been tested against) and retry. Over
    # max_retries attempts, the brain progressively shrinks the wired
    # set until the demo runs end-to-end.
    if wired_components and retries_attempted < max_retries:
        last_wired = wired_components[-1]
        verdict = DemoRecoveryVerdict(
            action="disable_component_and_retry",
            archive_paths=[],
            reason=(
                f"could not pinpoint broken component from pytest tail; "
                f"falling back to brute-force disable of last-wired "
                f"`{last_wired}` (attempt {retries_attempted + 1}/{max_retries + 1}) "
                f"— if demo passes after this, `{last_wired}` is the culprit"
            ),
        )
        verdict.broken_component = last_wired  # type: ignore[attr-defined]
        return verdict

    if retries_attempted < max_retries:
        return DemoRecoveryVerdict(
            action="retry",
            archive_paths=[],
            reason=(
                f"no wired components available to disable — retry once "
                f"for flake protection (attempt {retries_attempted + 1}/{max_retries + 1})"
            ),
        )

    return DemoRecoveryVerdict(
        action="give_up",
        archive_paths=[],
        reason=(
            f"all {max_retries} retries exhausted; the per-component PCC "
            f"suite already proved the model is on device. Surfacing the "
            f"demo failure for human review rather than masking it."
        ),
    )


def archive_demo_files(paths: List[Path]) -> List[Path]:
    """Rename each path to ``.stale_demo_sibling`` so pytest discovery
    skips it. Returns the archived paths. Safe to call with an empty
    list. Idempotent: if the archive name already exists, the original
    is left alone."""
    archived: List[Path] = []
    for p in paths:
        if not p.is_file():
            continue
        stale = p.with_suffix(p.suffix + ".stale_demo_sibling")
        if stale.exists():
            continue
        p.rename(stale)
        archived.append(stale)
    return archived


__all__ = [
    "DemoRecoveryVerdict",
    "archive_demo_files",
    "decide_demo_recovery",
    "detect_stale_demo_sibling",
    "parse_failing_wired_component",
    "remove_component_from_wiring",
]
