# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CI gate: ``pcc_thresholds.json`` may only ratchet UP, never down.

This guards ``models/demos/gemma4/tests/pcc_thresholds.json`` against silent
threshold relaxation. After an optimization lands we *raise* the recorded PCC
threshold for the affected module/variant/topology; we must never lower a
recorded value to make a regression pass. This test diffs the working-tree
thresholds against their git baseline and fails the build if any entry dropped.

Two complementary layers of protection exist:

1. Runtime gate (already enforced per test): each PCC unit test asserts measured
   PCC >= ``get_pcc_threshold(request)`` (the recorded value in
   ``pcc_thresholds.json``), so a real numerical regression fails the build at
   the test that measures it.
2. Source gate (this test): the recorded thresholds themselves can only go up,
   so a PR cannot hide a regression by editing the table down.

Baseline resolution order:

* ``$GEMMA4_PCC_BASELINE_REF`` if set (an exact ref, e.g. the PR base branch);
* the merge-base of ``HEAD`` with ``origin/main`` / ``main`` (the fork point);
* ``origin/main`` / ``main`` directly.

If no baseline can be resolved (e.g. a shallow clone with no ``main`` ref), the
test skips with a clear message rather than failing spuriously. Set
``GEMMA4_PCC_BASELINE_REF`` to make the gate deterministic in such an
environment.

Runnable standalone for local checks::

    python models/demos/gemma4/tests/unit/test_pcc_threshold_ratchet.py
"""

import json
import os
import subprocess

import pytest

_THRESHOLDS_RELPATH = "models/demos/gemma4/tests/pcc_thresholds.json"
_THRESHOLDS_ABSPATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pcc_thresholds.json"))
# Float comparison slack so JSON round-trip noise never trips the gate.
_EPS = 1e-9


def _git(root, *args):
    """Run a git command in ``root``; returns the CompletedProcess (never raises)."""
    return subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_root():
    """Repo top-level, or None if this file isn't inside a git checkout."""
    start = os.path.dirname(__file__)
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=start,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _flatten(table):
    """Flatten the nested threshold table to ``{(model, mesh, node): float}``.

    Non-numeric values (the ``_comment_*`` annotation strings) are skipped, so
    documentation entries never participate in the ratchet comparison.
    """
    flat = {}
    for model, meshes in table.items():
        if not isinstance(meshes, dict):
            continue
        for mesh, entries in meshes.items():
            if not isinstance(entries, dict):
                continue
            for node, value in entries.items():
                # bool is an int subclass — exclude it explicitly.
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                flat[(model, mesh, node)] = float(value)
    return flat


def _candidate_refs(root):
    """Ordered candidate baseline refs to diff the thresholds against."""
    refs = []
    env_ref = os.getenv("GEMMA4_PCC_BASELINE_REF")
    if env_ref:
        refs.append(env_ref)
    for base in ("origin/main", "main"):
        if _git(root, "rev-parse", "--verify", "--quiet", base).returncode != 0:
            continue
        merge_base = _git(root, "merge-base", "HEAD", base)
        if merge_base.returncode == 0 and merge_base.stdout.strip():
            refs.append(merge_base.stdout.strip())
        refs.append(base)
    # De-dupe preserving order.
    seen = set()
    return [r for r in refs if not (r in seen or seen.add(r))]


def _resolve_baseline(root):
    """Return ``(ref, baseline_table)`` for the first resolvable baseline, else ``(None, None)``."""
    for ref in _candidate_refs(root):
        show = _git(root, "show", f"{ref}:{_THRESHOLDS_RELPATH}")
        if show.returncode != 0:
            continue
        try:
            return ref, json.loads(show.stdout)
        except json.JSONDecodeError:
            continue
    return None, None


def find_lowered_thresholds():
    """Core comparison, decoupled from pytest so it can run standalone.

    Returns ``(status, payload)`` where status is one of:
      * ``"skip"`` — payload is a human reason string;
      * ``"ok"``   — payload is ``(ref, added_count, removed_count)``;
      * ``"fail"`` — payload is ``(ref, lowered)`` with lowered a list of
        ``((model, mesh, node), baseline_value, current_value)``.
    """
    root = _git_root()
    if root is None:
        return "skip", "not a git checkout; cannot resolve a PCC baseline"

    ref, baseline = _resolve_baseline(root)
    if baseline is None:
        return "skip", (
            "could not resolve a git baseline for pcc_thresholds.json "
            "(no origin/main or main); set GEMMA4_PCC_BASELINE_REF to enable the ratchet gate"
        )

    with open(_THRESHOLDS_ABSPATH) as f:
        current = json.load(f)

    cur_flat = _flatten(current)
    base_flat = _flatten(baseline)

    lowered = [
        (key, base_flat[key], cur_flat[key])
        for key in base_flat
        if key in cur_flat and cur_flat[key] < base_flat[key] - _EPS
    ]
    if lowered:
        return "fail", (ref, sorted(lowered))

    added = len(set(cur_flat) - set(base_flat))
    removed = len(set(base_flat) - set(cur_flat))
    return "ok", (ref, added, removed)


def test_pcc_thresholds_only_ratchet_up():
    """A PR must not lower any recorded PCC threshold (thresholds ratchet up)."""
    status, payload = find_lowered_thresholds()

    if status == "skip":
        pytest.skip(payload)

    if status == "fail":
        ref, lowered = payload
        lines = [f"  {model} / {mesh} / {node}: {base:.4f} -> {cur:.4f}" for (model, mesh, node), base, cur in lowered]
        pytest.fail(
            f"pcc_thresholds.json lowered {len(lowered)} recorded PCC threshold(s) vs baseline "
            f"'{ref}' — thresholds may only ratchet UP:\n"
            + "\n".join(lines)
            + "\n\nIf the regression is real, fix the model rather than lowering the bar. "
            "If the drop is genuinely intentional, justify it in review and override the "
            "baseline via GEMMA4_PCC_BASELINE_REF."
        )

    ref, added, removed = payload
    from loguru import logger

    logger.info(
        f"PCC ratchet gate OK vs baseline '{ref}': no thresholds lowered " f"({added} added, {removed} removed)."
    )


if __name__ == "__main__":
    _status, _payload = find_lowered_thresholds()
    if _status == "skip":
        print(f"SKIP: {_payload}")
        raise SystemExit(0)
    if _status == "fail":
        _ref, _lowered = _payload
        print(f"FAIL: {len(_lowered)} threshold(s) lowered vs '{_ref}':")
        for (_m, _mesh, _node), _b, _c in _lowered:
            print(f"  {_m} / {_mesh} / {_node}: {_b:.4f} -> {_c:.4f}")
        raise SystemExit(1)
    _ref, _added, _removed = _payload
    print(f"OK vs '{_ref}': no thresholds lowered ({_added} added, {_removed} removed).")
    raise SystemExit(0)
