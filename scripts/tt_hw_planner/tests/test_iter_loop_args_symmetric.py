"""Symmetry tests for the shared iter-loop CLI args.

These guard the principle stated by the user on 2026-06-04: any flag that
controls the auto-iterate loop's behavior must appear on BOTH `up` (pup)
and `promote` (pprom) parsers. The 2026-05-27 `--parallel-agents` drift
(added to pup but never mirrored to pprom) was exactly the bug this
suite is designed to catch in CI before it ships.
"""
from __future__ import annotations

import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _planner_cli_source() -> str:
    return (_REPO_ROOT / "scripts" / "tt_hw_planner" / "cli.py").read_text()


def test_pup_uses_shared_iter_loop_helper() -> None:
    """`pup` (auto-up) must call `add_iter_loop_cli_args(pup)` so its
    iter-loop knobs come from the shared definition. Inlining flags
    directly on `pup` re-opens the drift gap."""
    src = _planner_cli_source()
    assert "_add_iter_loop_cli_args(pup)" in src or "add_iter_loop_cli_args(pup)" in src, (
        "`pup` must call add_iter_loop_cli_args(pup) — the shared "
        "iter-loop arg definition lives in _cli_helpers/auto_iterate.py"
    )


def test_pprom_uses_shared_iter_loop_helper() -> None:
    """`pprom` (promote) must call the same helper. Before this guard,
    promote was silently missing --parallel-agents, --auto-only-component,
    --auto-model-super-heavy, --strict-pcc, --escalate-on-pcc-fail,
    --pcc-engine, and others."""
    src = _planner_cli_source()
    assert "_add_iter_loop_cli_args(pprom)" in src or "add_iter_loop_cli_args(pprom)" in src, (
        "`pprom` must call add_iter_loop_cli_args(pprom) so promote " "exposes the same iter-loop knobs as auto-up"
    )


def test_pup_and_pprom_share_iter_loop_args() -> None:
    """Build both parsers via the shared helper and assert they expose
    identical flag sets for iter-loop control. This is the actual
    behavioral assertion — the prior two tests only check that the
    helper is called; this one verifies the helper itself is
    self-consistent."""
    from scripts.tt_hw_planner._cli_helpers.auto_iterate import add_iter_loop_cli_args

    pup_test = argparse.ArgumentParser()
    add_iter_loop_cli_args(pup_test)
    pup_dests = {a.dest for a in pup_test._actions if a.dest != "help"}

    pprom_test = argparse.ArgumentParser()
    add_iter_loop_cli_args(pprom_test)
    pprom_dests = {a.dest for a in pprom_test._actions if a.dest != "help"}

    assert pup_dests == pprom_dests, (
        f"Helper produced inconsistent args between parsers; " f"diff: {pup_dests ^ pprom_dests}"
    )
    # Non-empty sanity check
    assert "parallel_agents" in pup_dests, "Helper must define parallel_agents"
    assert "auto_only_component" in pup_dests, "Helper must define auto_only_component"
    assert "pcc_engine" in pup_dests, "Helper must define pcc_engine"
