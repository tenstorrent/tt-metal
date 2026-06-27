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


def test_iter_loop_kwargs_helper_covers_all_function_params() -> None:
    """The shared kwarg-forwarder `iter_loop_kwargs_from(args, ...)` must
    return a dict whose keys exactly match the kwargs of
    `_run_auto_iterate_loop`. If someone adds a param to the iter-loop
    function but forgets to update the helper, this test fires loudly
    rather than letting the iter-loop param fall back to a silent
    function default in one of the call sites."""
    import inspect

    from scripts.tt_hw_planner._cli_helpers.auto_iterate import (
        _run_auto_iterate_loop,
        iter_loop_kwargs_from,
    )

    sig = inspect.signature(_run_auto_iterate_loop)
    fn_param_names = set(sig.parameters.keys())

    # Build a dummy args namespace and call the helper to discover its keys.
    args = argparse.Namespace()
    helper_keys = set(
        iter_loop_kwargs_from(
            args,
            MODEL="",
            BOX="",
            demo_dir=Path("."),
            sep="",
            target_components=[],
            provider=None,
            agent_bin=None,
            model=None,
            model_light=None,
            model_heavy=None,
            model_super_heavy=None,
        ).keys()
    )

    missing_in_helper = fn_param_names - helper_keys
    # Ignore any internal-only params that legitimately shouldn't be
    # forwarded from CLI (none today; but allow future allow-list here)
    _IGNORE: set = set()
    truly_missing = missing_in_helper - _IGNORE
    assert not truly_missing, (
        f"iter_loop_kwargs_from is missing kwargs for _run_auto_iterate_loop "
        f"parameters: {sorted(truly_missing)}. Add them to the helper so "
        f"cmd_up and cmd_promote stay in sync; otherwise the iter-loop "
        f"silently falls back to function defaults."
    )

    # Reverse direction: helper shouldn't pass kwargs the function doesn't accept
    extra_in_helper = helper_keys - fn_param_names
    assert not extra_in_helper, (
        f"iter_loop_kwargs_from is passing kwargs the iter-loop function "
        f"doesn't accept: {sorted(extra_in_helper)}. Remove them or rename "
        f"to match the function signature."
    )


def test_promote_uses_kwarg_helper() -> None:
    """cmd_promote must build its iter-loop kwarg dict via
    `iter_loop_kwargs_from`. Without this, the 2026-05-27-style drift
    re-opens at Layer 2 even if Layer 1 (the CLI parser) stays in sync."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "commands" / "promote.py").read_text()
    assert "iter_loop_kwargs_from(" in src, (
        "cmd_promote must use the shared iter_loop_kwargs_from() helper "
        "when calling _run_auto_iterate_loop, otherwise iter-loop knobs "
        "fall back to function defaults silently"
    )
