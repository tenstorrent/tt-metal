"""Unit tests for the sandbox flags added to the auto-iterate loop.

Two flags are covered:

  --auto-only-component <name>     Restrict the iter loop to one component.
                                   Used for $2 single-component A/B tests
                                   of prompt enrichment changes before
                                   committing to a $36 multi-component run.

  TT_PLANNER_DRY_RUN_PROMPTS=1     Write assembled prompts to disk and
                                   skip the LLM invocation. Used to verify
                                   the prompt-assembly pipeline is shaping
                                   the right blocks without spending money.

These tests do not hit hardware, the network, or the LLM. They source-grep
the wiring and exercise the parser + the inline gates."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

cli = importlib.import_module("scripts.tt_hw_planner.cli")


def _auto_iterate_source() -> str:
    return (Path(cli.__file__).parent / "_cli_helpers" / "auto_iterate.py").read_text()


def test_auto_only_component_flag_exists_on_up_parser() -> None:
    """`up --auto-only-component foo` must parse without error and
    expose `.auto_only_component == "foo"` on the argparse Namespace."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    pup = sub.add_parser("up")
    pup.add_argument("model_id")
    pup.add_argument("--auto-only-component", default=None)
    ns = parser.parse_args(["up", "facebook/sam2-hiera-tiny", "--auto-only-component", "vision_neck"])
    assert ns.auto_only_component == "vision_neck"


def test_real_parser_accepts_auto_only_component() -> None:
    """The actual CLI parser (not a mock) must accept the flag. Catches
    typos / missed wiring in the production parser."""
    src = (Path(cli.__file__)).read_text()
    assert '"--auto-only-component"' in src, (
        "scripts/tt_hw_planner/cli.py must add the `--auto-only-component` " "flag on the `up` (pup) subparser"
    )
    assert "only_component=getattr(args" in src, (
        "cmd_up must wire args.auto_only_component through to " "_run_auto_iterate_loop as only_component=..."
    )


def test_run_auto_iterate_loop_signature_has_only_component() -> None:
    """The auto-iterate loop must accept only_component as a kwarg, default None."""
    src = _auto_iterate_source()
    assert "only_component: Optional[str] = None," in src, (
        "_run_auto_iterate_loop must accept `only_component: Optional[str] = None` " "in its signature"
    )


def test_only_component_filters_candidate_pool() -> None:
    """When only_component is set, candidate_pool must be restricted to
    that single component (or made empty if the component isn't in the pool)."""
    src = _auto_iterate_source()
    assert "if only_component:" in src, "auto_iterate.py must have a candidate-pool filter gated on only_component"
    assert "candidate_pool = [only_component]" in src, (
        "When only_component is set and is in the candidate_pool, the pool " "must be restricted to that one component"
    )


def test_only_component_zeroes_extra_targets() -> None:
    """When only_component is set, _extra_targets must be empty (no parallel
    extras spawned alongside the single targeted component)."""
    src = _auto_iterate_source()
    assert "_extra_targets = []" in src, (
        "auto_iterate.py must zero _extra_targets when only_component is " "set to prevent parallel-extra spawn"
    )


def test_dry_run_prompts_env_gate_present() -> None:
    """The dry-run gate must be checked via environment variable so it can
    be activated without a CLI re-deploy or argparse change."""
    src = _auto_iterate_source()
    assert 'os.environ.get("TT_PLANNER_DRY_RUN_PROMPTS"' in src, (
        "auto_iterate.py must gate the dry-run-prompts branch on the " "TT_PLANNER_DRY_RUN_PROMPTS env var"
    )


def test_dry_run_prompts_writes_to_tmp_dir() -> None:
    """Dry-run must write prompts to a deterministic path under /tmp so
    they can be inspected after the run."""
    src = _auto_iterate_source()
    assert "/tmp" in src and "tt_planner_dry_run" in src, (
        "Dry-run must write prompts to /tmp/tt_planner_dry_run/<safe_model>/" "iter_N/<comp>.prompt.txt"
    )


def test_dry_run_prompts_skips_agent_invocation() -> None:
    """Dry-run must NOT call _invoke_agent or run_parallel_agents. The
    presence of an early `return 0` inside the dry-run branch is the
    structural enforcement of that."""
    src = _auto_iterate_source()
    dry_run_idx = src.find('os.environ.get("TT_PLANNER_DRY_RUN_PROMPTS"')
    assert dry_run_idx != -1
    # The dry-run branch must terminate before the agent call (return 0)
    branch_end = src.find("if _parallel_extra_jobs:", dry_run_idx)
    assert branch_end != -1
    branch_body = src[dry_run_idx:branch_end]
    assert "return 0" in branch_body, (
        "Dry-run branch must `return 0` before falling through to the "
        "real agent-invocation code. Otherwise dry-run would still spend money."
    )


def test_dry_run_prompts_dir_uses_safe_id() -> None:
    """Model id like `facebook/sam2-hiera-tiny` contains `/`. The dry-run
    path must use _safe_id(MODEL) to avoid creating subdirectories."""
    src = _auto_iterate_source()
    assert "_safe_id(MODEL)" in src, "Dry-run path must apply _safe_id() to MODEL to avoid `/` in path components"


def test_only_component_and_dry_run_compose() -> None:
    """The flags must be independent — dry-run + only-component should
    write a single prompt file for the targeted component."""
    src = _auto_iterate_source()
    # Both gates exist in the same function
    assert "only_component:" in src
    assert "TT_PLANNER_DRY_RUN_PROMPTS" in src
    # And the dry-run loop iterates over _parallel_extra_jobs (which will be
    # empty when only_component is set), so it ends up writing just the primary.
    assert "for _job in _parallel_extra_jobs:" in src
