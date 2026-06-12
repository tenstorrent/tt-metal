"""Pin the zero-flag `auto-up` subcommand wiring.

The user wants a one-line entry: `tt-hw-planner auto-up <model_id>`.
The brain handles every decision. Power users can still fall back to
`up` with explicit flags.

These tests verify the subcommand is registered and locks in the
brain-orchestrated defaults."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_auto_up_subcommand_registered() -> None:
    """`auto-up` must appear in the cli's subcommand list and accept
    only the positional model_id (plus optional --box)."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "cli.py").read_text()
    # The subparser registration.
    assert '"auto-up"' in src, "cli must register `auto-up` subcommand"
    # The handler function.
    assert "def cmd_bringup(" in src, "cli must define cmd_bringup as the handler"
    assert "set_defaults(func=cmd_bringup)" in src, "auto-up must wire to cmd_bringup"


def test_auto_up_locks_in_brain_defaults() -> None:
    """Pin: the auto-up handler sets every brain-orchestrated default
    so the user doesn't need any flags."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "cli.py").read_text()
    handler_idx = src.find("def cmd_bringup(")
    assert handler_idx >= 0
    handler_body = src[handler_idx : handler_idx + 4000]

    # Brain-orchestrated defaults that MUST be locked in:
    assert '"auto": True' in handler_body, "auto must be True"
    assert '"auto_agent": "claude"' in handler_body, "auto_agent must be claude"
    assert '"auto_model_tiered": True' in handler_body, "tiered model must be on"
    assert '"auto_max_iters": 24' in handler_body, "iter budget must be 24"
    assert '"auto_max_attempts_per_component": 5' in handler_body, "per-component cap must be 5"
    assert '"isolation": "worktree"' in handler_body, "isolation must be worktree"


def test_auto_up_delegates_to_cmd_up() -> None:
    """Pin: auto-up doesn't duplicate the bring-up logic — it sets
    defaults and delegates to cmd_up. Without this, every bug fix to
    cmd_up would need to be ported to cmd_bringup."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "cli.py").read_text()
    handler_idx = src.find("def cmd_bringup(")
    assert handler_idx >= 0
    # Scan until the NEXT def so we cover the whole function regardless
    # of length growth.
    next_def = src.find("\ndef ", handler_idx + 5)
    end = next_def if next_def > 0 else handler_idx + 8000
    handler_body = src[handler_idx:end]
    assert "return cmd_up(" in handler_body, "cmd_bringup must delegate to cmd_up, not duplicate its logic"


def test_auto_up_help_mentions_brain_orchestration() -> None:
    """User-visible help text must make it clear the brain is orchestrating
    so the user knows what they're getting from the zero-flag command."""
    src = (_REPO_ROOT / "scripts" / "tt_hw_planner" / "cli.py").read_text()
    # Find the auto-up help string.
    paut_idx = src.find('"auto-up"')
    assert paut_idx >= 0
    help_region = src[paut_idx : paut_idx + 800]
    assert "brain" in help_region.lower()
    # box + mesh are now mandatory — the help must say so.
    assert "required" in help_region.lower()


def test_auto_up_runtime_smoke() -> None:
    """The subcommand parses without error when invoked with --help.
    Light-touch end-to-end check that registration is healthy."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.tt_hw_planner", "auto-up", "--help"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=30,
    )
    assert proc.returncode == 0, f"auto-up --help failed: {proc.stderr}"
    assert "model_id" in proc.stdout
    assert "HuggingFace" in proc.stdout


def test_auto_up_exposes_mesh_flag() -> None:
    """auto-up must expose --mesh (now a required flag, not an override)."""
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.tt_hw_planner", "auto-up", "--help"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=30,
    )
    assert proc.returncode == 0
    assert "--mesh" in proc.stdout, "auto-up must expose --mesh"


def test_auto_up_requires_box_and_mesh() -> None:
    """The user-mandated contract: auto-up will NOT run without an
    explicit --box and --mesh. Omitting either must error (argparse
    exit code 2) and name the missing flag — no silent QB2 / auto-mesh
    default."""
    import subprocess

    def _run(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "scripts.tt_hw_planner", "auto-up", "some/model", *extra],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
            timeout=30,
        )

    # neither flag
    p = _run()
    assert p.returncode == 2, p.stderr
    assert "--box" in p.stderr and "--mesh" in p.stderr

    # box only -> mesh still required
    p = _run("--box", "QB2")
    assert p.returncode == 2, p.stderr
    assert "--mesh" in p.stderr

    # mesh only -> box still required
    p = _run("--mesh", "2,2")
    assert p.returncode == 2, p.stderr
    assert "--box" in p.stderr


def test_promote_requires_box_and_mesh() -> None:
    """Same mandatory-flag contract for promote."""
    import subprocess

    p = subprocess.run(
        [sys.executable, "-m", "scripts.tt_hw_planner", "promote", "some/model"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        timeout=30,
    )
    assert p.returncode == 2, p.stderr
    assert "--box" in p.stderr and "--mesh" in p.stderr
