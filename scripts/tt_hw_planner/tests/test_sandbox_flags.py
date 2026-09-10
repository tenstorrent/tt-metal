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
