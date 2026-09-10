"""Tests for the template-list / template-promote / template-demote
CLI commands (Item 8 follow-on)."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from scripts.tt_hw_planner._cli_helpers.family_template_registry import (
    register_template,
)
from scripts.tt_hw_planner.cli import (
    cmd_template_demote,
    cmd_template_list,
    cmd_template_promote,
)


@contextmanager
def _registry_in_tmp(tmp_path: Path):
    """Redirect the registry's repo-root resolution to tmp_path for
    the duration of a test."""
    with patch(
        "scripts.tt_hw_planner._cli_helpers.family_template_registry._registry_path",
        return_value=tmp_path / "learned_chained_templates.json",
    ):
        yield


# ─── cmd_template_list ──────────────────────────────────────────────


def test_list_empty_registry(tmp_path: Path, capsys) -> None:
    with _registry_in_tmp(tmp_path):
        rc = cmd_template_list(argparse.Namespace(all=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "no chained-template" in out


def test_list_shows_registered_entry(tmp_path: Path, capsys) -> None:
    with _registry_in_tmp(tmp_path):
        register_template(
            family_key="sam2",
            template_demo_source="models/demos/sam2/demo.py",
            source_model_id="facebook/sam2-hiera-tiny",
            final_pcc=0.983,
        )
        rc = cmd_template_list(argparse.Namespace(all=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "sam2" in out
    assert "REGISTERED" in out  # not yet promoted


def test_list_skips_demoted_by_default(tmp_path: Path, capsys) -> None:
    from scripts.tt_hw_planner._cli_helpers.family_template_registry import demote_template

    with _registry_in_tmp(tmp_path):
        register_template(family_key="sam2", template_demo_source="x", source_model_id="m1")
        demote_template(family_key="sam2", reason="regressed")
        rc = cmd_template_list(argparse.Namespace(all=False))
    out = capsys.readouterr().out
    assert "DEMOTED" not in out
    assert "no non-demoted entries" in out


def test_list_shows_demoted_when_all(tmp_path: Path, capsys) -> None:
    from scripts.tt_hw_planner._cli_helpers.family_template_registry import demote_template

    with _registry_in_tmp(tmp_path):
        register_template(family_key="sam2", template_demo_source="x", source_model_id="m1")
        demote_template(family_key="sam2", reason="regressed")
        rc = cmd_template_list(argparse.Namespace(all=True))
    out = capsys.readouterr().out
    assert "DEMOTED" in out


# ─── cmd_template_promote ───────────────────────────────────────────


def test_promote_returns_error_for_unknown_family(tmp_path: Path, capsys) -> None:
    with _registry_in_tmp(tmp_path):
        rc = cmd_template_promote(argparse.Namespace(family_key="never-registered"))
    assert rc == 1
    err = capsys.readouterr().err
    assert "could not promote" in err


def test_promote_force_promotes_with_threshold_1(tmp_path: Path, capsys) -> None:
    """Operator force-promotes a single-confirmed entry. Bypasses
    the default 2-model threshold by using threshold=1."""
    with _registry_in_tmp(tmp_path):
        register_template(family_key="sam2", template_demo_source="x", source_model_id="m1")
        rc = cmd_template_promote(argparse.Namespace(family_key="sam2"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "promoted family=sam2" in out


def test_promote_returns_error_for_empty_family_key(capsys) -> None:
    rc = cmd_template_promote(argparse.Namespace(family_key=""))
    assert rc == 2


# ─── cmd_template_demote ────────────────────────────────────────────


def test_demote_returns_error_for_unknown_family(tmp_path: Path, capsys) -> None:
    with _registry_in_tmp(tmp_path):
        rc = cmd_template_demote(argparse.Namespace(family_key="never-registered", reason=""))
    assert rc == 1


def test_demote_succeeds_with_reason(tmp_path: Path, capsys) -> None:
    with _registry_in_tmp(tmp_path):
        register_template(family_key="sam2", template_demo_source="x", source_model_id="m1")
        rc = cmd_template_demote(argparse.Namespace(family_key="sam2", reason="HF v5 regression"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "demoted family=sam2" in out
    assert "HF v5" in out


def test_demote_returns_error_for_empty_family_key() -> None:
    rc = cmd_template_demote(argparse.Namespace(family_key="", reason=""))
    assert rc == 2
