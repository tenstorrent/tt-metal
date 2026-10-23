#!/usr/bin/env python3
"""Tests for validate_test_combination (the fail-fast filter pre-check)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "validate_test_combination.py"


@pytest.fixture
def tests_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "tests.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            - name: A
              model: meta-llama/Llama-3.1-8B-Instruct
              skus:
                wh_llmbox:
                  timeout: 10
                  tier: 2
                bh_quietbox_2:
                  timeout: 10
                  tier: 2
            - name: B
              model: google/gemma-4-E2B-it
              skus:
                wh_n150:
                  timeout: 10
                  tier: 3
            - name: C
              model: openai/gpt-oss-120b
              skus:
                wh_galaxy_perf:
                  timeout: 10
                  tier: 1
            """
        )
    )
    return path


def run(tests_yaml: Path, enabled: str, tier: str, model: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tests_yaml), enabled, tier, model],
        capture_output=True,
        text=True,
        check=False,
    )


def test_all_matches(tests_yaml: Path):
    r = run(tests_yaml, "wh_llmbox,bh_quietbox_2,wh_n150,wh_galaxy_perf", "all", "all")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "match the selected combination" in r.stdout


def test_substring_model_match(tests_yaml: Path):
    # "gemma" is a case-insensitive substring of google/gemma-4-E2B-it.
    r = run(tests_yaml, "wh_n150", "all", "GEMMA")
    assert r.returncode == 0, r.stdout + r.stderr


def test_comma_separated_or_list(tests_yaml: Path):
    r = run(tests_yaml, "wh_llmbox,wh_n150", "all", "llama,does-not-exist")
    assert r.returncode == 0, r.stdout + r.stderr


def test_tier_filter_selects(tests_yaml: Path):
    r = run(tests_yaml, "wh_galaxy_perf", "1", "all")
    assert r.returncode == 0, r.stdout + r.stderr


def test_tier_filter_excludes(tests_yaml: Path):
    # gpt-oss-120b is tier 1 only; asking for tier 3 on its SKU yields nothing.
    r = run(tests_yaml, "wh_galaxy_perf", "3", "all")
    assert r.returncode == 1
    assert "matches no tests" in r.stdout


def test_sku_filter_excludes(tests_yaml: Path):
    # Llama is not configured on wh_n150 in the fixture.
    r = run(tests_yaml, "wh_n150", "all", "llama")
    assert r.returncode == 1


def test_no_match_lists_valid_options(tests_yaml: Path):
    r = run(tests_yaml, "wh_n150", "all", "Qwen/Qwen3-32B")
    assert r.returncode == 1
    # The diagnostic must surface the valid catalogue.
    assert "Valid models:" in r.stdout
    assert "Valid tiers:" in r.stdout
    assert "Valid SKUs:" in r.stdout
    assert "Valid combinations" in r.stdout


def test_usage_error_on_wrong_argc(tests_yaml: Path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), str(tests_yaml), "wh_n150"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2
    assert "usage:" in r.stdout
