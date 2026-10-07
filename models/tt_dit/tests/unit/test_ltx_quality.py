# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host coverage for the LTX serving quality-tier env expansion (utils.ltx.apply_quality_env).

These helpers mutate import-time configuration, so a typo or a precedence regression would only
surface deep in the device-serving stack. Exercise the tier mapping, the invalid-tier guard, and
the LTX_FAST fallback on a throwaway env dict so os.environ is never touched."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from ...utils import ltx

PIPELINE = Path(__file__).resolve().parents[2] / "pipelines/ltx/pipeline_ltx_distilled.py"


def test_high_tier_keeps_pipeline_defaults():
    # high = pipeline defaults: no quant/sigma overrides, only the perf-only knobs.
    env = {"LTX_QUALITY": "high"}
    ltx.apply_quality_env(env)
    assert "LTX_QUANT" not in env
    assert "LTX_S1_SIGMAS" not in env
    assert "LTX_S2_SIGMAS" not in env
    assert env["LTX_TRACED"] == "1"
    assert env["TT_DIT_HOST_WEIGHT_CACHE"] == "1"


def test_medium_tier_is_the_fast_bundle():
    env = {"LTX_QUALITY": "medium"}
    ltx.apply_quality_env(env)
    assert env["LTX_QUANT"] == ltx.FAST_QUANT
    assert env["LTX_S1_SIGMAS"] == ltx.FAST_S1_SIGMAS
    assert env["LTX_S2_SIGMAS"] == ltx.FAST_S2_SIGMAS


def test_fast_tier_collapses_stage1():
    env = {"LTX_QUALITY": "fast"}
    ltx.apply_quality_env(env)
    assert env["LTX_QUANT"] == ltx.FAST_QUANT
    assert env["LTX_S1_SIGMAS"] == ltx.FAST_S1_SIGMAS_N3
    assert env["LTX_S2_SIGMAS"] == ltx.FAST_S2_SIGMAS


def test_tier_is_case_and_whitespace_insensitive():
    env = {"LTX_QUALITY": "  Medium  "}
    ltx.apply_quality_env(env)
    assert env["LTX_QUANT"] == ltx.FAST_QUANT


def test_invalid_tier_raises(expect_error):
    with expect_error(ValueError, "LTX_QUALITY must be one of"):
        ltx.apply_quality_env({"LTX_QUALITY": "ultra"})


def test_ltx_fast_fallback_when_quality_unset():
    # LTX_QUALITY unset must still honor the legacy LTX_FAST=1 switch.
    env = {"LTX_FAST": "1"}
    ltx.apply_quality_env(env)
    assert env["LTX_QUANT"] == ltx.FAST_QUANT
    assert env["LTX_S1_SIGMAS"] == ltx.FAST_S1_SIGMAS
    assert env["LTX_S2_SIGMAS"] == ltx.FAST_S2_SIGMAS


def test_no_tier_no_fast_is_a_noop():
    env: dict[str, str] = {}
    ltx.apply_quality_env(env)
    assert env == {}


def test_explicit_var_survives_the_tier():
    # setdefault: an explicitly-set var must win over the tier's value.
    env = {"LTX_QUALITY": "medium", "LTX_QUANT": "custom"}
    ltx.apply_quality_env(env)
    assert env["LTX_QUANT"] == "custom"


def _pipeline_sigma_defs():
    # The pipeline module needs ttnn to import; pull the schedule defaults and the override parser
    # out of its AST so this stays a host-only check.
    tree = ast.parse(PIPELINE.read_text())
    keep = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name == "_sigma_override")
        or (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_DEFAULT_S2_SIGMAS")
    ]
    ns = {"os": os}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(PIPELINE), "exec"), ns)
    return ns


def test_stage2_default_is_two_steps(monkeypatch):
    ns = _pipeline_sigma_defs()
    assert ns["_DEFAULT_S2_SIGMAS"] == [0.909375, 0.421875, 0.0]
    monkeypatch.delenv("LTX_S2_SIGMAS", raising=False)
    assert ns["_sigma_override"]("LTX_S2_SIGMAS", ns["_DEFAULT_S2_SIGMAS"]) == [0.909375, 0.421875, 0.0]


@pytest.mark.parametrize("raw", ["0.909375,0.725,0.421875,0.0", ltx.FAST_S2_SIGMAS])
def test_stage2_schedule_still_selectable(raw, monkeypatch):
    ns = _pipeline_sigma_defs()
    monkeypatch.setenv("LTX_S2_SIGMAS", raw)
    assert ns["_sigma_override"]("LTX_S2_SIGMAS", ns["_DEFAULT_S2_SIGMAS"]) == [float(x) for x in raw.split(",")]
