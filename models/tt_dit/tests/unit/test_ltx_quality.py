# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host coverage for the LTX serving quality-tier env expansion (utils.ltx.apply_quality_env).

These helpers mutate import-time configuration, so a typo or a precedence regression would only
surface deep in the device-serving stack. Exercise the tier mapping, the invalid-tier guard, and
the LTX_FAST fallback on a throwaway env dict so os.environ is never touched."""

from __future__ import annotations

from ...utils import ltx


def test_high_tier_keeps_pipeline_defaults():
    # high = shipped baseline: no quant/sigma overrides, only the perf-only knobs.
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
