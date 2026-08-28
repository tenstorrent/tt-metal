#!/usr/bin/env python3
"""Tests for the failed-job -> Jira title derivation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from failure_title import family, name, title  # noqa: E402

PRE = "build-test-publish (Ubuntu 22.04) / release-demo-tests / release-demo-tests / "
MULTI = "build-test-publish (Ubuntu 22.04) / release-demo-tests / release-demo-tests / models-e2e-tests-multihost / "


def test_family_is_the_second_segment_humanised():
    assert family(PRE + "Gemma-4-31B e2e tests [bh_p150]") == "demo tests"
    assert family("a / create-docker-release-image (release-models) / smoke [x]") == "create docker release image"
    assert family("unnested-job") == ""


def test_name_strips_the_runner_tag_and_test_kind():
    assert name(PRE + "Gemma-4-31B e2e tests [bh_quietbox_2]") == "Gemma-4-31B"
    assert name(MULTI + "TT-DiT Wan2.2-T2V-A14B multihost quad e2e tests [bh_sc4]") == "TT-DiT Wan2.2-T2V-A14B"


def test_a_runner_qualifier_after_e2e_tests_is_dropped():
    assert name(PRE + "TT-DiT Wan2.2-T2V-A14B e2e tests (BH QuietBox 2) [bh_quietbox_2]") == "TT-DiT Wan2.2-T2V-A14B"


def test_a_parenthetical_that_is_the_model_is_kept():
    """ "Demo Test with Perf Metrics" alone identifies nothing."""
    job = MULTI + "Demo Test with Perf Metrics (DeepSeek V3 B1 Supercluster 16 aka Superpod 4) [bh_sc16]"
    assert name(job) == "Demo Test with Perf Metrics (DeepSeek V3 B1 Supercluster 16 aka Superpod 4)"


def test_one_suite_is_named_with_its_models():
    jobs = [PRE + "Gemma-4-31B e2e tests [r]", PRE + "GPT-OSS 120B e2e tests [r]"]
    assert title(jobs, "v1.0") == "Release v1.0 — 2 demo tests failed: Gemma-4-31B, GPT-OSS 120B"


def test_a_single_failure_reads_singular():
    assert title([PRE + "Gemma-4-31B e2e tests [r]"], "stable") == "Release stable — 1 demo test failed: Gemma-4-31B"


def test_overlong_names_are_skipped_not_truncated():
    """One real job name is 74 characters and would crowd out every other."""
    long_job = MULTI + "Demo Test with Perf Metrics (DeepSeek V3 B1 Supercluster 16 aka Superpod 4) [r]"
    got = title([long_job, PRE + "Gemma-4-31B e2e tests [r]"], "v1.0")
    assert got == "Release v1.0 — 2 demo tests failed: Gemma-4-31B +1 more"


def test_every_name_overlong_falls_back_to_the_bare_count():
    long_job = MULTI + "Demo Test with Perf Metrics (DeepSeek V3 B1 Supercluster 16 aka Superpod 4) [r]"
    assert title([long_job], "v1.0") == "Release v1.0 — 1 demo test failed"


def test_mixed_suites_are_listed_with_counts():
    jobs = [
        PRE + "Gemma-4-31B e2e tests [r]",
        PRE + "GPT-OSS 120B e2e tests [r]",
        "build-test-publish (Ubuntu 22.04) / create-docker-release-image (release-models) / smoke-test [r]",
    ]
    assert title(jobs, "main") == "Release main — 3 jobs failed: demo tests (2), create docker release image"


def test_unnested_jobs_are_named_by_themselves():
    assert title(["create-docker-release-image", "publish-docs"], "stable") == (
        "Release stable — 2 jobs failed: create-docker-release-image, publish-docs"
    )


def test_duplicate_models_on_two_runners_count_once_in_the_names():
    jobs = [PRE + "Gemma-4-31B e2e tests [bh_p150]", PRE + "Gemma-4-31B e2e tests [wh_n150]"]
    assert title(jobs, "stable") == "Release stable — 2 demo tests failed: Gemma-4-31B"


def test_title_never_exceeds_the_jira_limit():
    jobs = [PRE + f"Model-{i}-{'x' * 20} e2e tests [r]" for i in range(200)]
    assert len(title(jobs, "v1.0")) <= 255
