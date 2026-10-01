# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Artifact metadata controls with an in-memory collector; no benchmark is run."""

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import report as reporter


@pytest.mark.parametrize("device_name", ["P150x4", "QB2"])
def test_report_preserves_explicit_hardware_and_subset_scope(monkeypatch, device_name):
    # Unit fixture only; the fake collector writes no measurement artifact.
    summary = dict(
        completed_samples=64,
        requested_samples=64,
        accuracy=0.5,
        mean_ttft_s=None,
        mean_stream_decode_tokens_per_s=None,
        aggregate_output_tokens_per_s=None,
        measurement_start="2026-09-29T00:00:00+00:00",
        measurement_end="2026-09-29T00:01:00+00:00",
        scope="first 64 of 1319",
        checkpoint_revision="unit-fixture",
        dataset_revision="unit-fixture",
        dataset_sha256="unit-fixture",
        max_output_tokens=512,
        ignore_eos=False,
        timing_note="unit fixture, not measured performance",
        accuracy_floor=0.8,
        passed=False,
    )
    recorded = {}
    fake = SimpleNamespace(
        add_measurement=lambda *args: None, save_partial_run_json=lambda profiler, **kwargs: recorded.update(kwargs)
    )
    monkeypatch.setattr(reporter, "BenchmarkData", lambda: fake)
    reporter.report(summary, device_name=device_name)
    assert recorded["device_name"] == device_name
    assert recorded["dataset_name"] == "GSM8K: first 64 of 1319"
    assert recorded["config_params"]["passed"] is False
    assert recorded["config_params"]["completed_samples"] == 64


def test_report_rejects_logical_mesh_as_hardware_identity(expect_error):
    with expect_error(ValueError, match="qualified physical allocation"):
        reporter.report({}, device_name="1x4")


def test_report_rejects_incomplete_run_with_valid_hardware(expect_error):
    with expect_error(ValueError, match="incomplete task run"):
        reporter.report({"completed_samples": 63, "requested_samples": 64}, device_name="P150x4")


def test_cli_requires_explicit_hardware(expect_error, tmp_path):
    with expect_error(SystemExit):
        reporter.main([str(tmp_path / "summary.json")])
