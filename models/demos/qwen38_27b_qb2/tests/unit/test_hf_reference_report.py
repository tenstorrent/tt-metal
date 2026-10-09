# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep real Transformers loading metadata from hiding a reference failure."""

import json

from models.demos.qwen38_27b_qb2.demo.probe_hf_head import save_report


def test_empty_loading_sets_allow_reference_progress(tmp_path):
    report = dict(state="reference_forward", loading_info=dict(missing_keys=set(), unexpected_keys=set()))
    save_report(tmp_path, report)
    assert json.loads((tmp_path / "progress.json").read_text()) == dict(
        state="reference_forward", loading_info=dict(missing_keys=[], unexpected_keys=[])
    )


def test_failed_load_preserves_nonempty_key_evidence(tmp_path):
    report = dict(state="loading_reference")
    save_report(tmp_path, report)
    report.update(state="failed", loading_info=dict(missing_keys={"layer.b", "layer.a"}), error="ValueError")
    save_report(tmp_path, report)
    result = json.loads((tmp_path / "progress.json").read_text())
    assert result["state"] == "failed"
    assert result["error"] == "ValueError"
    assert result["loading_info"]["missing_keys"] == ["layer.a", "layer.b"]
    assert not (tmp_path / "progress.json.tmp").exists()


def test_unknown_metadata_cannot_silently_become_a_string(tmp_path, expect_error):
    save_report(tmp_path, dict(state="loading_reference"))
    before = (tmp_path / "progress.json").read_bytes()
    with expect_error(TypeError, "Unsupported reference report value"):
        save_report(tmp_path, dict(state="completed", unknown=object()))
    assert (tmp_path / "progress.json").read_bytes() == before
