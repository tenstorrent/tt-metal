#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional tests for load_ttnn_ops_data_v2.py.

Tests the end-to-end workflows and manifest permutations documented in
model_tracer/GUIDE.md.  Every test creates a realistic manifest YAML on disk,
runs the real resolution / reconstruction code, and asserts outcomes.

Organisation:
  §1  Manifest target permutations (all forms from the Resolution rules table)
  §2  Registry status lifecycle (draft → active → deprecated)
  §3  Model-filtered reconstruction pipeline (resolve → reconstruct → merge)
  §4  Load → draft append → promote → reconstruct end-to-end
  §5  Production manifest mirroring (tests against the actual repo structure)
  §6  JSON round-trip: load → reconstruct → verify
"""

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.sweep_framework.load_ttnn_ops_data_v2 import (
    _append_registry_entries,
    _load_manifest,
    _resolve_manifest_with_models,
    find_config_line_numbers,
    load_data,
    reconstruct_from_manifest,
    reconstruct_from_trace_run,
    resolve_manifest,
    verify_reconstruction,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _write_manifest(path, content):
    path.write_text(textwrap.dedent(content))


def _make_master_json(
    tmp_path,
    ops,
    *,
    source="models/demos/deepseek_v3/demo/demo.py",
    board_type="Wormhole",
    device_series="n300",
    card_count=1,
    filename="master.json",
):
    """Build a realistic master JSON matching the tracer output format.

    Args:
        ops: dict mapping op_name → list of (config_hash, arguments_dict) tuples.
    """
    operations = {}
    for op_name, configs in ops.items():
        cfgs = []
        for config_hash, arguments in configs:
            cfgs.append(
                {
                    "arguments": arguments,
                    "config_hash": config_hash,
                    "executions": [
                        {
                            "source": source,
                            "machine_info": {
                                "board_type": board_type,
                                "device_series": device_series,
                                "card_count": card_count,
                            },
                            "count": 1,
                        }
                    ],
                }
            )
        operations[op_name] = {"configurations": cfgs}

    data = {
        "operations": operations,
        "metadata": {
            "models": [source],
            "unique_operations": len(operations),
            "total_configurations": sum(len(c) for c in ops.values()),
            "trace_uid": "fixture-trace-uid",
        },
    }
    p = tmp_path / filename
    p.write_text(json.dumps(data, indent=2))
    return str(p)


# =====================================================================
# §1  Manifest Target Permutations
#     (GUIDE.md → Sweep Manifest → Targets → Resolution rules table)
# =====================================================================


class TestTargetFormModelStringNoTrace:
    """Form: model: X (no trace) → latest active per device_series."""

    def test_single_model_single_hw(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 500
                status: active
                models: [whisper, llama]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == [500]

    def test_picks_highest_trace_id_when_multiple_active(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 100
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
              - trace_id: 300
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
              - trace_id: 200
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == [300]

    def test_resolves_per_device_series(self, tmp_path):
        """One trace per unique device_series where model appears."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 10
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
              - trace_id: 20
                status: active
                models: [whisper]
                hardware: {board_type: Blackhole, device_series: p150b, card_count: 1}
              - trace_id: 30
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        # n300: latest=30, p150b: latest=20
        assert set(ids) == {30, 20}


class TestTargetFormModelStringWithHardware:
    """Form: model: X, hardware: H → latest active where device_series == H."""

    def test_filters_to_specified_hw(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                  hardware: p150b
            registry:
              - trace_id: 10
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
              - trace_id: 20
                status: active
                models: [whisper]
                hardware: {board_type: Blackhole, device_series: p150b, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == [20]

    def test_no_match_on_hw(self, tmp_path, capsys):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                  hardware: tt-galaxy-wh
            registry:
              - trace_id: 10
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == []
        assert "no active traces match" in capsys.readouterr().out.lower()


class TestTargetFormModelWithPinnedTrace:
    """Form: model: X, trace: N → trace N exactly, filtered to model X."""

    def test_single_pinned(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - model: deepseek_v3
                  trace: 35
            registry: []
            """,
        )
        assert resolve_manifest(str(f)) == [35]
        models = _resolve_manifest_with_models(str(f))
        assert models == {35: {"deepseek_v3"}}

    def test_pinned_list(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: [whisper, llama-3.2-1b-instruct]
                  trace: [538, 1]
            registry: []
            """,
        )
        ids = resolve_manifest(str(f))
        assert set(ids) == {538, 1}
        models = _resolve_manifest_with_models(str(f))
        assert models[538] == {"whisper", "llama-3.2-1b-instruct"}
        assert models[1] == {"whisper", "llama-3.2-1b-instruct"}

    def test_pinned_bypasses_registry(self, tmp_path):
        """Pinned traces don't need a registry entry and ignore status."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                  trace: 999
            registry:
              - trace_id: 999
                status: deprecated
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == [999]


class TestTargetFormTraceOnly:
    """Form: trace: N (no model) → all configs in trace, no model filter."""

    def test_trace_only(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - trace: 42
            registry: []
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == [42]
        models = _resolve_manifest_with_models(str(f))
        assert models[42] is None  # None = all models


class TestTargetFormModelList:
    """Form: model: [X, Y, Z] (no trace) → each resolved independently."""

    def test_all_in_same_trace(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model:
                    - whisper
                    - llama-3.2-1b-instruct
                    - efficientnetb0
            registry:
              - trace_id: 538
                status: active
                models: [whisper, llama-3.2-1b-instruct, efficientnetb0, falcon7b]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == [538]
        models = _resolve_manifest_with_models(str(f))
        assert models[538] == {"whisper", "llama-3.2-1b-instruct", "efficientnetb0"}

    def test_models_split_across_traces(self, tmp_path):
        """Different models may resolve to different traces."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model:
                    - deepseek_v3
                    - whisper
            registry:
              - trace_id: 35
                status: active
                models: [deepseek_v3]
                hardware: {board_type: Wormhole, device_series: tt-galaxy-wh, card_count: 32}
              - trace_id: 538
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert set(ids) == {35, 538}
        models = _resolve_manifest_with_models(str(f))
        assert models[35] == {"deepseek_v3"}
        assert models[538] == {"whisper"}

    def test_partial_model_match(self, tmp_path, capsys):
        """Some models in the list match, others don't — warnings for misses."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model:
                    - whisper
                    - nonexistent_model
            registry:
              - trace_id: 538
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert 538 in ids
        assert "no active traces match model 'nonexistent_model'" in capsys.readouterr().out.lower()


class TestMixedTargetEntries:
    """Entries mixing pinned and auto-resolved in the same group."""

    def test_mixed_pinned_and_resolved(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: deepseek_v3
                  trace: 35
                - model: whisper
            registry:
              - trace_id: 538
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert set(ids) == {35, 538}

    def test_both_scopes_combined(self, tmp_path):
        """scope=None merges lead_models + model_traced."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - model: deepseek_v3
                  trace: 35
              model_traced:
                - model: whisper
                  trace: 538
            registry: []
            """,
        )
        all_ids = resolve_manifest(str(f), scope=None)
        lead_ids = resolve_manifest(str(f), scope="lead_models")
        traced_ids = resolve_manifest(str(f), scope="model_traced")
        assert set(all_ids) == {35, 538}
        assert lead_ids == [35]
        assert traced_ids == [538]

    def test_cross_scope_dedup(self, tmp_path):
        """Same trace_id in both groups gets deduplicated."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - model: deepseek_v3
                  trace: 538
              model_traced:
                - model: whisper
                  trace: 538
            registry: []
            """,
        )
        ids = resolve_manifest(str(f), scope=None)
        assert ids == [538]
        models = _resolve_manifest_with_models(str(f))
        assert models[538] == {"deepseek_v3", "whisper"}


class TestMultipleEntriesSameGroup:
    """Multiple entries within a single scope group."""

    def test_separate_entries_per_model(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                  trace: 1
                - model: llama-3.2-1b-instruct
                  trace: 538
                - model: deepseek_v3
                  trace: 35
            registry: []
            """,
        )
        ids = resolve_manifest(str(f))
        assert set(ids) == {1, 538, 35}

    def test_separate_entries_accumulate_models_for_same_trace(self, tmp_path):
        """Two entries both pinning trace 538 → model sets are merged."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                  trace: 538
                - model: llama-3.2-1b-instruct
                  trace: 538
            registry: []
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == [538]
        models = _resolve_manifest_with_models(str(f))
        assert models[538] == {"whisper", "llama-3.2-1b-instruct"}


# =====================================================================
# §2  Registry Status Lifecycle
#     (GUIDE.md → Registry → status: active | draft | deprecated)
# =====================================================================


class TestRegistryStatusLifecycle:
    """Draft → active → deprecated lifecycle behaviour."""

    def test_draft_invisible_to_auto_resolve(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 100
                status: draft
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == []

    def test_deprecated_invisible_to_auto_resolve(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 100
                status: deprecated
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == []

    def test_only_active_considered(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 50
                status: draft
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
              - trace_id: 100
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
              - trace_id: 200
                status: deprecated
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == [100]

    def test_promote_draft_to_active_changes_resolution(self, tmp_path):
        """Simulates the promote workflow: initially draft → promote → re-resolve."""
        f = tmp_path / "m.yaml"
        # Phase 1: draft only
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: new_model
            registry:
              - trace_id: 999
                status: draft
                models: [new_model]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == []

        # Phase 2: promote to active
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: new_model
            registry:
              - trace_id: 999
                status: active
                models: [new_model]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == [999]

    def test_pinned_trace_ignores_status(self, tmp_path):
        """Pinned traces work regardless of status."""
        for status in ("draft", "active", "deprecated"):
            f = tmp_path / f"m_{status}.yaml"
            _write_manifest(
                f,
                f"""\
                targets:
                  model_traced:
                    - model: whisper
                      trace: 42
                registry:
                  - trace_id: 42
                    status: {status}
                    models: [whisper]
                    hardware: {{board_type: Wormhole, device_series: n300, card_count: 1}}
                """,
            )
            assert resolve_manifest(str(f)) == [42], f"Failed for status={status}"

    def test_new_active_supersedes_old_active(self, tmp_path):
        """When a newer trace is loaded and promoted, it takes over."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 538
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
              - trace_id: 600
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        assert resolve_manifest(str(f)) == [600]


# =====================================================================
# §3  Model-Filtered Reconstruction Pipeline
#     (GUIDE.md → Step 5 + "Important" note about model filtering)
# =====================================================================


class TestModelFilteredReconstruction:
    """reconstruct-manifest only includes configs for the listed models."""

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.reconstruct_from_trace_run")
    def test_single_model_from_multi_model_trace(self, mock_recon, tmp_path):
        """Trace 538 has 19 models but target lists only [whisper]."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                  trace: 538
            registry: []
            """,
        )
        mock_recon.return_value = {
            "operations": {"ttnn::add": {"configurations": [{"config_hash": "h1", "arguments": {}}]}},
            "metadata": {"models": ["models/demos/audio/whisper/demo/demo.py"]},
        }
        result = reconstruct_from_manifest(str(f), scope="model_traced")
        mock_recon.assert_called_once_with(538, schema="ttnn_ops_v6", model_names={"whisper"})
        assert result["operations"]["ttnn::add"]["configurations"][0]["config_hash"] == "h1"

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.reconstruct_from_trace_run")
    def test_multiple_traces_merged(self, mock_recon, tmp_path):
        """Configs from traces 538 and 1 are merged, deduped by config_hash."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                  trace: [538, 1]
            registry: []
            """,
        )
        mock_recon.side_effect = [
            {
                "operations": {
                    "ttnn::add": {"configurations": [{"config_hash": "h1", "arguments": {}}]},
                    "ttnn::multiply": {"configurations": [{"config_hash": "h2", "arguments": {}}]},
                },
                "metadata": {"models": ["src_a"]},
            },
            {
                "operations": {
                    "ttnn::add": {
                        "configurations": [
                            {"config_hash": "h1", "arguments": {}},  # duplicate
                            {"config_hash": "h3", "arguments": {}},  # unique
                        ]
                    },
                },
                "metadata": {"models": ["src_b"]},
            },
        ]
        result = reconstruct_from_manifest(str(f), scope="model_traced")
        # h1 appears in both traces: only counted once
        add_hashes = [c["config_hash"] for c in result["operations"]["ttnn::add"]["configurations"]]
        assert sorted(add_hashes) == ["h1", "h3"]
        assert "ttnn::multiply" in result["operations"]

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.reconstruct_from_trace_run")
    def test_lead_models_scope(self, mock_recon, tmp_path):
        """reconstruct-manifest with scope='lead_models' only processes that group."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - model: deepseek_v3
                  trace: 35
              model_traced:
                - model: whisper
                  trace: 538
            registry: []
            """,
        )
        mock_recon.return_value = {
            "operations": {"ttnn::linear": {"configurations": [{"config_hash": "lh1", "arguments": {}}]}},
            "metadata": {"models": ["models/demos/deepseek_v3/demo/demo.py"]},
        }
        result = reconstruct_from_manifest(str(f), scope="lead_models")
        mock_recon.assert_called_once_with(35, schema="ttnn_ops_v6", model_names={"deepseek_v3"})
        assert "ttnn::linear" in result["operations"]

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.reconstruct_from_trace_run")
    def test_all_scope_merges_both_groups(self, mock_recon, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - model: deepseek_v3
                  trace: 35
              model_traced:
                - model: whisper
                  trace: 538
            registry: []
            """,
        )
        mock_recon.side_effect = [
            {
                "operations": {"ttnn::linear": {"configurations": [{"config_hash": "lead_h1", "arguments": {}}]}},
                "metadata": {"models": ["deepseek_src"]},
            },
            {
                "operations": {"ttnn::add": {"configurations": [{"config_hash": "traced_h1", "arguments": {}}]}},
                "metadata": {"models": ["whisper_src"]},
            },
        ]
        result = reconstruct_from_manifest(str(f), scope=None)
        assert set(result["metadata"]["trace_run_ids"]) == {35, 538}
        assert "ttnn::linear" in result["operations"]
        assert "ttnn::add" in result["operations"]

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.reconstruct_from_trace_run")
    def test_no_traces_returns_empty(self, mock_recon, tmp_path, capsys):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: nonexistent
            registry: []
            """,
        )
        result = reconstruct_from_manifest(str(f))
        assert result == {"operations": {}, "metadata": {}}
        mock_recon.assert_not_called()


# =====================================================================
# §4  Load → Draft Append → Promote → Reconstruct
#     (GUIDE.md → End-to-End Workflow → Steps 2-5)
# =====================================================================


class TestLoadToDraftToPromotePipeline:
    """Simulates the full lifecycle from GUIDE.md Quick Start."""

    def test_append_draft_then_promote(self, tmp_path):
        """Load auto-appends draft; promote to active enables auto-resolve."""
        manifest_file = tmp_path / "manifest.yaml"
        _write_manifest(
            manifest_file,
            """\
            targets:
              model_traced:
                - model: new_model
            registry:
            """,
        )

        # Step 1: Simulate what load_data does — append a draft entry
        new_entry = {
            "trace_id": 42,
            "status": "draft",
            "models": ["new_model"],
            "hardware": {"board_type": "Wormhole", "device_series": "n300", "card_count": 1},
            "tt_metal_sha": "abc123",
            "config_count": 100,
            "loaded_at": "2026-03-21",
            "notes": "auto-appended by load",
        }
        _append_registry_entries([new_entry], str(manifest_file))

        # Still draft: auto-resolve should not find it
        data, _ = _load_manifest(str(manifest_file))
        assert any(e["trace_id"] == 42 and e["status"] == "draft" for e in data["registry"])
        assert resolve_manifest(str(manifest_file)) == []

        # Step 2: Promote to active by rewriting the appended entry
        content = manifest_file.read_text()
        manifest_file.write_text(content.replace("status: draft", "status: active"))

        # Now auto-resolve should find it
        assert resolve_manifest(str(manifest_file)) == [42]
        models = _resolve_manifest_with_models(str(manifest_file))
        assert models[42] == {"new_model"}

    def test_multiple_loads_append_multiple_drafts(self, tmp_path):
        """Each load appends a separate draft entry."""
        manifest_file = tmp_path / "manifest.yaml"
        _write_manifest(
            manifest_file,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
            """,
        )

        for trace_id in [100, 101, 102]:
            _append_registry_entries(
                [
                    {
                        "trace_id": trace_id,
                        "status": "draft",
                        "models": ["whisper"],
                        "hardware": {"board_type": "Wormhole", "device_series": "n300", "card_count": 1},
                        "tt_metal_sha": None,
                        "config_count": 50,
                        "loaded_at": "2026-03-21",
                        "notes": "",
                    }
                ],
                str(manifest_file),
            )

        data, _ = _load_manifest(str(manifest_file))
        draft_ids = [e["trace_id"] for e in data["registry"] if e["status"] == "draft"]
        assert sorted(draft_ids) == [100, 101, 102]

        # None are active, so auto-resolve returns nothing
        assert resolve_manifest(str(manifest_file)) == []


# =====================================================================
# §4b  Snowflake load -> reconstruct integration (GATED)
#      Real round-trip against a throwaway schema; replaces the removed
#      mocked-psycopg2 DB tests. Skipped unless Snowflake creds are set.
# =====================================================================

_HAS_SF_CREDS = bool(
    os.environ.get("SNOWFLAKE_ACCOUNT")
    and os.environ.get("SNOWFLAKE_USER")
    and (os.environ.get("SNOWFLAKE_PRIVATE_KEY") or os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH"))
)


@pytest.mark.skipif(
    not _HAS_SF_CREDS,
    reason="Snowflake creds not set (SNOWFLAKE_ACCOUNT/USER/PRIVATE_KEY[_PATH])",
)
class TestSnowflakeLoadReconstructIntegration:
    """Real Snowflake load -> reconstruct round-trip against a throwaway schema.

    This class replaces the old mocked-``psycopg2`` tests (the loader is now
    Snowflake-only, so those mocks no longer apply). It is GATED on Snowflake
    credentials and skipped otherwise, so the default local/CI run stays offline.

    Test input is generated at runtime by reconstructing the smallest real V6
    trace (read-only) into a temp file — no data fixture is committed to the repo
    and the input never drifts from the schema. Each test builds an empty
    throwaway schema from the v6 DDL, loads that JSON, reconstructs it, and drops
    the schema in teardown. ``SELF_SERVE.TTNN_OPS_V6`` is only read, never written.
    """

    # Provenance/aggregation keys added by load+reconstruct that are not part of
    # the config body we assert equality on.
    _PROVENANCE_KEYS = {"executions", "trace_run_ids", "pytest_args", "pytest_args_seen"}

    # Fixed throwaway schema (unqualified name; the loader qualifies + uppercases
    # it to SELF_SERVE.TTNN_OPS_PYTEST_INTEG). It is DROP+CREATEd per test so
    # counts are deterministic and no state leaks between runs.
    TEST_SCHEMA = "ttnn_ops_pytest_integ"

    @pytest.fixture
    def sf_schema(self):
        """Create an empty throwaway schema from the v6 DDL; drop it on teardown.

        Yields ``(schema_name, connection)``. The DDL itself begins with
        ``DROP SCHEMA IF EXISTS ...; CREATE SCHEMA ...`` so it is idempotent.
        """
        import tests.sweep_framework.load_ttnn_ops_data_v2 as _ldr

        ddl_path = Path(_ldr.__file__).parent.parent.parent / "model_tracer" / "create_ttnn_ops_schema_v6_snowflake.sql"
        ddl = ddl_path.read_text().replace("TTNN_OPS_V6", "TTNN_OPS_PYTEST_INTEG")

        conn = _ldr._connect(autocommit=True)
        qualified = f"SELF_SERVE.{self.TEST_SCHEMA.upper()}"
        try:
            for _ in conn.execute_string(ddl):
                pass
            yield self.TEST_SCHEMA, conn
        finally:
            try:
                conn.cursor().execute(f"DROP SCHEMA IF EXISTS {qualified}")
            finally:
                conn.close()

    @pytest.fixture
    def sample_master(self, tmp_path):
        """Reconstruct the smallest real V6 trace into a temp master JSON.

        Generated at runtime (V6 read-only) so no data fixture lives in the repo
        and the input always matches the current schema. Returns (path, doc).
        """
        import tests.sweep_framework.load_ttnn_ops_data_v2 as _ldr

        conn = _ldr._get_read_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT trace_run_id FROM SELF_SERVE.TTNN_OPS_V6.TRACE_RUN "
                "WHERE config_count IS NOT NULL ORDER BY config_count ASC, trace_run_id ASC LIMIT 1"
            )
            row = cur.fetchone()
        finally:
            conn.close()
        assert row is not None, "no traces in SELF_SERVE.TTNN_OPS_V6 to sample"

        doc = reconstruct_from_trace_run(int(row[0]), schema="ttnn_ops_v6")
        assert doc and doc.get("operations"), f"empty reconstruct for trace {row[0]}"
        doc.setdefault("metadata", {})["trace_uid"] = "PYTEST_INTEG_SAMPLE"

        path = tmp_path / "sample_master.json"
        path.write_text(json.dumps(doc))
        return str(path), doc

    # ── helpers ──────────────────────────────────────────────────────────

    def _config_bodies(self, doc):
        """Map (operation, config_hash) -> config body sans provenance keys."""
        out = {}
        for op_name, op_data in doc["operations"].items():
            for cfg in op_data["configurations"]:
                body = {k: v for k, v in cfg.items() if k not in self._PROVENANCE_KEYS}
                out[(op_name, cfg["config_hash"])] = body
        return out

    def _config_counts(self, doc):
        """Map (operation, config_hash) -> sorted execution count list."""
        out = {}
        for op_name, op_data in doc["operations"].items():
            for cfg in op_data["configurations"]:
                out[(op_name, cfg["config_hash"])] = sorted(e["count"] for e in cfg["executions"])
        return out

    # ── tests ────────────────────────────────────────────────────────────

    def test_load_reconstruct_roundtrip(self, sf_schema, sample_master):
        """load_data -> reconstruct_from_trace_run preserves ops/configs/counts."""
        schema, conn = sf_schema
        sample_path, fixture = sample_master

        # Loading appends a draft to model_tracer/trace_selection_registry.yaml;
        # no-op it so the working tree stays clean.
        with patch("tests.sweep_framework.load_ttnn_ops_data_v2._append_manifest_drafts"):
            load_data(json_path=sample_path, tt_metal_sha="pytest-integ", dry_run=False, schema=schema)

        # Fresh schema -> the loaded trace is trace_run_id 1.
        result = reconstruct_from_trace_run(1, schema=schema)
        assert result is not None

        # Operation set matches the fixture.
        assert set(result["operations"]) == set(fixture["operations"])

        # Per-op config_hash sets match.
        for op_name in fixture["operations"]:
            fix_hashes = {c["config_hash"] for c in fixture["operations"][op_name]["configurations"]}
            got_hashes = {c["config_hash"] for c in result["operations"][op_name]["configurations"]}
            assert got_hashes == fix_hashes, op_name

        # Per-config body (arguments etc., excluding provenance keys) matches.
        fix_bodies = self._config_bodies(fixture)
        got_bodies = self._config_bodies(result)
        assert set(got_bodies) == set(fix_bodies)
        for key, body in fix_bodies.items():
            assert got_bodies[key] == body, key

        # Execution count values are preserved.
        assert self._config_counts(result) == self._config_counts(fixture)

        # Direct query: base_operation_name is populated for every operation.
        cur = conn.cursor()
        cur.execute(
            f"SELECT COUNT(*) FROM SELF_SERVE.{schema.upper()}.TTNN_OPERATION WHERE BASE_OPERATION_NAME IS NULL"
        )
        assert cur.fetchone()[0] == 0

    def test_dry_run_persists_nothing(self, sf_schema, sample_master):
        """dry_run=True rolls back every write — the schema stays empty."""
        schema, conn = sf_schema
        sample_path, _ = sample_master

        with patch("tests.sweep_framework.load_ttnn_ops_data_v2._append_manifest_drafts"):
            load_data(json_path=sample_path, tt_metal_sha="pytest-integ", dry_run=True, schema=schema)

        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM SELF_SERVE.{schema.upper()}.TTNN_CONFIGURATION")
        assert cur.fetchone()[0] == 0

    def test_emit_id_file_written(self, sf_schema, sample_master, tmp_path):
        """A real (non-dry-run) load writes the new trace_run_id to emit_id_file."""
        schema, _conn = sf_schema
        sample_path, _ = sample_master
        emit_path = tmp_path / "trace_run_id.txt"

        with patch("tests.sweep_framework.load_ttnn_ops_data_v2._append_manifest_drafts"):
            load_data(
                json_path=sample_path,
                tt_metal_sha="pytest-integ",
                dry_run=False,
                schema=schema,
                emit_id_file=str(emit_path),
            )

        assert emit_path.exists()
        # Fresh schema -> the sole new trace_run_id is 1.
        assert emit_path.read_text().strip() == "1"


# =====================================================================
# §5  Production Manifest Mirroring
#     (Tests against the actual trace_selection_registry.yaml patterns)
# =====================================================================


class TestProductionManifestPatterns:
    """Reproduce the exact structure of the real trace_selection_registry.yaml."""

    @pytest.fixture
    def prod_manifest(self, tmp_path):
        """Clone of the production manifest with all 3 traces and 19 models."""
        f = tmp_path / "manifest.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - model:
                    - deepseek_v3

              model_traced:
                - model:
                    - deepseek-llm-7b-chat
                    - deepseek-r1-distill-qwen-32b
                    - efficientnetb0
                    - falcon7b
                    - llama-3.2-11b-vision-instruct
                    - llama-3.2-1b-instruct
                    - llama-krikri-8b-instruct
                    - mistral-7b-instruct-v0.3
                    - phi-3-mini-128k-instruct
                    - qwen2.5-coder-32b
                    - qwen2.5-coder-7b-instruct
                    - qwen3-32b
                    - qwen3-32b-test
                    - segmentation
                    - sentence_bert
                    - stable_diffusion_xl_base
                    - vit-base-patch16-224
                    - vit_performant_imagenet_inference
                    - whisper

            registry:
              - trace_id: 1
                status: active
                models: [whisper]
                hardware: {board_type: Blackhole, device_series: p150b, card_count: 1}
              - trace_id: 35
                status: active
                models: [deepseek_v3]
                hardware: {board_type: Wormhole, device_series: tt-galaxy-wh, card_count: 32}
              - trace_id: 538
                status: active
                models:
                  - deepseek-llm-7b-chat
                  - deepseek-r1-distill-qwen-32b
                  - efficientnetb0
                  - falcon7b
                  - llama-3.2-11b-vision-instruct
                  - llama-3.2-1b-instruct
                  - llama-krikri-8b-instruct
                  - mistral-7b-instruct-v0.3
                  - phi-3-mini-128k-instruct
                  - qwen2.5-coder-32b
                  - qwen2.5-coder-7b-instruct
                  - qwen3-32b
                  - qwen3-32b-test
                  - segmentation
                  - sentence_bert
                  - stable_diffusion_xl_base
                  - vit-base-patch16-224
                  - vit_performant_imagenet_inference
                  - whisper
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        return str(f)

    def test_lead_models_resolves_to_35(self, prod_manifest):
        ids = resolve_manifest(prod_manifest, scope="lead_models")
        assert 35 in ids
        models = _resolve_manifest_with_models(prod_manifest, scope="lead_models")
        assert models[35] == {"deepseek_v3"}

    def test_model_traced_resolves_to_538_and_1(self, prod_manifest):
        """All 19 model_traced models → traces 538 (n300) + 1 (p150b) for whisper."""
        ids = resolve_manifest(prod_manifest, scope="model_traced")
        # whisper lives on both n300 (538) and p150b (1)
        assert 538 in ids
        assert 1 in ids

    def test_model_traced_model_set(self, prod_manifest):
        models = _resolve_manifest_with_models(prod_manifest, scope="model_traced")
        assert 538 in models
        assert "whisper" in models[538]
        assert "llama-3.2-1b-instruct" in models[538]
        assert len(models[538]) == 19
        # whisper also resolves to p150b trace 1
        assert 1 in models
        assert models[1] == {"whisper"}

    def test_all_scope_includes_everything(self, prod_manifest):
        ids = resolve_manifest(prod_manifest, scope=None)
        assert set(ids) == {1, 35, 538}

    def test_deepseek_v3_not_in_model_traced(self, prod_manifest):
        """deepseek_v3 is only in lead_models, not model_traced."""
        models = _resolve_manifest_with_models(prod_manifest, scope="model_traced")
        for model_set in models.values():
            if model_set is not None:
                assert "deepseek_v3" not in model_set

    def test_adding_new_trace_with_higher_id_supersedes(self, tmp_path, prod_manifest):
        """Simulate loading a new trace 600 that replaces 538 for n300."""
        data, path = _load_manifest(prod_manifest)
        data["registry"].append(
            {
                "trace_id": 600,
                "status": "active",
                "models": data["registry"][2]["models"],  # same 19 models
                "hardware": {"board_type": "Wormhole", "device_series": "n300", "card_count": 1},
            }
        )
        import yaml

        new_f = tmp_path / "updated.yaml"
        new_f.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

        ids = resolve_manifest(str(new_f), scope="model_traced")
        # 600 should supersede 538 for n300; 1 for p150b unchanged
        assert 600 in ids
        assert 538 not in ids
        assert 1 in ids  # p150b still has whisper


# =====================================================================
# §6  JSON Round-Trip: Load → Reconstruct → Verify
#     (GUIDE.md → verify [original] [reconstructed])
# =====================================================================


class TestJSONRoundTrip:
    """Verify that reconstructed JSON matches the original structure."""

    def test_verify_identical(self, tmp_path):
        json_path = _make_master_json(
            tmp_path,
            {
                "ttnn::add": [
                    ("ha1", {"arg0": {"type": "int", "value": 1}}),
                    ("ha2", {"arg0": {"type": "int", "value": 2}}),
                ],
                "ttnn::multiply": [
                    ("hm1", {"arg0": {"type": "float", "value": 3.0}}),
                ],
            },
        )
        result = verify_reconstruction(json_path, json_path)
        assert result["missing_ops"] == []
        assert result["extra_ops"] == []
        assert result["config_diffs"] == []

    def test_verify_detects_extra_ops(self, tmp_path):
        orig = _make_master_json(
            tmp_path,
            {"ttnn::add": [("h1", {})]},
            filename="orig.json",
        )
        recon = _make_master_json(
            tmp_path,
            {"ttnn::add": [("h1", {})], "ttnn::sub": [("h2", {})]},
            filename="recon.json",
        )
        result = verify_reconstruction(orig, recon)
        assert "ttnn::sub" in result["extra_ops"]

    def test_verify_detects_missing_ops(self, tmp_path):
        orig = _make_master_json(
            tmp_path,
            {"ttnn::add": [("h1", {})], "ttnn::sub": [("h2", {})]},
            filename="orig.json",
        )
        recon = _make_master_json(
            tmp_path,
            {"ttnn::add": [("h1", {})]},
            filename="recon.json",
        )
        result = verify_reconstruction(orig, recon)
        assert "ttnn::sub" in result["missing_ops"]

    def test_verify_detects_config_expansion(self, tmp_path):
        """Reconstruction may expand configs due to machine_info normalization."""
        orig = _make_master_json(
            tmp_path,
            {"ttnn::add": [("h1", {})]},
            filename="orig.json",
        )
        recon = _make_master_json(
            tmp_path,
            {"ttnn::add": [("h1", {}), ("h2", {})]},
            filename="recon.json",
        )
        result = verify_reconstruction(orig, recon)
        assert len(result["config_diffs"]) == 1
        assert result["config_diffs"][0] == ("ttnn::add", 1, 2)


class TestFindConfigLineNumbersFunctional:
    """find-lines with realistic multi-operation JSON files."""

    def test_multi_op_json(self, tmp_path):
        json_path = _make_master_json(
            tmp_path,
            {
                "ttnn::add": [
                    ("h1", {"a": 1}),
                    ("h2", {"a": 2}),
                ],
                "ttnn::multiply": [
                    ("h3", {"b": 1}),
                ],
            },
        )
        result = find_config_line_numbers(json_path, "ttnn::add", [0, 1])
        assert result[0] is not None
        assert result[1] is not None
        assert result[0] < result[1]

    def test_second_operation(self, tmp_path):
        json_path = _make_master_json(
            tmp_path,
            {
                "ttnn::add": [("h1", {"a": 1})],
                "ttnn::multiply": [("h2", {"b": 1}), ("h3", {"b": 2})],
            },
        )
        result = find_config_line_numbers(json_path, "ttnn::multiply", [0, 1])
        assert result[0] is not None
        assert result[1] is not None


# =====================================================================
# §8  Edge-Case Manifest Permutations
# =====================================================================


class TestManifestEdgeCases:
    """Unusual but valid manifest configurations."""

    def test_empty_model_list_in_target(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: []
            registry: []
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == []

    def test_empty_registry(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry: []
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == []

    def test_target_with_no_entries(self, tmp_path):
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models: []
              model_traced: []
            registry: []
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == []

    def test_single_model_string_not_list(self, tmp_path):
        """model: whisper (string) instead of model: [whisper] (list)."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                  trace: 538
            registry: []
            """,
        )
        assert resolve_manifest(str(f)) == [538]
        models = _resolve_manifest_with_models(str(f))
        assert models[538] == {"whisper"}

    def test_single_trace_int_not_list(self, tmp_path):
        """trace: 35 (int) instead of trace: [35] (list)."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - model: deepseek_v3
                  trace: 35
            registry: []
            """,
        )
        assert resolve_manifest(str(f)) == [35]

    def test_registry_with_many_models_and_hardware_variants(self, tmp_path):
        """Registry with same model on 3 different hardware platforms."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 1
                status: active
                models: [whisper]
                hardware: {board_type: Blackhole, device_series: p150b, card_count: 1}
              - trace_id: 538
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
              - trace_id: 35
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: tt-galaxy-wh, card_count: 32}
            """,
        )
        ids = resolve_manifest(str(f))
        # One trace per device_series: p150b=1, n300=538, tt-galaxy-wh=35
        assert set(ids) == {1, 538, 35}

    def test_model_in_registry_but_not_in_targets(self, tmp_path):
        """Models exist in registry but aren't requested in targets → not resolved."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 538
                status: active
                models: [whisper, llama-3.2-1b-instruct, falcon7b]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        models = _resolve_manifest_with_models(str(f))
        assert models[538] == {"whisper"}  # only whisper, not llama/falcon

    def test_custom_scope_group_scope_semantics(self, tmp_path):
        """Custom groups are included for scope=None but ignored for explicit lead_models/model_traced scopes."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              custom_group:
                - model: whisper
                  trace: 1
              lead_models:
                - model: deepseek_v3
                  trace: 35
            registry: []
            """,
        )
        # scope=None processes all groups including custom_group
        ids_all = resolve_manifest(str(f), scope=None)
        assert set(ids_all) == {1, 35}

        # scope=lead_models only
        ids_lead = resolve_manifest(str(f), scope="lead_models")
        assert ids_lead == [35]

        # scope=model_traced misses custom_group
        ids_traced = resolve_manifest(str(f), scope="model_traced")
        assert ids_traced == []

    def test_duplicate_models_in_registry_entry(self, tmp_path):
        """Duplicate model names in a registry entry don't cause issues."""
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 538
                status: active
                models: [whisper, whisper, llama]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == [538]

    def test_large_model_list_resolution(self, tmp_path):
        """20+ models all resolving to the same trace."""
        models = [f"model_{i}" for i in range(25)]
        models_yaml = "\n                    - ".join(models)
        f = tmp_path / "m.yaml"
        _write_manifest(
            f,
            f"""\
            targets:
              model_traced:
                - model:
                    - {models_yaml}
            registry:
              - trace_id: 1000
                status: active
                models: [{', '.join(models)}]
                hardware: {{board_type: Wormhole, device_series: n300, card_count: 1}}
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == [1000]
        model_map = _resolve_manifest_with_models(str(f))
        assert len(model_map[1000]) == 25
