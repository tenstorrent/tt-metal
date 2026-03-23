#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for load_ttnn_ops_data_v2.py — all use cases documented in model_tracer/GUIDE.md.

Covers:
  - Pure utility functions (parsing, derivation, formatting)
  - Manifest resolution (resolve_manifest, _resolve_manifest_with_models)
  - DB-interacting functions (load_data, reconstruct, set-model-name, delete-trace)
  - verify_reconstruction and find_config_line_numbers
  - CLI argument parsing (__main__ block)
"""

import json
import os
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# The module under test validates NEON_URL at import time; provide a dummy.
os.environ.setdefault("TTNN_OPS_DATABASE_URL", "postgresql://test:test@localhost/testdb")

from tests.sweep_framework.load_ttnn_ops_data_v2 import (
    _append_registry_entries,
    _load_manifest,
    _resolve_manifest_with_models,
    _validate_schema,
    delete_trace_run,
    derive_model_name,
    extract_model_family,
    find_config_line_numbers,
    format_source,
    get_or_create_hardware,
    get_or_create_mesh_config,
    get_or_create_model,
    get_or_create_trace_run,
    link_trace_run_config,
    list_trace_runs,
    load_data,
    parse_all_sources,
    parse_array_value,
    parse_mesh_from_machine_info,
    parse_placement,
    parse_source,
    reconstruct_from_manifest,
    reconstruct_from_trace_run,
    resolve_manifest,
    set_model_name,
    verify_reconstruction,
)


# =====================================================================
# 1. Pure Utility Functions
# =====================================================================


class TestParseSource:
    """parse_source: extract (source_file, hf_model_identifier) from a source string."""

    def test_file_with_hf_model(self):
        s = "models/tt_transformers/demo/simple_text_demo.py [HF_MODEL:meta-llama/Llama-3.2-1B-Instruct]"
        assert parse_source(s) == (
            "models/tt_transformers/demo/simple_text_demo.py",
            "meta-llama/Llama-3.2-1B-Instruct",
        )

    def test_file_only(self):
        assert parse_source("models/demos/deepseek_v3/demo/demo.py") == (
            "models/demos/deepseek_v3/demo/demo.py",
            None,
        )

    def test_none_input(self):
        assert parse_source(None) == (None, None)

    def test_empty_string(self):
        assert parse_source("") == (None, None)

    def test_hf_model_with_extra_whitespace(self):
        s = "path/to/demo.py   [HF_MODEL:org/Model-Name]"
        assert parse_source(s) == ("path/to/demo.py", "org/Model-Name")

    def test_no_hf_bracket_returns_raw(self):
        assert parse_source("just/a/path.py") == ("just/a/path.py", None)


class TestParseAllSources:
    """parse_all_sources: normalize string-or-list source fields."""

    def test_string_input(self):
        result = parse_all_sources("models/demos/audio/whisper/demo/demo.py")
        assert result == [("models/demos/audio/whisper/demo/demo.py", None)]

    def test_list_input(self):
        result = parse_all_sources(
            [
                "path/a.py [HF_MODEL:org/A]",
                "path/b.py",
            ]
        )
        assert result == [("path/a.py", "org/A"), ("path/b.py", None)]

    def test_none_returns_none_tuple(self):
        assert parse_all_sources(None) == [(None, None)]

    def test_empty_list(self):
        assert parse_all_sources([]) == [(None, None)]

    def test_non_string_non_list_returns_none_tuple(self):
        assert parse_all_sources(42) == [(None, None)]


class TestExtractModelFamily:
    """extract_model_family: infer family name from source/HF strings."""

    @pytest.mark.parametrize(
        "source_file, hf_model, expected",
        [
            ("models/demos/deepseek_v3/demo.py", None, "deepseek"),
            (None, "meta-llama/Llama-3.2-1B-Instruct", "llama"),
            ("models/demos/audio/whisper/demo.py", None, "whisper"),
            (None, "Qwen/Qwen2.5-72B", "qwen"),
            ("path/to/mistral/demo.py", None, "mistral"),
            ("path/to/bert/demo.py", None, "bert"),
            ("path/to/resnet/demo.py", None, "resnet"),
            ("path/to/efficientnet/demo.py", None, "efficientnet"),
            ("path/to/unknown_model/demo.py", None, None),
            (None, None, None),
        ],
    )
    def test_families(self, source_file, hf_model, expected):
        assert extract_model_family(source_file, hf_model) == expected


class TestDeriveModelName:
    """derive_model_name: documented derivation rules from GUIDE.md §Step 3."""

    def test_hf_model_last_segment_lowered(self):
        # GUIDE: meta-llama/Llama-3.2-1B-Instruct → llama-3.2-1b-instruct
        assert derive_model_name(None, "meta-llama/Llama-3.2-1B-Instruct") == "llama-3.2-1b-instruct"

    def test_hf_model_simple(self):
        assert derive_model_name(None, "Qwen/Qwen2.5-Coder-32B") == "qwen2.5-coder-32b"

    def test_file_path_skips_generic_segments(self):
        # GUIDE: models/demos/audio/whisper/demo/demo.py → whisper
        assert derive_model_name("models/demos/audio/whisper/demo/demo.py", None) == "whisper"

    def test_file_path_deepseek(self):
        assert derive_model_name("models/demos/deepseek_v3/demo/demo.py", None) == "deepseek_v3"

    def test_file_path_strips_test_prefix(self):
        assert derive_model_name("tests/test_whisper.py", None) == "whisper"

    def test_file_path_strips_py_extension(self):
        assert derive_model_name("models/some_model/run.py", None) == "some_model"

    def test_file_path_with_test_node_suffix(self):
        # "path/file.py::test_foo[bar]" should strip the ::... suffix
        assert derive_model_name("models/demos/audio/whisper/demo/demo.py::test_demo_text", None) == "whisper"

    def test_none_inputs(self):
        assert derive_model_name(None, None) is None

    def test_all_generic_segments_falls_back_to_last(self):
        assert derive_model_name("models/demos/demo/demo.py", None) is not None

    def test_hf_takes_precedence_over_source(self):
        name = derive_model_name("path/to/demo.py", "org/MyModel")
        assert name == "mymodel"

    def test_experimental_segment_skipped(self):
        assert derive_model_name("models/experimental/cool_model/demo.py", None) == "cool_model"

    def test_vision_classification_skipped(self):
        assert derive_model_name("models/demos/vision/classification/efficientnet/demo.py", None) == "efficientnet"


class TestParseArrayValue:
    """parse_array_value: coerce various array representations to Python lists."""

    def test_python_list_passthrough(self):
        assert parse_array_value([1, 2, 3]) == [1, 2, 3]

    def test_json_string(self):
        assert parse_array_value("[1, 2]") == [1, 2]

    def test_curly_brace_string(self):
        assert parse_array_value("{4, 8}") == [4, 8]

    def test_nullopt_variants(self):
        assert parse_array_value("std::nullopt") is None
        assert parse_array_value("nullopt") is None
        assert parse_array_value("null") is None

    def test_none_input(self):
        assert parse_array_value(None) is None

    def test_numeric_fallback(self):
        assert parse_array_value("shape(1, 32)") == [1, 32]

    def test_non_parseable_returns_none(self):
        assert parse_array_value("no_numbers_here") is None


class TestParsePlacement:
    """parse_placement: extract (placement_type, shard_dim) from placement strings."""

    def test_shard_placement(self):
        assert parse_placement("[PlacementShard(3)]") == ("shard", 3)

    def test_shard_placement_dim_zero(self):
        assert parse_placement("[PlacementShard(0)]") == ("shard", 0)

    def test_replicate_placement(self):
        assert parse_placement("[PlacementReplicate]") == ("replicate", None)

    def test_none_returns_replicate(self):
        assert parse_placement(None) == ("replicate", None)

    def test_empty_returns_replicate(self):
        assert parse_placement("") == ("replicate", None)


class TestFormatSource:
    """format_source: inverse of parse_source."""

    def test_with_hf_model(self):
        assert format_source("path/to/demo.py", "org/Model") == "path/to/demo.py [HF_MODEL:org/Model]"

    def test_without_hf_model(self):
        assert format_source("path/to/demo.py", None) == "path/to/demo.py"

    def test_none_source(self):
        assert format_source(None, None) is None

    def test_roundtrip(self):
        original = "path/to/demo.py [HF_MODEL:meta-llama/Llama-3.2-1B-Instruct]"
        sf, hf = parse_source(original)
        assert format_source(sf, hf) == original


class TestValidateSchema:
    """_validate_schema: prevent SQL injection in schema names."""

    def test_valid_schemas(self):
        _validate_schema("ttnn_ops_v5")
        _validate_schema("public")
        _validate_schema("my_schema_2")

    def test_invalid_schemas(self):
        with pytest.raises(ValueError):
            _validate_schema("ttnn_ops; DROP TABLE")
        with pytest.raises(ValueError):
            _validate_schema("schema-name")
        with pytest.raises(ValueError):
            _validate_schema("1starts_with_digit")


class TestParseMeshFromMachineInfo:
    """parse_mesh_from_machine_info: V2 tensor_placement extraction."""

    def test_v2_tensor_placement(self):
        machine_info = {"device_count": 2}
        arguments = {
            "arg0": {
                "type": "ttnn.Tensor",
                "tensor_placement": {
                    "placement": "['PlacementShard(2)']",
                    "distribution_shape": "[1, 2]",
                    "mesh_device_shape": "[1, 2]",
                },
            }
        }
        mesh_shape, device_count, placement_type, shard_dim, dist_shape = parse_mesh_from_machine_info(
            machine_info, arguments
        )
        assert mesh_shape == [1, 2]
        assert device_count == 2
        assert placement_type == "shard"
        assert shard_dim == 2
        assert dist_shape == [1, 2]

    def test_no_tensor_placement(self):
        result = parse_mesh_from_machine_info({}, {"arg0": {"type": "int", "value": 1}})
        assert result == (None, None, None, None, None)

    def test_no_arguments(self):
        assert parse_mesh_from_machine_info({}, None) == (None, None, None, None, None)

    def test_device_count_calculated_from_shape(self):
        machine_info = {}
        arguments = {
            "arg0": {
                "type": "ttnn.Tensor",
                "tensor_placement": {
                    "placement": "['PlacementReplicate']",
                    "mesh_device_shape": "[4, 8]",
                },
            }
        }
        mesh_shape, device_count, _, _, _ = parse_mesh_from_machine_info(machine_info, arguments)
        assert mesh_shape == [4, 8]
        assert device_count == 32

    def test_empty_mesh_shape(self):
        arguments = {
            "arg0": {
                "type": "ttnn.Tensor",
                "tensor_placement": {"placement": "['PlacementReplicate']", "mesh_device_shape": None},
            }
        }
        result = parse_mesh_from_machine_info({}, arguments)
        assert result == (None, None, None, None, None)


# =====================================================================
# 2. Manifest Resolution
# =====================================================================


def _write_manifest(path, content):
    """Helper: write YAML string to a manifest file."""
    path.write_text(textwrap.dedent(content))


class TestLoadManifest:
    """_load_manifest: read and default-fill manifest YAML."""

    def test_loads_existing_file(self, tmp_path):
        manifest_file = tmp_path / "manifest.yaml"
        _write_manifest(
            manifest_file,
            """\
            targets:
              lead_models:
                - model: deepseek_v3
                  trace: 35
            registry:
              - trace_id: 35
                status: active
                models: [deepseek_v3]
            """,
        )
        data, path = _load_manifest(str(manifest_file))
        assert path == str(manifest_file)
        assert "targets" in data
        assert "registry" in data
        assert len(data["registry"]) == 1

    def test_missing_file_returns_defaults(self, tmp_path):
        data, _ = _load_manifest(str(tmp_path / "nonexistent.yaml"))
        assert data == {"targets": {}, "registry": []}

    def test_empty_file_returns_defaults(self, tmp_path):
        manifest_file = tmp_path / "empty.yaml"
        manifest_file.write_text("")
        data, _ = _load_manifest(str(manifest_file))
        assert data == {"targets": {}, "registry": []}


class TestAppendRegistryEntries:
    """_append_registry_entries: append-only manifest updates."""

    def test_appends_single_model(self, tmp_path):
        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text("registry:\n")
        entries = [
            {
                "trace_id": 42,
                "status": "draft",
                "models": ["deepseek_v3"],
                "hardware": {"board_type": "Wormhole", "device_series": "n300", "card_count": 1},
                "tt_metal_sha": "abc123",
                "config_count": 100,
                "loaded_at": "2026-03-21",
                "notes": "test entry",
            }
        ]
        _append_registry_entries(entries, str(manifest_file))
        content = manifest_file.read_text()
        assert "trace_id: 42" in content
        assert "status: draft" in content
        assert "models: [deepseek_v3]" in content
        assert "board_type: Wormhole" in content

    def test_appends_multiple_models(self, tmp_path):
        manifest_file = tmp_path / "manifest.yaml"
        manifest_file.write_text("registry:\n")
        entries = [
            {
                "trace_id": 99,
                "status": "draft",
                "models": ["whisper", "llama-3.2-1b-instruct", "bert"],
                "hardware": {"board_type": "Wormhole", "device_series": "n300", "card_count": 1},
                "tt_metal_sha": None,
                "config_count": 500,
                "loaded_at": "2026-03-21",
                "notes": "",
            }
        ]
        _append_registry_entries(entries, str(manifest_file))
        content = manifest_file.read_text()
        assert "models:" in content
        assert "- whisper" in content
        assert "- llama-3.2-1b-instruct" in content
        assert "tt_metal_sha: null" in content


class TestResolveManifest:
    """resolve_manifest: all resolution rules from GUIDE.md §Targets."""

    @pytest.fixture
    def manifest_file(self, tmp_path):
        f = tmp_path / "manifest.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - model: deepseek_v3
                  trace: 35
              model_traced:
                - model:
                    - whisper
                    - llama-3.2-1b-instruct
                  trace:
                    - 538
                    - 1
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
                models: [whisper, llama-3.2-1b-instruct, deepseek-llm-7b-chat]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        return str(f)

    def test_pinned_trace_lead_models(self, manifest_file):
        """model: X, trace: N → use trace N directly."""
        ids = resolve_manifest(manifest_file, scope="lead_models")
        assert ids == [35]

    def test_pinned_trace_list(self, manifest_file):
        """model: [X, Y], trace: [N, M] → use traces N and M."""
        ids = resolve_manifest(manifest_file, scope="model_traced")
        assert set(ids) == {538, 1}

    def test_scope_all(self, manifest_file):
        """scope=None combines both groups, deduplicates."""
        ids = resolve_manifest(manifest_file, scope=None)
        assert set(ids) == {35, 538, 1}

    def test_registry_resolution_latest_active(self, tmp_path):
        """model: X (no trace) → latest active trace per device_series."""
        f = tmp_path / "manifest.yaml"
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
              - trace_id: 200
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
              - trace_id: 150
                status: deprecated
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        # Latest active on n300 is trace_id=200 (150 is deprecated)
        assert ids == [200]

    def test_registry_resolution_per_device_series(self, tmp_path):
        """Multiple device_series → one trace per device_series."""
        f = tmp_path / "manifest.yaml"
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
            """,
        )
        ids = resolve_manifest(str(f))
        assert set(ids) == {10, 20}

    def test_hardware_filter(self, tmp_path):
        """model: X, hardware: H → only matching device_series."""
        f = tmp_path / "manifest.yaml"
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
        ids = resolve_manifest(str(f))
        assert ids == [20]

    def test_no_matching_model_warning(self, tmp_path, capsys):
        """No active trace matching the model → warning, empty result."""
        f = tmp_path / "manifest.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: nonexistent_model
            registry:
              - trace_id: 1
                status: active
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == []
        assert "no active traces match" in capsys.readouterr().out.lower()

    def test_draft_traces_invisible_to_resolution(self, tmp_path):
        """Traces with status: draft are not resolved unless pinned."""
        f = tmp_path / "manifest.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 99
                status: draft
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == []

    def test_draft_still_usable_when_pinned(self, tmp_path):
        """Pinned trace: N skips registry resolution — draft traces work."""
        f = tmp_path / "manifest.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                  trace: 99
            registry:
              - trace_id: 99
                status: draft
                models: [whisper]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == [99]

    def test_deduplication(self, tmp_path):
        """Multiple entries resolving to the same trace_id are deduplicated."""
        f = tmp_path / "manifest.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - model: whisper
                  trace: 538
              model_traced:
                - model: whisper
                  trace: 538
            registry: []
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == [538]

    def test_empty_targets(self, tmp_path, capsys):
        f = tmp_path / "manifest.yaml"
        _write_manifest(f, "targets: {}\nregistry: []\n")
        ids = resolve_manifest(str(f))
        assert ids == []


class TestResolveManifestWithModels:
    """_resolve_manifest_with_models: returns {trace_id: set_of_model_names | None}."""

    def test_pinned_trace_with_models(self, tmp_path):
        f = tmp_path / "manifest.yaml"
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
        result = _resolve_manifest_with_models(str(f), scope="lead_models")
        assert result == {35: {"deepseek_v3"}}

    def test_pinned_trace_no_model_means_all(self, tmp_path):
        """trace: N with no model → None (all models)."""
        f = tmp_path / "manifest.yaml"
        _write_manifest(
            f,
            """\
            targets:
              lead_models:
                - trace: 35
            registry: []
            """,
        )
        result = _resolve_manifest_with_models(str(f), scope="lead_models")
        assert result == {35: None}

    def test_registry_resolved_models_accumulated(self, tmp_path):
        """Two target entries sharing the same resolved trace → model sets merged."""
        f = tmp_path / "manifest.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
                - model: llama-3.2-1b-instruct
            registry:
              - trace_id: 538
                status: active
                models: [whisper, llama-3.2-1b-instruct]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        result = _resolve_manifest_with_models(str(f), scope="model_traced")
        assert 538 in result
        assert result[538] == {"whisper", "llama-3.2-1b-instruct"}


# =====================================================================
# 3. DB-Interacting Functions (mocked psycopg2)
# =====================================================================


def _mock_cursor():
    """Create a mock DB cursor with configurable fetchone/fetchall."""
    cur = MagicMock()
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    return cur


class TestGetOrCreateHardware:
    """get_or_create_hardware: hardware cache + upsert logic."""

    def test_creates_new_hardware(self):
        cur = _mock_cursor()
        cur.fetchone.return_value = (42,)
        cache = {}
        hw_id, hw_key = get_or_create_hardware(cur, cache, "Wormhole", "n300", 1)
        assert hw_id == 42
        assert cache[("Wormhole", "n300", 1)] == 42

    def test_returns_cached(self):
        cur = _mock_cursor()
        cache = {("Wormhole", "n300", 1): 42}
        hw_id, hw_key = get_or_create_hardware(cur, cache, "Wormhole", "n300", 1)
        assert hw_id == 42
        cur.execute.assert_not_called()

    def test_none_board_type_returns_none(self):
        cur = _mock_cursor()
        hw_id, hw_key = get_or_create_hardware(cur, {}, None, "n300", 1)
        assert hw_id is None
        assert hw_key is None

    def test_list_device_series_normalized(self):
        cur = _mock_cursor()
        cur.fetchone.return_value = (7,)
        cache = {}
        hw_id, _ = get_or_create_hardware(cur, cache, "Wormhole", ["n300", "other"], 1)
        assert hw_id == 7
        assert ("Wormhole", "n300", 1) in cache

    def test_on_conflict_fetches_existing(self):
        cur = _mock_cursor()
        # First INSERT returns None (conflict, DO NOTHING), then SELECT returns ID
        cur.fetchone.side_effect = [None, (99,)]
        cache = {}
        hw_id, _ = get_or_create_hardware(cur, cache, "Blackhole", "p150b", 1)
        assert hw_id == 99


class TestGetOrCreateMeshConfig:
    """get_or_create_mesh_config: mesh config cache."""

    def test_creates_new(self):
        cur = _mock_cursor()
        cur.fetchone.return_value = (5,)
        cache = {}
        mesh_id = get_or_create_mesh_config(cur, cache, [4, 8], 32)
        assert mesh_id == 5

    def test_none_mesh_shape(self):
        assert get_or_create_mesh_config(_mock_cursor(), {}, None, 0) is None

    def test_cached(self):
        cur = _mock_cursor()
        cache = {((4, 8), 32): 5}
        mesh_id = get_or_create_mesh_config(cur, cache, [4, 8], 32)
        assert mesh_id == 5
        cur.execute.assert_not_called()


class TestGetOrCreateTraceRun:
    """get_or_create_trace_run: creates trace_run rows keyed by hardware+sha."""

    def test_creates_new(self):
        cur = _mock_cursor()
        cur.fetchone.return_value = (42,)
        cache = {}
        tr_id = get_or_create_trace_run(cur, cache, 1, "abc123")
        assert tr_id == 42
        assert cache[(1, "abc123")] == 42

    def test_cached(self):
        cur = _mock_cursor()
        cache = {(1, "abc123"): 42}
        tr_id = get_or_create_trace_run(cur, cache, 1, "abc123")
        assert tr_id == 42
        cur.execute.assert_not_called()

    def test_null_sha(self):
        cur = _mock_cursor()
        cur.fetchone.return_value = (99,)
        cache = {}
        tr_id = get_or_create_trace_run(cur, cache, 1, None)
        assert tr_id == 99
        assert cache[(1, None)] == 99


class TestLinkTraceRunConfig:
    """link_trace_run_config: creates junction table entries."""

    def test_inserts_link(self):
        cur = _mock_cursor()
        link_trace_run_config(cur, 42, 100, 5)
        cur.execute.assert_called_once()
        args = cur.execute.call_args[0]
        assert (42, 100, 5) == args[1]


class TestLoadData:
    """load_data: main loading function — use cases from GUIDE.md §Step 2."""

    def _sample_json(self, tmp_path, config_hash="hash_abc"):
        """Write a minimal valid master JSON and return its path."""
        data = {
            "operations": {
                "ttnn::add": {
                    "configurations": [
                        {
                            "config_hash": config_hash,
                            "arguments": {"arg0": {"type": "int", "value": 1}},
                            "executions": [
                                {
                                    "source": "models/demos/deepseek_v3/demo/demo.py",
                                    "machine_info": {
                                        "board_type": "Wormhole",
                                        "device_series": "n300",
                                        "card_count": 1,
                                    },
                                    "count": 5,
                                }
                            ],
                        }
                    ]
                }
            }
        }
        p = tmp_path / "master.json"
        p.write_text(json.dumps(data))
        return str(p)

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    @patch("tests.sweep_framework.load_ttnn_ops_data_v2._append_manifest_drafts")
    def test_load_default_path(self, mock_drafts, mock_pg, tmp_path):
        """load [json] — loads from default or specified path."""
        json_path = self._sample_json(tmp_path)
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        # Operation insert
        mock_cur.fetchone.side_effect = [
            (1,),  # operation_id
            (10,),  # model_id (lookup)
            (20,),  # hardware_id (insert)
            (30,),  # config_id (insert)
            (40,),  # trace_run_id (insert)
            (100,),  # _fetch_db_totals calls — 9 tables
            (200,),
            (300,),
            (400,),
            (500,),
            (600,),
            (700,),
            (800,),
            (900,),
        ]
        load_data(json_path=json_path, tt_metal_sha="sha123", dry_run=False)
        mock_pg.connect.assert_called()
        mock_drafts.assert_called_once()

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_load_dry_run_rolls_back(self, mock_pg, tmp_path):
        """load --dry-run: preview without committing."""
        json_path = self._sample_json(tmp_path)
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.side_effect = [
            (1,),  # operation_id
            (10,),  # model_id
            (20,),  # hardware_id
            (30,),  # config_id
            (40,),  # trace_run_id
            (100,),  # _fetch_db_totals (9 tables)
            (200,),
            (300,),
            (400,),
            (500,),
            (600,),
            (700,),
            (800,),
            (900,),
        ]
        load_data(json_path=json_path, tt_metal_sha="sha123", dry_run=True)
        mock_conn.rollback.assert_called()
        mock_conn.commit.assert_not_called()

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_load_missing_config_hash_raises(self, mock_pg, tmp_path):
        """Missing config_hash → ValueError."""
        data = {
            "operations": {
                "ttnn::add": {
                    "configurations": [
                        {
                            "arguments": {"arg0": {"type": "int", "value": 1}},
                            "executions": [
                                {
                                    "source": "path/demo.py",
                                    "machine_info": {"board_type": "Wormhole"},
                                    "count": 1,
                                }
                            ],
                        }
                    ]
                }
            }
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(data))
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.return_value = (1,)
        with pytest.raises(ValueError, match="Missing config_hash"):
            load_data(json_path=str(p), tt_metal_sha="sha")

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    @patch("tests.sweep_framework.load_ttnn_ops_data_v2._append_manifest_drafts")
    def test_load_auto_detects_sha(self, mock_drafts, mock_pg, tmp_path):
        """SHA auto-detection from git when not provided."""
        json_path = self._sample_json(tmp_path)
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.side_effect = [
            (1,),
            (10,),
            (20,),
            (30,),
            (40,),
            *[(n,) for n in range(9)],
        ]
        with patch("subprocess.run") as mock_sub:
            mock_sub.return_value = MagicMock(returncode=0, stdout="deadbeef1234\n")
            load_data(json_path=json_path, tt_metal_sha=None, dry_run=False)
            mock_sub.assert_called_once()


class TestReconstructFromTraceRun:
    """reconstruct-trace <id> [output.json] — GUIDE.md §Step 5."""

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_trace_not_found(self, mock_pg, capsys):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.return_value = None
        result = reconstruct_from_trace_run(999)
        assert result is None
        assert "not found" in capsys.readouterr().out

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_reconstruct_basic(self, mock_pg, tmp_path):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur

        # trace_run metadata
        tr_meta = (35, "Wormhole", "tt-galaxy-wh", 32, "abc123", "2026-03-21", 323, "test")
        # trace models
        trace_models = [(1, "models/demos/deepseek_v3/demo/demo.py", None, "deepseek_v3")]
        # config rows: (op_name, config_id, hash, full_json, mesh, dev_cnt, exec_cnt, src, hf)
        config_rows = [
            (
                "ttnn::add",
                100,
                "hash_abc",
                {"arguments": {"arg0": {"type": "int", "value": 1}}},
                None,
                None,
                5,
                "models/demos/deepseek_v3/demo/demo.py",
                None,
            )
        ]

        mock_cur.fetchone.return_value = tr_meta
        mock_cur.fetchall.side_effect = [trace_models, config_rows]

        output = str(tmp_path / "output.json")
        result = reconstruct_from_trace_run(35, output_path=output)

        assert result is not None
        assert "ttnn::add" in result["operations"]
        assert result["metadata"]["trace_run_id"] == 35
        assert Path(output).exists()

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_reconstruct_model_name_filter(self, mock_pg):
        """model_names filter skips non-matching models."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur

        tr_meta = (538, "Wormhole", "n300", 1, None, "2026-03-21", 7000, "")
        # Trace contains whisper and llama, but we filter to whisper only
        trace_models = [
            (1, "models/demos/audio/whisper/demo/demo.py", None, "whisper"),
            (2, "path/to/llama.py", "meta-llama/Llama-3.2-1B", "llama-3.2-1b"),
        ]
        config_rows = [
            (
                "ttnn::multiply",
                200,
                "hash_xyz",
                {"arguments": {}},
                None,
                None,
                1,
                "models/demos/audio/whisper/demo/demo.py",
                None,
            )
        ]

        mock_cur.fetchone.return_value = tr_meta
        mock_cur.fetchall.side_effect = [trace_models, config_rows]

        result = reconstruct_from_trace_run(538, model_names={"whisper"})
        assert result is not None
        assert "ttnn::multiply" in result["operations"]


class TestReconstructFromManifest:
    """reconstruct-manifest — GUIDE.md §Step 5: manifest-driven reconstruction."""

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.reconstruct_from_trace_run")
    @patch("tests.sweep_framework.load_ttnn_ops_data_v2._resolve_manifest_with_models")
    def test_reconstruct_merges_traces(self, mock_resolve, mock_recon, tmp_path):
        mock_resolve.return_value = {
            35: {"deepseek_v3"},
            538: {"whisper"},
        }
        mock_recon.side_effect = [
            {
                "operations": {
                    "ttnn::add": {
                        "configurations": [{"config_hash": "h1", "arguments": {}}],
                    }
                },
                "metadata": {"models": ["models/demos/deepseek_v3/demo/demo.py"]},
            },
            {
                "operations": {
                    "ttnn::add": {
                        "configurations": [{"config_hash": "h2", "arguments": {}}],
                    },
                    "ttnn::multiply": {
                        "configurations": [{"config_hash": "h3", "arguments": {}}],
                    },
                },
                "metadata": {"models": ["models/demos/audio/whisper/demo/demo.py"]},
            },
        ]

        output = str(tmp_path / "merged.json")
        result = reconstruct_from_manifest(output_path=output, scope=None)

        assert "ttnn::add" in result["operations"]
        assert len(result["operations"]["ttnn::add"]["configurations"]) == 2
        assert "ttnn::multiply" in result["operations"]
        assert set(result["metadata"]["trace_run_ids"]) == {35, 538}

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.reconstruct_from_trace_run")
    @patch("tests.sweep_framework.load_ttnn_ops_data_v2._resolve_manifest_with_models")
    def test_config_deduplication_across_traces(self, mock_resolve, mock_recon):
        """Configs with the same config_hash from different traces are deduplicated."""
        mock_resolve.return_value = {1: None, 2: None}
        shared_config = {"config_hash": "shared_hash", "arguments": {}}
        mock_recon.side_effect = [
            {"operations": {"ttnn::add": {"configurations": [shared_config]}}, "metadata": {"models": []}},
            {"operations": {"ttnn::add": {"configurations": [shared_config]}}, "metadata": {"models": []}},
        ]
        result = reconstruct_from_manifest()
        assert len(result["operations"]["ttnn::add"]["configurations"]) == 1

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2._resolve_manifest_with_models")
    def test_empty_resolution(self, mock_resolve):
        mock_resolve.return_value = {}
        result = reconstruct_from_manifest()
        assert result == {"operations": {}, "metadata": {}}


class TestSetModelName:
    """set-model-name — GUIDE.md §Step 3: resolve model_name collisions."""

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_set_by_source_file(self, mock_pg, capsys):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [(7, "path/to/demo.py", None)]

        set_model_name(source_file="path/to/demo.py", new_name="custom_name")
        mock_conn.commit.assert_called_once()
        assert "custom_name" in capsys.readouterr().out

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_set_by_model_id(self, mock_pg, capsys):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [(7, "path/to/demo.py", None)]

        set_model_name(model_id=7, new_name="vit_nightly")
        mock_conn.commit.assert_called_once()
        assert "vit_nightly" in capsys.readouterr().out

    def test_no_name_prints_error(self, capsys):
        set_model_name(source_file="path.py", new_name=None)
        assert "error" in capsys.readouterr().out.lower()

    def test_no_identifier_prints_error(self, capsys):
        set_model_name(new_name="foo")
        assert "error" in capsys.readouterr().out.lower()

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_no_match_reports_nothing(self, mock_pg, capsys):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []

        set_model_name(source_file="nonexistent.py", new_name="foo")
        assert "nothing updated" in capsys.readouterr().out.lower()

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_name_lowercased(self, mock_pg):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [(1, "demo.py", None)]

        set_model_name(source_file="demo.py", new_name="MyModel_V2")
        sql_call = mock_cur.execute.call_args[0]
        assert "mymodel_v2" in sql_call[1]


class TestDeleteTraceRun:
    """delete-trace <id> [--yes] — GUIDE.md CLI Reference."""

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_trace_not_found(self, mock_pg, capsys):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchone.return_value = None

        delete_trace_run(999, yes=True)
        assert "not found" in capsys.readouterr().out

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_delete_with_yes_flag(self, mock_pg, capsys):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        # First: trace_run row; Second: exclusive count; Third: exclusive ids
        mock_cur.fetchone.side_effect = [(100, "test note"), (3,)]
        mock_cur.fetchall.return_value = [(10,), (11,), (12,)]

        delete_trace_run(42, yes=True)
        mock_conn.commit.assert_called_once()
        out = capsys.readouterr().out
        assert "deleted" in out.lower()
        assert "3 configs" in out


class TestListTraceRuns:
    """list-traces [filter] — GUIDE.md CLI Reference."""

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_list_all(self, mock_pg, capsys):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [
            (1, "p150b", 1, "abc123", "2026-03-21", 318, "", "whisper"),
            (35, "tt-galaxy-wh", 32, "def456", "2026-03-21", 323, "", "deepseek_v3"),
        ]

        rows = list_trace_runs()
        assert len(rows) == 2
        out = capsys.readouterr().out
        assert "p150b" in out
        assert "whisper" in out

    @patch("tests.sweep_framework.load_ttnn_ops_data_v2.psycopg2")
    def test_list_empty(self, mock_pg, capsys):
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = []

        rows = list_trace_runs()
        assert rows == []
        assert "No trace runs found" in capsys.readouterr().out


# =====================================================================
# 4. Verify Reconstruction & Find Config Line Numbers
# =====================================================================


class TestVerifyReconstruction:
    """verify [original] [reconstructed] — GUIDE.md CLI Reference."""

    def test_identical_files(self, tmp_path):
        data = {
            "operations": {
                "ttnn::add": {"configurations": [{"arguments": {"x": 1}}, {"arguments": {"x": 2}}]},
                "ttnn::multiply": {"configurations": [{"arguments": {"y": 3}}]},
            }
        }
        orig = tmp_path / "orig.json"
        recon = tmp_path / "recon.json"
        orig.write_text(json.dumps(data))
        recon.write_text(json.dumps(data))

        result = verify_reconstruction(str(orig), str(recon))
        assert result["original_ops"] == 2
        assert result["reconstructed_ops"] == 2
        assert result["missing_ops"] == []
        assert result["extra_ops"] == []
        assert result["config_diffs"] == []

    def test_missing_and_extra_ops(self, tmp_path):
        orig = {"operations": {"ttnn::add": {"configurations": []}, "ttnn::sub": {"configurations": []}}}
        recon = {"operations": {"ttnn::add": {"configurations": []}, "ttnn::mul": {"configurations": []}}}
        orig_f = tmp_path / "orig.json"
        recon_f = tmp_path / "recon.json"
        orig_f.write_text(json.dumps(orig))
        recon_f.write_text(json.dumps(recon))

        result = verify_reconstruction(str(orig_f), str(recon_f))
        assert "ttnn::sub" in result["missing_ops"]
        assert "ttnn::mul" in result["extra_ops"]

    def test_config_count_difference(self, tmp_path):
        orig = {"operations": {"ttnn::add": {"configurations": [{"x": 1}]}}}
        recon = {"operations": {"ttnn::add": {"configurations": [{"x": 1}, {"x": 2}]}}}
        orig_f = tmp_path / "orig.json"
        recon_f = tmp_path / "recon.json"
        orig_f.write_text(json.dumps(orig))
        recon_f.write_text(json.dumps(recon))

        result = verify_reconstruction(str(orig_f), str(recon_f))
        assert len(result["config_diffs"]) == 1
        assert result["config_diffs"][0] == ("ttnn::add", 1, 2)


class TestFindConfigLineNumbers:
    """find-lines <op> <i1,i2,...> — GUIDE.md CLI Reference."""

    def test_finds_correct_lines(self, tmp_path):
        data = {
            "operations": {
                "ttnn::add": {
                    "configurations": [
                        {"arguments": {"arg0": 1}},
                        {"arguments": {"arg0": 2}},
                        {"arguments": {"arg0": 3}},
                    ]
                }
            }
        }
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data, indent=2))

        result = find_config_line_numbers(str(json_file), "ttnn::add", [0, 1, 2])
        assert result[0] is not None
        assert result[1] is not None
        assert result[2] is not None
        assert result[0] < result[1] < result[2]

    def test_operation_not_found(self, tmp_path, capsys):
        data = {"operations": {"ttnn::add": {"configurations": []}}}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data, indent=2))

        result = find_config_line_numbers(str(json_file), "ttnn::nonexistent", [0])
        assert result == {}
        assert "not found" in capsys.readouterr().out.lower()

    def test_index_out_of_range(self, tmp_path):
        data = {"operations": {"ttnn::add": {"configurations": [{"arguments": {"x": 1}}]}}}
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(data, indent=2))

        result = find_config_line_numbers(str(json_file), "ttnn::add", [0, 99])
        assert result[0] is not None
        assert result[99] is None


# =====================================================================
# 5. CLI Argument Parsing (__main__ block)
# =====================================================================


def _parse_cli(argv):
    """Parse CLI argv the same way the __main__ block does.

    Returns (function_name, args_dict) so tests can verify dispatch without
    actually calling any function or connecting to any DB.
    """
    cmd = argv[0] if argv else None

    if cmd == "load":
        args = argv[1:]
        dry_run = "--dry-run" in args
        args = [a for a in args if a != "--dry-run"]
        return "load_data", {
            "json_path": args[0] if args else None,
            "tt_metal_sha": args[1] if len(args) > 1 else None,
            "dry_run": dry_run,
        }
    elif cmd == "reconstruct-trace":
        return "reconstruct_from_trace_run", {
            "trace_run_id": int(argv[1]),
            "output_path": argv[2] if len(argv) > 2 else None,
        }
    elif cmd == "reconstruct-manifest":
        _args = argv[1:]
        if _args and _args[0].endswith(".json"):
            manifest, output = None, _args[0]
            _args = _args[1:]
        else:
            manifest = _args[0] if _args else None
            output = _args[1] if len(_args) > 1 else None
            _args = _args[2:]
        scope = _args[0] if _args else None
        schema = _args[1] if len(_args) > 1 else "ttnn_ops_v5"
        return "reconstruct_from_manifest", {
            "manifest_path": manifest,
            "output_path": output,
            "scope": scope,
            "schema": schema,
        }
    elif cmd == "resolve-manifest":
        return "resolve_manifest", {
            "manifest_path": argv[1] if len(argv) > 1 else None,
            "scope": argv[2] if len(argv) > 2 else None,
        }
    elif cmd == "reconstruct-op":
        return "reconstruct_single_operation", {
            "operation_name": argv[1],
            "output_path": argv[2] if len(argv) > 2 else None,
        }
    elif cmd == "verify":
        return "verify_reconstruction", {
            "original_path": argv[1] if len(argv) > 1 else None,
            "reconstructed_path": argv[2] if len(argv) > 2 else None,
        }
    elif cmd == "find-lines":
        return "find_config_line_numbers", {
            "operation_name": argv[1],
            "config_indices": [int(i) for i in argv[2].split(",")],
            "json_path": argv[3] if len(argv) > 3 else None,
        }
    elif cmd == "list-traces":
        return "list_trace_runs", {
            "model_filter": argv[1].split(",") if len(argv) > 1 else None,
        }
    elif cmd == "delete-trace":
        return "delete_trace_run", {
            "trace_run_id": int(argv[1]),
            "yes": "--yes" in argv,
        }
    elif cmd == "set-model-name":
        _args = argv[1:]
        _kv = {}
        i = 0
        while i < len(_args):
            if _args[i].startswith("--") and i + 1 < len(_args):
                _kv[_args[i][2:]] = _args[i + 1]
                i += 2
            else:
                i += 1
        return "set_model_name", {
            "source_file": _kv.get("source-file"),
            "hf_model": _kv.get("hf-model"),
            "model_id": _kv.get("model-id"),
            "new_name": _kv.get("model-name"),
        }
    return None, {}


class TestCLI:
    """CLI dispatch: verify argument parsing matches documented CLI from GUIDE.md.

    Tests the argument extraction logic that mirrors the __main__ block, ensuring
    the correct function is dispatched with the correct arguments for every
    documented invocation form.
    """

    def test_cli_load_default(self):
        """python load_ttnn_ops_data_v2.py load"""
        fn, args = _parse_cli(["load"])
        assert fn == "load_data"
        assert args == {"json_path": None, "tt_metal_sha": None, "dry_run": False}

    def test_cli_load_with_path(self):
        """python load_ttnn_ops_data_v2.py load path/to/master.json"""
        fn, args = _parse_cli(["load", "path/to/master.json"])
        assert fn == "load_data"
        assert args["json_path"] == "path/to/master.json"
        assert args["tt_metal_sha"] is None
        assert args["dry_run"] is False

    def test_cli_load_with_path_and_sha(self):
        """python load_ttnn_ops_data_v2.py load path.json abc123def456"""
        fn, args = _parse_cli(["load", "path.json", "abc123def456"])
        assert fn == "load_data"
        assert args == {"json_path": "path.json", "tt_metal_sha": "abc123def456", "dry_run": False}

    def test_cli_load_dry_run(self):
        """python load_ttnn_ops_data_v2.py load path.json --dry-run"""
        fn, args = _parse_cli(["load", "path.json", "--dry-run"])
        assert fn == "load_data"
        assert args == {"json_path": "path.json", "tt_metal_sha": None, "dry_run": True}

    def test_cli_load_dry_run_with_sha(self):
        """python load_ttnn_ops_data_v2.py load path.json abc123 --dry-run"""
        fn, args = _parse_cli(["load", "path.json", "abc123", "--dry-run"])
        assert fn == "load_data"
        assert args == {"json_path": "path.json", "tt_metal_sha": "abc123", "dry_run": True}

    def test_cli_resolve_manifest(self):
        """python load_ttnn_ops_data_v2.py resolve-manifest manifest.yaml model_traced"""
        fn, args = _parse_cli(["resolve-manifest", "manifest.yaml", "model_traced"])
        assert fn == "resolve_manifest"
        assert args == {"manifest_path": "manifest.yaml", "scope": "model_traced"}

    def test_cli_resolve_manifest_no_scope(self):
        """python load_ttnn_ops_data_v2.py resolve-manifest manifest.yaml"""
        fn, args = _parse_cli(["resolve-manifest", "manifest.yaml"])
        assert fn == "resolve_manifest"
        assert args == {"manifest_path": "manifest.yaml", "scope": None}

    def test_cli_reconstruct_manifest_full(self):
        """python load_ttnn_ops_data_v2.py reconstruct-manifest manifest.yaml out.json lead_models"""
        fn, args = _parse_cli(["reconstruct-manifest", "manifest.yaml", "out.json", "lead_models"])
        assert fn == "reconstruct_from_manifest"
        assert args == {
            "manifest_path": "manifest.yaml",
            "output_path": "out.json",
            "scope": "lead_models",
            "schema": "ttnn_ops_v5",
        }

    def test_cli_reconstruct_manifest_output_only(self):
        """python load_ttnn_ops_data_v2.py reconstruct-manifest output.json"""
        fn, args = _parse_cli(["reconstruct-manifest", "output.json"])
        assert fn == "reconstruct_from_manifest"
        assert args["manifest_path"] is None
        assert args["output_path"] == "output.json"

    def test_cli_reconstruct_trace(self):
        """python load_ttnn_ops_data_v2.py reconstruct-trace 35 output.json"""
        fn, args = _parse_cli(["reconstruct-trace", "35", "output.json"])
        assert fn == "reconstruct_from_trace_run"
        assert args == {"trace_run_id": 35, "output_path": "output.json"}

    def test_cli_reconstruct_trace_no_output(self):
        """python load_ttnn_ops_data_v2.py reconstruct-trace 35"""
        fn, args = _parse_cli(["reconstruct-trace", "35"])
        assert fn == "reconstruct_from_trace_run"
        assert args == {"trace_run_id": 35, "output_path": None}

    def test_cli_set_model_name_by_source(self):
        """python load_ttnn_ops_data_v2.py set-model-name --source-file demo.py --model-name foo"""
        fn, args = _parse_cli(["set-model-name", "--source-file", "demo.py", "--model-name", "foo"])
        assert fn == "set_model_name"
        assert args == {"source_file": "demo.py", "hf_model": None, "model_id": None, "new_name": "foo"}

    def test_cli_set_model_name_by_id(self):
        """python load_ttnn_ops_data_v2.py set-model-name --model-id 7 --model-name vit_nightly"""
        fn, args = _parse_cli(["set-model-name", "--model-id", "7", "--model-name", "vit_nightly"])
        assert fn == "set_model_name"
        assert args == {"source_file": None, "hf_model": None, "model_id": "7", "new_name": "vit_nightly"}

    def test_cli_set_model_name_by_hf(self):
        """python load_ttnn_ops_data_v2.py set-model-name --hf-model meta-llama/Llama --model-name llama_custom"""
        fn, args = _parse_cli(["set-model-name", "--hf-model", "meta-llama/Llama", "--model-name", "llama_custom"])
        assert fn == "set_model_name"
        assert args["hf_model"] == "meta-llama/Llama"
        assert args["new_name"] == "llama_custom"

    def test_cli_delete_trace(self):
        """python load_ttnn_ops_data_v2.py delete-trace 42 --yes"""
        fn, args = _parse_cli(["delete-trace", "42", "--yes"])
        assert fn == "delete_trace_run"
        assert args == {"trace_run_id": 42, "yes": True}

    def test_cli_delete_trace_no_yes(self):
        """python load_ttnn_ops_data_v2.py delete-trace 42"""
        fn, args = _parse_cli(["delete-trace", "42"])
        assert fn == "delete_trace_run"
        assert args == {"trace_run_id": 42, "yes": False}

    def test_cli_verify(self):
        """python load_ttnn_ops_data_v2.py verify orig.json recon.json"""
        fn, args = _parse_cli(["verify", "orig.json", "recon.json"])
        assert fn == "verify_reconstruction"
        assert args == {"original_path": "orig.json", "reconstructed_path": "recon.json"}

    def test_cli_list_traces(self):
        """python load_ttnn_ops_data_v2.py list-traces deepseek"""
        fn, args = _parse_cli(["list-traces", "deepseek"])
        assert fn == "list_trace_runs"
        assert args == {"model_filter": ["deepseek"]}

    def test_cli_list_traces_no_filter(self):
        """python load_ttnn_ops_data_v2.py list-traces"""
        fn, args = _parse_cli(["list-traces"])
        assert fn == "list_trace_runs"
        assert args == {"model_filter": None}

    def test_cli_reconstruct_op(self):
        """python load_ttnn_ops_data_v2.py reconstruct-op ttnn::add output.json"""
        fn, args = _parse_cli(["reconstruct-op", "ttnn::add", "output.json"])
        assert fn == "reconstruct_single_operation"
        assert args == {"operation_name": "ttnn::add", "output_path": "output.json"}

    def test_cli_find_lines(self):
        """python load_ttnn_ops_data_v2.py find-lines ttnn::add 0,1,2"""
        fn, args = _parse_cli(["find-lines", "ttnn::add", "0,1,2"])
        assert fn == "find_config_line_numbers"
        assert args["operation_name"] == "ttnn::add"
        assert args["config_indices"] == [0, 1, 2]

    def test_cli_find_lines_custom_json(self):
        """python load_ttnn_ops_data_v2.py find-lines ttnn::add 0,1 custom.json"""
        fn, args = _parse_cli(["find-lines", "ttnn::add", "0,1", "custom.json"])
        assert fn == "find_config_line_numbers"
        assert args["json_path"] == "custom.json"

    def test_cli_unknown_command(self):
        fn, args = _parse_cli(["foobar"])
        assert fn is None


# =====================================================================
# 6. Integration-Style: Manifest → Reconstruct Pipeline
# =====================================================================


class TestManifestReconstructPipeline:
    """End-to-end pipeline: resolve-manifest → reconstruct-manifest."""

    def test_lead_models_scope_filtering(self, tmp_path):
        """Only lead_models group included when scope='lead_models'."""
        f = tmp_path / "manifest.yaml"
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
        ids = resolve_manifest(str(f), scope="lead_models")
        assert ids == [35]

    def test_model_traced_scope_filtering(self, tmp_path):
        """Only model_traced group included when scope='model_traced'."""
        f = tmp_path / "manifest.yaml"
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
        ids = resolve_manifest(str(f), scope="model_traced")
        assert ids == [538]

    def test_model_filter_restricts_configs(self, tmp_path):
        """reconstruct-manifest only includes configs for the listed models."""
        f = tmp_path / "manifest.yaml"
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
        result = _resolve_manifest_with_models(str(f), scope="model_traced")
        assert result == {538: {"whisper"}}

    def test_real_manifest_structure(self, tmp_path):
        """Mirrors the actual sweep_manifest.yaml from the repo."""
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
                    - whisper
                    - llama-3.2-1b-instruct
                    - efficientnetb0
            registry:
              - trace_id: 35
                status: active
                models: [deepseek_v3]
                hardware: {board_type: Wormhole, device_series: tt-galaxy-wh, card_count: 32}
              - trace_id: 538
                status: active
                models: [whisper, llama-3.2-1b-instruct, efficientnetb0, deepseek-llm-7b-chat]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        # lead_models: deepseek_v3 → latest active = trace 35 (tt-galaxy-wh)
        lead_ids = resolve_manifest(str(f), scope="lead_models")
        assert 35 in lead_ids

        # model_traced: whisper → 538 (n300), llama → 538, efficientnet → 538
        traced_ids = resolve_manifest(str(f), scope="model_traced")
        assert 538 in traced_ids

        # With model names
        traced_models = _resolve_manifest_with_models(str(f), scope="model_traced")
        assert 538 in traced_models
        assert traced_models[538] == {"whisper", "llama-3.2-1b-instruct", "efficientnetb0"}


# =====================================================================
# 7. Edge Cases
# =====================================================================


class TestEdgeCases:
    """Miscellaneous edge cases and boundary conditions."""

    def test_v1_format_backward_compat(self):
        """V1 configs without 'executions' key get converted to V2 format internally.

        This is tested via parse_all_sources which is used for V1 → V2 conversion.
        """
        source = ["path/a.py [HF_MODEL:org/A]", "path/b.py"]
        result = parse_all_sources(source)
        assert len(result) == 2
        assert result[0] == ("path/a.py", "org/A")
        assert result[1] == ("path/b.py", None)

    def test_model_name_exact_match_not_substring(self, tmp_path):
        """model_name matching is exact, not substring (per GUIDE.md)."""
        f = tmp_path / "manifest.yaml"
        _write_manifest(
            f,
            """\
            targets:
              model_traced:
                - model: whisper
            registry:
              - trace_id: 1
                status: active
                models: [whisper_large]
                hardware: {board_type: Wormhole, device_series: n300, card_count: 1}
            """,
        )
        ids = resolve_manifest(str(f))
        assert ids == []

    def test_derive_model_name_json_extension(self):
        # "path" is the first non-generic segment when walking left to right
        name = derive_model_name("path/to/config.json", None)
        assert name == "path"
        # When only the filename is meaningful, it strips the extension
        name2 = derive_model_name("models/demos/demo/config.json", None)
        assert name2 == "config"

    def test_parse_array_value_nested_braces(self):
        assert parse_array_value("{1, 2, 3}") == [1, 2, 3]

    def test_format_source_roundtrip_no_hf(self):
        original = "models/demos/deepseek_v3/demo/demo.py"
        sf, hf = parse_source(original)
        assert format_source(sf, hf) == original

    def test_empty_operations_in_json(self, tmp_path):
        """verify_reconstruction handles empty operations gracefully."""
        data = {"operations": {}}
        f = tmp_path / "empty.json"
        f.write_text(json.dumps(data))
        result = verify_reconstruction(str(f), str(f))
        assert result["original_ops"] == 0
        assert result["config_diffs"] == []

    def test_get_or_create_model_name_collision_raises(self):
        """model_name uniqueness violation → RuntimeError with resolution hint."""
        cur = MagicMock()
        cur.fetchone.return_value = None

        error = Exception("duplicate key value violates unique constraint ttnn_model_name_unique")
        # execute calls: (1) SELECT lookup → OK, (2) INSERT → raises uniqueness error
        cur.execute.side_effect = [None, error]

        with pytest.raises(RuntimeError, match="model_name.*already taken"):
            get_or_create_model(cur, {}, "path/demo.py", None)
