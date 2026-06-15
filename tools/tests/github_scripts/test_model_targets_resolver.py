# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from models.demos.utils import model_targets
from models.demos.utils.model_targets import resolve_accuracy_targets, resolve_perf_targets, resolve_target_entry


def test_model_targets_resolver_prefers_specific_batch_and_seq(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "aliases": ["Demo-Model"],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {"batch_size": None, "seq_len": None, "status": "active", "perf": {"decode_t/s/u": 10}},
                            {"batch_size": 1, "seq_len": 128, "status": "active", "perf": {"decode_t/s/u": 20}},
                        ]
                    }
                },
            }
        },
    }
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(yaml.safe_dump(targets), encoding="utf-8")
    monkeypatch.setattr(model_targets, "TARGETS_YAML_PATH_DEFAULT", str(yaml_path))

    generic = resolve_perf_targets("demo-model", "wh_n150")
    specific = resolve_perf_targets("Demo-Model", "wh_n150", 1, 128)

    assert generic["decode_t/s/u"] == 10
    assert specific["decode_t/s/u"] == 20


def test_model_targets_resolver_accuracy_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {"batch_size": 1, "seq_len": 128, "status": "active", "perf": {}, "accuracy": {"top1": 91}}
                        ]
                    }
                },
            }
        },
    }
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(yaml.safe_dump(targets), encoding="utf-8")
    monkeypatch.setattr(model_targets, "TARGETS_YAML_PATH_DEFAULT", str(yaml_path))

    accuracy = resolve_accuracy_targets("demo-model", "wh_n150", 1, 128)
    assert accuracy["top1"] == 91


def test_model_targets_resolver_does_not_collapse_distinct_skus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [{"batch_size": 1, "seq_len": 128, "status": "active", "perf": {"decode_t/s/u": 10}}]
                    },
                    "wh_llmbox_perf": {
                        "entries": [{"batch_size": 1, "seq_len": 128, "status": "active", "perf": {"decode_t/s/u": 20}}]
                    },
                },
            }
        },
    }
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(yaml.safe_dump(targets), encoding="utf-8")
    monkeypatch.setattr(model_targets, "TARGETS_YAML_PATH_DEFAULT", str(yaml_path))

    n150 = resolve_perf_targets("demo-model", "wh_n150", 1, 128)
    llmbox = resolve_perf_targets("demo-model", "wh_llmbox_perf", 1, 128)
    assert n150["decode_t/s/u"] == 10
    assert llmbox["decode_t/s/u"] == 20


def test_model_targets_resolver_none_query_prefers_generic_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "aliases": [],
                "skus": {
                    "wh_n150": {
                        "entries": [
                            {"batch_size": None, "seq_len": None, "status": "active", "perf": {"decode_t/s/u": 10}},
                            {"batch_size": 1, "seq_len": 128, "status": "active", "perf": {"decode_t/s/u": 20}},
                        ]
                    }
                },
            }
        },
    }
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(yaml.safe_dump(targets), encoding="utf-8")
    monkeypatch.setattr(model_targets, "TARGETS_YAML_PATH_DEFAULT", str(yaml_path))

    generic_entry = resolve_target_entry("demo-model", "wh_n150", batch_size=None, seq_len=None)
    assert generic_entry["perf"]["decode_t/s/u"] == 10


def test_model_targets_resolver_blackhole_alias_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    targets = {
        "version": 1,
        "targets": {
            "demo-model": {
                "aliases": [],
                "skus": {
                    "bh_p150": {
                        "entries": [
                            {"batch_size": 1, "seq_len": 128, "status": "active", "perf": {"decode_t/s/u": 42.0}}
                        ]
                    }
                },
            }
        },
    }
    yaml_path = tmp_path / "targets.yaml"
    yaml_path.write_text(yaml.safe_dump(targets), encoding="utf-8")
    monkeypatch.setattr(model_targets, "TARGETS_YAML_PATH_DEFAULT", str(yaml_path))

    p150_alias = resolve_perf_targets("demo-model", "P150", 1, 128)
    canonical = resolve_perf_targets("demo-model", "bh_p150", 1, 128)
    assert p150_alias["decode_t/s/u"] == 42.0
    assert canonical["decode_t/s/u"] == 42.0
