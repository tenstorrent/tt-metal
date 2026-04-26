# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

from models.demos.utils.model_targets import resolve_accuracy_targets, resolve_perf_targets


def test_model_targets_resolver_prefers_specific_batch_and_seq(tmp_path: Path):
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

    generic = resolve_perf_targets("demo-model", "wh_n150", targets_yaml_path=str(yaml_path))
    specific = resolve_perf_targets("Demo-Model", "wh_n150", 1, 128, str(yaml_path))

    assert generic["decode_t/s/u"] == 10
    assert specific["decode_t/s/u"] == 20


def test_model_targets_resolver_accuracy_lookup(tmp_path: Path):
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

    accuracy = resolve_accuracy_targets("demo-model", "wh_n150", 1, 128, str(yaml_path))
    assert accuracy["top1"] == 91
