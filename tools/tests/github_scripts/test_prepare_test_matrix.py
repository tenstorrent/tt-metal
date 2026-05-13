#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / ".github/scripts/utils/prepare_test_matrix.py"


def _load_prepare_matrix_module():
    spec = importlib.util.spec_from_file_location("prepare_test_matrix", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_test_matrix_preserves_strict_perf_target_checks_override():
    module = _load_prepare_matrix_module()
    tests = [
        {
            "name": "Demo",
            "cmd": "echo hello",
            "model": "demo-model",
            "owner_id": "U000000",
            "team": "models",
            "skus": {
                "wh_n150": {
                    "timeout": 5,
                    "tier": 1,
                    "strict_perf_target_checks": False,
                }
            },
        }
    ]
    enabled_skus = ["wh_n150"]
    sku_config = {"wh_n150": {"runs_on": ["runner"]}}

    matrix = module.build_test_matrix(tests, enabled_skus, sku_config)

    assert len(matrix) == 1
    assert matrix[0]["strict_perf_target_checks"] is False


def test_build_test_matrix_defaults_to_strict_checks_when_override_is_missing():
    module = _load_prepare_matrix_module()
    tests = [
        {
            "name": "Demo",
            "cmd": "echo hello",
            "model": "demo-model",
            "owner_id": "U000000",
            "team": "models",
            "skus": {
                "wh_n150": {
                    "timeout": 5,
                    "tier": 1,
                }
            },
        }
    ]
    enabled_skus = ["wh_n150"]
    sku_config = {"wh_n150": {"runs_on": ["runner"]}}

    matrix = module.build_test_matrix(tests, enabled_skus, sku_config)

    assert len(matrix) == 1
    assert "strict_perf_target_checks" not in matrix[0]
