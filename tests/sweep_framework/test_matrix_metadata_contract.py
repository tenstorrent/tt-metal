#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_FRAMEWORK_DIR = REPO_ROOT / "tests" / "sweep_framework"
FRAMEWORK_DIR = SWEEP_FRAMEWORK_DIR / "framework"

for path in (REPO_ROOT, SWEEP_FRAMEWORK_DIR, FRAMEWORK_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import compute_sweep_matrix as matrix_builder
import sweeps_parameter_generator as sweeps_parameter_generator


def _write_vector_file(path, suite_name, vector_entries):
    path.write_text(json.dumps({suite_name: vector_entries}, indent=2), encoding="utf-8")


def test_merge_duplicate_serialized_vectors_unions_trace_ids():
    existing_vector = {
        "traced_source": ["model_a.py"],
        "traced_machine_info": {"board_type": "Wormhole", "device_series": "n300", "card_count": 1},
        "trace_ids": [5, 1],
    }
    new_vector = {
        "traced_source": ["model_b.py"],
        "traced_machine_info": {"board_type": "Wormhole", "device_series": "n300", "card_count": 1},
        "trace_ids": [3, 5, 2],
    }

    merged = sweeps_parameter_generator._merge_duplicate_serialized_vectors(existing_vector, new_vector)

    assert merged["trace_ids"] == [1, 2, 3, 5]


def test_compute_validation_matrix_aggregates_trace_ids_per_hardware_group(tmp_path):
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()

    _write_vector_file(
        vectors_dir / "model_traced.add.hw_wormhole_n300_1c.json",
        "model_traced",
        {
            "cfg_a": {"trace_ids": [3, 1]},
            "cfg_b": {"trace_ids": [2]},
        },
    )
    _write_vector_file(
        vectors_dir / "model_traced.sub.hw_wormhole_n300_1c.json",
        "model_traced",
        {
            "cfg_c": {"trace_ids": [4, 3]},
        },
    )
    _write_vector_file(
        vectors_dir / "model_traced.mul.hw_blackhole_p150b_0c.json",
        "model_traced",
        {
            "cfg_d": {"trace_ids": [9]},
        },
    )

    modules, vector_metadata = matrix_builder._load_vector_metadata(vectors_dir)
    include_entries, batches, ccl_batches = matrix_builder.compute_validation_matrix(
        modules,
        "model_traced",
        vector_metadata,
    )

    assert ccl_batches == []
    assert len(batches) == 2
    assert [entry["test_group_name"] for entry in include_entries] == [
        "validation-blackhole_p150b_0c",
        "validation-wormhole_n300_1c",
    ]

    wormhole_entry = next(entry for entry in include_entries if entry["hardware_group"] == "wormhole_n300_1c")
    assert wormhole_entry["module_selector"] == "model_traced.add,model_traced.sub"
    assert wormhole_entry["batch_display"] == "wormhole/n300/1c"
    assert wormhole_entry["trace_ids"] == [1, 2, 3, 4]


def test_compute_standard_matrix_model_traced_includes_metadata(tmp_path):
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()

    _write_vector_file(
        vectors_dir / "model_traced.add.hw_wormhole_n300_1c.json",
        "model_traced",
        {"cfg_a": {"trace_ids": [11, 10]}},
    )

    modules, vector_metadata = matrix_builder._load_vector_metadata(vectors_dir)
    include_entries, batches, ccl_batches = matrix_builder.compute_standard_matrix(
        modules,
        batch_size=10,
        suite_name="model_traced",
        vector_metadata_by_module=vector_metadata,
    )

    assert batches == ["model_traced.add"]
    assert ccl_batches == []
    assert len(include_entries) == 1
    assert include_entries[0]["hardware_group"] == "wormhole_n300_1c"
    assert include_entries[0]["trace_ids"] == [10, 11]
