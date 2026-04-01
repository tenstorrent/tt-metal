#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

FRAMEWORK_ROOT = Path(__file__).parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from framework import compute_sweep_matrix, compute_validation_matrix, vector_source


def _write_vector_file(path: Path, *, board_type: str, device_series: str, card_count: int, trace_id: int) -> None:
    path.write_text(
        json.dumps(
            {
                "model_traced": {
                    "vector": {
                        "trace_ids": [trace_id],
                        "traced_machine_info": {
                            "board_type": board_type,
                            "device_series": device_series,
                            "card_count": card_count,
                            "mesh_device_shape": [1, 1],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_compute_sweep_matrix_prefilters_modules_by_active_profile(tmp_path, monkeypatch, capsys):
    _write_vector_file(
        tmp_path / "model_traced.add_model_traced.hw_wormhole_n300_1c.json",
        board_type="Wormhole",
        device_series="n300",
        card_count=1,
        trace_id=3,
    )
    _write_vector_file(
        tmp_path / "model_traced.add_model_traced.hw_wormhole_tt_galaxy_wh_32c.json",
        board_type="Wormhole",
        device_series="tt_galaxy_wh",
        card_count=32,
        trace_id=4,
    )

    monkeypatch.setenv("TT_SWEEP_CAPABILITY_PROFILE", "wormhole_n300_2c_host")
    monkeypatch.setenv("VECTORS_DIR", str(tmp_path))
    monkeypatch.setenv("SWEEP_NAME", "ALL SWEEPS (Model Traced)")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("MEASURE_DEVICE_PERF", "false")

    compute_sweep_matrix.main()
    stdout = capsys.readouterr().out.strip().splitlines()
    matrix_payload = json.loads(stdout[0].split("=", 1)[1])

    assert matrix_payload["module"] == ["model_traced.add_model_traced.hw_wormhole_n300_1c"]
    assert matrix_payload["include"][0]["test_group_name"] == "wormhole-n300-sweeps"


def test_compute_validation_matrix_emits_trace_ids_and_hardware_group(tmp_path, monkeypatch):
    _write_vector_file(
        tmp_path / "model_traced.add_model_traced.hw_wormhole_n300_1c.json",
        board_type="Wormhole",
        device_series="n300",
        card_count=1,
        trace_id=3,
    )

    monkeypatch.setenv("TT_SWEEP_CAPABILITY_PROFILE", "wormhole_n300_2c_host")

    matrix_payload = compute_validation_matrix.compute_validation_matrix(tmp_path, "model_traced")

    assert matrix_payload["module"] == ["model_traced.add_model_traced.hw_wormhole_n300_1c"]
    assert matrix_payload["include"][0]["trace_ids"] == [3]
    assert matrix_payload["include"][0]["hardware_group"] == "wormhole_n300_1c"


def test_vector_export_source_filters_grouped_variants_by_active_profile(tmp_path, monkeypatch):
    _write_vector_file(
        tmp_path / "model_traced.add_model_traced.hw_wormhole_n300_1c.json",
        board_type="Wormhole",
        device_series="n300",
        card_count=1,
        trace_id=3,
    )
    _write_vector_file(
        tmp_path / "model_traced.add_model_traced.hw_wormhole_tt_galaxy_wh_32c.json",
        board_type="Wormhole",
        device_series="tt_galaxy_wh",
        card_count=32,
        trace_id=4,
    )

    monkeypatch.setenv("TT_SWEEP_CAPABILITY_PROFILE", "wormhole_n300_2c_host")

    source = vector_source.VectorExportSource(tmp_path)
    vectors = source.load_vectors("model_traced.add_model_traced", suite_name="model_traced")

    assert len(vectors) == 1
    assert vectors[0]["trace_ids"] == [3]
