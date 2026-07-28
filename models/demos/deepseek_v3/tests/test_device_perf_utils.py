# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import csv

import pytest

from models.demos.deepseek_v3.utils.device_perf_utils import _parse_signposts, filter_profile_csv, process_profile_stats
from models.demos.deepseek_v3.utils.signpost_names import (
    DECODE_EXECUTE_TRACE_SAMPLE_ON_DEVICE_SIGNPOST,
    DECODE_EXECUTE_TRACE_SIGNPOST,
    DECODE_TRACE_CAPTURE_SIGNPOST,
    DECODE_WARMUP_SIGNPOST,
    FIRST_DENSE_LAYER_SIGNPOST,
    FIRST_MOE_LAYER_SIGNPOST,
    WARMUP_MODEL_SIGNPOST,
)

HEADER = [
    "OP CODE",
    "DEVICE ID",
    "OP TYPE",
    "ATTRIBUTES",
    "OP TO OP LATENCY [ns]",
    "DEVICE KERNEL DURATION [ns]",
]


def _signpost(name):
    return [name, "", "signpost", "", "", ""]


def _op(name, *, device_id="0", attributes="", latency_ns="1000", duration_ns="2000"):
    return [name, device_id, "operation", attributes, latency_ns, duration_ns]


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_parse_signposts_uses_nested_decode_warmup_inside_warmup_model():
    rows = [
        _signpost(WARMUP_MODEL_SIGNPOST),
        _op("prefill_op"),
        _signpost(DECODE_WARMUP_SIGNPOST),
        _op("embedding_op"),
        _signpost(FIRST_DENSE_LAYER_SIGNPOST),
        _op("dense_op"),
        _signpost(FIRST_DENSE_LAYER_SIGNPOST),
        _signpost(FIRST_MOE_LAYER_SIGNPOST),
        _op("moe_op"),
        _signpost(FIRST_MOE_LAYER_SIGNPOST),
        _op("tail_op"),
        _signpost(DECODE_WARMUP_SIGNPOST),
        _signpost(DECODE_TRACE_CAPTURE_SIGNPOST),
        _op("trace_capture_op"),
        _signpost(DECODE_TRACE_CAPTURE_SIGNPOST),
        _signpost(WARMUP_MODEL_SIGNPOST),
        _signpost(DECODE_EXECUTE_TRACE_SIGNPOST),
        _op("trace_embedding_op"),
        _op("trace_dense_op"),
        _op("trace_moe_op"),
        _op("trace_tail_op"),
        _signpost(DECODE_EXECUTE_TRACE_SAMPLE_ON_DEVICE_SIGNPOST),
        _op("sample_op"),
        _signpost(DECODE_EXECUTE_TRACE_SIGNPOST),
        _op("trace_embedding_op_1"),
        _op("trace_dense_op_1"),
        _op("trace_moe_op_1"),
        _op("trace_tail_op_1"),
        _signpost(DECODE_EXECUTE_TRACE_SAMPLE_ON_DEVICE_SIGNPOST),
    ]

    info = _parse_signposts(rows, op_code_idx=0, op_type_idx=2)

    assert info["warmup_range"] == (3, 10)
    assert info["warmup_structure"] == {
        "embedding": (3, 3),
        "dense": (5, 5),
        "moe": (8, 8),
        "tail": (10, 10),
    }
    assert info["trace_capture_range"] == (12, 14)
    assert info["trace_execution_ranges"] == [(17, 20), (24, 27)]


def test_filter_profile_csv_ignores_outer_warmup_model_and_labels_decode_warmup(tmp_path):
    rows = [
        _signpost(WARMUP_MODEL_SIGNPOST),
        _op("prefill_op"),
        _signpost(DECODE_WARMUP_SIGNPOST),
        _op("embedding_op"),
        _signpost(FIRST_DENSE_LAYER_SIGNPOST),
        _op("dense_op"),
        _signpost(FIRST_DENSE_LAYER_SIGNPOST),
        _signpost(FIRST_MOE_LAYER_SIGNPOST),
        _op("moe_op", attributes="cluster_axis=0"),
        _signpost(FIRST_MOE_LAYER_SIGNPOST),
        _op("tail_op"),
        _signpost(DECODE_WARMUP_SIGNPOST),
        _signpost(DECODE_TRACE_CAPTURE_SIGNPOST),
        _op("trace_capture_op"),
        _signpost(DECODE_TRACE_CAPTURE_SIGNPOST),
        _signpost(WARMUP_MODEL_SIGNPOST),
        _signpost(DECODE_EXECUTE_TRACE_SIGNPOST),
        _op("trace_embedding_op"),
        _op("trace_dense_op"),
        _op("trace_moe_op", attributes="cluster_axis=0"),
        _op("trace_tail_op"),
        _signpost(DECODE_EXECUTE_TRACE_SAMPLE_ON_DEVICE_SIGNPOST),
        _op("sample_op"),
        _signpost(DECODE_EXECUTE_TRACE_SIGNPOST),
        _op("trace_embedding_op_1"),
        _op("trace_dense_op_1"),
        _op("trace_moe_op_1", attributes="cluster_axis=0"),
        _op("trace_tail_op_1"),
        _signpost(DECODE_EXECUTE_TRACE_SAMPLE_ON_DEVICE_SIGNPOST),
    ]
    input_path = tmp_path / "ops.csv"
    output_path = tmp_path / "filtered.csv"
    _write_csv(input_path, HEADER, rows)

    filter_profile_csv(input_path, output_path)

    with open(output_path, "r", newline="", encoding="utf-8") as f:
        filtered_rows = list(csv.DictReader(f))

    assert [(row["RUN TYPE"], row["OP CODE"], row["OP LEVEL"]) for row in filtered_rows] == [
        ("warmup", "embedding_op", "Embedding"),
        ("warmup", "dense_op", "Dense Decoder"),
        ("warmup", "moe_op", "MoE Decoder"),
        ("warmup", "tail_op", "Tail"),
        ("trace_execution_0", "trace_embedding_op", "Embedding"),
        ("trace_execution_0", "trace_dense_op", "Dense Decoder"),
        ("trace_execution_0", "trace_moe_op", "MoE Decoder"),
        ("trace_execution_0", "trace_tail_op", "Tail"),
        ("trace_execution_1", "trace_embedding_op_1", "Embedding"),
        ("trace_execution_1", "trace_dense_op_1", "Dense Decoder"),
        ("trace_execution_1", "trace_moe_op_1", "MoE Decoder"),
        ("trace_execution_1", "trace_tail_op_1", "Tail"),
    ]
    assert filtered_rows[2]["OP TYPE"] == "CCL"


def test_process_profile_stats_rejects_missing_warmup_reference(tmp_path):
    filtered_path = tmp_path / "filtered.csv"
    merged_path = tmp_path / "merged.csv"
    _write_csv(
        filtered_path,
        [
            "OP CODE",
            "DEVICE ID",
            "RUN TYPE",
            "OP TYPE",
            "OP LEVEL",
            "OP TO OP LATENCY [us]",
            "DEVICE KERNEL DURATION [us]",
        ],
        [["trace_op", "0", "trace_execution_0", "Non CCL", "Embedding", "1.0", "2.0"]],
    )

    with pytest.raises(ValueError, match="No warmup reference ops found"):
        process_profile_stats(filtered_path, merged_path)
