# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
ResNet-50 end-to-end graph capture and offline import.

Moved from tests/ttnn/unit_tests/base_functionality/test_graph_report.py because it runs the full ResNet-50 demo,
which is too slow for per-PR and simulator runs. TestLinearModelE2E in that file keeps the per-PR end-to-end coverage.
"""

import sqlite3
from pathlib import Path

import pytest

import ttnn
from ttnn import graph_report

from tt_py_test_utils_common.utility_functions import is_wormhole_b0


@pytest.fixture
def imagenet_label_dict():
    import ast

    path = Path("models/sample_data/imagenet_class_labels.txt")
    assert path.exists(), f"ImageNet labels not found at {path}"
    with open(path, "r") as f:
        return ast.literal_eval(f.read())


@pytest.mark.skipif(not is_wormhole_b0(), reason="Requires Wormhole B0")
@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "batch_size, input_loc",
    ((16, "models/demos/vision/classification/resnet50/ttnn_resnet/demo/images/"),),
)
def test_resnet50_e2e_graph_capture(
    mesh_device, batch_size, input_loc, imagenet_label_dict, model_location_generator, tmp_path
):
    """Run ResNet50 inference, capture graph, import to DB, and validate structural properties."""
    from models.demos.vision.classification.resnet50.ttnn_resnet.demo.demo import run_resnet_inference

    report_path = tmp_path / "resnet50_report.json"

    ttnn.graph.enable_python_stack_traces()
    try:
        with ttnn.manage_config("enable_fast_runtime_mode", False):
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            run_resnet_inference(batch_size, input_loc, imagenet_label_dict, mesh_device, model_location_generator)
            ttnn.graph.end_graph_capture_to_file(report_path)
    finally:
        ttnn.graph.disable_python_stack_traces()

    assert report_path.exists(), "Report JSON should be created"

    output_dir = tmp_path / "output"
    db_path = graph_report.import_report(report_path, output_dir)
    assert db_path.exists(), "Imported DB should be created"

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM operations")
    op_count = c.fetchone()[0]
    assert op_count >= 300, f"Expected at least 300 operations, got {op_count}"

    c.execute("SELECT name, COUNT(*) FROM operations GROUP BY name ORDER BY COUNT(*) DESC")
    op_dist = dict(c.fetchall())
    assert op_dist.get("ttnn.conv2d", 0) == 159, f"Expected 159 conv2d ops, got {op_dist.get('ttnn.conv2d', 0)}"
    assert (
        op_dist.get("ttnn.deallocate", 0) == 51
    ), f"Expected 51 deallocate ops, got {op_dist.get('ttnn.deallocate', 0)}"
    assert op_dist.get("ttnn.add_", 0) == 48, f"Expected 48 add_ ops, got {op_dist.get('ttnn.add_', 0)}"

    c.execute("SELECT COUNT(*) FROM tensors WHERE device_id IS NULL")
    host_count = c.fetchone()[0]
    assert host_count > 0, "Expected host tensors (weights)"

    c.execute("SELECT COUNT(*) FROM tensors WHERE device_id IS NOT NULL")
    device_count = c.fetchone()[0]
    assert device_count > 0, "Expected device tensors"

    c.execute("SELECT COUNT(*) FROM input_tensors")
    input_count = c.fetchone()[0]
    assert input_count >= 600, f"Expected at least 600 input_tensor rows, got {input_count}"

    c.execute("SELECT COUNT(*) FROM output_tensors")
    output_count = c.fetchone()[0]
    assert output_count >= 300, f"Expected at least 300 output_tensor rows, got {output_count}"

    c.execute(
        """
        SELECT COUNT(*) FROM input_tensors it
        LEFT JOIN tensors t ON it.tensor_id = t.tensor_id
        WHERE t.tensor_id IS NULL
    """
    )
    dangling = c.fetchone()[0]
    assert dangling == 0, f"{dangling} dangling input_tensors references"

    c.execute(
        """
        SELECT COUNT(*) FROM output_tensors ot
        LEFT JOIN tensors t ON ot.tensor_id = t.tensor_id
        WHERE t.tensor_id IS NULL
    """
    )
    dangling = c.fetchone()[0]
    assert dangling == 0, f"{dangling} dangling output_tensors references"

    c.execute(
        """
        SELECT COUNT(*) FROM operations o
        WHERE o.name = 'ttnn.deallocate'
        AND EXISTS (SELECT 1 FROM input_tensors it WHERE it.operation_id = o.operation_id)
        AND NOT EXISTS (SELECT 1 FROM output_tensors ot WHERE ot.operation_id = o.operation_id)
    """
    )
    dealloc_correct = c.fetchone()[0]
    assert dealloc_correct == 51, f"Expected 51 deallocate ops with input+no-output, got {dealloc_correct}"

    c.execute("SELECT COUNT(*) FROM buffers")
    buf_count = c.fetchone()[0]
    assert buf_count > 0, "Expected non-zero buffers"

    c.execute("SELECT COUNT(*) FROM captured_graph")
    cg_count = c.fetchone()[0]
    assert cg_count == op_count, f"Expected {op_count} captured_graph rows, got {cg_count}"

    c.execute("SELECT COUNT(*) FROM stack_traces")
    st_count = c.fetchone()[0]
    assert st_count > 0, f"Expected at least one stack_trace row (from Python), got {st_count}"

    conn.close()
