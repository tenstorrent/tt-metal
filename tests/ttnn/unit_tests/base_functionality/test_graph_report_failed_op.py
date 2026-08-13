# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression test for tenstorrent/tt-metal#28836.

Provokes the issue's failure - conv2d's static circular buffers overrunning the L1 window
left by an allocated L1 buffer, so ``validate_circular_buffer_region`` throws mid-flight and
the operation never reports a ``function_end`` - then asserts it reaches the report database.
"""

import sqlite3
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "ttnn" / "ttnn"))

import graph_report

import ttnn

TILE_BYTES = 32 * 32 * 2  # bfloat16 32x32 tile

# Conv whose static CB region on the L1 path is a few hundred KB per core.
CONV = dict(batch=1, in_channels=256, out_channels=256, height=80, width=80, kernel=3, act_block_h=64)

# Free share of the L1 window: large enough for conv2d's tensors, small enough for the CBs to overrun it.
FREE_FRACTION = 0.26

CLASH_MESSAGE = "clash with L1 buffers"


def _pin_lowest_l1_address(device):
    """Pin ``lowest_occupied_compute_l1_address`` low, capping the CB budget of every core range."""
    info = ttnn._ttnn.reports.get_device_info(device)
    usable = info.worker_l1_size - info.address_at_first_l1_cb_buffer
    shard_bytes = int(usable * (1.0 - FREE_FRACTION)) // TILE_BYTES * TILE_BYTES
    shard_shape = [shard_bytes // TILE_BYTES * 32, 32]
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
            shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    return ttnn.from_torch(
        torch.zeros(shard_shape, dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
    )


def _run_conv2d(device):
    batch, in_c, out_c = CONV["batch"], CONV["in_channels"], CONV["out_channels"]
    height, width, kernel = CONV["height"], CONV["width"], CONV["kernel"]

    torch_input = torch.randn(batch, in_c, height, width, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1).reshape(1, 1, batch * height * width, in_c).contiguous(),
        ttnn.bfloat16,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_weight = ttnn.from_torch(torch.randn(out_c, in_c, kernel, kernel, dtype=torch.bfloat16), ttnn.bfloat16)

    conv_config = ttnn.Conv2dConfig(
        weights_dtype=ttnn.bfloat16,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_act_double_buffer=False,
    )
    conv_config.act_block_h_override = CONV["act_block_h"]

    return ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=in_c,
        out_channels=out_c,
        device=device,
        batch_size=batch,
        input_height=height,
        input_width=width,
        kernel_size=(kernel, kernel),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        groups=1,
        conv_config=conv_config,
        compute_config=ttnn.init_device_compute_kernel_config(device.arch()),
        dtype=ttnn.bfloat16,
        # As in the issue's yolov11m: L1_FULL skips the DRAM auto-slicing that would avoid the clash.
        slice_config=ttnn.Conv2dL1FullSliceConfig,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_failed_op_recorded_in_report(device, tmp_path, expect_error):
    """The failing conv2d must reach ``operations``, with an error row holding the clash diagnostic."""
    report_path = tmp_path / "report.json"

    hog = _pin_lowest_l1_address(device)
    try:
        with ttnn.manage_config("enable_logging", True):
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                with expect_error(RuntimeError, CLASH_MESSAGE):
                    _run_conv2d(device)
            finally:
                ttnn.graph.end_graph_capture_to_file(report_path)
    finally:
        ttnn.deallocate(hog)

    db_path = graph_report.import_report(report_path, tmp_path / "db")
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT operation_id, name FROM operations")
        operations = cursor.fetchall()
        conv_operations = [(op_id, name) for op_id, name in operations if "conv2d" in name]
        assert len(conv_operations) == 1, f"expected the failing conv2d in operations, got {operations}"
        conv_operation_id = conv_operations[0][0]

        cursor.execute("SELECT operation_id, operation_name, error_type, error_message FROM errors")
        errors = cursor.fetchall()
        assert len(errors) == 1, f"expected exactly one error row, got {errors}"
        operation_id, operation_name, error_type, error_message = errors[0]

        assert operation_id == conv_operation_id, "error row must be joinable to the failing operation"
        assert operation_name == "ttnn.conv2d"
        assert error_type == "RuntimeError", f"expected the raised exception type, got {error_type}"
        assert CLASH_MESSAGE in error_message, f"expected the clash diagnostic, got {error_message!r}"
        assert "static circular buffer region ends at" in error_message

        # Arguments, inputs and sub-graph are what make the failing operation debuggable.
        cursor.execute("SELECT COUNT(*) FROM operation_arguments WHERE operation_id = ?", (conv_operation_id,))
        assert cursor.fetchone()[0] > 0, "expected the failing operation's arguments to be recorded"
        cursor.execute("SELECT COUNT(*) FROM input_tensors WHERE operation_id = ?", (conv_operation_id,))
        assert cursor.fetchone()[0] > 0, "expected the failing operation's input tensors to be recorded"
        cursor.execute("SELECT COUNT(*) FROM captured_graph WHERE operation_id = ?", (conv_operation_id,))
        assert cursor.fetchone()[0] == 1, "expected the failing operation's captured sub-graph to be recorded"
    finally:
        conn.close()
