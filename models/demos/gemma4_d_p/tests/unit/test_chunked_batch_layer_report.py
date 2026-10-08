# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guard the profiler's parallel-device aggregation and operation alignment."""

from models.demos.gemma4_d_p.scripts.chunked_batch_layer_report import MATMULS, merge_device_ops


def device_rows():
    rows = []
    for device in range(2):
        for index, code in enumerate(["MatmulDeviceOperation"] * 5 + ["AllGatherDeviceOperation"]):
            rows.append(
                {
                    "OP TYPE": "tt_dnn_device",
                    "OP CODE": code,
                    "DEVICE ID": str(device),
                    "GLOBAL CALL COUNT": str(index * 1024 + device),
                    "DEVICE KERNEL DURATION [ns]": str((device + 1) * (index + 1) * 1000),
                    "METAL TRACE ID": "1",
                    "METAL TRACE REPLAY SESSION ID": "4",
                }
            )
    return rows


def test_merge_parallel_devices_in_execution_order():
    ops = merge_device_ops(list(reversed(device_rows())), expected_devices=2)
    assert [op["label"] for op in ops[:5]] == list(MATMULS)
    assert [op["us"] for op in ops] == [2, 4, 6, 8, 10, 9]
    assert len(ops) == 6


def test_reject_mismatched_device_sequence(expect_error):
    rows = device_rows()
    rows[-1]["OP CODE"] = "ReduceScatterDeviceOperation"
    with expect_error(AssertionError, "Mismatched operation"):
        merge_device_ops(rows, expected_devices=2)


def test_reject_missing_device_data(expect_error):
    with expect_error(AssertionError, "Expected 2 devices"):
        merge_device_ops(device_rows()[:6], expected_devices=2)
