# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Temporary home for the KDA perf-label contract."""

from models.demos.deepseek_v3_d_p.tests.kda.perf.test_layer_perf import _device_program_label


def test_device_program_label_preserves_material_operation_identity() -> None:
    assert (
        _device_program_label(("src/operations/experimental/kda/recurrent_chunk_scan/kernel.cpp",))
        == "experimental.kda.recurrent_chunk_scan"
    )
    assert _device_program_label(("src/operations/ccl/all_gather_async/kernel.cpp",)) == "ccl.all_gather_async"
    assert _device_program_label(("src/operations/matmul/kernel.cpp",)) == "matmul"
    assert _device_program_label(("src/operations/experimental/matmul/kernel.cpp",)) == "experimental.matmul"
    assert (
        _device_program_label(("src/operations/experimental/ccl/reduce_scatter/kernel.cpp",))
        == "experimental.ccl.reduce_scatter"
    )
    assert _device_program_label(("src/reader.cpp", "src/writer.cpp")) == "reader+writer"
    assert _device_program_label(()) == "unknown"
