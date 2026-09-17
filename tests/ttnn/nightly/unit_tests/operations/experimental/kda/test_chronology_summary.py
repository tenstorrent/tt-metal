# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh chronology summary replay; separate from module-scoped device tests."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.recurrent_chunk_scan_test_utils import (
    device_protocol,
    host_protocol,
)
from tests.ttnn.nightly.unit_tests.operations.experimental.kda.test_summarize_chunk_recurrence import (
    _segmented_summary_oracle,
)
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate

pytestmark = run_for_blackhole()


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 2_000_000}], indirect=True)
@pytest.mark.parametrize("groups", [1, 4])
def test_device_chronology_summaries_single_capture(
    mesh_device: ttnn.MeshDevice, groups: int, device_params: dict
) -> None:
    """Validate only live head/tail slots while one trace changes split and boundary rank."""
    device = mesh_device
    chunks_per_group = 4
    rows = groups * chunks_per_group * 32
    host = host_protocol(2 * groups, chunks_per_group, 32, 32, seed=981)
    inputs = device_protocol(host, device)

    def scalar(value: int) -> ttnn.Tensor:
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )

    actual_start = scalar(0)

    def run():
        return ttnn.experimental.kda.summarize_chunk_recurrence(
            *inputs,
            groups_per_head=groups,
            emit_tail_summaries=True,
            actual_start=actual_start,
            sequence_parallel_axis=0,
        )

    for _ in range(2):
        for tensor in run():
            ttnn.deallocate(tensor)
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    outputs = run()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    try:
        for offset in range(0, 2 * rows + 32, 32):
            source = scalar(offset)
            ttnn.copy(source, actual_start)
            ttnn.deallocate(source)
            ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
            split = offset % rows != 0 and (offset // rows) % 2 == 0
            wrap = (rows - offset % rows) // 32 if split else rows // 32
            expected = _segmented_summary_oracle(host, groups, chunks_per_group, wrap)
            actual = [ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float() for t in outputs]
            assert all(t.dtype == ttnn.bfloat16 for t in outputs)
            for folded_head in range(2 * groups):
                group = folded_head % groups
                for part in range(4):
                    active = (
                        group * chunks_per_group < wrap if part < 2 else split and (group + 1) * chunks_per_group > wrap
                    )
                    if active:
                        assert_accurate(
                            expected[part][folded_head].bfloat16().float(),
                            actual[part][folded_head],
                            name=f"summary G={groups} start={offset} group={group} part={part}",
                            pcc_threshold=0.999,
                        )
    finally:
        ttnn.release_trace(device, trace)
