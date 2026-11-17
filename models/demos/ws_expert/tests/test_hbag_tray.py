# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""high_bw_all_gather on the whole 2 x 4 tray (cluster_axis None: one snake ring over all 8 chips, 2 links, both
directions - the lane design's cycle and links) as a fabric reference for the weight lanes. Uses the op test's perf
helper (realtime profiler, effective receive bandwidth a chip = shard bytes x 7 / median time)."""

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.experimental.test_high_bw_all_gather import (
    _device_params,
    _run_high_bw_all_gather_perf,
)

ROWS = [8192, 14400, 65536]  # 14400 bf16 rows = 16.6 MB a chip: one lane launch (4 own experts x 4.15 MB)


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [_device_params(ttnn.FabricConfig.FABRIC_2D)],  # (the op's perf helper requires its 14 KB payload)
    ids=["pay14336"],
    indirect=True,
)
@pytest.mark.parametrize("rows", ROWS, ids=[f"rows{r}" for r in ROWS])
@pytest.mark.parametrize(
    "dtype,width,layout,page",
    [(ttnn.bfloat16, 576, ttnn.ROW_MAJOR_LAYOUT, 1152), (ttnn.bfloat8_b, 576, ttnn.TILE_LAYOUT, 1088)],
    ids=["bf16_rows", "bfp8_tiles"],
)
def test_hbag_tray_ring(mesh_device, rows, dtype, width, layout, page):
    _run_high_bw_all_gather_perf(
        mesh_device, dtype, width, layout, page, min_bandwidth_gbps=0.0, cluster_axis=None, rows_per_device=rows
    )
