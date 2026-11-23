# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""high_bw_all_gather on all 8 LoudBox chips opened as a 1 x 8 mesh with FABRIC_1D_RING (one ring through every chip,
2 links, both directions), using the op test's perf helper (realtime profiler; effective receive bandwidth a chip =
shard bytes x 7 / median time). A ring all-gather moves 7 shards into every chip over 4 link directions (2 links x 2
directions), so a link direction carries 1.75 shards: link-direction rate = receive rate / 4."""

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.experimental.test_high_bw_all_gather import (
    _device_params,
    _run_high_bw_all_gather_perf,
)

ROWS = [8192, 14400, 65536]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [_device_params(ttnn.FabricConfig.FABRIC_1D_RING)], ids=["ring1d"], indirect=True
)
@pytest.mark.parametrize("rows", ROWS, ids=[f"rows{r}" for r in ROWS])
@pytest.mark.parametrize(
    "dtype,width,layout,page",
    [(ttnn.bfloat16, 576, ttnn.ROW_MAJOR_LAYOUT, 1152), (ttnn.bfloat8_b, 576, ttnn.TILE_LAYOUT, 1088)],
    ids=["bf16_rows", "bfp8_tiles"],
)
def test_hbag_ring8(mesh_device, rows, dtype, width, layout, page):
    bw, ns = _run_high_bw_all_gather_perf(
        mesh_device, dtype, width, layout, page, min_bandwidth_gbps=0.0, cluster_axis=1, rows_per_device=rows
    )
    print(f"HBAG_RING8 rows {rows} {dtype}: receive {bw:.1f} GB/s a chip = {bw / 4:.1f} GB/s a link direction")
