# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.2: the fork paths PACKED_DESIGN.md section 8 names as unprobed at 8192 (R = 256 rows a chip on the 8x4), on the
model's mesh (resident runner, before / without a model load), smallest first:
  - P.83 mhc_post fast (MHC_POST_FAST): the fast program vs the old one at [256, 5120] a chip (with and without the
    feed, and the plain separate-mix case), the shapes the model's residual passes take at 8192.
  - P.81 grouped_matmul (ATTN_ROOFLINE, wo_a): whether it plans at M = 256 (G 8, K 4096, N 1024) with the model's
    block_w / depth; when it plans, against the stock batched matmul and float64 (the fork's own check); when it does
    not, the model falls back to the stock matmul (printed)."""

import pytest

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"


@mesh_parametrize(timeout_s=900)
def test_chunk8192_ops(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from ttnn.bringup.grouped_matmul_ttnn import supported
    from ttnn.bringup.mhc_post_ttnn.tests.unit.test_mhc_post_fast import check_fast

    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    rows = 8192 // mesh_device.get_num_devices()
    for case, name in [
        ((rows, 5120, True, False, True, True), "norm"),
        ((rows, 5120, True, True, True, True), "norm_feed"),
        ((rows, 5120, False, False, False, False), "plain_sep"),
    ]:
        check_fast(mesh_device, *case)
        print(f"PB.2 mhc_post fast == old at [{rows}, 5120]: {name} ok", flush=True)

    from ttnn.bringup.grouped_matmul_ttnn.tests.unit.test_grouped_matmul import check

    bw, depth = settings.get("WO_A_GMM_BLOCK_W"), settings.get("WO_A_GMM_DEPTH")
    g = mesh_device.compute_with_storage_grid_size()
    grid = (min(g.x, 11), min(g.y, 10))
    shape = (1, 8, rows, 4096), (1, 8, 4096, 1024)
    ok = supported(*shape, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, mesh_device, ttnn.CoreCoord(*grid), bw)
    print(f"PB.2 grouped_matmul wo_a at M {rows} bw {bw} depth {depth}: plans = {ok}", flush=True)
    if ok:
        check(mesh_device, 8, rows, 4096, 1024, ttnn.bfloat8_b, grid, bw, depth)
        print(f"PB.2 grouped_matmul wo_a at M {rows}: == stock / float64 ok", flush=True)
