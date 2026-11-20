# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.7 (DS41-339): moe_head's shared expert reduced by columns (MoEBlockConfig.shared_scatter_columns) on the
model's mesh (resident runner, no model load): rows moved across TP chips give bit-identical rows (off: they differ,
the reproduction), deterministic, accurate, one row block at 2048 and 8192.
ttnn/ttnn/bringup/moe_block_ttnn/tests/test_shared_scatter_columns.py's check."""

import pytest

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_LAYERS = "executed"


@mesh_parametrize(timeout_s=1500)
def test_moe_head_scatter_columns(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    from ttnn.bringup.moe_block_ttnn.tests.test_shared_scatter_columns import check_scatter_columns

    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

    g = max_grid(mesh_device)  # (the model's core cap - MAX_WORKER_CORES - fixed at the process's first plan)
    for r in check_scatter_columns(mesh_device, max_grid=(int(g.x), int(g.y)) if g is not None else (0, 0)):
        print(
            f"PB.7 moe_head scatter columns={r['columns']} chunk {r['chunk']}: moved max abs {r['moved_max_abs_diff']}"
            f" ({r['moved_rows_differ']} rows; routing rows {r['moved_routing_rows_differ']}), determinism {r['determinism_max_abs_diff']}, padded real rows "
            f"{r['padded_real_rows_max_abs_diff']}, pcc {r['pcc']:.6f}, "
            f"row blocks {r['row_blocks']}",
            flush=True,
        )
