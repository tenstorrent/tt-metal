# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP fracture correctness on a real mesh: a column-fractured matmul + all_gather must reproduce the
dense single-chip matmul (PCC ~ 1). Skips when ttnn / a multi-chip mesh is unavailable, so it is inert
in the offline venv and runs only on hardware. Proven on a QB2 (2,2) mesh: PCC 0.99997 across shapes."""
import pytest


@pytest.mark.parametrize("m,k,n", [(128, 512, 1024), (256, 1024, 2048), (64, 2048, 4096)])
def test_column_fracture_matches_dense(m, k, n):
    ttnn = pytest.importorskip("ttnn")
    from cc_optimize.tp_fracture import verify_fracture

    if not hasattr(ttnn, "open_mesh_device"):
        pytest.skip("ttnn mesh API unavailable")
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    except Exception as exc:
        pytest.skip(f"no multi-chip mesh available: {exc}")
    try:
        r = verify_fracture(mesh, m=m, k=k, n=n, tp=4)
        assert r["pcc"] > 0.99, r
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
