# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The probe cards.py runs (through scripts/run_safe_pytest.sh) under a TT_VISIBLE_DEVICES subset: it writes the system
mesh shape the visible cards form to $BRINGUP_CARD_PROBE_OUT, or ok=false when they do not open."""

import json
import os


def test_card_probe():
    out = os.environ["BRINGUP_CARD_PROBE_OUT"]
    try:
        import ttnn

        shape = ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()
        dims = [int(shape[i]) for i in range(shape.dims())]
        n = 1
        for d in dims:
            n *= d
        # Opening the mesh with a 2D fabric is what the multi-device tests do, so that is what has to work (a subset
        # whose links do not train fails here, in the fabric router handshake).
        fabric = n > 1
        if fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
        try:
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(*dims))
            ttnn.close_mesh_device(mesh)
        finally:
            if fabric:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        res = {"ok": True, "shape": dims if len(dims) == 2 else [1, n]}
    except Exception as e:  # noqa: BLE001 - any failure means the subset does not open
        res = {"ok": False, "error": repr(e)[:400]}
    with open(out, "w") as f:
        json.dump(res, f)
