# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Print the TT_VISIBLE_DEVICES list of each carve of a live 8x4 Blackhole Galaxy.

The carve -> physical device map is a per-Galaxy property, and a wrong list does not error: it opens a mesh whose
"rows" are not Galaxy rows and the numbers come out quietly wrong (Pavle's gen_pipeline_binding.py measured every
rank differ between two Galaxies). So derive it: open the whole 8x4 with the torus fabric and ask create_submesh for
each carve's device ids, the same carving a single-process (8,4)-mesh test does.

Carves (rows of the 8x4 mesh; axis 1 is the 4-wide in-tray ring):
  2x4      rows 0-1, 2-3, 4-5, 6-7: "a Galaxy playing a LoudBox" (axis 1 a ring of 4, axis 0 a line of 2)
  4x4      rows 0-3 (top half), 4-7 (bottom half): LINE x RING, FABRIC_2D_TORUS_X
  4x4mid   rows 2-5: RING x RING (FABRIC_2D_TORUS_XY) on a subtorus Galaxy only (rows 2 and 5 cabled together)
  8x1      columns 0-3
Usage (Galaxy idle, from the repo root):
  python models/demos/deepseek_v3_d_p/utils/galaxy_carve.py [2x4 4x4 4x4mid 8x1]
then e.g.  export TT_VISIBLE_DEVICES=<the 2x4 rows 0-1 line>  and the matching single-mesh descriptor (GALAXY_RUNBOOK.md).
"""

import os
import sys

CARVES = {
    "2x4": ((2, 4), [(0, 0), (2, 0), (4, 0), (6, 0)]),
    "4x4": ((4, 4), [(0, 0), (4, 0)]),
    "4x4mid": ((4, 4), [(2, 0)]),
    "8x1": ((8, 1), [(0, 0), (0, 1), (0, 2), (0, 3)]),
}


def main(names):
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, ttnn.FabricReliabilityMode.RELAXED_INIT)
    md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
    try:
        full = list(md.get_device_ids())
        print("8x4 device ids, row-major:")
        for r in range(8):
            print(f"  row {r}: {full[4 * r : 4 * r + 4]}")
        for name in names:
            shape, offsets = CARVES[name]
            for off in offsets:
                sm = md.create_submesh(ttnn.MeshShape(*shape), ttnn.MeshCoordinate(*off))
                ids = ",".join(map(str, sm.get_device_ids()))
                print(f"{name} @ {off}: TT_VISIBLE_DEVICES={ids}")
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    sys.path.insert(0, os.environ.get("TT_METAL_HOME", "."))
    args = sys.argv[1:] or list(CARVES)
    bad = [a for a in args if a not in CARVES]
    if bad:
        raise SystemExit(f"unknown carve(s) {bad}; choose from {list(CARVES)}")
    main(args)
