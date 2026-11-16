"""skew_topo.py: print mesh position <-> device id <-> physical chip, bank cores. Run on whole 4x8 mesh (RUN VIA run_skew.sh)."""
import json

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import skinny as SK
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh

md = open_mesh()
try:
    ids = list(md.get_device_ids())
    print("RES", json.dumps({"device_ids_row_major": ids}))
    try:
        print("RES", json.dumps({"mesh_shape": list(md.shape)}))
    except Exception as e:
        print("shape err", e)
    for r in range(4):
        for c in range(8):
            try:
                d = md.get_device(ttnn.MeshCoordinate(r, c))
                print("RES", json.dumps({"r": r, "c": c, "id": d.id()}))
            except Exception as e:
                print("coord err", r, c, e)
                break
    cs = SK.bank_cores(md)
    print(
        "RES",
        json.dumps(
            {
                "bank_cores": [(c.x, c.y) for c in cs],
                "grid": [md.compute_with_storage_grid_size().x, md.compute_with_storage_grid_size().y],
            }
        ),
    )
finally:
    close_mesh(md)
