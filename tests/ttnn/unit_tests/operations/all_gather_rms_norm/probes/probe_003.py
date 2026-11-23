import torch, ttnn
from eval.golden_tests.all_gather_rms_norm import helpers
from eval.golden_tests.all_gather_rms_norm.topology import FABRIC_CONFIG, system_mesh_shape
import ttnn.operations.all_gather_rms_norm.all_gather_rms_norm_program_descriptor as pd

orig_cb = pd._cb
log = []


def _cb(index, num_pages, page_size, dtype, core_ranges):
    log.append((index, num_pages, page_size, num_pages * page_size))
    return orig_cb(index, num_pages, page_size, dtype, core_ranges)


pd._cb = _cb
ttnn.set_fabric_config(FABRIC_CONFIG)
md = ttnn.open_mesh_device(ttnn.MeshShape(*system_mesh_shape()))
try:
    call, check = (
        helpers.prepare_all_gather_rms_norm(
            ((128, 8192),),
            mesh_device=md,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            fp32_dest_acc_en=True,
            gamma_mode="gamma",
            gamma_dtype=ttnn.bfloat16,
            gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
            beta_mode="beta",
            norm_type="layer_norm",
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            cluster_axis=0,
            topology=ttnn.Topology.Ring,
            num_links=2,
            rank=2,
            alignment="tile_aligned",
        )
        if False
        else (None, None)
    )
except Exception as e:
    print("ERR", e)
import inspect

print("ERR sig", inspect.signature(helpers.prepare_all_gather_rms_norm))
ttnn.close_mesh_device(md)
