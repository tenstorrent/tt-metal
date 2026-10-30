import torch, ttnn, traceback
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get(None)
TSL = ttnn.TensorMemoryLayout


def view():
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    return v.largest_contiguous_bytes_free_per_bank, v.total_bytes_allocated_per_bank


kw = dict(
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    fp32_dest_acc_en=True,
    gamma_mode="gamma",
    gamma_dtype=ttnn.float32,
    gamma_layout=ttnn.TILE_LAYOUT,
    cluster_axis=0,
    topology=ttnn.Topology.Linear,
    num_links=1,
)
print("PROBE start", view())
for shp, ml in [
    ((1, 1, 32, 1792), TSL.INTERLEAVED),
    ((1, 1, 2048, 2048), TSL.HEIGHT_SHARDED),
    ((1, 1, 4096, 1024), TSL.INTERLEAVED),
    ((1, 1, 4096, 1024), TSL.WIDTH_SHARDED),
    ((1, 1, 32, 4096), TSL.HEIGHT_SHARDED),
    ((1, 1, 2048, 2048), TSL.HEIGHT_SHARDED),
    ((1, 1, 640, 1792), TSL.HEIGHT_SHARDED),
]:
    try:
        run_distributed_rms_norm(((shp),), mesh_device=mesh, memory_layout=ml, **kw)
        r = "ok"
    except Exception as e:
        r = (type(e).__name__ + ": " + str(e).split("\n")[0])[:160]
    print("PROBE", shp, str(ml).split(".")[-1], r, "view", view())
cache.close()
