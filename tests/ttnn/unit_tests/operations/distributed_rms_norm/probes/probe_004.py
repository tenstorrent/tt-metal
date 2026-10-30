import gc, torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
from eval.golden_tests.distributed_rms_norm.feature_spec import INPUTS
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get(None)
TSL = ttnn.TensorMemoryLayout
base = dict(
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    fp32_dest_acc_en=True,
    gamma_mode="gamma",
    gamma_dtype=ttnn.float32,
    gamma_layout=ttnn.TILE_LAYOUT,
    cluster_axis=0,
    topology=ttnn.Topology.Linear,
    num_links=1,
)


def view():
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    return v.largest_contiguous_bytes_free_per_bank, v.total_bytes_allocated_per_bank


def run(shape, ml):
    try:
        run_distributed_rms_norm(((shape),), mesh_device=mesh, memory_layout=ml, **base)
        r = "ok"
    except Exception as e:
        r = type(e).__name__ + ": " + str(e)[:60]
    land = {k: v[0] for k, v in pd._LANDING_CACHE.items()}
    print("PROBE", shape, str(ml).split(".")[-1], "view", view(), "landing_words", list(land.values()), r)


print("PROBE start view", view())
run((1, 1, 32, 4096), TSL.HEIGHT_SHARDED)
for (shp,) in INPUTS:
    run(shp, TSL.INTERLEAVED)
run((1, 1, 32, 4096), TSL.HEIGHT_SHARDED)
gc.collect()
print("PROBE after gc", view())
run((1, 1, 32, 4096), TSL.HEIGHT_SHARDED)
cache.close()
