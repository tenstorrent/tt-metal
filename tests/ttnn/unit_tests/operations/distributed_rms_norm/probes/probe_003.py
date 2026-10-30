import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
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


def run(shape, ml):
    try:
        run_distributed_rms_norm(((shape),), mesh_device=mesh, memory_layout=ml, **base)
        r = "ok"
    except Exception as e:
        r = type(e).__name__ + ": " + str(e)[:80]
    print(
        "PROBE",
        shape,
        str(ml).split(".")[-1],
        "budget",
        pd.l1_budget(mesh),
        "landings",
        sorted(k[2] for k in pd._LANDING_CACHE),
        r,
    )


run((1, 1, 32, 4096), TSL.HEIGHT_SHARDED)
for shp in [(1, 1, 640, 1792), (1, 1, 2048, 1280), (1, 1, 4096, 1024), (4, 1, 512, 512)]:
    run(shp, TSL.INTERLEAVED)
run((1, 1, 32, 4096), TSL.HEIGHT_SHARDED)
cache.close()
