import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get(None)
TSL = ttnn.TensorMemoryLayout


def view():
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    return v.largest_contiguous_bytes_free_per_bank, v.total_bytes_allocated_per_bank


orig = pd.select_plan


def spy(*a):
    print("PROBE   select_plan budget", a[-1], "view", view())
    return orig(*a)


pd.select_plan = spy
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
for ml in (TSL.HEIGHT_SHARDED, TSL.INTERLEAVED, TSL.HEIGHT_SHARDED):
    try:
        run_distributed_rms_norm(((1, 1, 2048, 2048),), mesh_device=mesh, memory_layout=ml, **kw)
        r = "ok"
    except Exception as e:
        r = str(e)[:70]
    print("PROBE", str(ml).split(".")[-1], r, "view", view(), "landing", [v[0] for v in pd._LANDING_CACHE.values()])
cache.close()
