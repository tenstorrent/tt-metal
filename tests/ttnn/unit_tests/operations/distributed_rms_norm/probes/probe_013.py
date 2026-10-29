import ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm

cache = _FabricMeshCache()
mesh = cache.get()
axes = dict(
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    fp32_dest_acc_en=True,
    gamma_mode="gamma",
    gamma_dtype=ttnn.float32,
    gamma_layout=ttnn.TILE_LAYOUT,
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    cluster_axis=0,
    topology=ttnn.Topology.Linear,
    num_links=1,
)
res = []
for i in range(12):
    a = dict(axes)
    a["cluster_axis"] = i % 2
    a["num_links"] = 1 + (i // 2) % 2
    try:
        run_distributed_rms_norm(((1, 1, 4096, 1024),), mesh_device=mesh, extras={"seed": i}, **a)
        res.append("ok")
    except Exception as e:
        res.append("FAIL:" + str(e)[:90])
print("PROBE", res)
