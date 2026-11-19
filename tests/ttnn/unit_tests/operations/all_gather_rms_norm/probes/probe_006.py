import ttnn
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.all_gather_rms_norm.helpers import run_all_gather_rms_norm

c = _FabricMeshCache()
m = c.get((2, 2))
print("OPEN1 OK", m.shape)
ax = dict(
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    fp32_dest_acc_en=True,
    gamma_mode="no_gamma",
    gamma_dtype="none",
    gamma_layout="none",
    memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
    cluster_axis=1,
    topology=ttnn.Topology.Linear,
    num_links=1,
)
run_all_gather_rms_norm(((1, 1, 32, 64),), mesh_device=m, **ax)
print("OP OK")
ttnn.synchronize_device(m)
c.close()
print("CLOSED")
m = c.get((2, 2))
print("OPEN2 OK", m.shape)
c.close()
print("DONE")
