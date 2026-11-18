# Perf 1 zone breakdown driver: AGRN_PERF_ZONES=1 TT_METAL_DEVICE_PROFILER=1 scripts/tt-probe.sh all_gather_rms_norm_kimi < this
import os

os.environ["AGRN_PERF_ZONES"] = "1"
import torch, ttnn
from eval import ccl_perf
from eval.golden_tests.all_gather_rms_norm_kimi.conftest import _FabricMeshCache
from eval.golden_tests.all_gather_rms_norm_kimi.topology import focus_mesh_shape
from eval.golden_tests.all_gather_rms_norm_kimi.helpers import create_ttnn_input_tensor, create_ttnn_gamma
from ttnn.operations.all_gather_rms_norm_kimi import all_gather_rms_norm_kimi

cache = _FabricMeshCache()
mesh = cache.get(focus_mesh_shape()[0])
rows, cols = tuple(mesh.shape)
shape = (1, 1, int(os.environ.get("AGRN_M", "640")), 384)
torch.manual_seed(0)
x = torch.randn((rows, cols, *shape)).to(torch.bfloat16)
g = torch.randn((cols * 384,)).to(torch.bfloat16)
tx = create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
tg = create_ttnn_gamma(g, mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
spin = ccl_perf.spin_op(mesh, cycles=40_000_000)
call = lambda: all_gather_rms_norm_kimi(tx, gamma=tg, cluster_axis=1, num_links=2)
call()
ttnn.synchronize_device(mesh)
for _ in range(6):
    spin()
    call()
ttnn.synchronize_device(mesh)
ttnn.ReadDeviceProfiler(mesh)
print("PROBE DONE")
cache.close()
