import os

os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
from eval import ccl_perf

for k, v in ccl_perf.PROFILER_ENV.items():
    os.environ.setdefault(k, v)
import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache
from eval.golden_tests.all_to_all_matmul.test_golden import _loose_cases
from eval.golden_tests.all_to_all_matmul.helpers import prepare_all_to_all_matmul
from ttnn.operations.all_to_all_matmul import all_to_all_matmul

inputs, axes, extras = _loose_cases()[int(os.environ.get("CASE", "0"))].values
cache = _FabricMeshCache()
mesh = cache.get((1, 4), trace_region_size=ccl_perf.TRACE_REGION_SIZE, payload_bytes=extras.get("fabric_payload_bytes"))
call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, op=all_to_all_matmul, **axes)
os.environ["A2AMM_ABLATE"] = "zones"
for _ in range(3):
    call()
ttnn.synchronize_device(mesh)
ttnn.ReadDeviceProfiler(mesh)
os.environ["A2AMM_ABLATE"] = ""
cache.close()
