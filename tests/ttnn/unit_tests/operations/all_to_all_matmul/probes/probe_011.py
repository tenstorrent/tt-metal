import os

os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
os.environ["A2AMM_ABLATE"] = os.environ.get("ABL", "zones")
import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache
from eval.golden_tests.all_to_all_matmul.test_golden import _loose_cases
from eval.golden_tests.all_to_all_matmul.helpers import prepare_all_to_all_matmul
from ttnn.operations.all_to_all_matmul import all_to_all_matmul

idx = int(os.environ.get("CASE", "1"))
inputs, axes, extras = _loose_cases()[idx].values
cache = _FabricMeshCache()
mesh = cache.get((1, 4), payload_bytes=extras.get("fabric_payload_bytes"))
call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, op=all_to_all_matmul, **axes)
for i in range(3):
    out = call()
ttnn.synchronize_device(mesh)
ttnn.ReadDeviceProfiler(mesh)
cache.close()
