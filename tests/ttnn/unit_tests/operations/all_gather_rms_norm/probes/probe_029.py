import ttnn, torch, os
from eval.golden_tests.all_gather_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.all_gather_rms_norm.helpers import prepare_all_gather_rms_norm
from eval.golden_tests.all_gather_rms_norm.test_golden import _loose_cases
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

c = _loose_cases()[int(os.environ.get("CASE", "0"))]
inputs, axes, extras = c.values
cache = _FabricMeshCache()
mesh = cache.get((1, 4))
call, check = prepare_all_gather_rms_norm(inputs, mesh_device=mesh, extras=extras, op=all_gather_rms_norm, **axes)
for _ in range(3):
    out = call()
import os

if not os.environ.get("AGRMS_ABLATE"):
    check(out)
ttnn.synchronize_device(mesh)
ttnn.ReadDeviceProfiler(mesh)
cache.close()
print("PROBE_DONE")
