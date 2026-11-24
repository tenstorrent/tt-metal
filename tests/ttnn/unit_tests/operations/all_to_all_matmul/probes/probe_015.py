import os
from eval import ccl_perf

for k, v in ccl_perf.PROFILER_ENV.items():
    os.environ.setdefault(k, v)
from pathlib import Path
import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache
from eval.golden_tests.all_to_all_matmul.test_golden import _loose_cases
from eval.golden_tests.all_to_all_matmul.helpers import prepare_all_to_all_matmul
from ttnn.operations.all_to_all_matmul import all_to_all_matmul

# Refinement 3 breakdown on the perf focus (LOOSE_CASES[0], Kimi model config: L1 out, HiFi2, 16-bit DEST).
inputs, axes, extras = _loose_cases()[0].values
cache = _FabricMeshCache()
mesh = cache.get((1, 4), trace_region_size=ccl_perf.TRACE_REGION_SIZE, payload_bytes=extras.get("fabric_payload_bytes"))
call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, op=all_to_all_matmul, **axes)
for v in os.environ.get(
    "ABLS", "none;nowrite;noweight;nocompute;nowrite,noweight;nowrite,noweight,nocompute;noportread"
).split(";"):
    os.environ["A2AMM_ABLATE"] = "" if v == "none" else v
    ccl_perf.run(mesh, call, experiments=3, out_path=Path(f"/tmp/r3/abl/{v.replace(',', '+')}.json"), label=v)
os.environ["A2AMM_ABLATE"] = ""
cache.close()
