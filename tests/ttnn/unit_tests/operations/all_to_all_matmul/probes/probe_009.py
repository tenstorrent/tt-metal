import os, sys
from eval import ccl_perf

for k, v in ccl_perf.PROFILER_ENV.items():
    os.environ.setdefault(k, v)
from pathlib import Path
import ttnn
from eval.golden_tests.all_to_all_matmul.conftest import _FabricMeshCache
from eval.golden_tests.all_to_all_matmul.test_golden import _loose_cases
from eval.golden_tests.all_to_all_matmul.helpers import prepare_all_to_all_matmul
from ttnn.operations.all_to_all_matmul import all_to_all_matmul

idx = int(os.environ.get("CASE", "1"))
case = _loose_cases()[idx]
inputs, axes, extras = case.values
cache = _FabricMeshCache()
mesh = cache.get((1, 4), trace_region_size=ccl_perf.TRACE_REGION_SIZE, payload_bytes=extras.get("fabric_payload_bytes"))
call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, op=all_to_all_matmul, **axes)
variants = os.environ.get("VARIANTS", "").split(";")
for v in variants:
    os.environ["A2AMM_ABLATE"] = v
    r = ccl_perf.run(
        mesh, call, experiments=3, out_path=Path(f"/tmp/a2amm_perf/abl_{v or 'none'}.json"), label=v or "none"
    )


cache.close()
