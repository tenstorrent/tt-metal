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

# Verifier breakdown on the Phase 0 perf case (LOOSE_CASES[1], Kimi, 7232 B): which stage is exposed.
inputs, axes, extras = _loose_cases()[1].values
cache = _FabricMeshCache()
mesh = cache.get((1, 4), trace_region_size=ccl_perf.TRACE_REGION_SIZE, payload_bytes=extras.get("fabric_payload_bytes"))
call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, op=all_to_all_matmul, **axes)
for v in ["", "nowrite", "noweight", "nocompute", "nowrite,noweight", "nowrite,noweight,nocompute", "noportread"]:
    os.environ["A2AMM_ABLATE"] = v
    ccl_perf.run(
        mesh,
        call,
        experiments=3,
        out_path=Path(f".eval/verify_abl/{(v or 'none').replace(',', '+')}.json"),
        label=v or "none",
    )
os.environ["A2AMM_ABLATE"] = ""
cache.close()
