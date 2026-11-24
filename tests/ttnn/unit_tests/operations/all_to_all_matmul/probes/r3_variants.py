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

# Refinement 3/4 variants (OUTDIR env, default /tmp/r3/var): VARS="abl|xp;abl|xp" (A2AMM_ABLATE stubs | A2AMM_XP experiment knobs), CASE = LOOSE index.
inputs, axes, extras = _loose_cases()[int(os.environ.get("CASE", "0"))].values
cache = _FabricMeshCache()
mesh = cache.get((1, 4), trace_region_size=ccl_perf.TRACE_REGION_SIZE, payload_bytes=extras.get("fabric_payload_bytes"))
call, check = prepare_all_to_all_matmul(inputs, mesh_device=mesh, extras=extras, op=all_to_all_matmul, **axes)
for v in os.environ["VARS"].split(";"):
    abl, xp = (v.split("|") + [""])[:2]
    os.environ["A2AMM_ABLATE"] = abl
    os.environ["A2AMM_XP"] = xp
    if not abl and os.environ.get("CHECK", "1") == "1":
        check(call())
        print("CHECK OK", v)
    ccl_perf.run(
        mesh,
        call,
        experiments=int(os.environ.get("EXP", "3")),
        out_path=Path(os.environ.get("OUTDIR", "/tmp/r3/var")) / (f"{v.replace(',', '+').replace('|','__')}.json"),
        label=v,
    )
os.environ["A2AMM_ABLATE"] = ""
os.environ["A2AMM_XP"] = ""
cache.close()
