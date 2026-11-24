import os
from eval import ccl_perf

for k, v in ccl_perf.PROFILER_ENV.items():
    os.environ.setdefault(k, v)
from pathlib import Path
import ttnn
from eval.golden_tests.q_stem.conftest import _FabricMeshCache, focus_mesh_or_skip
from eval.golden_tests.q_stem.helpers import prepare_q_stem
from eval.golden_tests.q_stem.test_golden import _loose_cases
from ttnn.operations.q_stem import q_stem

case = _loose_cases()[1]
inputs, axes, extras = case.values
cache = _FabricMeshCache()
mesh = cache.get(
    focus_mesh_or_skip(), trace_region_size=ccl_perf.TRACE_REGION_SIZE, payload_bytes=extras.get("fabric_payload_bytes")
)
call, check = prepare_q_stem(inputs, mesh_device=mesh, extras=extras, op=q_stem, **axes)
variants = [
    "",
    "SEND_PAYLOAD",
    "OUT_WRITE",
    "SEND_PAYLOAD,OUT_WRITE",
    "S3,OUT_WRITE,WKV_READ",
    "S1_COMPUTE",
    "X_READ,X_MCAST",
    "QB_READ",
]
for v in variants:
    os.environ["QSTEM_ABLATE"] = v
    call()
    ttnn.synchronize_device(mesh)
    d = ccl_perf.run(
        mesh,
        call,
        experiments=3,
        out_path=Path(f"/tmp/q_stem_verify/ablate/{v or 'none'}.json"),
        label=v or "none",
        log=None,
    )
    s = d["summary"]
    print(
        "ABL",
        repr(v),
        {
            k: (round(s[k]["mean"], 1) if isinstance(s.get(k), dict) and "mean" in s[k] else s.get(k))
            for k in ("fastest", "slowest")
        },
    )
os.environ["QSTEM_ABLATE"] = ""
cache.close()
