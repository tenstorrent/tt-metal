# Scratch (not for commit): per-program device time of one MLA layer forward (realtime profiler, slowest chip per
# program), labelled by kernel sources, to see where the expanded-KV layer spends its time.
import os
from collections import defaultdict

import pytest
from loguru import logger

import models.demos.deepseek_v3_d_p.tests.test_mla as T
from models.demos.deepseek_v3_d_p.reference.kimi_k2_7_config import KimiK27Config
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import (
    moe_fabric_payload,
    torus_xy_device_params,
    torus_y_device_params,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology


def _torus_device_params(**kw):
    """Full torus (TORUS_XY) by default; SCRATCH_TORUS=y for a Galaxy whose TP rings do not close."""
    return (torus_y_device_params if os.environ.get("SCRATCH_TORUS", "xy") == "y" else torus_xy_device_params)(**kw)


def _mesh(cfg, variant_id):
    return pytest.param(
        (8, 4),
        _torus_device_params(l1_small_size=1152, fabric_payload_size=moe_fabric_payload(cfg)),
        variant_id,
        id=f"torus-8x4-{variant_id}",
    )


@pytest.mark.parametrize(
    "mesh_device,device_params,variant",
    [_mesh(KimiK3Config, "kimi_k3"), _mesh(KimiK27Config, "kimi_k2_7")],
    indirect=["mesh_device", "device_params", "variant"],
)
@pytest.mark.parametrize("mode", ["expandedV", "expandedKV"])
@pytest.mark.timeout(0)
def test_breakdown(request, mesh_device, device_params, variant, mode, monkeypatch):
    calls = []
    orig = T.profile_realtime_program

    def capture(*a, **kw):
        result, records = orig(*a, **kw)
        calls.append(records)
        return result, records

    monkeypatch.setattr(T, "profile_realtime_program", capture)
    monkeypatch.setattr(T, "DETERMINISM_REPS", 3)
    T._run_chunked_prefill(
        request,
        mesh_device,
        reference=None,
        topology=per_axis_topology(device_params["fabric_config"]),
        iters_isl=[5120],
        chunk_size_global=5120,
        prefill_len=524288,
        use_metadata_tensor=True,
        profile=True,
        determinism_check=True,
        mla_kwargs=dict(expanded_k=mode == "expandedKV"),
    )
    records = calls[-1]  # a warm forward
    per_program = {}
    for r in records:
        if not r["runtime_id"]:
            continue
        e = per_program.setdefault(r["runtime_id"], [0.0, r["kernel_sources"]])
        e[0] = max(e[0], r["duration_ns"])
    by_op = defaultdict(lambda: [0.0, 0])
    for ns, srcs in per_program.values():
        label = ",".join(sorted({os.path.basename(s).split(".")[0] for s in srcs}))[:110]
        by_op[label][0] += ns
        by_op[label][1] += 1
    total = sum(v[0] for v in by_op.values())
    for label, (ns, n) in sorted(by_op.items(), key=lambda kv: -kv[1][0])[:12]:
        logger.info(f"BREAKDOWN {mode} {variant.name} {ns / 1e6:8.3f} ms x{n} {label}")
    logger.info(f"BREAKDOWN {mode} {variant.name} TOTAL {total / 1e6:.3f} ms")
