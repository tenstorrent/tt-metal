import os

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.sparse_mla.perf_cases import (
    BOX,
    GATE_CASES,
    build_sparse_perf_case,
    gate_params,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_determinism import _check, _fill_prefix
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name",
    gate_params(),
    indirect=["mesh_device", "device_params", "variant"],
)
def test_det_push(mesh_device, device_params, variant, case_name, config_only):
    import models.demos.deepseek_v3_d_p.tests.measure.determinism as det

    det.RUNS = int(os.environ.get("DET_RUNS", det.RUNS))
    cache = int(os.environ.get("DET_CACHE", 14080))
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=BOX.chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        shared_layer=GATE_CASES[case_name][1],
        sp_axis=BOX.sp_axis,
    )
    print(
        f"DETPUSH {case_name} push={case.mla._push is not None} full_push_env={os.environ.get('SPARSE_MLA_FULL_PUSH', 'default')}",
        flush=True,
    )
    ttnn.deallocate(case.forward(cache))
    ttnn.synchronize_device(mesh_device)
    results = []
    for prefix in os.environ.get("DET_PREFIXES", "zero,filled").split(","):
        if prefix == "filled":
            if GATE_CASES[case_name][1]:
                continue
            _fill_prefix(mesh_device, case, cache, BOX.chunk, config_only.hidden_size)
        failure = _check(mesh_device, case, cache)
        print(
            f"DETPUSH {case_name} {prefix}: {'NONDETERMINISTIC ' + failure.splitlines()[0] if failure else 'identical'}",
            flush=True,
        )
        results.append(failure)
    assert not any(results)
