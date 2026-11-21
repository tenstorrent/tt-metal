import os

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.tests.sparse_mla.perf_cases import (
    BOX,
    LONG_CACHE_TOKENS,
    build_sparse_perf_case,
    gate_params,
)
from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_perf import _time_traced, _time_untraced
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "mesh_device, device_params, variant, case_name",
    [p for p in gate_params() if not p.id.endswith("no_indexer")],
    indirect=["mesh_device", "device_params", "variant"],
)
def test_long_full(mesh_device, device_params, variant, case_name, config_only):
    cache = int(os.environ.get("LONGFULL_CACHE", LONG_CACHE_TOKENS))
    case = build_sparse_perf_case(
        mesh_device,
        variant,
        config_only,
        cache=cache,
        chunk=BOX.chunk,
        topology=per_axis_topology(device_params["fabric_config"]),
        sp_axis=BOX.sp_axis,
    )
    start = int(os.environ.get("LONGFULL_START", cache))
    if os.environ.get("LONGFULL_META") == "1":
        import torch

        def _scalar(v):
            return ttnn.from_torch(
                torch.tensor([v], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        meta = (_scalar(0), _scalar(start), _scalar(start + BOX.chunk))

    if os.environ.get("LONGFULL_FILL") == "1":
        import time

        from models.demos.deepseek_v3_d_p.tests.sparse_mla.test_sparse_mla_determinism import _fill_prefix

        t0 = time.time()
        _fill_prefix(mesh_device, case, start, BOX.chunk, config_only.hidden_size)
        print(f"filled prefix in {time.time() - t0:.1f} s", flush=True)
    if os.environ.get("LONGFULL_META") == "1":
        orig_forward = case.mla.forward
        case.mla.forward = lambda *a, **k: orig_forward(*a, **{**k, "metadata": meta})
    u, ref = _time_untraced(mesh_device, case, start)
    if os.environ.get("LONGFULL_SAVE"):
        import torch

        torch.save(ref, os.environ["LONGFULL_SAVE"] + f"_{case_name}.pt")
    t, tout = _time_traced(mesh_device, case, start)
    if os.environ.get("LONGFULL_SAVE"):
        import torch

        torch.save(tout, os.environ["LONGFULL_SAVE"] + f"_{case_name}_traced.pt")
    print(
        f"\nLONGFULL {case_name} cache={cache} start={start}: untraced {min(u):.3f} ms (samples {' '.join(f'{x:.3f}' for x in u)}), traced {t[0]:.3f} ms"
    )
