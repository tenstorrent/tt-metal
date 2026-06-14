# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Task 2 (device): ring-40 prefetcher global_cb allocates on BH 8x4.

Builds TtLlamaPrefetcherSetup(mode="decode", ring40=True) on the real 32-chip BH
mesh and allocates the global circular buffer with the 4-bank x 10-receiver ring-40
override. Each step is wrapped so the first failure is reported as the boundary
(same pattern as test_prefetcher_bh_spike), so a config error surfaces cleanly
rather than as a bare crash.

GO = step1 setup (4-sender / 40-receiver sub-devices, no "SubDevices intersect")
and step2 create_global_cb both OK.

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    MESH_DEVICE=BH_GLX python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_prefetcher_bh_ring40_global_cb.py -v -s
"""
from __future__ import annotations

import pytest

import ttnn
from models.common.utility_functions import is_blackhole

_MESH_SHAPE = (8, 4)


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*_MESH_SHAPE), trace_region_size=20_000_000)
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.hardware
def test_ring40_global_cb(bh_glx_mesh):
    from models.demos.qwen3_6_galaxy_v2.tt.prefetcher_common import TtLlamaPrefetcherSetup

    mesh = bh_glx_mesh
    print(f"\n[ring40-cb] is_blackhole={is_blackhole()}", flush=True)
    results = {}

    pf = None
    try:
        pf = TtLlamaPrefetcherSetup(mesh, n_tensors=3, n_layers=1, mode="decode", is_qwen=True, ring40=True)
        results["step1_setup"] = "OK"
        print(
            f"[ring40-cb] step1 setup OK: senders={len(pf.all_sender_cores)} "
            f"dram_banks={len(pf.dram_cores)} receiver_sets={len(pf.all_receiver_cores)}",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        results["step1_setup"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
        print(f"[ring40-cb] step1 FAILED: {results['step1_setup']}", flush=True)

    if pf is not None:
        try:
            pf.create_global_cb()
            ttnn.synchronize_device(mesh)
            results["step2_global_cb"] = "OK" if pf.global_circular_buffer is not None else "FAIL: None"
            print(f"[ring40-cb] step2 create_global_cb: {results['step2_global_cb']}", flush=True)
        except Exception as e:  # noqa: BLE001
            results["step2_global_cb"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
            print(f"[ring40-cb] step2 FAILED: {results['step2_global_cb']}", flush=True)
    else:
        results["step2_global_cb"] = "SKIPPED (step1 failed)"

    print("\n============== RING-40 GLOBAL_CB RESULT ==============", flush=True)
    for k, v in results.items():
        print(f"  {k:18s}: {v}", flush=True)
    go = all(v == "OK" for v in results.values())
    print(f"  VERDICT: {'GO — ring-40 global_cb allocates on BH' if go else 'NO-GO (see first FAIL)'}", flush=True)
    print("=" * 53, flush=True)

    assert results["step1_setup"] == "OK", results["step1_setup"]
    assert results["step2_global_cb"] == "OK", results["step2_global_cb"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
