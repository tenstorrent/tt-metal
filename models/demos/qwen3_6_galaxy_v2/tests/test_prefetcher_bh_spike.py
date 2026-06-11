# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefetcher / global_circular_buffer feasibility spike on Blackhole 8x4 GLX.

GOAL (GO / NO-GO): the largest decode CCL fusions (matmul + reduce_scatter via
``ttnn.experimental.llama_rs_matmul``, all_gather + matmul via
``llama_all_gather_matmul_async``) all require the DRAM prefetcher's
``global_circular_buffer`` (``global_cb``). qwen3.6 v2 force-disables the
prefetcher on BH (``use_prefetcher=False``) with the claim that BH's 8-DRAM-bank
topology is incompatible with the prefetcher, which hardcodes WH TG's 12 banks
in ``get_core_ranges`` (``all_dram_cores = [CoreCoord(idx,0) for idx in range(12)]``).

This spike attempts, in increasing order of DRAM-topology dependence, on the
REAL 32-chip BH mesh:

  step 1: build ``TtLlamaPrefetcherSetup(mode="decode")``      (sub-device setup)
  step 2: ``create_global_cb()``                                (global CB alloc)
  step 3: ``ttnn.dram_prefetcher([...], tensor_addrs, global_cb)`` (weight read via 12-bank map)

Each step is wrapped so the FIRST failure is captured and reported as the
NO-GO boundary. If all three succeed, the prefetcher mechanism is viable on BH
and the matmul+CCL fusions become reachable (GO).

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_prefetcher_bh_spike.py -v -s
"""
from __future__ import annotations

import pytest
import torch

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
def test_prefetcher_bh_global_cb_spike(bh_glx_mesh):
    """GO/NO-GO: can the DRAM prefetcher global_cb be built + used on BH 8x4?"""
    from models.demos.qwen3_6_galaxy_v2.tt.prefetcher_common import TtLlamaPrefetcherSetup

    mesh = bh_glx_mesh
    grid = mesh.compute_with_storage_grid_size()
    try:
        n_dram = mesh.dram_grid_size()
        dram_str = f"({n_dram.x},{n_dram.y})"
    except Exception as e:
        dram_str = f"<dram_grid_size unavailable: {type(e).__name__}>"
    print(f"\n[spike] arch is_blackhole={is_blackhole()}  compute_grid=({grid.x},{grid.y})  dram_grid={dram_str}")
    print(f"[spike] get_core_ranges hardcodes 12 DRAM banks (all_dram_cores=[CoreCoord(idx,0) for idx in range(12)])")

    results = {}

    # --- step 1: build prefetcher setup (sub-device split) ---
    pf = None
    try:
        pf = TtLlamaPrefetcherSetup(
            mesh,
            n_tensors=5,
            n_layers=1,
            mode="decode",
            is_qwen=True,
        )
        results["step1_setup"] = "OK"
        print("[spike] step1 TtLlamaPrefetcherSetup(decode): OK")
        print(f"[spike]   dram_cores={len(pf.dram_cores)}  sender_cores={len(pf.all_sender_cores)}")
    except Exception as e:
        results["step1_setup"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
        print(f"[spike] step1 FAILED: {results['step1_setup']}")

    # --- step 2: allocate the global circular buffer ---
    if pf is not None:
        try:
            pf.create_global_cb()
            ttnn.synchronize_device(mesh)
            results["step2_global_cb"] = "OK" if pf.global_circular_buffer is not None else "FAIL: None"
            print(f"[spike] step2 create_global_cb(): {results['step2_global_cb']}")
        except Exception as e:
            results["step2_global_cb"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
            print(f"[spike] step2 FAILED: {results['step2_global_cb']}")
    else:
        results["step2_global_cb"] = "SKIPPED (step1 failed)"

    # --- step 3: exercise the DRAM prefetcher (weight read via 12-bank map) ---
    if pf is not None and results.get("step2_global_cb") == "OK":
        try:
            # Build one small DRAM-sharded weight tensor on the 12-bank dram_core_range_set
            # and a tensor_addrs tensor, then call dram_prefetcher. This is the op that
            # actually reads from the (assumed-12) DRAM banks via the sender cores.
            n_banks = max(1, len(pf.dram_cores))
            # Width must tile-divide across the (assumed-12) DRAM banks.
            width = 2176
            per_bank = ((width // n_banks + 31) // 32) * 32
            w = torch.randn(1, 1, 1280, per_bank * n_banks, dtype=torch.bfloat16)
            dram_shard = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(pf.dram_core_range_set, [1280, per_bank], ttnn.ShardOrientation.ROW_MAJOR),
            )
            w_tt = ttnn.from_torch(
                w,
                device=mesh,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=dram_shard,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            out = ttnn.dram_prefetcher(
                [w_tt],
                num_layers=1,
                global_cb=pf.global_circular_buffer,
            )
            ttnn.synchronize_device(mesh)
            results["step3_dram_prefetcher"] = "OK"
            print("[spike] step3 dram_prefetcher: OK")
        except Exception as e:
            results["step3_dram_prefetcher"] = f"FAIL: {type(e).__name__}: {str(e)[:240]}"
            print(f"[spike] step3 FAILED: {results['step3_dram_prefetcher']}")
    else:
        results["step3_dram_prefetcher"] = "SKIPPED (step2 not OK)"

    print("\n================= PREFETCHER BH SPIKE RESULT =================")
    for k, v in results.items():
        print(f"  {k:24s}: {v}")
    go = all(v == "OK" for v in results.values())
    print(f"  VERDICT: {'GO — prefetcher viable on BH' if go else 'NO-GO — prefetcher blocked on BH (see first FAIL)'}")
    print("=" * 61)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
