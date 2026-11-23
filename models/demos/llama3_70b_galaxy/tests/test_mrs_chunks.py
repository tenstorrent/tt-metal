# matmul_reduce_scatter_prefill chunk-split sweep (TT_MRS_FORCE_CHUNKS) at the DP4 x TP8 WO / FF2 shapes under the L1
# budget the full model leaves (l1_headroom ~256 KB reproduces the in-model plans). Each candidate gets a distinct
# headroom (+16 B steps) so the plan cache and program cache see a new key.
#   TT_MRS_BF8_TRANSPORT=1 MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_mrs_chunks.py -s
# env: MC_M (16384), MC_HR (262144)
import itertools
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H = 8192
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_mrs_chunks(mesh_device, device_params):
    mesh = mesh_device
    rep = ttnn.ReplicateTensorToMesh(mesh)
    hr0 = int(os.environ.get("MC_HR") or "262144")
    step = [0]
    plan_fn = ttnn._ttnn.operations.experimental.ccl_experimental._matmul_reduce_scatter_prefill_last_plan
    for M in [int(m) for m in (os.environ.get("MC_M") or "16384").split(",")]:
        for name, K in (("ff2", 3584), ("wo", 1024)):
            ta = ttnn.from_torch(
                torch.randn(1, 1, M, K),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=DRAM,
                mesh_mapper=rep,
            )
            tw = ttnn.from_torch(
                torch.randn(K, H) * 0.02,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=DRAM,
                mesh_mapper=rep,
            )
            Mt, Kt, Nt = M // 32, K // 32, H // 32
            results = []
            for nm, nn in [(0, 0)] + list(itertools.product((1, 2, 3, 4, 6), (1, 2, 3, 4, 5, 6, 8, 11, 16))):
                if nm:
                    os.environ["TT_MRS_FORCE_CHUNKS"] = f"{Mt},{Kt},{Nt},{nm},{nn}"
                else:
                    os.environ.pop("TT_MRS_FORCE_CHUNKS", None)
                step[0] += 1
                hr = hr0 + 16 * step[0]
                f = lambda: ttnn.experimental.matmul_reduce_scatter_prefill(
                    ta,
                    tw,
                    cluster_axis=0,
                    scatter_dim=-2,
                    topology=ttnn.Topology.Ring,
                    num_links=2,
                    compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi),
                    memory_config=DRAM,
                    l1_headroom=hr,
                )
                try:
                    ttnn.deallocate(f())
                    plan = dict(plan_fn())
                    ttnn.synchronize_device(mesh)
                    t0 = time.perf_counter()
                    for _ in range(6):
                        ttnn.deallocate(f())
                    ttnn.synchronize_device(mesh)
                    ms = (time.perf_counter() - t0) / 6 * 1e3
                    tag = "auto" if not nm else f"force({nm},{nn})"
                    logger.info(
                        f"MRSC M={M} {name} {tag}: {ms:.3f} ms chunks {plan['chunks']} grid {plan['grid']} "
                        f"{plan['orient']} kb {plan['kb']} depths {plan['depths']} sb {plan['super_block']} "
                        f"fp {plan['footprint_kib']} KiB"
                    )
                    results.append((ms, tag, plan["chunks"]))
                except Exception as e:
                    logger.warning(f"MRSC M={M} {name} ({nm},{nn}) ERROR {str(e)[:120]}")
            os.environ.pop("TT_MRS_FORCE_CHUNKS", None)
            results.sort()
            logger.info(f"MRSC M={M} {name} TOP " + " | ".join(f"{t} {c} {ms:.3f}" for ms, t, c in results[:6]))
            ttnn.deallocate(ta), ttnn.deallocate(tw)
