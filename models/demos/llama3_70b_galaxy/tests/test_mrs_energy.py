# matmul_reduce_scatter_prefill chunk splits ranked by ENERGY at the power cap (the DP4 prefill is energy-bound): each
# split runs ~ME_SECS seconds back to back on all 32 chips under the in-model L1 budget (l1_headroom, as in
# test_mrs_chunks.py) while a thread polls tt-smi. Reports sustained ms, chip power and dynamic mJ = (power - idle) x ms.
#   TT_MRS_BF8_TRANSPORT=1 MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_mrs_energy.py -s
# env: ME_CASES ("ff2b4:16384:4x6,2x6,...;ff2:4096:auto,1x2;wo:4096:auto,..."), ME_SECS (6), ME_HR (262144)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.llama3_70b_galaxy.tests.test_op_energy import Poller

H = 8192
DRAM = ttnn.DRAM_MEMORY_CONFIG
DEFAULT_CASES = (
    "ff2b4:16384:4x6,2x6,4x4,8x3,4x3,8x2,2x4,3x6,2x8,6x4;"
    "ff2:4096:auto,1x2,2x2,2x3,1x4,4x2,1x3;"
    "wo:4096:auto,1x2,2x2,1x4,2x4,4x2,1x3"
)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_mrs_energy(mesh_device, device_params):
    mesh = mesh_device
    rep = ttnn.ReplicateTensorToMesh(mesh)
    secs = float(os.environ.get("ME_SECS") or 6)
    hr0 = int(os.environ.get("ME_HR") or "262144")
    ck = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi)
    poll = Poller()
    time.sleep(3)
    t0 = time.time()
    time.sleep(8)
    idle, _, _, _ = poll.window(t0 + 1, time.time())
    logger.info(f"MRSE idle {idle:.1f} W")
    keep = os.environ.get("TT_MRS_FORCE_CHUNKS", "")
    step = [0]
    rows = []
    for case in (os.environ.get("ME_CASES") or DEFAULT_CASES).split(";"):
        name, M, splits = case.split(":")
        M = int(M)
        K = 1024 if name.startswith("wo") else 3584
        wdt = ttnn.bfloat4_b if name.endswith("b4") else ttnn.bfloat8_b
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
            dtype=wdt,
            memory_config=DRAM,
            mesh_mapper=rep,
        )
        Mt, Kt, Nt = M // 32, K // 32, H // 32
        for sp in splits.split(","):
            if sp == "auto":
                os.environ["TT_MRS_FORCE_CHUNKS"] = ""
            else:
                nm, nn = sp.split("x")
                os.environ["TT_MRS_FORCE_CHUNKS"] = f"{Mt},{Kt},{Nt},{nm},{nn}"
            step[0] += 1
            hr = hr0 + 16 * step[0]  # distinct key: new plan + program per split

            def fn():
                return ttnn.experimental.matmul_reduce_scatter_prefill(
                    ta,
                    tw,
                    cluster_axis=0,
                    scatter_dim=-2,
                    topology=ttnn.Topology.Ring,
                    num_links=2,
                    compute_kernel_config=ck,
                    memory_config=DRAM,
                    l1_headroom=hr,
                )

            try:
                ttnn.deallocate(fn())
                ttnn.synchronize_device(mesh)
                plan = dict(
                    ttnn._ttnn.operations.experimental.ccl_experimental._matmul_reduce_scatter_prefill_last_plan()
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"MRSE {name} M={M} {sp} ERROR {str(e)[:160]}")
                continue
            time.sleep(4)
            t1 = time.perf_counter()
            for _ in range(10):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh)
            ms_cool = (time.perf_counter() - t1) / 10 * 1e3
            n = max(20, int(secs * 1e3 / ms_cool))
            w0 = time.time()
            t1 = time.perf_counter()
            for _ in range(n):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t1) / n * 1e3
            p, amin, _, ns = poll.window(w0 + 2.0, time.time() - 0.2)
            mj = (p - idle) * ms
            rows.append((name, M, sp, ms_cool, ms, p, mj))
            chunk = {k: plan[k] for k in plan if "chunk" in k or k in ("kb", "core_m", "core_n")}
            logger.info(
                f"MRSE {name} M={M} {sp}: {ms_cool:.3f} ms cool, {ms:.3f} ms sustained, {p:.1f} W, aiclk min "
                f"{amin:.0f}, dyn {mj:.1f} mJ ({ns} samples) plan {chunk}"
            )
        ttnn.deallocate(ta), ttnn.deallocate(tw)
    os.environ["TT_MRS_FORCE_CHUNKS"] = keep
    poll.stop = True
    for r in rows:
        logger.info("MRSE SUMMARY %s M=%d %s cool %.3f sust %.3f W %.1f dyn_mJ %.1f" % r)
