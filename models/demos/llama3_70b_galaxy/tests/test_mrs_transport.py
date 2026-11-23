# matmul_reduce_scatter_prefill on the column axis (cluster_axis 0, 8 chips, scatter rows) at the DP4 x TP8 WO / FF2
# shapes: time and accuracy vs torch. Run twice: TT_MRS_BF8_TRANSPORT unset (bf16 staging) and =1 (bfp8 staging).
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_mrs_transport.py -s      env: MRS_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, TP = 8192, 8
MS = [int(m) for m in (os.environ.get("MRS_M") or "4096,16384").split(",")]
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_mrs_transport(mesh_device, device_params):
    mesh = mesh_device
    tag = "bf8" if os.environ.get("TT_MRS_BF8_TRANSPORT") == "1" else "bf16"
    rows = ttnn.ShardTensor2dMesh(mesh, dims=(0, None), mesh_shape=(8, 4))
    for M in MS:
        for name, K in (("wo", 1024), ("ff2", 3584)):
            torch.manual_seed(0)
            a = torch.randn(TP, 1, M, K)
            w = torch.randn(TP, K, H) * 0.02
            ta = ttnn.from_torch(a, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rows)
            tw = ttnn.from_torch(
                w.reshape(TP * K, H), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rows
            )
            for fid in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi):
                f = lambda: ttnn.experimental.matmul_reduce_scatter_prefill(
                    ta,
                    tw,
                    cluster_axis=0,
                    scatter_dim=-2,
                    topology=ttnn.Topology.Ring,
                    num_links=2,
                    compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=fid),
                    memory_config=DRAM,
                )
                o = f()
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                for _ in range(10):
                    ttnn.deallocate(f())
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - t0) / 10 * 1e3
                # golden: sum over the 8 chips of a_r @ w_r, rows block r on chip r (column 0)
                gold_full = sum(a[r][0] @ w[r] for r in range(TP))
                devs = ttnn.get_device_tensors(o)
                pccs = []
                for r in (0, 3, 7):
                    got = ttnn.to_torch(devs[r * 4]).float().reshape(M // TP, H)
                    gold = gold_full[r * M // TP : (r + 1) * M // TP]
                    pccs.append(torch.corrcoef(torch.stack([gold.flatten(), got.flatten()]))[0, 1].item())
                logger.info(
                    f"MRSX {tag} M={M} {name} {fid}: {ms:.3f} ms out {o.dtype} pcc r0/r3/r7 "
                    + " ".join(f"{p:.5f}" for p in pccs)
                )
                ttnn.deallocate(o)
            ttnn.deallocate(ta), ttnn.deallocate(tw)
