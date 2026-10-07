import torch, ttnn, sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/matmul_reduce_scatter")
from tests.scripts.common import get_updated_device_params
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter as M
import ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter as mod
import test_matmul_reduce_scatter as T

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D, ttnn.FabricReliabilityMode.STRICT_INIT)
params = get_updated_device_params({"fabric_config": ttnn.FabricConfig.FABRIC_2D})
params.pop("fabric_config")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4), **params)
try:
    print(
        "PROBE l1 bank",
        mod._l1_cb_capacity(mesh),
        "unreserved",
        ttnn.get_max_worker_l1_unreserved_size(),
        "alloc base",
        ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1),
    )
    cases = [
        ((1, 1, 640, 2048), (2048, 6144), 1, -2, ttnn.bfloat8_b, True),
        ((1, 1, 2048, 2048), (2048, 4096), 0, -2, ttnn.bfloat8_b, True),
        ((1, 1, 640, 1536), (1536, 7168), 1, -2, ttnn.bfloat8_b, True),
        ((1, 1, 640, 2048), (2048, 7168), 1, -1, ttnn.bfloat8_b, False),
        ((1, 1, 640, 8448), (8448, 7168), 1, -2, ttnn.bfloat16, True),
        ((1, 1, 640, 4608), (4608, 7168), 1, -2, ttnn.bfloat8_b, True),
    ]
    for a_shape, w_shape, ax, sd, wdt, fp32 in cases:
        try:
            T._run_and_check(
                mesh,
                a_shape,
                w_shape,
                cluster_axis=ax,
                scatter_dim=sd,
                weight_dtype=wdt,
                num_links=2,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32
                ),
            )
            res = "ok"
        except Exception as e:
            res = f"FAIL {type(e).__name__}: {str(e)[:150]}"
        plan = [
            v[1] for k, v in mod._PLAN_CACHE.items() if k[3:6] == (a_shape[-2], a_shape[-1], w_shape[-1]) and k[6] == sd
        ][-1]
        b = plan["blk"]
        print(
            "PROBE",
            a_shape,
            w_shape,
            ax,
            sd,
            wdt,
            fp32,
            b.regime,
            f"core {b.core_m_tiles}x{b.core_n_tiles} lines {b.m_lines}x{b.n_lines} kbt {b.k_block_tiles} sb {b.out_subblock_h}x{b.out_subblock_w}",
            res,
        )
finally:
    ttnn.close_mesh_device(mesh)
