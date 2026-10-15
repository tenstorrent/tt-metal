import torch, ttnn
from tests.ttnn.unit_tests.operations.matmul_reduce_scatter.test_matmul_reduce_scatter import (
    _open_mesh,
    _close_mesh,
    _to_mesh,
    _stacked,
)
import ttnn.operations.matmul_reduce_scatter as pkg
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

mesh = _open_mesh((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
try:
    print("base", ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1))

    def l1bufs():
        return sorted(
            (b.address, b.max_size_per_bank)
            for b in ttnn._ttnn.reports.get_buffers(mesh)
            if b.buffer_type == ttnn.BufferType.L1 and b.device_id == 0
        )

    for shp in [((1, 1, 256, 512), (512, 1024)), ((1, 1, 640, 2048), (2048, 7168))]:
        a = _to_mesh(_stacked(mesh, shp[0]), mesh, ttnn.bfloat16)
        w = _to_mesh(_stacked(mesh, shp[1]), mesh, ttnn.bfloat8_b)
        out = matmul_reduce_scatter(a, w, cluster_axis=1, scatter_dim=-2, topology=ttnn.Topology.Ring, num_links=1)
        ttnn.synchronize_device(mesh)
        print("plan", pkg._LAST_PLAN)
        print("after", l1bufs())
finally:
    _close_mesh(mesh)
