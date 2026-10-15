import torch, ttnn
from tests.ttnn.unit_tests.operations.matmul_reduce_scatter.test_matmul_reduce_scatter import (
    _open_mesh,
    _close_mesh,
    _to_mesh,
    _stacked,
)
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

mesh = _open_mesh((1, 4), ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
try:
    print("base", ttnn._ttnn.device.GetAllocatorBaseAddress(mesh, ttnn.BufferType.L1))
    di = ttnn._ttnn.reports.get_device_info(mesh)
    print(
        "l1_bank_size", di.l1_bank_size, "addr_first_bank", di.address_at_first_l1_bank, "worker_l1", di.worker_l1_size
    )

    def l1bufs():
        out = []
        for b in ttnn._ttnn.reports.get_buffers(mesh):
            if b.buffer_type == ttnn.BufferType.L1 and b.device_id == 0:
                out.append((b.address, b.max_size_per_bank, str(b.buffer_layout)))
        return sorted(out)

    print("before", l1bufs())
    a = _to_mesh(_stacked(mesh, (1, 1, 256, 512)), mesh, ttnn.bfloat16)
    w = _to_mesh(_stacked(mesh, (512, 1024)), mesh, ttnn.bfloat16)
    out = matmul_reduce_scatter(a, w, cluster_axis=1, scatter_dim=-1, topology=ttnn.Topology.Ring, num_links=1)
    ttnn.synchronize_device(mesh)
    print("after", l1bufs())
finally:
    _close_mesh(mesh)
