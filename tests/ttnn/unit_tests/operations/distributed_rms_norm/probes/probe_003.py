import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import production_router_config

ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    production_router_config(),
)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2))
try:
    g = mesh.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    lo, hi = 1400 * 1024, 1536 * 1024
    while hi - lo > 1024:
        mid = (lo + hi) // 2 // 1024 * 1024
        try:
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
                    (1, mid // 4),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, mid // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mc)
            t.deallocate()
            lo = mid
        except Exception:
            hi = mid
    print("BASE approx max alloc", lo, "base", 1572864 - lo)
finally:
    ttnn.close_mesh_device(mesh)
