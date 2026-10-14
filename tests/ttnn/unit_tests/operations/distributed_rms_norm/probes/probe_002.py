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
    print("L1 size", mesh.l1_size_per_core() if hasattr(mesh, "l1_size_per_core") else None)
    for a in dir(mesh):
        if "l1" in a.lower() or "alloc" in a.lower():
            print("ATTR", a)
    from eval.sharding import auto_shard_config

    for kb in (64, 512, 1024, 1200, 1300, 1400):
        try:
            words = kb * 256
            mc = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
                    (1, words),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mc)
            print("ALLOC", kb, "KB at", t.buffer_address())
            t.deallocate()
        except Exception as e:
            print("FAIL", kb, str(e).splitlines()[0][:200])
finally:
    ttnn.close_mesh_device(mesh)
