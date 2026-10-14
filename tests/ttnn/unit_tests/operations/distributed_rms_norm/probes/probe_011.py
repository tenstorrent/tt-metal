import ttnn
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
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
            (1, 4096),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    t = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 4096]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, mc)
    print("GV addr", t.buffer_address())
    mv = ttnn._ttnn.device.GetMemoryView(mesh, ttnn.BufferType.L1)
    print(
        "MV",
        mv.num_banks,
        mv.total_bytes_per_bank,
        mv.total_bytes_allocated_per_bank,
        mv.largest_contiguous_bytes_free_per_bank,
    )
    for b in mv.block_table[:6]:
        print("MV block", b)
finally:
    ttnn.close_mesh_device(mesh)
