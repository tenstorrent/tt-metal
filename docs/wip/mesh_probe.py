import torch, ttnn
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

md = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
try:
    print(f"PROBE mesh_device            id()={md.id()} py_id={id(md)}")
    # (a) how the persistent cache is made: allocate_tensor_on_device
    cache = ttnn.allocate_tensor_on_device(
        ttnn.Shape([2, 1, 64, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
    )
    # (b) how the scratch is made: from_torch(device=..., mesh_mapper=Replicate)
    ccl = get_tt_ccl(md)
    print(f"PROBE ccl.mesh_device        id()={ccl.mesh_device.id()} py_id={id(ccl.mesh_device)} same_obj={ccl.mesh_device is md}")
    buf = ttnn.from_torch(
        torch.zeros(1, 1, 64, 64), device=md, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    cd, bd = cache.device(), buf.device()
    print(f"PROBE cache.device()  id()={cd.id()} py_id={id(cd)}")
    print(f"PROBE buf.device()    id()={bd.id()} py_id={id(bd)}")
    print(f"PROBE EQUAL(cache.device()==buf.device()) -> {cd == bd}")
    print(f"PROBE cache.device()==md -> {cd == md}   buf.device()==md -> {bd == md}")
finally:
    ttnn.close_mesh_device(md)
