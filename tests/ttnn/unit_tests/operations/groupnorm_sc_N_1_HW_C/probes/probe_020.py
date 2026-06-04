import torch, ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
try:
    print("MESH num_devices:", mesh.get_num_devices())
    t = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )
    ttnn.graph.up_front_begin_collect()
    try:
        ttnn.to_torch(ttnn.exp(x), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    except Exception:
        pass
    finally:
        ttnn.graph.up_front_end_collect()
    n_prog, n_err, _, _ = ttnn.graph.up_front_compile(mesh, 4, True)
    print(f"COMPILED {n_prog} programs errors={n_err}")
finally:
    ttnn.close_mesh_device(mesh)
