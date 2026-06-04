import torch, ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))  # 2-chip mesh (chip0 harvest 0x201, chip1 0x204)
try:
    print("MESH num_devices:", mesh.get_num_devices())
    torch.manual_seed(0)
    t = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )

    # COLLECT across the mesh under NO_DISPATCH
    ttnn.graph.up_front_begin_collect()
    try:
        y = ttnn.exp(x)
        _ = ttnn.to_torch(y, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))  # readback under NO_DISPATCH
    except Exception as e:
        print("body raised under NO_DISPATCH (swallowed):", repr(e)[:120])
    finally:
        ttnn.graph.up_front_end_collect()

    print("COLLECTED:", ttnn.graph.up_front_num_collected(), "UNIQUE:", ttnn.graph.up_front_num_unique())
    n_prog, n_err, used, wall = ttnn.graph.up_front_compile(mesh, 4, True)  # compiles on devices.front() = chip0
    print(f"COMPILED {n_prog} programs on the mesh, errors={n_err}, wall={wall:.2f}s")

    # WARM run on the mesh — if one compile warmed BOTH chips (shared build_key), this hits cache.
    y2 = ttnn.exp(x)
    out = ttnn.to_torch(y2, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
    expected = torch.exp(t.float())
    print("warm out shape", tuple(out.shape))
    for i in range(out.shape[0]):
        sl = out[i : i + 1].flatten()
        pcc = torch.corrcoef(torch.stack([sl, expected.flatten()]))[0, 1].item()
        print(f"  chip{i} warm PCC = {pcc:.5f}")
finally:
    ttnn.close_mesh_device(mesh)
