import itertools, ttnn

for cfg, shape in ((ttnn.FabricConfig.FABRIC_2D, (2, 2)), (ttnn.FabricConfig.FABRIC_2D_TORUS_XY, (4, 1))):
    try:
        ttnn.set_fabric_config(cfg)
        mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape))
        print(f"CFG {cfg} mesh={shape}")
        coords = [(r, c) for r in range(shape[0]) for c in range(shape[1])]
        for a, b in itertools.permutations(coords, 2):
            na = mesh.get_fabric_node_id(ttnn.MeshCoordinate(*a))
            nb = mesh.get_fabric_node_id(ttnn.MeshCoordinate(*b))
            try:
                n = len(ttnn.get_forwarding_link_indices(na, nb))
            except Exception as e:
                n = f"err {type(e).__name__}"
            print(f"LINKS {a}->{b} chips {na.chip_id}->{nb.chip_id}: {n}")
        ttnn.close_mesh_device(mesh)
    except Exception as e:
        print(f"ERR {cfg}: {e}")
    finally:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
