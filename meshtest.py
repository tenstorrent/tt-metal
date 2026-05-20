import ttnn

mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4))
try:
    print("=== ttnn.visualize_mesh_device ===")
    ttnn.visualize_mesh_device(mesh)
    print()
    print("=== (row, col) -> device_id ===")
    for r in range(8):
        print(" ", "  ".join(f"(r={r},c={c})→{mesh.get_device_id(ttnn.MeshCoordinate(r,c)):>2}" for c in range(4)))
    print()
    print("=== dispatch group → 8 physical device IDs ===")
    for c in range(4):
        devids = [mesh.get_device_id(ttnn.MeshCoordinate(r, c)) for r in range(8)]
        print(f"  col {c}: {devids}")
finally:
    ttnn.close_mesh_device(mesh)
