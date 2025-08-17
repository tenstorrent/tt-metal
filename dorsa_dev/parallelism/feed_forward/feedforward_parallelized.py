import torch
import ttnn

"""
FF(V) = (V @ W1) @ W2

Parallelized in 2D (3 mesh axes: x, y, z)

Steps:
   1. Initialize activation matrix V (BLE)
   2. Initialize W1 matrix EF
   3. Shard V's E across x, y and z => BLE_xyz
   4. All-gather y and z => BLE_x
   4. Shard W1's E across x and F across y, z => E_xF_yz
   5. Matmul BLE_x @ E_xF_yz => BLF_yz (partial-sum x)
   5. Reduce-scatter x => BLF_xyz
   6. Apply GELU to BLF_xyz
   7. All-gather x => BLF_yz
   8. Initialize W2 matrix FE
   9. Shard W2's F across y, z and E across x => F_yzE_x
   10. Matmul BLF_yz @ F_yzE_x => BLE_x
   11. Reduce-scatter yz => BLE_xyz

"""

V_B, V_L, V_E, W1_E, W1_F = 12, 10, 10, 10, 10

assert V_E == W1_E, "V's E must equal W1's E"
# assert a_rows % 4 == 0, "A's rows must be divisible by 4"

# seed
torch.manual_seed(0)

V = torch.randn(V_B, V_L, V_E)
W1 = torch.randn(W1_E, W1_F)

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))  # or (2, 2)
ttnn.visualize_mesh_device(mesh_device)

"""
                  MeshDevice(rows=2, cols=2):                       x: rows of the mesh (x=2). y: cols of the mesh (y=2). z: ignored, no third axis (z=1)
┌──────────────────────────────┬──────────────────────────────┐
│          Dev. ID: 4          │          Dev. ID: 0          │
│            (0, 0)            │            (0, 1)            │
│                              │                              │
├──────────────────────────────┼──────────────────────────────┤
│          Dev. ID: 7          │          Dev. ID: 3          │
│            (1, 0)            │            (1, 1)            │
│                              │                              │
└──────────────────────────────┴──────────────────────────────┘
"""

# convert to ttnn tensor + shared
mesh_tensor = ttnn.from_torch(
    V,
    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device),
    mesh_device=mesh_device,
)

W1_mesh_tensor = ttnn.from_torch(
    W1,
    mesh_mapper=ttnn.SharedTensor2dToMesh(mesh_device),
    mesh_device=mesh_device,
)
