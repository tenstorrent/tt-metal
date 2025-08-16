import torch
import ttnn

"""
1) Split the rows of A

Each device has:
    - n rows of A
    - full B

Plan:
    1. Shard the rows of A across n devices (so each device has A/n rows)
    2. Replicate full B across all n devices
    3. Do matmul A_[i/n, j] @ B on each device
    4. All gather (concatenate rows)
    5. how to get it all on one device?
"""

a_rows, a_cols, b_rows, b_cols = 12, 10, 10, 10

assert a_cols == b_rows, "A's columns must equal B's rows"
assert a_rows % 4 == 0, "A's rows must be divisible by 4"

# seed
torch.manual_seed(0)

A = torch.randn(a_rows, a_cols)
B = torch.randn(b_rows, b_cols)

C = torch.zeros(a_rows, b_cols)


# T3K mesh device management

mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))  # or (2, 2)
ttnn.visualize_mesh_device(mesh_device)
"""                                                 MeshDevice(rows=1, cols=4):
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│          Dev. ID: 4          │          Dev. ID: 0          │          Dev. ID: 1          │          Dev. ID: 5          │
│            (0, 0)            │            (0, 1)            │            (0, 2)            │            (0, 3)            │
│                              │                              │                              │                              │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
"""

# === Shard A ===

# convert torch tensor to ttnn tensor + shard A (rows, dim=0) across all devices; MeshTensor holds buffers to two shards in host-memory
mesh_tensor_A = ttnn.from_torch(
    A,
    mesh_mapper=ttnn.ShardTensorToMesh(
        mesh_device, dim=0
    ),  # shard rows, dim=0    -    if MeshShape was 2d (2, 2), need to use ShardTensor2dMesh
    layout=ttnn.TILE_LAYOUT,
)
# these shards still reside in host-memory
print("shards in host-memory:", mesh_tensor_A)
print("mesh_tensor_A.shape:", mesh_tensor_A.shape)

# now transfer data to device (see how printed output says ex. "device_id: 0, MeshCoordinate([0, 1])")
mesh_tensor_A = ttnn.to_device(mesh_tensor_A, mesh_device)
print("shards on devices:", mesh_tensor_A)

ttnn.visualize_mesh_device(mesh_device, tensor=mesh_tensor_A)

"""
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│          Dev. ID: 4          │          Dev. ID: 0          │          Dev. ID: 1          │          Dev. ID: 5          │
│            (0, 0)            │            (0, 1)            │            (0, 2)            │            (0, 3)            │
│    ttnn.Shape([3, 10])       │    ttnn.Shape([3, 10])       │    ttnn.Shape([3, 10])       │    ttnn.Shape([3, 10])       │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
"""

# === Replicate B ===


replicate_B = ttnn.ReplicateTensorToMesh(mesh_device)  # returns a CppTensorToMesh object

# convert torch tensor to ttnn tensor + replicate B (rows, dim=0) across all devices; MeshTensor holds buffers to two shards in host-memory
mesh_tensor_B = ttnn.from_torch(
    B,
    mesh_mapper=replicate_B,
    layout=ttnn.TILE_LAYOUT,
)
print("replicated B in host-memory:", mesh_tensor_B)

# now transfer data to device (see how printed output says ex. "device_id: 0, MeshCoordinate([0, 1])")
mesh_tensor_B = ttnn.to_device(mesh_tensor_B, mesh_device)
print("full B on devices:", mesh_tensor_B)

ttnn.visualize_mesh_device(mesh_device, tensor=mesh_tensor_B)

"""
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│          Dev. ID: 4          │          Dev. ID: 0          │          Dev. ID: 1          │          Dev. ID: 5          │
│            (0, 0)            │            (0, 1)            │            (0, 2)            │            (0, 3)            │
│      Shape([10, 10])         │      Shape([10, 10])         │      Shape([10, 10])         │        Shape([10, 10])       │
└──────────────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────────────┘
"""

# === Matmul ===

# Perform distributed matmul. Because `mesh_tensor_A` is row-sharded and
# `mesh_tensor_B` is fully replicated, `ttnn.matmul` produces a row-sharded
# result where each device computes its slice independently.
mesh_tensor_C = ttnn.matmul(mesh_tensor_A, mesh_tensor_B)

print("C on devices:", mesh_tensor_C)
print("C.shape:", mesh_tensor_C.shape)  # each C shard has shape (3, 10)

# === Gather results back to the host ===
# Concatenate the row shards along dim=0 so that we get the full matrix.

# Gather the row-sharded result across devices. The output tensor is now
# replicated, so each device holds the full C matrix with rows in order.

if len(mesh_tensor_C.shape) < 4:
    # pad dims to make it 4D
    mesh_tensor_C = ttnn.unsqueeze_to_4D(mesh_tensor_C)  # (3, 10) -> (1, 1, 3, 10)

C_gathered = ttnn.all_gather(
    mesh_tensor_C,
    dim=2,  # gather along row dimension (dim=2 because of 4D tensor)
    num_links=1,  # single-link gather (default in most demos)
    topology=ttnn.Topology.Linear,
)

# move the gathered tensor to host for validation
C_from_devices = ttnn.to_torch(C_gathered, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

# C_from_devices.shape: (4, 1, 12, 10)
# remove the first two dims to get (12, 10)
C_output = C_from_devices[0, 0]


# === Verification ===
C_torch = torch.matmul(A, B)
print("C_torch:\n", C_torch)
print("C_torch.shape:", C_torch.shape)

print("C_output:\n", C_output)
print("C_output.shape:", C_output.shape)

assert C_torch.shape == C_output.shape, "Shapes should match"
assert torch.allclose(C_torch, C_output, atol=1e-1), "Results differ!"

print("[SUCCESS] Distributed matmul matches torch\n")

ttnn.close_mesh_device(mesh_device)
