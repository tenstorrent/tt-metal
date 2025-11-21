import time
import ttnn
import torch


def test_open_mesh():
    mesh_shape = ttnn.MeshShape(4, 32)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.RELAXED_INIT)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    print(f"opened mesh device with shape {mesh_device.shape}")
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_open_submeshes():
    mesh_shape = ttnn.MeshShape(4, 32)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.RELAXED_INIT)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    submeshes = [mesh_device.create_submesh(ttnn.MeshShape(2, 32))]
    for submesh in submeshes:
        print(f"submesh with shape {submesh.shape}")
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


# test that submeshes execute in parallel
def test_async_submeshes():
    mesh_shape = ttnn.MeshShape(4, 32)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.RELAXED_INIT)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    submeshes = mesh_device.create_submeshes(ttnn.MeshShape(2, 32))
    for submesh in submeshes:
        print(f"submesh with shape {submesh.shape}")

    a_0 = ttnn.from_torch(torch.randn(8192, 8192), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=submeshes[0])
    b_0 = ttnn.from_torch(torch.randn(8192, 8192), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=submeshes[0])

    a_1 = ttnn.from_torch(torch.randn(8192, 8192), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=submeshes[1])
    b_1 = ttnn.from_torch(torch.randn(8192, 8192), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=submeshes[1])

    # warmup
    ttnn.experimental.minimal_matmul(a_0, b_0)
    ttnn.synchronize_device(submeshes[0])
    ttnn.synchronize_device(submeshes[1])

    start = time.perf_counter()

    for i in range(100):
        o = ttnn.experimental.minimal_matmul(a_0, b_0)
    ttnn.synchronize_device(submeshes[0])
    ttnn.synchronize_device(submeshes[1])
    submesh_0_time = time.perf_counter() - start

    print(f"submesh 0 time: {submesh_0_time}")

    # warmup submesh 1
    ttnn.experimental.minimal_matmul(a_1, b_1)
    ttnn.synchronize_device(submeshes[0])
    ttnn.synchronize_device(submeshes[1])

    start = time.perf_counter()

    for i in range(1000):
        o = ttnn.experimental.minimal_matmul(a_0, b_0)

    for i in range(1000):
        o = ttnn.experimental.minimal_matmul(a_1, b_1)

    ttnn.synchronize_device(submeshes[0])
    ttnn.synchronize_device(submeshes[1])
    duration = time.perf_counter() - start
    print(f"both submeshes time: {duration}")

    for submesh in submeshes:
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def test_matmul_stress():
    mesh_shape = ttnn.MeshShape(4, 32)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.RELAXED_INIT)
    mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape)

    a = ttnn.from_torch(torch.randn(8192, 8192), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device)
    b = ttnn.from_torch(torch.randn(8192, 8192), layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=mesh_device)

    # warmup
    ttnn.experimental.minimal_matmul(a, b)
    ttnn.synchronize_device(mesh_device)

    for i in range(250):
        print(f"iteration {i}")
        start = time.perf_counter()

        for i in range(1000):
            o = ttnn.experimental.minimal_matmul(a, b)
        ttnn.synchronize_device(mesh_device)
        duration = time.perf_counter() - start
        print(f"matmul time: {duration}")

    ttnn.synchronize_device(mesh_device)

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
