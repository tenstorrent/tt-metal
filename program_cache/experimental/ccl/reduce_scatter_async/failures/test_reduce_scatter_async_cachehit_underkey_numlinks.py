import pytest, torch, ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "device_params, topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 266240}, ttnn.Topology.Ring)],
    indirect=["device_params"],
)
def test_reduce_scatter_async_program_cache_underkeys_num_links(t3k_mesh_device, topology):
    """
    Purpose: Expose under-keying in program hash for experimental CCL reduce_scatter_async.

    Suspected issue: compute_program_hash(...) excludes the preferred number of links. The same hash is generated even
    when num_links changes, reusing a program compiled for a different fabric setting on cache-hit.

    Files/lines:
      - ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_op.cpp
        ReduceScatterAsync::compute_program_hash(...): L226-L241 — hashes reduce op, scatter_dim, ring_size, topology,
        cluster_axis, and input[0] properties. It omits num_links_preferred.
      - ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_async/device/reduce_scatter_async_program.cpp
        override callback built for concrete worker command streams; changing num_links changes the built program.

    Expected failure: On the second run (cache-hit), differing num_links should lead to PCC mismatch or instability.
    We assert PCC and let it FAIL on cache-hit if hash under-keys.
    """
    torch.manual_seed(0)

    # Shape and configs: keep hashed properties constant across runs
    num_devices = 2
    per_chip_output_shape = [1, 1, 64, 256]
    dim = 3
    input_shape = per_chip_output_shape.copy()
    input_shape[dim] *= num_devices

    layout = ttnn.TILE_LAYOUT
    dtype = ttnn.bfloat16
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    # Sub-device and semaphores setup (mirrors unit tests)
    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    ccrs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccrs])
    sub_device_manager = t3k_mesh_device.create_sub_device_manager([worker_sub_device], 0)
    t3k_mesh_device.load_sub_device_manager(sub_device_manager)
    sub_device_id = ttnn.SubDeviceId(0)
    t3k_mesh_device.set_sub_device_stall_group([sub_device_id])

    # Input and golden
    x1 = torch.randn(input_shape).bfloat16()
    x2 = torch.randn(input_shape).bfloat16()
    golden1 = torch.chunk(x1, num_devices, dim)
    golden1 = sum(golden1).bfloat16()  # all-reduce then scatter equivalent across devices
    golden2 = torch.chunk(x2, num_devices, dim)
    golden2 = sum(golden2).bfloat16()

    tt_x1 = ttnn.from_torch(
        x1,
        dtype=dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            t3k_mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(0)], ttnn.MeshShape(1, num_devices)),
        ),
    )
    tt_x2 = ttnn.from_torch(
        x2,
        dtype=dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(
            t3k_mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(0)], ttnn.MeshShape(1, num_devices)),
        ),
    )

    # Semaphores per run
    from_sem_run1 = ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0)
    to_sem_run1 = ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0)
    from_sem_run2 = ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0)
    to_sem_run2 = ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0)

    # 1) First run: compile and seed cache with num_links=1
    logger.debug("Executing first run (num_links=1)")
    out1 = ttnn.experimental.reduce_scatter_async(
        tt_x1,
        dim=dim,
        from_remote_multi_device_global_semaphore=from_sem_run1,
        to_remote_multi_device_global_semaphore=to_sem_run1,
        math_op=ttnn.ReduceType.Sum,
        num_links=1,
        memory_config=mem_config,
        topology=topology,
        subdevice_id=sub_device_id,
    )
    out1_host = ttnn.to_torch(ttnn.from_device(out1))
    ok1, pcc1 = comp_pcc(out1_host, golden1)
    logger.debug(f"First run PCC: ok={ok1}, pcc={pcc1}")
    assert ok1, f"First run PCC failed: {pcc1}"

    # 2) Second run: cache-hit with num_links=2 (hash unchanged if under-keyed)
    logger.debug("Executing second run (num_links=2, cache-hit expected)")
    out2 = ttnn.experimental.reduce_scatter_async(
        tt_x2,
        dim=dim,
        from_remote_multi_device_global_semaphore=from_sem_run2,
        to_remote_multi_device_global_semaphore=to_sem_run2,
        math_op=ttnn.ReduceType.Sum,
        num_links=2,
        memory_config=mem_config,
        topology=topology,
        subdevice_id=sub_device_id,
    )
    out2_host = ttnn.to_torch(ttnn.from_device(out2))
    ok2, pcc2 = comp_pcc(out2_host, golden2)
    logger.debug(f"Second run PCC: ok={ok2}, pcc={pcc2}")
    # Let this assertion FAIL on cache-hit if hash under-keys num_links
    assert ok2, "PCC mismatch on cache-hit path (expected failure if program hash omits num_links)"
