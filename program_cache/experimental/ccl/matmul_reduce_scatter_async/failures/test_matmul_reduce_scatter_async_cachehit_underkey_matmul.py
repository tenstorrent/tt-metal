import pytest, torch, ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "device_params, rs_topology",
    [({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 266240}, ttnn.Topology.Ring)],
    indirect=["device_params"],
)
def test_matmul_reduce_scatter_async_program_cache_underkeys_matmul_program_config(t3k_mesh_device, rs_topology):
    """
    Purpose: Expose under-keying in program hash for experimental CCL matmul_reduce_scatter_async.

    Suspected issue: compute_program_hash excludes matmul program configuration and weight tensor properties, so the
    same hash is reused even when matmul program selection changes between runs. This reuses a program compiled with a
    different matmul kernel configuration on cache-hit, leading to PCC mismatch or a hang.

    Files/lines:
      - ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_op.cpp
        compute_program_hash(...): L164-L191 — only RS fields and input[0] properties are hashed; matmul_struct and
        weight tensor are omitted.
      - ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_async/device/multi_core/matmul_reduce_scatter_async_op_multi_core.cpp
        fused override wiring: L121-L151 — overrides for matmul and reduce-scatter are applied on cache-hit.

    Expected failure: On the second run (cache-hit), differing matmul program_config should cause either PCC mismatch
    or runtime instability. The assertion below expects PCC to be OK; if under-keyed, it will FAIL on the second run.
    """
    torch.manual_seed(0)

    # Shapes and configs (keep input/RS the same across runs to force same hash)
    num_devices = 2
    rs_input_shape = [2, 1, 512, 2048]  # [B, H, M, N]
    mm_weights_shape = [1, 1, 2048, 1024]  # [B, H, K, O]
    rs_scatter_dim = 3  # scatter along width

    layout = ttnn.TILE_LAYOUT
    rs_input_dtype = ttnn.bfloat16
    matmul_weights_dtype = ttnn.bfloat16

    mem_config_input = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_mm = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    mem_config_rs = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    compute_grid = (8, 6)

    # Two different matmul program configs to force different compiled programs but same hash
    prog_cfg_run1 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=compute_grid,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=rs_input_shape[2] // 32 // compute_grid[1] or 1,
        per_core_N=mm_weights_shape[3] // 32 // compute_grid[0] or 1,
        out_block_w=1,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    prog_cfg_run2 = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=compute_grid,
        in0_block_w=2,  # different from run1
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=rs_input_shape[2] // 32 // compute_grid[1] or 1,
        per_core_N=mm_weights_shape[3] // 32 // compute_grid[0] or 1,
        out_block_w=2,  # different from run1
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Setup semaphores and persistent buffers
    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    ccrs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccrs])
    sub_device_manager = t3k_mesh_device.create_sub_device_manager([worker_sub_device], 0)
    t3k_mesh_device.load_sub_device_manager(sub_device_manager)
    sub_device_id = ttnn.SubDeviceId(0)
    t3k_mesh_device.set_sub_device_stall_group([sub_device_id])

    semaphores_run1 = (
        ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0),
        ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0),
        ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0),
    )
    semaphores_run2 = (
        ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0),
        ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0),
        ttnn.create_global_semaphore(t3k_mesh_device, ccrs, 0),
    )

    rs_num_batches = rs_input_shape[0]
    single_batch_input_shape = rs_input_shape[:]
    single_batch_input_shape[2] //= rs_num_batches
    rs_output_shape = rs_input_shape[:]
    rs_output_shape[3] //= num_devices

    persistent_intermediate_buf_run1 = ttnn.from_torch(
        torch.zeros(single_batch_input_shape),
        device=t3k_mesh_device,
        layout=layout,
        dtype=rs_input_dtype,
        memory_config=mem_config_rs,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )
    persistent_output_buf_run1 = ttnn.from_torch(
        torch.zeros(rs_output_shape),
        device=t3k_mesh_device,
        layout=layout,
        dtype=rs_input_dtype,
        memory_config=mem_config_rs,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )
    persistent_intermediate_buf_run2 = ttnn.from_torch(
        torch.zeros(single_batch_input_shape),
        device=t3k_mesh_device,
        layout=layout,
        dtype=rs_input_dtype,
        memory_config=mem_config_rs,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )
    persistent_output_buf_run2 = ttnn.from_torch(
        torch.zeros(rs_output_shape),
        device=t3k_mesh_device,
        layout=layout,
        dtype=rs_input_dtype,
        memory_config=mem_config_rs,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )

    # Shared inputs (identical shapes to keep hash same)
    weights = torch.randn(mm_weights_shape).bfloat16()
    weight_tt = ttnn.from_torch(
        weights,
        dtype=matmul_weights_dtype,
        layout=layout,
        device=t3k_mesh_device,
        memory_config=mem_config_mm,
        mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
    )

    mm_input_shape = [rs_input_shape[0], 1, rs_input_shape[2], mm_weights_shape[2]]
    mm_input_1 = torch.rand(mm_input_shape).bfloat16()
    mm_input_2 = torch.rand(mm_input_shape).bfloat16()
    tt_in1 = ttnn.from_torch(
        mm_input_1,
        device=t3k_mesh_device,
        layout=layout,
        dtype=rs_input_dtype,
        memory_config=mem_config_input,
        mesh_mapper=ttnn.create_mesh_mapper(
            t3k_mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(1, num_devices)),
        ),
    )
    tt_in2 = ttnn.from_torch(
        mm_input_2,
        device=t3k_mesh_device,
        layout=layout,
        dtype=rs_input_dtype,
        memory_config=mem_config_input,
        mesh_mapper=ttnn.create_mesh_mapper(
            t3k_mesh_device,
            ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], ttnn.MeshShape(1, num_devices)),
        ),
    )

    # 1) First run (compile + seed cache)
    logger.debug("Executing first run")
    out_mm_1, out_rs_1 = ttnn.experimental.matmul_reduce_scatter_async(
        tt_in1,
        weight_tt,
        persistent_intermediate_buffer=persistent_intermediate_buf_run1,
        persistent_output_buffer=persistent_output_buf_run1,
        dim=rs_scatter_dim,
        multi_device_global_semaphore=list(semaphores_run1),
        reduce_scatter_core_grid_offset=(0, 6),
        num_links=1,
        memory_config_rs=mem_config_rs,
        topology=rs_topology,
        subdevice_id=sub_device_id,
        memory_config_mm=mem_config_mm,
        program_config=prog_cfg_run1,
        compute_kernel_config=compute_kernel_config,
    )
    out1_host = ttnn.to_torch(ttnn.from_device(out_rs_1))
    golden1 = torch.chunk(torch.matmul(mm_input_1, weights), num_devices, rs_scatter_dim)
    golden1 = torch.cat(golden1, dim=rs_scatter_dim)
    ok1, pcc1 = comp_pcc(out1_host, golden1)
    logger.debug(f"First run PCC: ok={ok1}, pcc={pcc1}")
    assert ok1, f"First run PCC failed: {pcc1}"

    # 2) Second run (cache-hit) — change only matmul program_config
    logger.debug("Executing second run")
    out_mm_2, out_rs_2 = ttnn.experimental.matmul_reduce_scatter_async(
        tt_in2,
        weight_tt,
        persistent_intermediate_buffer=persistent_intermediate_buf_run2,
        persistent_output_buffer=persistent_output_buf_run2,
        dim=rs_scatter_dim,
        multi_device_global_semaphore=list(semaphores_run2),
        reduce_scatter_core_grid_offset=(0, 6),
        num_links=1,
        memory_config_rs=mem_config_rs,
        topology=rs_topology,
        subdevice_id=sub_device_id,
        memory_config_mm=mem_config_mm,
        program_config=prog_cfg_run2,  # different program config, but hash unchanged
        compute_kernel_config=compute_kernel_config,
    )
    out2_host = ttnn.to_torch(ttnn.from_device(out_rs_2))
    golden2 = torch.chunk(torch.matmul(mm_input_2, weights), num_devices, rs_scatter_dim)
    golden2 = torch.cat(golden2, dim=rs_scatter_dim)
    ok2, pcc2 = comp_pcc(out2_host, golden2)
    logger.debug(f"Second run PCC: ok={ok2}, pcc={pcc2}")
    # Let this assertion FAIL on cache-hit if under-keyed
    assert ok2, "PCC mismatch on cache-hit path (expected failure if program hash under-keys matmul config)"
