# rms_repro.py
#
# Pytest reproduction of the compiler-generated distributed RMS norm test.
# Source: test_distributed_rms_norm[emitpy-1-1x2-True-True-1x1x32x8192]
#
# Runs fused_rms_minimal on a 1x2 mesh device with:
#   - input:    [1, 1, 32, 8192] bf16, sharded on dim 3 across 2 devices
#   - weight:   [8192] bf16, sharded on dim 0 across 2 devices
#   - residual: [1, 1, 32, 8192] bf16, sharded on dim 3 across 2 devices
#
# Run: pytest rms_repro.py

import pytest
import ttnn


MESH_SHAPE = ttnn.MeshShape([1, 2])
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
CORE_GRID_8x8 = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
L1_WIDTH_SHARDED_8x8 = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(CORE_GRID_8x8, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
)


def create_global_semaphore(input_tensor):
    """Create a global semaphore from the input tensor's device and shard grid."""
    mesh_device = input_tensor.device()
    shard_spec = input_tensor.memory_config().shard_spec
    return ttnn.create_global_semaphore(mesh_device, shard_spec.grid, 0)


def create_inputs():
    """Create host-side input tensors (all ones for reproducibility)."""
    inp = ttnn.ones(
        shape=ttnn.Shape([1, 1, 32, 8192]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
    )
    weight = ttnn.ones(
        shape=ttnn.Shape([8192]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
    )
    residual = ttnn.ones(
        shape=ttnn.Shape([1, 1, 32, 8192]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
    )
    return inp, weight, residual


def distribute_and_prepare(inp, weight, residual, mesh_device):
    """Distribute tensors across the mesh and move to device in TILE layout."""

    # -- Distribute across the 1x2 mesh --
    # input & residual: replicate dim 0, shard dim 3
    shard_dim3_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(3)], MESH_SHAPE),
    )
    # weight: replicate dim 0, shard dim 0 (the only dim)
    shard_dim0_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(0)], MESH_SHAPE),
    )

    dist_inp = ttnn.distribute_tensor(inp, mapper=shard_dim3_mapper)
    ttnn.deallocate(inp, False)

    dist_weight = ttnn.distribute_tensor(weight, mapper=shard_dim0_mapper)
    ttnn.deallocate(weight, False)

    dist_residual = ttnn.distribute_tensor(residual, mapper=shard_dim3_mapper)
    ttnn.deallocate(residual, False)

    # -- Move to device (DRAM) and convert to TILE layout --
    def to_device_tile(t):
        on_dev = ttnn.to_device(t, device=mesh_device, memory_config=DRAM_INTERLEAVED)
        ttnn.deallocate(t, False)
        tiled = ttnn.to_layout(on_dev, ttnn.Layout.TILE, None, memory_config=None)
        ttnn.deallocate(on_dev, False)
        return tiled

    inp_tiled = to_device_tile(dist_inp)
    weight_tiled = to_device_tile(dist_weight)
    residual_tiled = to_device_tile(dist_residual)

    # -- Shard input & residual to L1 (WIDTH_SHARDED, 8x8 grid, [32, 64] per core) --
    inp_sharded = ttnn.to_memory_config(inp_tiled, L1_WIDTH_SHARDED_8x8)
    ttnn.deallocate(inp_tiled, False)

    residual_sharded = ttnn.to_memory_config(residual_tiled, L1_WIDTH_SHARDED_8x8)
    ttnn.deallocate(residual_tiled, False)

    # -- Reshape weight to [128, 32] and convert to ROW_MAJOR for the kernel --
    weight_reshaped = ttnn.reshape(weight_tiled, [128, 32], memory_config=DRAM_INTERLEAVED)
    ttnn.deallocate(weight_tiled, False)
    weight_rm = ttnn.to_layout(weight_reshaped, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(weight_reshaped, False)

    return inp_sharded, weight_rm, residual_sharded


def run_fused_rms_norm(inp_sharded, weight_rm, residual_sharded, mesh_device):
    """Execute fused_rms_minimal and return the on-device result."""

    # Allocate stats accumulator: [1, 1, 32, 32] f32, sharded on a single core
    single_core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    stats = ttnn.empty(
        ttnn.Shape([1, 1, 32, 32]),
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        mesh_device,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(single_core, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )

    global_semaphore = create_global_semaphore(inp_sharded)

    # CCL sub-device: required so the all-gather's dispatch knows which cores
    # are workers. Without this, fused_rms_minimal hangs in the CCL phase.
    # Must be a superset of the layernorm compute grid (8x8 here).
    ccl_sub_device_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_manager = mesh_device.create_sub_device_manager([worker_sub_device], 0)
    mesh_device.load_sub_device_manager(sub_device_manager)
    mesh_device.set_sub_device_stall_group([worker_sub_device_id])

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(8, 8),
        subblock_w=1,
        block_h=1,
        block_w=2,
        inplace=False,
    )
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    result = ttnn.fused_rms_minimal(
        inp_sharded,
        program_config,
        1,  # num_links
        mesh_device,
        global_semaphore,
        stats=stats,
        subdevice_id=None,
        compute_kernel_config=compute_config,
        memory_config=L1_WIDTH_SHARDED_8x8,
        residual_input_tensor=residual_sharded,
        epsilon=1e-5,
        weight=weight_rm,
    )

    # Cleanup intermediate tensors
    ttnn.deallocate(stats, False)
    ttnn.deallocate(residual_sharded, False)
    ttnn.deallocate(weight_rm, False)
    ttnn.deallocate(inp_sharded, False)

    return result


def collect_result(result, mesh_device):
    """Move result off device and aggregate shards back into a single tensor."""
    # DRAM interleaved -> ROW_MAJOR -> host
    result_dram = ttnn.to_memory_config(result, DRAM_INTERLEAVED)
    ttnn.deallocate(result, False)

    result_rm = ttnn.to_layout(result_dram, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(result_dram, False)

    result_host = ttnn.from_device(result_rm)
    ttnn.deallocate(result_rm, False)

    # Aggregate shards: concat dims [2, 3] from the 1x2 mesh
    composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig([2, 3], MESH_SHAPE),
    )
    aggregated = ttnn.aggregate_tensor(result_host, composer=composer)
    ttnn.deallocate(result_host, False)

    return aggregated


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param((1, 2), id="1x2_grid")], indirect=True)
def test_fused_rms_minimal_repro(mesh_device):
    inp, weight, residual = create_inputs()
    inp_sharded, weight_rm, residual_sharded = distribute_and_prepare(inp, weight, residual, mesh_device)
    result = run_fused_rms_norm(inp_sharded, weight_rm, residual_sharded, mesh_device)
    output = collect_result(result, mesh_device)

    print("Output shape:", output.shape)
    assert output is not None
