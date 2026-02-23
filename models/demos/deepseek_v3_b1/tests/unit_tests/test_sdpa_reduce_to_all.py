# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import skip_for_wormhole_b0
from models.demos.deepseek_v3_b1.micro_ops.sdpa_reduce_to_all.op import SdpaReduceToAll
from tests.ttnn.unit_tests.operations.ccl.blackhole_CI.box.nightly.test_all_gather_nightly import validate_test


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def compute_forwarder_scratch_size(
    batch_size: int,
    l_width: int,
    num_cores: int,
    tile_height: int = 8,
    tile_width: int = 32,
    bytes_per_element: int = 2,
    num_links: int = 2,
):
    input_page_size_bytes = tile_height * tile_width * bytes_per_element
    input_l_num_pages = (batch_size // tile_height) * (l_width // tile_width)

    PNH = 8
    DH = input_l_num_pages * tile_width
    DHt = DH // tile_width
    PNHt = PNH // tile_height
    out_tiles = PNHt * DHt

    max_tiles_per_chunk = 8
    min_num_l_chunks = (out_tiles + max_tiles_per_chunk - 1) // max_tiles_per_chunk
    num_l_chunks = max(min_num_l_chunks, 4)
    if out_tiles % num_l_chunks != 0:
        raise ValueError("out_tiles must be divisible by num_l_chunks")

    tiles_per_l_chunk = out_tiles // num_l_chunks
    l_chunk_size_bytes = tiles_per_l_chunk * input_page_size_bytes

    header_size = ttnn.get_tt_fabric_packet_header_size_bytes()
    l1_alignment = 16
    slot_size = _round_up(header_size + l_chunk_size_bytes, l1_alignment)

    num_workers_per_link = num_cores // num_links
    workers_per_type = num_workers_per_link // 2
    slots_per_worker = 1 + num_l_chunks
    slots_per_round = workers_per_type * slots_per_worker

    return 2 * slots_per_round * slot_size * 2


@skip_for_wormhole_b0("This test is for blackhole")
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}],
    indirect=["device_params"],
)
@pytest.mark.parametrize("scatter_enabled", [False, True], ids=["reduce_only", "reduce_and_scatter"])
@pytest.mark.parametrize(
    "position_vector", [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
)
def test_sdpa_reduce_to_all(bh_1d_mesh_device, scatter_enabled, position_vector):
    num_devices = 4
    num_cores = 8
    l_width = 512
    ms_width = 32
    batch_size = 8

    l_shape = [batch_size, l_width * num_cores]
    ms_shape = [batch_size, ms_width * num_cores]
    intermediate_shape = [batch_size, (l_width + ms_width) * num_cores]

    scale_value = 1.0

    topology = ttnn.Topology.Torus
    validate_test(num_devices, topology, bh_1d_mesh_device.shape, 0)
    submesh_device = bh_1d_mesh_device.create_submesh(ttnn.MeshShape((num_devices, 1)))

    dtype = ttnn.bfloat16
    layout = ttnn.TILE_LAYOUT
    tile = ttnn.Tile((8, 32))

    forwarder_cores = [ttnn.CoreCoord(6, 8), ttnn.CoreCoord(6, 9)]

    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(2, 8), ttnn.CoreCoord(5, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(2, 9), ttnn.CoreCoord(5, 9)),
        }
    )
    shard_spec_l = ttnn.ShardSpec(shard_grid, [batch_size, l_width], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_ms = ttnn.ShardSpec(shard_grid, [batch_size, ms_width], ttnn.ShardOrientation.ROW_MAJOR)
    shard_spec_int = ttnn.ShardSpec(shard_grid, [batch_size, (l_width + ms_width)], ttnn.ShardOrientation.ROW_MAJOR)

    mem_config_l = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_l
    )
    mem_config_ms = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_ms
    )
    mem_config_int = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec_int
    )

    mesh_mapper_config = ttnn.MeshMapperConfig(
        [ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config)

    mesh_mapper_config2 = ttnn.MeshMapperConfig(
        [ttnn.PlacementReplicate(), ttnn.PlacementReplicate()], submesh_device.shape
    )
    mesh_mapper2 = ttnn.create_mesh_mapper(submesh_device, mesh_mapper_config2)

    torch.manual_seed(42)
    l_data_per_device = [torch.randn(l_shape, dtype=torch.float32).to(torch.bfloat16) for _ in range(num_devices)]
    ms_data_per_device = [torch.randn(ms_shape, dtype=torch.float32).to(torch.bfloat16) for _ in range(num_devices)]

    position_mask = torch.tensor(position_vector, dtype=torch.bfloat16)
    final_reduction = (position_mask.sum() > 1.0).item()

    m_data_per_device = []
    s_data_per_device = []
    for d in range(num_devices):
        ms_device = ms_data_per_device[d]
        m_device = torch.zeros((ms_shape[0], num_cores), dtype=torch.bfloat16)
        s_device = torch.zeros((ms_shape[0], num_cores), dtype=torch.bfloat16)
        for core_idx in range(num_cores):
            m_device[:, core_idx] = torch.randn(ms_shape[0], dtype=torch.bfloat16) * 0.5 - 1.0
            s_device[:, core_idx] = torch.abs(torch.randn(ms_shape[0], dtype=torch.bfloat16)) + 0.1
            ms_device[:, core_idx * ms_width + 0] = m_device[:, core_idx]
            ms_device[:, core_idx * ms_width + 1] = s_device[:, core_idx]
        m_data_per_device.append(m_device)
        s_data_per_device.append(s_device)

    l_data_f32 = [t.float() for t in l_data_per_device]
    s_data_f32 = [t.float() for t in s_data_per_device]
    m_data_f32 = [t.float() for t in m_data_per_device]

    ref_l, _, _ = SdpaReduceToAll.golden(
        l_data_f32, s_data_f32, m_data_f32, num_cores, scale_value, position_mask, final_reduction
    )
    ref_l = ref_l.to(torch.bfloat16)

    l_data_all = torch.stack(l_data_per_device, dim=0)
    ms_data_all = torch.stack(ms_data_per_device, dim=0)

    position_shape = [num_cores, 32]
    position_data_base = torch.zeros(position_shape, dtype=torch.int32)

    # Fill all rows identically: replicate position mask across each row
    for row in range(num_cores):
        for d in range(num_devices):
            position_data_base[row, d] = int(position_mask[d])

    # Replicate this same tensor on all devices
    position_data_per_device = [position_data_base.clone() for _ in range(num_devices)]
    position_data_all = torch.stack(position_data_per_device, dim=0)

    # HEIGHT_SHARDED: each core gets [1, 32] shard
    shard_spec_position = ttnn.ShardSpec(shard_grid, [1, 32], ttnn.ShardOrientation.ROW_MAJOR)
    position_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec_position
    )

    position_mesh = ttnn.from_torch(
        position_data_all,
        device=submesh_device,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=ttnn.uint32,
        memory_config=position_mem_config,
        mesh_mapper=mesh_mapper,
    )
    if position_vector == [1.0, 1.0, 1.0, 1.0]:
        position_mesh = None

    input_l_mesh = ttnn.from_torch(
        l_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_l,
        mesh_mapper=mesh_mapper,
    )
    input_ms_mesh = ttnn.from_torch(
        ms_data_all,
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_ms,
        mesh_mapper=mesh_mapper,
    )

    output_l_mesh = ttnn.from_torch(
        torch.zeros_like(l_data_all),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_l,
        mesh_mapper=mesh_mapper,
    )

    r1_recv_mesh = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )
    r2_recv_mesh = ttnn.from_torch(
        torch.zeros(intermediate_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=mem_config_int,
        mesh_mapper=mesh_mapper2,
    )

    # Per-core scratch size (covers BRISC + NCRISC regions).
    # The shard is split across forwarder cores; each core gets the full per-core size.
    forwarder_buffer_size_bytes = compute_forwarder_scratch_size(
        batch_size=batch_size, l_width=l_width, num_cores=num_cores, tile_height=8, tile_width=32, bytes_per_element=2
    )

    num_forwarder_cores = 2
    # Convert per-core bytes to shard width in elements (bfloat16 = 2 bytes).
    # BRISC and NCRISC share the same per-core buffer region.
    forwarder_shard_width_elements = forwarder_buffer_size_bytes // (tile.tile_shape[0] * 2)
    forwarder_shard_width_elements = _round_up(forwarder_shard_width_elements, tile.tile_shape[1])

    forwarder_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(forwarder_cores[0], forwarder_cores[1])})
    forwarder_shard_spec = ttnn.ShardSpec(
        forwarder_shard_grid, [tile.tile_shape[0], forwarder_shard_width_elements], ttnn.ShardOrientation.ROW_MAJOR
    )
    forwarder_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, forwarder_shard_spec
    )
    forwarder_scratch_shape = [tile.tile_shape[0], forwarder_shard_width_elements * num_forwarder_cores]
    forwarder_scratch_mesh = ttnn.from_torch(
        torch.zeros(forwarder_scratch_shape, dtype=torch.bfloat16),
        device=submesh_device,
        layout=layout,
        tile=tile,
        dtype=dtype,
        memory_config=forwarder_mem_config,
        mesh_mapper=mesh_mapper2,
    )

    # ========================================================================
    # Scatter destination tensor (optional): HEIGHT_SHARDED, 1x32 tiles, 8x8 grid
    # Each of the 64 cores gets [1, 512] after scatter
    # ========================================================================
    scatter_dest_mesh = None
    scatter_grid = None
    num_scatter_cores = num_cores * batch_size  # 8 * 8 = 64

    if scatter_enabled:
        scatter_tile = ttnn.Tile((1, 32))
        scatter_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, batch_size - 1))}
        )
        scatter_shard_shape = [1, l_width]  # [1, 512] per core
        scatter_shard_spec = ttnn.ShardSpec(scatter_grid, scatter_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        scatter_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, scatter_shard_spec
        )
        scatter_dest_mesh = ttnn.from_torch(
            torch.zeros([num_scatter_cores, l_width], dtype=torch.bfloat16),
            device=submesh_device,
            layout=layout,
            tile=scatter_tile,
            dtype=dtype,
            memory_config=scatter_mem_config,
            mesh_mapper=mesh_mapper2,
        )

    semaphores = [ttnn.create_global_semaphore(submesh_device, shard_grid, 0) for _ in range(2)]
    ttnn.synchronize_device(submesh_device)

    logger.info(f"Running SDPA reduce-to-all (scatter={'enabled' if scatter_enabled else 'disabled'})...")
    output_mesh = SdpaReduceToAll.op(
        input_l_mesh,
        input_ms_mesh,
        output_l_mesh,
        r1_recv_mesh,
        r2_recv_mesh,
        forwarder_scratch_mesh,
        semaphores,
        scale_fp32=scale_value,
        cluster_axis=0,
        input_forwarder_cores=forwarder_cores,
        scatter_dest_tensor_mesh=scatter_dest_mesh,
        scatter_dest_grid=scatter_grid,
        position_tensor_mesh=position_mesh,
        final_reduction=final_reduction,
    )
    ttnn.synchronize_device(submesh_device)

    # ========================================================================
    # Verify L output (original reduce-to-all correctness check)
    # ========================================================================
    output_l_torch = ttnn.to_torch(output_mesh, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0))
    out_l_root = output_l_torch[0]

    max_diff = torch.max(torch.abs(out_l_root.flatten().float() - ref_l.flatten().float())).item()
    match = max_diff < 0.07

    logger.info(f"L tensor match: {match}, max_diff: {max_diff:.4f}")
    assert match, f"L tensor mismatch! Max diff: {max_diff}"

    # ========================================================================
    # Verify scatter output (only when scatter is enabled)
    # Each core (x=i, y=j) should have ref_l[j, i*l_width:(i+1)*l_width]
    # In HEIGHT_SHARDED ROW_MAJOR order: tensor row (j * num_cores + i) = core (x=i, y=j)
    # ========================================================================
    if scatter_enabled:
        scatter_out_torch = ttnn.to_torch(
            scatter_dest_mesh, mesh_composer=ttnn.ConcatMeshToTensor(submesh_device, dim=0)
        )
        scatter_out_root = scatter_out_torch[:num_scatter_cores, :]  # First device

        scatter_max_diff = 0.0
        for j in range(batch_size):
            for i in range(num_cores):
                row_idx = j * num_cores + i
                expected = ref_l[j, i * l_width : (i + 1) * l_width].float()
                actual = scatter_out_root[row_idx, :].float()
                diff = torch.max(torch.abs(actual - expected)).item()
                scatter_max_diff = max(scatter_max_diff, diff)

        scatter_match = scatter_max_diff < 0.07
        logger.info(f"Scatter output match: {scatter_match}, max_diff: {scatter_max_diff:.4f}")
        assert scatter_match, f"Scatter output mismatch! Max diff: {scatter_max_diff}"
