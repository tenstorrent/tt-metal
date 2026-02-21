# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN Post SDPA Fused Op Test with optional SDPA Reduce-to-All and TP All-Reduce

Tests the full post_sdpa fused operation which implements:

When SDPA is enabled:
- SDPA Reduce-to-All: Reduce L/MS tensors across devices, scatter [1,512] to matmul1 cores
  - SDPA Workers (8 cores): Execute SDPA reduction and scatter
  - SDPA Forwarders (2 cores): Forward fabric packets for SDPA CCL

Post-SDPA phases:
- Matmul1: [1, 512] x [512, 128] -> [1, 128] per core on 64 cores (8x8)
- Gather1: Collect to [1, 8192] on gather core (12, 9)
- Mcast: Broadcast [1, 8192] to 130 cores (13x10 rectangular grid)
- Matmul2: [1, 8192] x [8192, 64] -> [1, 64] per core on 112 active cores
- Gather2: Collect to [1, 7168] on gather core (12, 9)
- TP All-Reduce: Exchange [1, 7168] between devices, reduce (local + remote + residual)

The mcast grid (13x10=130 cores) includes 18 inactive cores that receive mcast data
but skip matmul2 via is_matmul2_core=false (col 12 rows 0-8 + row 9 cols 4-11).

Core Layout:
- SDPA Workers: (2,8)-(5,8), (2,9)-(5,9) = 8 cores
- SDPA Forwarders: (6,9), (7,9) = 2 cores
- TP All-Reduce Receiver = Gather core (12, 9): already has local data after Gather2
- TP All-Reduce Sender = Adjacent core (11, 9): reads from gather core, sends via fabric

Full operation: [1, 512] @ [512, 8192] @ [8192, 7168] -> [1, 7168] per device,
then all-reduce across devices with optional residual add.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.post_sdpa.op import PostSDPA
from models.demos.deepseek_v3_b1.micro_ops.sdpa_reduce_to_all.op import SdpaReduceToAll


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _round_up(value: int, alignment: int) -> int:
    """Round up value to the nearest multiple of alignment."""
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
    """
    Compute the total forwarder scratch buffer size in bytes for SDPA reduce-to-all.

    This matches the calculation in sdpa_reduce_to_all/op.py for proper L1 allocation.
    """
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


@pytest.mark.parametrize("mesh_rows, mesh_cols", [(1, 1), (4, 2)], ids=["single_device", "multi_device"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K1, intermediate, K2, output_size, in0_dtype, in1_dtype",
    [
        (1, 512, 8192, 8192, 7168, ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("fuse_residual_add", [False, True])
@pytest.mark.parametrize("ccl_enabled", [True, False], ids=["ccl_on", "ccl_off"])
def test_post_sdpa(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    M,
    K1,
    intermediate,
    K2,
    output_size,
    in0_dtype,
    in1_dtype,
    cluster_axis,
    fuse_residual_add,
    ccl_enabled,
):
    """Test post_sdpa fused operation with optional CCL all-reduce"""

    num_devices = mesh_rows * mesh_cols

    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # CCL requires multiple devices
    if num_devices == 1 and ccl_enabled:
        pytest.skip("CCL requires multiple devices")

    # Residual add is part of CCL reduction; skip when CCL is disabled
    if not ccl_enabled and fuse_residual_add:
        pytest.skip("Residual add requires CCL to be enabled")

    # Create submesh - fabric requires opening full system mesh first
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    # Set up sub-device
    compute_grid_size = submesh.compute_with_storage_grid_size()

    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # 1x32 tiles for input/activation
    b_tile = ttnn.Tile([32, 32])  # 32x32 tiles for weights

    # ========================================================================
    # Grid configuration
    # ========================================================================
    # Matmul1 grid: 8x8 = 64 cores
    MATMUL1_GRID_X = 8
    MATMUL1_GRID_Y = 8
    num_matmul1_cores = MATMUL1_GRID_X * MATMUL1_GRID_Y  # 64

    # Mcast grid: 13x10 = 130 cores (rectangular for efficient mcast)
    MCAST_GRID_X = 13
    MCAST_GRID_Y = 10
    num_mcast_cores = MCAST_GRID_X * MCAST_GRID_Y  # 130

    # Active Matmul2 cores: 112 (rows 0-8 full 12 cols + row 9 cols 0-3)
    # Non-rectangular grid: 12*9 + 4 = 108 + 4 = 112
    num_matmul2_cores = 112

    # Per-core dimensions
    n1_per_core = intermediate // num_matmul1_cores  # 8192 / 64 = 128
    n2_per_core = output_size // num_matmul2_cores  # 7168 / 112 = 64

    logger.info(f"Testing full post_sdpa fused op with TP All-Reduce:")
    logger.info(f"  Matmul1: [{M}, {K1}] x [{K1}, {intermediate}] on {num_matmul1_cores} cores")
    logger.info(f"  Mcast: [{M}, {intermediate}] to {num_mcast_cores} cores (13x10 grid)")
    logger.info(f"  Matmul2: [{M}, {K2}] x [{K2}, {output_size}] on {num_matmul2_cores} active cores")
    logger.info(f"  TP All-Reduce: [{M}, {output_size}] across {num_devices} devices")
    logger.info(f"  Output: [{M}, {output_size}] (fuse_residual_add={fuse_residual_add})")

    # Create core grids
    matmul1_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(MATMUL1_GRID_X - 1, MATMUL1_GRID_Y - 1))]
    )
    # Active matmul2 cores: non-rectangular grid (112 cores)
    # - Rows 0-8: all 12 columns = 108 cores
    # - Row 9: columns 0-3 = 4 cores
    matmul2_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 8)),  # 12x9 = 108 cores
            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(3, 9)),  # 4x1 = 4 cores
        ]
    )
    gather_core = ttnn.CoreCoord(12, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    # ========================================================================
    # Create PyTorch tensors (per-device)
    # ========================================================================
    torch.manual_seed(0)

    # Weights are shared across all devices (replicated)
    # Weights1: [512, 8192]
    torch_weights1 = torch.randn((K1, intermediate), dtype=torch.bfloat16)

    # Weights2: [8192, 7168]
    torch_weights2 = torch.randn((K2, output_size), dtype=torch.bfloat16)

    # Input per device: [1, 512]
    device_inputs = []
    device_input_replicated = []  # For sharding
    for device_idx in range(num_devices):
        torch_input_single = torch.randn((M, K1), dtype=torch.bfloat16)
        device_inputs.append(torch_input_single)
        # Replicate for matmul1 cores
        torch_input = torch_input_single.repeat(num_matmul1_cores, 1)  # [64, 512]
        device_input_replicated.append(torch_input)

    # Residual tensor (optional, shared across devices)
    if fuse_residual_add:
        torch_residual = torch.randn((M, output_size), dtype=torch.bfloat16)
    else:
        torch_residual = None

    # ========================================================================
    # Compute golden reference
    # ========================================================================
    if ccl_enabled:
        # Per-pair CCL all-reduce: each row of devices forms an independent pair
        num_pairs = mesh_rows
        devices_per_pair = mesh_cols
        torch_expected_per_pair = []
        for pair_idx in range(num_pairs):
            pair_inputs = []
            for col in range(devices_per_pair):
                device_idx = pair_idx * devices_per_pair + col
                pair_inputs.append(device_inputs[device_idx].float())
            golden = PostSDPA.golden(
                pair_inputs,
                torch_weights1.float(),
                torch_weights2.float(),
                torch_residual.float() if torch_residual is not None else None,
            ).bfloat16()
            torch_expected_per_pair.append(golden)
        logger.info(f"Golden output shape (per-pair all-reduced): {torch_expected_per_pair[0].shape}")
    else:
        # Per-device matmul only: input @ W1 @ W2
        torch_expected_per_device = []
        for inp in device_inputs:
            result = (inp.float() @ torch_weights1.float() @ torch_weights2.float()).bfloat16()
            torch_expected_per_device.append(result)
        logger.info(f"Golden output shape (per-device): {torch_expected_per_device[0].shape}")

    # ========================================================================
    # Create mesh mapper - shard along dim 0 across all devices in the mesh
    # ========================================================================
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)

    # ========================================================================
    # Create input tensor (height-sharded across matmul1 cores)
    # Each core gets [1, 512]
    # ========================================================================
    input_shard_shape = (M, K1)  # [1, 512] per core
    input_shard_spec = ttnn.ShardSpec(
        matmul1_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Concatenate per-device inputs for mesh tensor
    mesh_input_torch = torch.cat(device_input_replicated, dim=0)  # [num_devices * 64, 512]
    ttnn_input = ttnn.from_torch(
        mesh_input_torch,
        device=submesh,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_mem_config,
        tile=a_tile,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created input tensor: shard {input_shard_shape} on {num_matmul1_cores} cores per device")

    # ========================================================================
    # Create weights1 tensor (width-sharded across matmul1 cores, replicated)
    # Each core gets [512, 128]
    # ========================================================================
    weights1_shard_shape = (K1, n1_per_core)  # [512, 128] per core
    weights1_shard_spec = ttnn.ShardSpec(
        matmul1_grid,
        weights1_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights1_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights1_shard_spec
    )

    # Get single device for weights (they're replicated, so we just need one)
    single_device = ttnn.get_device_tensors(ttnn_input)[0].device()
    ttnn_weights1 = ttnn.from_torch(
        torch_weights1,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=single_device,
        memory_config=weights1_mem_config,
        tile=b_tile,
    )
    logger.info(f"Created weights1 tensor: shard {weights1_shard_shape} on {num_matmul1_cores} cores")

    # ========================================================================
    # Create weights2 tensor (width-sharded across 112 active matmul2 cores, replicated)
    # Each core gets [8192, 64]
    # ========================================================================
    weights2_shard_shape = (K2, n2_per_core)  # [8192, 64] per core
    weights2_shard_spec = ttnn.ShardSpec(
        matmul2_grid,  # Non-rectangular grid of 112 active cores
        weights2_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    weights2_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights2_shard_spec
    )

    ttnn_weights2 = ttnn.from_torch(
        torch_weights2,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=single_device,
        memory_config=weights2_mem_config,
        tile=b_tile,
    )
    logger.info(f"Created weights2 tensor: shard {weights2_shard_shape} on {num_matmul2_cores} active cores")

    # ========================================================================
    # Create gather1 output tensor (intermediate [1, 8192] on gather core, replicated)
    # ========================================================================
    gather1_output_shard_shape = (M, intermediate)  # [1, 8192]
    gather1_output_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        gather1_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gather1_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gather1_output_shard_spec
    )

    torch_gather1_zeros = torch.zeros((M, intermediate), dtype=torch.bfloat16)
    ttnn_gather1_output = ttnn.from_torch(
        torch_gather1_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=single_device,
        memory_config=gather1_output_mem_config,
        tile=a_tile,
    )
    logger.info(f"Created gather1 output tensor: {gather1_output_shard_shape} on gather core")

    # ========================================================================
    # Create gather2 output tensor (intermediate [1, 7168] on gather core, per device)
    # This tensor backs CB7 and holds gather2 output for CCL to read
    # ========================================================================
    gather2_output_shard_shape = (M, output_size)  # [1, 7168]
    gather2_output_shard_spec = ttnn.ShardSpec(
        gather_core_grid,
        gather2_output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gather2_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gather2_output_shard_spec
    )

    torch_gather2_zeros = torch.zeros((M, output_size), dtype=torch.bfloat16)
    mesh_gather2_torch = torch.cat([torch_gather2_zeros] * num_devices, dim=0)
    ttnn_gather2_output = ttnn.from_torch(
        mesh_gather2_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=gather2_output_mem_config,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created gather2 output tensor: {gather2_output_shard_shape} on gather core per device")

    # ========================================================================
    # Create CCL tensors and semaphores (only when CCL is enabled)
    # ========================================================================
    ttnn_ccl_intermediate = None
    ttnn_output = None
    ttnn_residual = None
    semaphores = None

    if ccl_enabled:
        # CCL intermediate tensor (1x32 tiles to match gather2 output format)
        # Shape: [1, 7168] = 224 tiles of 1x32
        ccl_intermediate_shape = [M, output_size]  # [1, 7168]
        ccl_intermediate_shard_shape = tuple(ccl_intermediate_shape)
        ccl_intermediate_shard_spec = ttnn.ShardSpec(
            gather_core_grid,
            ccl_intermediate_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        ccl_intermediate_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ccl_intermediate_shard_spec
        )

        torch_ccl_intermediate = torch.zeros(ccl_intermediate_shape, dtype=torch.bfloat16)
        mesh_ccl_intermediate_torch = torch.cat([torch_ccl_intermediate] * num_devices, dim=0)
        ttnn_ccl_intermediate = ttnn.from_torch(
            mesh_ccl_intermediate_torch,
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            tile=a_tile,  # 1x32 tiles to match gather2 output
            dtype=ttnn.bfloat16,
            memory_config=ccl_intermediate_mem_config,
            mesh_mapper=mesh_mapper,
        )
        logger.info(f"Created CCL intermediate tensor: {ccl_intermediate_shape} on gather core per device")

        # Final output tensor ([1, 7168] on gather core)
        output_shard_shape = (M, output_size)  # [1, 7168]
        output_shard_spec = ttnn.ShardSpec(
            gather_core_grid,
            output_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        output_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec
        )

        torch_output_zeros = torch.zeros((M, output_size), dtype=torch.bfloat16)
        mesh_output_torch = torch.cat([torch_output_zeros] * num_devices, dim=0)
        ttnn_output = ttnn.from_torch(
            mesh_output_torch,
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            tile=a_tile,
            dtype=ttnn.bfloat16,
            memory_config=output_mem_config,
            mesh_mapper=mesh_mapper,
        )
        logger.info(f"Created output tensor: {output_shard_shape} on gather core per device")

        # Residual tensor (optional)
        if fuse_residual_add:
            mesh_residual_torch = torch.cat([torch_residual] * num_devices, dim=0)
            ttnn_residual = ttnn.from_torch(
                mesh_residual_torch,
                device=submesh,
                layout=ttnn.TILE_LAYOUT,
                tile=a_tile,
                dtype=ttnn.bfloat16,
                memory_config=output_mem_config,
                mesh_mapper=mesh_mapper,
            )
            logger.info(f"Created residual tensor: {output_shard_shape} on gather core per device")

        # Global semaphores for CCL
        num_cores = compute_grid_size.x * compute_grid_size.y
        available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
        semaphore1 = ttnn.create_global_semaphore(submesh, available_cores, 0)
        semaphore2 = ttnn.create_global_semaphore(submesh, available_cores, 0)
        semaphores = [semaphore1, semaphore2]
        logger.info("Created global semaphores for CCL synchronization")

    # ========================================================================
    # SDPA KV Cache tensor for CB overlap
    # Matches flash_mla's double-buffered KV CB sizing: shard = (256, 576) per core,
    # giving 156672 bytes/core in bfloat8_b — same as flash_mla's cb_k_in.
    # ========================================================================
    KNOPE_DIM = 512
    KROPE_DIM = 64
    kvpe_dim = KNOPE_DIM + KROPE_DIM  # 576
    sdpa_kv_cache_num_cores_x = compute_grid_size.x
    sdpa_kv_cache_num_cores_y = compute_grid_size.y
    sdpa_kv_cache_num_cores = sdpa_kv_cache_num_cores_x * sdpa_kv_cache_num_cores_y
    sdpa_kv_cache_shard_height = 256  # 2 * k_chunk_size (128), matching flash_mla double-buffer
    sdpa_kv_cache_total_height = sdpa_kv_cache_shard_height * sdpa_kv_cache_num_cores

    sdpa_kv_cache_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(sdpa_kv_cache_num_cores_x - 1, sdpa_kv_cache_num_cores_y - 1),
                )
            }
        ),
        (sdpa_kv_cache_shard_height, kvpe_dim),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sdpa_kv_cache_buffer = ttnn.from_torch(
        torch.randn((1, 1, sdpa_kv_cache_total_height, kvpe_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            sdpa_kv_cache_shard_spec,
        ),
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    logger.info(
        f"Created sdpa_kv_cache buffer for CB overlap: shard ({sdpa_kv_cache_shard_height}, {kvpe_dim}) on {sdpa_kv_cache_num_cores} cores"
    )

    # ========================================================================
    # Run fused operation
    # ========================================================================
    logger.info(f"Running post_sdpa fused operation (ccl_enabled={ccl_enabled})...")
    ttnn_result = PostSDPA.op(
        ttnn_input,
        ttnn_weights1,
        ttnn_weights2,
        ttnn_gather1_output,
        ttnn_gather2_output,
        ttnn_ccl_intermediate,
        ttnn_output,
        semaphores,
        cluster_axis=cluster_axis,
        residual_tensor_mesh=ttnn_residual,
        fp32_dest_acc_en=False,
        ccl_enabled=ccl_enabled,
        sdpa_kv_cache_buffer=sdpa_kv_cache_buffer,
    )
    ttnn.synchronize_device(submesh)

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )
    logger.info(f"Output shape: {output_torch.shape}")

    # ========================================================================
    # Verify results using PCC (Pearson Correlation Coefficient)
    # ========================================================================
    all_passed = True
    pcc_threshold = 0.99  # Require 99% correlation
    for device_idx in range(num_devices):
        received = output_torch[device_idx : device_idx + 1, :]

        # Output should be [1, 7168]
        expected_shape = (M, output_size)
        assert received.shape == expected_shape, f"Expected shape {expected_shape}, got {received.shape}"

        if ccl_enabled:
            # Both devices in a pair should have the same all-reduced result
            pair_idx = device_idx // mesh_cols
            golden = torch_expected_per_pair[pair_idx]
        else:
            # Each device has its own independent result
            golden = torch_expected_per_device[device_idx]

        passing, pcc_message = comp_pcc(golden, received, pcc_threshold)
        if not passing:
            logger.error(f"Device {device_idx}: PCC check FAILED - {pcc_message}")
            logger.error(f"Expected: {golden[:, :5]}")
            logger.error(f"Received: {received[:, :5]}")
            all_passed = False
        else:
            logger.info(f"Device {device_idx}: PASSED - {pcc_message}")

    assert all_passed, "Not all devices have the correct output"
    logger.info(f"✓ Post SDPA fused op test passed (ccl_enabled={ccl_enabled})!")


@pytest.mark.parametrize("mesh_rows, mesh_cols", [(4, 2)], ids=["multi_device"])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "M, K1, intermediate, K2, output_size, in0_dtype, in1_dtype",
    [
        (1, 512, 8192, 8192, 7168, ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize("cluster_axis", [1])
@pytest.mark.parametrize("fuse_residual_add", [False])
def test_post_sdpa_with_sdpa_phase(
    bh_2d_mesh_device,
    mesh_rows,
    mesh_cols,
    M,
    K1,
    intermediate,
    K2,
    output_size,
    in0_dtype,
    in1_dtype,
    cluster_axis,
    fuse_residual_add,
):
    """Test post_sdpa fused operation with SDPA reduce-to-all phase enabled"""

    num_devices = mesh_rows * mesh_cols

    # Validate mesh size
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    # Create submesh - fabric requires opening full system mesh first
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    # Set up sub-device (not supported in slow dispatch mode)
    compute_grid_size = submesh.compute_with_storage_grid_size()
    # Tile dimensions
    a_tile = ttnn.Tile([M, 32])  # 1x32 tiles for input/activation
    b_tile = ttnn.Tile([32, 32])  # 32x32 tiles for weights
    sdpa_tile = ttnn.Tile([8, 32])  # 8x32 tiles for SDPA L and MS tensors (matches original SDPA op)

    # ========================================================================
    # Grid configuration
    # ========================================================================
    # Matmul1 grid: 8x8 = 64 cores
    MATMUL1_GRID_X = 8
    MATMUL1_GRID_Y = 8
    num_matmul1_cores = MATMUL1_GRID_X * MATMUL1_GRID_Y  # 64

    # Mcast grid: 13x10 = 130 cores (rectangular for efficient mcast)
    MCAST_GRID_X = 13
    MCAST_GRID_Y = 10
    num_mcast_cores = MCAST_GRID_X * MCAST_GRID_Y  # 130

    # Active Matmul2 cores: 112 (rows 0-8 full 12 cols + row 9 cols 0-3)
    num_matmul2_cores = 112

    # SDPA configuration (matching original sdpa_reduce_to_all test)
    NUM_SDPA_WORKERS = 8
    SDPA_L_HEIGHT = 8  # 8 rows in L tensor (matches 8x32 tile)
    SDPA_L_WIDTH = 512 * NUM_SDPA_WORKERS  # 512 per worker = 4096 total
    SDPA_MS_WIDTH = 32 * NUM_SDPA_WORKERS  # 32 per worker = 256 total

    # Per-core dimensions
    n1_per_core = intermediate // num_matmul1_cores  # 8192 / 64 = 128
    n2_per_core = output_size // num_matmul2_cores  # 7168 / 112 = 64

    logger.info(f"Testing post_sdpa fused op with SDPA reduce-to-all phase:")
    logger.info(f"  SDPA: [{SDPA_L_HEIGHT}, {SDPA_L_WIDTH}] L tensor, [{SDPA_L_HEIGHT}, {SDPA_MS_WIDTH}] MS tensor")
    logger.info(f"  Matmul1: [{M}, {K1}] x [{K1}, {intermediate}] on {num_matmul1_cores} cores")
    logger.info(f"  Mcast: [{M}, {intermediate}] to {num_mcast_cores} cores (13x10 grid)")
    logger.info(f"  Matmul2: [{M}, {K2}] x [{K2}, {output_size}] on {num_matmul2_cores} active cores")
    logger.info(f"  TP All-Reduce: [{M}, {output_size}] across {num_devices} devices")

    # Create core grids
    matmul1_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(MATMUL1_GRID_X - 1, MATMUL1_GRID_Y - 1))]
    )
    matmul2_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 8)),  # 12x9 = 108 cores
            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(3, 9)),  # 4x1 = 4 cores
        ]
    )
    gather_core = ttnn.CoreCoord(12, 9)
    gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

    # SDPA worker grid: 8 cores at (2,8)-(5,8), (2,9)-(5,9)
    sdpa_worker_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(2, 8), ttnn.CoreCoord(5, 8)),  # 4 cores
            ttnn.CoreRange(ttnn.CoreCoord(2, 9), ttnn.CoreCoord(5, 9)),  # 4 cores
        ]
    )

    # SDPA forwarder cores
    sdpa_forwarder_cores = [ttnn.CoreCoord(6, 9), ttnn.CoreCoord(7, 9)]
    sdpa_forwarder_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(sdpa_forwarder_cores[0], sdpa_forwarder_cores[0]),
            ttnn.CoreRange(sdpa_forwarder_cores[1], sdpa_forwarder_cores[1]),
        ]
    )

    # ========================================================================
    # Create PyTorch tensors (per-device)
    # ========================================================================
    torch.manual_seed(0)

    # Weights are shared across all devices (replicated)
    torch_weights1 = torch.randn((K1, intermediate), dtype=torch.bfloat16)
    torch_weights2 = torch.randn((K2, output_size), dtype=torch.bfloat16)

    # SDPA input tensors per device: L [8, 4096], MS [8, 256]
    # MS tensor layout: for each worker core c (0..7), within columns [c*32, (c+1)*32]:
    #   column 0 = M (max), column 1 = S (sum), rest unused
    # This matches the original sdpa_reduce_to_all test/op format.
    MS_WIDTH_PER_CORE = 32  # ms_width per SDPA worker core
    device_sdpa_l_inputs = []
    device_sdpa_ms_inputs = []
    device_sdpa_m_inputs = []  # M (max) per device: [8, num_workers]
    device_sdpa_s_inputs = []  # S (sum) per device: [8, num_workers]
    for device_idx in range(num_devices):
        torch_sdpa_l = torch.randn((SDPA_L_HEIGHT, SDPA_L_WIDTH), dtype=torch.bfloat16)
        torch_sdpa_ms = torch.randn((SDPA_L_HEIGHT, SDPA_MS_WIDTH), dtype=torch.bfloat16)

        # Create structured M and S values at correct positions
        m_device = torch.zeros((SDPA_L_HEIGHT, NUM_SDPA_WORKERS), dtype=torch.bfloat16)
        s_device = torch.zeros((SDPA_L_HEIGHT, NUM_SDPA_WORKERS), dtype=torch.bfloat16)
        for core_idx in range(NUM_SDPA_WORKERS):
            m_device[:, core_idx] = torch.randn(SDPA_L_HEIGHT, dtype=torch.bfloat16) * 0.5 - 1.0
            s_device[:, core_idx] = torch.abs(torch.randn(SDPA_L_HEIGHT, dtype=torch.bfloat16)) + 0.1
            torch_sdpa_ms[:, core_idx * MS_WIDTH_PER_CORE + 0] = m_device[:, core_idx]
            torch_sdpa_ms[:, core_idx * MS_WIDTH_PER_CORE + 1] = s_device[:, core_idx]

        device_sdpa_l_inputs.append(torch_sdpa_l)
        device_sdpa_ms_inputs.append(torch_sdpa_ms)
        device_sdpa_m_inputs.append(m_device)
        device_sdpa_s_inputs.append(s_device)

    # Residual tensor (optional)
    if fuse_residual_add:
        torch_residual = torch.randn((M, output_size), dtype=torch.bfloat16)
    else:
        torch_residual = None

    # ========================================================================
    # Compute golden reference (SDPA reduce-to-all -> scatter -> matmul1 -> gather -> matmul2 -> CCL)
    # ========================================================================
    # Step 1: Compute SDPA reduce-to-all for each column ring (sdpa_cluster_axis=0).
    # Devices in the same column form a ring. With a 4x2 mesh:
    #   Column 0 ring: device indices 0, 2, 4, 6 (row-major: (0,0), (1,0), (2,0), (3,0))
    #   Column 1 ring: device indices 1, 3, 5, 7 (row-major: (0,1), (1,1), (2,1), (3,1))
    sdpa_l_reduced_per_col = {}
    for col in range(mesh_cols):
        # Collect devices in this column's ring (along axis 0)
        ring_device_indices = [row * mesh_cols + col for row in range(mesh_rows)]
        ring_l_data = [device_sdpa_l_inputs[d].float() for d in ring_device_indices]
        ring_s_data = [device_sdpa_s_inputs[d].float() for d in ring_device_indices]
        ring_m_data = [device_sdpa_m_inputs[d].float() for d in ring_device_indices]

        l_reduced, _, _ = SdpaReduceToAll.golden(
            ring_l_data,
            ring_s_data,
            ring_m_data,
            num_cores=NUM_SDPA_WORKERS,
            scale_value=1.0,
        )
        sdpa_l_reduced_per_col[col] = l_reduced  # [8, 4096]
        logger.info(f"SDPA golden for column {col}: L_reduced shape {l_reduced.shape}")

    # Step 2: Model scatter + matmul1 + gather + matmul2 for each device.
    # The scatter maps SDPA output rows to matmul1 cores:
    #   core_idx = worker_i * scatter_rows + row_j
    #   core_idx gets L_reduced[row_j, worker_i*l_width:(worker_i+1)*l_width]
    # Matmul1: each core computes [1,512] x [512,128] -> [1,128] (weight cols = core_idx*128..+128)
    # Gather: concat 64 x [1,128] -> [1,8192]
    l_width = K1  # 512 per worker
    n_per_core = intermediate // num_matmul1_cores  # 8192/64 = 128

    def scatter_matmul_gather(l_reduced, weights1, weights2):
        """Model scatter -> matmul1 -> gather -> matmul2 pipeline."""
        gathered = torch.zeros(1, intermediate)  # [1, 8192]
        for worker_i in range(NUM_SDPA_WORKERS):
            for row_j in range(SDPA_L_HEIGHT):
                core_idx = worker_i * SDPA_L_HEIGHT + row_j
                input_slice = l_reduced[row_j, worker_i * l_width : (worker_i + 1) * l_width]  # [512]
                weight_slice = weights1[:, core_idx * n_per_core : (core_idx + 1) * n_per_core]  # [512, 128]
                gathered[0, core_idx * n_per_core : (core_idx + 1) * n_per_core] = input_slice @ weight_slice
        result = gathered @ weights2  # [1, 8192] x [8192, 7168] = [1, 7168]
        return result

    # Step 3: CCL all-reduce across row pairs (cluster_axis=1).
    # Each row has mesh_cols devices. Both devices in a row pair compute their own
    # scatter+matmul1+gather+matmul2 (using their column's SDPA output), then sum.
    num_pairs = mesh_rows
    torch_expected_per_pair = []
    for pair_idx in range(num_pairs):
        pair_result = torch.zeros(M, output_size)
        for col in range(mesh_cols):
            l_reduced = sdpa_l_reduced_per_col[col]
            device_result = scatter_matmul_gather(
                l_reduced,
                torch_weights1.float(),
                torch_weights2.float(),
            )
            pair_result += device_result
        # Add residual if provided
        if torch_residual is not None:
            pair_result += torch_residual.float()
        torch_expected_per_pair.append(pair_result.bfloat16())

    logger.info(f"Golden output shape (per-pair all-reduced): {torch_expected_per_pair[0].shape}")

    # ========================================================================
    # Create mesh mapper
    # ========================================================================
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)

    # ========================================================================
    # Create input tensor (height-sharded across matmul1 cores)
    # When SDPA is enabled, this tensor receives scatter output from SDPA workers
    # ========================================================================
    input_shard_shape = (M, K1)  # [1, 512] per core
    input_shard_spec = ttnn.ShardSpec(
        matmul1_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    # Initialize with zeros - SDPA scatter will populate this
    torch_input_zeros = torch.zeros((num_matmul1_cores, K1), dtype=torch.bfloat16)
    mesh_input_torch = torch.cat([torch_input_zeros] * num_devices, dim=0)
    ttnn_input = ttnn.from_torch(
        mesh_input_torch,
        device=submesh,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=input_mem_config,
        tile=a_tile,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created input tensor: shard {input_shard_shape} on {num_matmul1_cores} cores per device")

    # Get single device for replicated tensors
    single_device = ttnn.get_device_tensors(ttnn_input)[0].device()

    # ========================================================================
    # Create weights tensors (same as non-SDPA test)
    # ========================================================================
    weights1_shard_shape = (K1, n1_per_core)
    weights1_shard_spec = ttnn.ShardSpec(matmul1_grid, weights1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    weights1_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights1_shard_spec
    )
    ttnn_weights1 = ttnn.from_torch(
        torch_weights1,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=single_device,
        memory_config=weights1_mem_config,
        tile=b_tile,
    )

    weights2_shard_shape = (K2, n2_per_core)
    weights2_shard_spec = ttnn.ShardSpec(matmul2_grid, weights2_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    weights2_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weights2_shard_spec
    )
    ttnn_weights2 = ttnn.from_torch(
        torch_weights2,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=single_device,
        memory_config=weights2_mem_config,
        tile=b_tile,
    )

    # ========================================================================
    # Create gather output tensors
    # ========================================================================
    gather1_output_shard_shape = (M, intermediate)
    gather1_output_shard_spec = ttnn.ShardSpec(
        gather_core_grid, gather1_output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    gather1_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gather1_output_shard_spec
    )
    ttnn_gather1_output = ttnn.from_torch(
        torch.zeros((M, intermediate), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=single_device,
        memory_config=gather1_output_mem_config,
        tile=a_tile,
    )

    gather2_output_shard_shape = (M, output_size)
    gather2_output_shard_spec = ttnn.ShardSpec(
        gather_core_grid, gather2_output_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    gather2_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gather2_output_shard_spec
    )
    mesh_gather2_torch = torch.cat([torch.zeros((M, output_size), dtype=torch.bfloat16)] * num_devices, dim=0)
    ttnn_gather2_output = ttnn.from_torch(
        mesh_gather2_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=gather2_output_mem_config,
        mesh_mapper=mesh_mapper,
    )

    # ========================================================================
    # Create CCL tensors and semaphores
    # ========================================================================
    ccl_intermediate_shape = [M, output_size]
    ccl_intermediate_shard_spec = ttnn.ShardSpec(
        gather_core_grid, tuple(ccl_intermediate_shape), ttnn.ShardOrientation.ROW_MAJOR
    )
    ccl_intermediate_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ccl_intermediate_shard_spec
    )
    mesh_ccl_intermediate_torch = torch.cat(
        [torch.zeros(ccl_intermediate_shape, dtype=torch.bfloat16)] * num_devices, dim=0
    )
    ttnn_ccl_intermediate = ttnn.from_torch(
        mesh_ccl_intermediate_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=ccl_intermediate_mem_config,
        mesh_mapper=mesh_mapper,
    )

    output_shard_spec = ttnn.ShardSpec(gather_core_grid, (M, output_size), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    mesh_output_torch = torch.cat([torch.zeros((M, output_size), dtype=torch.bfloat16)] * num_devices, dim=0)
    ttnn_output = ttnn.from_torch(
        mesh_output_torch,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,
        mesh_mapper=mesh_mapper,
    )

    ttnn_residual = None
    if fuse_residual_add:
        mesh_residual_torch = torch.cat([torch_residual] * num_devices, dim=0)
        ttnn_residual = ttnn.from_torch(
            mesh_residual_torch,
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            tile=a_tile,
            dtype=ttnn.bfloat16,
            memory_config=output_mem_config,
            mesh_mapper=mesh_mapper,
        )

    # Global semaphores for CCL
    num_cores = compute_grid_size.x * compute_grid_size.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, row_wise=True)
    semaphore1 = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphore2 = ttnn.create_global_semaphore(submesh, available_cores, 0)
    semaphores = [semaphore1, semaphore2]

    # ========================================================================
    # Create SDPA tensors
    # ========================================================================
    # SDPA L input tensor: [8, 4096] width-sharded across 8 SDPA workers
    sdpa_l_per_worker = SDPA_L_WIDTH // NUM_SDPA_WORKERS  # 512 per worker
    sdpa_l_shard_shape = (SDPA_L_HEIGHT, sdpa_l_per_worker)  # [8, 512]
    sdpa_l_shard_spec = ttnn.ShardSpec(sdpa_worker_grid, sdpa_l_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_l_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sdpa_l_shard_spec)

    mesh_sdpa_l_torch = torch.cat(device_sdpa_l_inputs, dim=0)
    ttnn_sdpa_input_l = ttnn.from_torch(
        mesh_sdpa_l_torch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_l_mem_config,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created SDPA L input tensor: shard {sdpa_l_shard_shape} on {NUM_SDPA_WORKERS} workers")

    # SDPA MS input tensor: [8, 256] width-sharded across 8 SDPA workers
    sdpa_ms_per_worker = SDPA_MS_WIDTH // NUM_SDPA_WORKERS  # 32 per worker
    sdpa_ms_shard_shape = (SDPA_L_HEIGHT, sdpa_ms_per_worker)  # [8, 32]
    sdpa_ms_shard_spec = ttnn.ShardSpec(sdpa_worker_grid, sdpa_ms_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_ms_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sdpa_ms_shard_spec
    )

    mesh_sdpa_ms_torch = torch.cat(device_sdpa_ms_inputs, dim=0)
    ttnn_sdpa_input_ms = ttnn.from_torch(
        mesh_sdpa_ms_torch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_ms_mem_config,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )
    logger.info(f"Created SDPA MS input tensor: shard {sdpa_ms_shard_shape} on {NUM_SDPA_WORKERS} workers")

    # SDPA L output tensor (same shape as input)
    mesh_sdpa_l_out_torch = torch.zeros_like(mesh_sdpa_l_torch)
    ttnn_sdpa_output_l = ttnn.from_torch(
        mesh_sdpa_l_out_torch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_l_mem_config,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )

    # SDPA R1/R2 receive buffers: must hold BOTH L and MS data per core.
    # Buffer layout per core: [L_chunk0][L_chunk1]...[L_chunkN-1][MS]
    # The original sdpa_reduce_to_all op uses shard shape [batch, l_width + ms_width] per core.
    # Using only L-sized buffers causes MS writes to go past the allocated memory!
    sdpa_recv_per_worker = sdpa_l_per_worker + sdpa_ms_per_worker  # 512 + 32 = 544
    sdpa_recv_shard_shape = (SDPA_L_HEIGHT, sdpa_recv_per_worker)  # [8, 544]
    sdpa_recv_shard_spec = ttnn.ShardSpec(sdpa_worker_grid, sdpa_recv_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sdpa_recv_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sdpa_recv_shard_spec
    )
    # Full receive tensor: [8, (l_width + ms_width) * num_workers] = [8, 544*8] = [8, 4352]
    sdpa_recv_full_width = sdpa_recv_per_worker * NUM_SDPA_WORKERS
    mesh_sdpa_recv_torch = torch.cat(
        [torch.zeros((SDPA_L_HEIGHT, sdpa_recv_full_width), dtype=torch.bfloat16)] * num_devices, dim=0
    )
    ttnn_sdpa_r1_recv = ttnn.from_torch(
        mesh_sdpa_recv_torch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_recv_mem_config,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )
    ttnn_sdpa_r2_recv = ttnn.from_torch(
        mesh_sdpa_recv_torch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sdpa_recv_mem_config,
        tile=sdpa_tile,
        mesh_mapper=mesh_mapper,
    )

    # SDPA forwarder scratch buffer
    # Compute proper size using the same formula as sdpa_reduce_to_all op
    sdpa_l_per_worker = SDPA_L_WIDTH // NUM_SDPA_WORKERS  # 512 per worker
    sdpa_fwd_buffer_bytes = compute_forwarder_scratch_size(
        batch_size=SDPA_L_HEIGHT,
        l_width=sdpa_l_per_worker,
        num_cores=NUM_SDPA_WORKERS,
    )
    # Total elements (bfloat16 = 2 bytes per element)
    sdpa_fwd_total_elements = sdpa_fwd_buffer_bytes // 2
    # WIDTH_SHARDED across 2 forwarder cores, each gets half
    num_forwarders = 2
    sdpa_fwd_per_forwarder = sdpa_fwd_total_elements // num_forwarders
    sdpa_forwarder_shard_shape = (1, sdpa_fwd_per_forwarder)
    sdpa_forwarder_shard_spec = ttnn.ShardSpec(
        sdpa_forwarder_grid, sdpa_forwarder_shard_shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    sdpa_forwarder_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sdpa_forwarder_shard_spec
    )

    torch_forwarder_scratch = torch.zeros((1, sdpa_fwd_total_elements), dtype=torch.bfloat16)
    mesh_forwarder_scratch = torch.cat([torch_forwarder_scratch] * num_devices, dim=0)
    ttnn_sdpa_forwarder_scratch = ttnn.from_torch(
        mesh_forwarder_scratch,
        device=submesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=sdpa_forwarder_mem_config,
        mesh_mapper=mesh_mapper,
    )
    logger.info(
        f"Created SDPA forwarder scratch buffer: {sdpa_fwd_buffer_bytes} bytes total, {sdpa_fwd_per_forwarder} elements per forwarder"
    )

    # SDPA global semaphores - must be created on the SDPA worker grid (like original SDPA op)
    sdpa_semaphore1 = ttnn.create_global_semaphore(submesh, sdpa_worker_grid, 0)
    sdpa_semaphore2 = ttnn.create_global_semaphore(submesh, sdpa_worker_grid, 0)
    # SDPA global semaphores
    # sdpa_semaphore1 = ttnn.create_global_semaphore(submesh, available_cores, 0)
    # sdpa_semaphore2 = ttnn.create_global_semaphore(submesh, available_cores, 0)
    sdpa_semaphores = [sdpa_semaphore1, sdpa_semaphore2]

    # ========================================================================
    # Run fused operation with SDPA
    # ========================================================================
    logger.info("Running post_sdpa fused operation with SDPA phase...")
    ttnn_result = PostSDPA.op(
        ttnn_input,
        ttnn_weights1,
        ttnn_weights2,
        ttnn_gather1_output,
        ttnn_gather2_output,
        ttnn_ccl_intermediate,
        ttnn_output,
        semaphores,
        cluster_axis=cluster_axis,
        residual_tensor_mesh=ttnn_residual,
        fp32_dest_acc_en=False,
        ccl_enabled=True,
        # SDPA parameters
        sdpa_input_l_mesh=ttnn_sdpa_input_l,
        sdpa_input_ms_mesh=ttnn_sdpa_input_ms,
        sdpa_output_l_mesh=ttnn_sdpa_output_l,
        sdpa_r1_recv_mesh=ttnn_sdpa_r1_recv,
        sdpa_r2_recv_mesh=ttnn_sdpa_r2_recv,
        sdpa_forwarder_scratch_mesh=ttnn_sdpa_forwarder_scratch,
        sdpa_semaphores=sdpa_semaphores,
        sdpa_scale_fp32=1.0,
        sdpa_forwarder_cores=sdpa_forwarder_cores,
        sdpa_cluster_axis=0,  # SDPA reduces on axis 0 (rows), TP reduces on axis 1 (cols)
    )
    ttnn.synchronize_device(submesh)

    # Convert back to torch for comparison
    output_torch = ttnn.to_torch(
        ttnn_result,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )
    logger.info(f"Output shape: {output_torch.shape}")

    # ========================================================================
    # Verify results
    # ========================================================================
    all_passed = True
    pcc_threshold = 0.99
    for device_idx in range(num_devices):
        received = output_torch[device_idx : device_idx + 1, :]
        expected_shape = (M, output_size)
        assert received.shape == expected_shape, f"Expected shape {expected_shape}, got {received.shape}"

        pair_idx = device_idx // mesh_cols
        golden = torch_expected_per_pair[pair_idx]

        passing, pcc_message = comp_pcc(golden, received, pcc_threshold)
        if not passing:
            logger.error(f"Device {device_idx}: PCC check FAILED - {pcc_message}")
            all_passed = False
        else:
            logger.info(f"Device {device_idx}: PASSED - {pcc_message}")

    assert all_passed, "Not all devices have the correct output"
    logger.info("✓ Post SDPA fused op with SDPA phase test passed!")
