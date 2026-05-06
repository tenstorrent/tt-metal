# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
DRAM Zero Fill micro op.

Writes zeros to all pages of a DRAM tensor using a kernel on the 12x10 compute
grid.  Each core zeroes a single L1 tile and then writes it to its assigned
slice of DRAM pages via noc_async_write_page + TensorAccessor.

This replaces the host-side ttnn.from_torch / ttnn.zeros pattern for
zero-initialization, avoiding the host-to-device transfer entirely.
"""

import math

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.flash_mla.op import FlashMLADecode
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor

KERNEL_PATH = "models/demos/deepseek_v3_b1/micro_ops/dram_zero_fill/kernels/dram_zero_fill_kernel.cpp"

GRID_X = 12
GRID_Y = 10
NUM_CORES = GRID_X * GRID_Y  # 120

OUTPUT_CB = 0
TILE_32x32 = ttnn.Tile((32, 32))

DEFAULT_KVPE_DIM = 576
DEFAULT_MAX_SEQ_LEN = 1024 * 32
DEFAULT_K_CHUNK_SIZE = 128


class DRAMZeroFill:
    """Zero-fill a pre-allocated DRAM tensor using a device kernel."""

    @staticmethod
    def allocate_kv_cache_on_device(
        device: ttnn.MeshDevice,
        *,
        num_users: int = 1,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        kvpe_dim: int = DEFAULT_KVPE_DIM,
        dtype: ttnn.DataType = ttnn.bfloat8_b,
        mesh_shape: tuple[int, int] | None = None,
    ) -> ttnn.Tensor:
        """
        Allocate a zero-filled KV cache tensor in DRAM with the standard
        NdShardSpec layout used by FlashMLADecode.

        The sequence dimension is sharded across mesh rows, so each device
        receives ``max_seq_len // mesh_rows`` elements along dim 2.

        Args:
            device: Single device or MeshDevice to allocate on.
            num_users: Batch / user count (first dimension).
            max_seq_len: Total (global) sequence length across all mesh rows.
            kvpe_dim: Combined KNOPE + KROPE feature dimension.
            dtype: Data type for the cache (default bfloat8_b).
            mesh_shape: (rows, cols) mesh shape for the tensor topology.
                Dim 2 is sharded across rows, replicated across cols
                (matching ShardTensor2dMesh dims=(2, None)).
                Defaults to ``(device.shape[0], device.shape[1])``.

        Returns:
            A zero-filled DRAM tensor of per-device shape
            ``[num_users, 1, max_seq_len // mesh_rows, kvpe_dim]``.
        """
        mesh_rows = device.shape[0]
        mesh_cols = device.shape[1]
        if max_seq_len % mesh_rows != 0:
            raise ValueError(
                "max_seq_len must be divisible by mesh_rows for KV cache allocation: "
                f"got max_seq_len={max_seq_len}, mesh_rows={mesh_rows}"
            )
        per_device_seq = max_seq_len // mesh_rows

        if mesh_shape is None:
            mesh_shape = (mesh_rows, mesh_cols)

        program_config = FlashMLADecode.ProgramConfig(k_chunk_size=DEFAULT_K_CHUNK_SIZE, exp_approx_mode=False)
        if per_device_seq % program_config.k_chunk_size != 0:
            raise ValueError(
                "Per-device sequence length must be divisible by k_chunk_size for KV cache allocation: "
                f"got per_device_seq={per_device_seq}, k_chunk_size={program_config.k_chunk_size}, "
                f"max_seq_len={max_seq_len}, mesh_rows={mesh_rows}"
            )
        kv_nd_shard_spec = ttnn.NdShardSpec(
            shard_shape=[1, 1, program_config.k_chunk_size, kvpe_dim],
            grid=program_config.grid.optimal_dram_grid(),
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        kv_mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=kv_nd_shard_spec)

        shape = ttnn.Shape([num_users, 1, per_device_seq, kvpe_dim])
        tensor = ttnn.allocate_tensor_on_device(shape, dtype, ttnn.TILE_LAYOUT, device, kv_mem)
        DRAMZeroFill.op(tensor)

        # Set the correct topology so downstream ops see this tensor as TP=2, SP=4
        # which is what ShardTensor2dMesh() would produce
        dist_shape = ttnn.MeshShape(mesh_shape[0], mesh_shape[1])
        placements = [ttnn.PlacementShard(2), ttnn.PlacementReplicate()]
        coords = [
            ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())])
            for coord in ttnn.MeshCoordinateRange(dist_shape)
        ]
        tensor.update_tensor_topology(ttnn.TensorTopology(dist_shape, placements, coords))

        return tensor

    @staticmethod
    def op(output_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """
        Write zeros to every page of *output_tensor* on device.

        Args:
            output_tensor: A DRAM tensor previously allocated with
                ``ttnn.allocate_tensor_on_device`` (any NdShardSpec config).

        Returns:
            The same tensor, now filled with zeros.
        """
        page_size = TILE_32x32.get_tile_size(output_tensor.dtype)

        device_grid = output_tensor.device().compute_with_storage_grid_size()
        if device_grid.x < GRID_X or device_grid.y < GRID_Y:
            raise ValueError(
                "DRAMZeroFill requires at least a 12x10 compute grid: "
                f"required=({GRID_X}, {GRID_Y}), actual=({device_grid.x}, {device_grid.y})"
            )

        shape = output_tensor.shape
        if shape[-2] % 32 != 0 or shape[-1] % 32 != 0:
            raise ValueError(
                "DRAMZeroFill requires the last two tensor dimensions to be multiples of 32: "
                f"got shape={list(shape)}, shape[-2]={shape[-2]}, shape[-1]={shape[-1]}"
            )
        tile_rows = shape[-2] // 32
        tile_cols = shape[-1] // 32
        batch_tiles = 1
        for d in range(len(shape) - 2):
            batch_tiles *= shape[d]
        total_pages = batch_tiles * tile_rows * tile_cols
        pages_per_core = math.ceil(total_pages / NUM_CORES)

        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(GRID_X - 1, GRID_Y - 1))])

        tensor_accessor_args = ttnn.TensorAccessorArgs(output_tensor)
        ncrisc_compile_time_args = tensor_accessor_args.get_compile_time_args()

        ncrisc_named = [
            ("output_cb", OUTPUT_CB),
            ("total_pages", total_pages),
            ("pages_per_core", pages_per_core),
            ("page_size", page_size),
            ("grid_start_x", 0),
            ("grid_start_y", 0),
            ("grid_end_x", GRID_X - 1),
            ("grid_end_y", GRID_Y - 1),
        ]

        cb_format = ttnn.CBFormatDescriptor(
            buffer_index=OUTPUT_CB,
            data_format=output_tensor.dtype,
            page_size=page_size,
            tile=ttnn.TileDescriptor(TILE_32x32),
        )
        cb_descriptor = ttnn.CBDescriptor(
            total_size=page_size,
            core_ranges=core_grid,
            format_descriptors=[cb_format],
        )

        ncrisc_common_runtime_args = [output_tensor.buffer_address()]

        kernel_desc = UnifiedKernelDescriptor(
            kernel_source=KERNEL_PATH,
            core_ranges=core_grid,
            ncrisc_compile_time_args=ncrisc_compile_time_args,
            ncrisc_named_compile_time_args=ncrisc_named,
            ncrisc_common_runtime_args=ncrisc_common_runtime_args,
        )

        kernel_result = kernel_desc.get_kernel_descriptors()
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_result.kernels,
            cbs=[cb_descriptor],
        )

        return ttnn.generic_op([output_tensor, output_tensor], program_descriptor)
