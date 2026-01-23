# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import torch
from loguru import logger

import ttnn


class FusedResblock:
    WEIGHT_TILE_HEIGHT = 32
    WEIGHT_TILE_WIDTH = 32
    SEMAPHORE_RECEIVER_ID = 0
    SEMAPHORE_SENDER_ID = 1
    OUTPUT_TILES_PER_CORE = 1
    WEIGHT_SHARD_TILES_W = 1
    OUTPUT_SHARD_TILES_M = 1
    OUTPUT_SHARD_TILES_N = 1
    DEFAULT_MATH_FIDELITY = ttnn.MathFidelity.LoFi
    MIN_TENSOR_RANK = 2

    class MatmulCoreCBIndex:
        MM1_FULL_CB = 0
        WEIGHT0_CB = 1
        WEIGHT1_CB = 2
        OUT_CB = 3
        INTERMEDIATE_PREGATHER_CB = 4
        MM2_FULL_CB = 5

    class McastCoreCBIndex:
        MCAST_CORE_GATHER_CB = 6

    MCAST_CORE = ttnn.CoreCoord(8, 7)

    @staticmethod
    def golden(input_a: torch.Tensor, weights: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Golden reference implementation for fused ResBlock.

        Args:
            input_a: Input tensor of shape [B, K]
            weights: List of (weight0, weight1) tuples, one per layer.
                     Each weight0 and weight1 should be shape [K, K]

        Returns:
            Output tensor of shape [B, K]
        """

        def layer(input: torch.Tensor, weight0: torch.Tensor, weight1: torch.Tensor) -> torch.Tensor:
            x = input @ weight0
            x = torch.nn.functional.relu(x)
            x = x @ weight1
            x = x + input
            return x

        x = input_a
        for weight0, weight1 in weights:
            x = layer(x, weight0, weight1)

        return x

    @staticmethod
    def create_mcast_kernel(
        all_mcast_cores: ttnn.CoreRangeSet,
        all_sender_cores: ttnn.CoreRangeSet,
        data_format: ttnn.DataType,
        page_size: int,
        num_tiles: int,
        tile: ttnn.TileDescriptor,
        receiver_semaphore_descriptor: ttnn.SemaphoreDescriptor,
        sender_semaphore_descriptor: ttnn.SemaphoreDescriptor,
        device: ttnn.Device,
        debug: bool,
        num_layers: int,
        input_buffer_address: int,
    ) -> Tuple[ttnn.KernelDescriptor, ttnn.KernelDescriptor, ttnn.CBDescriptor]:
        logger.debug(f"All mcast cores: {all_mcast_cores}")
        logger.debug(f"Number of mcast cores: {all_mcast_cores.num_cores()}")
        logger.debug(f"All sender cores: {all_sender_cores}")
        logger.debug(f"Number of sender cores: {all_sender_cores.num_cores()}")

        number_of_senders = all_sender_cores.num_cores()

        # Create CBs for gather destination
        mcast_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
            data_format=data_format,
            page_size=page_size,
            tile=tile,
        )
        mcast_cb_descriptor = ttnn.CBDescriptor(
            total_size=page_size * num_tiles,
            core_ranges=all_sender_cores.merge(
                all_mcast_cores
            ),  # Create on all cores since otherwise the senders won't get the correct CB address
            format_descriptors=[mcast_cb_format],
        )

        def get_mcast_sender_noc_coords(all_sender_cores: ttnn.CoreRangeSet) -> Tuple[int, int, int, int]:
            all_sender_cores_list = (
                device.worker_core_from_logical_core(all_sender_cores.ranges()[0].start),
                device.worker_core_from_logical_core(all_sender_cores.ranges()[0].end),
            )
            return (
                all_sender_cores_list[0].x,
                all_sender_cores_list[0].y,
                all_sender_cores_list[1].x,
                all_sender_cores_list[1].y,
            )

        (
            mcast_sender_noc_coord_x_start,
            mcast_sender_noc_coord_y_start,
            mcast_sender_noc_coord_x_end,
            mcast_sender_noc_coord_y_end,
        ) = get_mcast_sender_noc_coords(all_sender_cores)
        logger.debug(
            f"Mcast sender NOC coords: {mcast_sender_noc_coord_x_start}, {mcast_sender_noc_coord_y_start}, {mcast_sender_noc_coord_x_end}, {mcast_sender_noc_coord_y_end}"
        )

        mcast_reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/mcast_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_mcast_cores,
            compile_time_args=[
                FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
                FusedResblock.MatmulCoreCBIndex.MM2_FULL_CB,
                FusedResblock.MatmulCoreCBIndex.MM1_FULL_CB,
                input_buffer_address,  # Base address of input tensor buffer
                number_of_senders,
                receiver_semaphore_descriptor.id,
                sender_semaphore_descriptor.id,
                mcast_sender_noc_coord_x_start,
                mcast_sender_noc_coord_y_start,
                mcast_sender_noc_coord_x_end,
                mcast_sender_noc_coord_y_end,
                num_layers,
            ],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=ttnn.NOC.NOC_0,
                noc_mode=ttnn.NOC_MODE.DM_DEDICATED_NOC,
            ),
        )
        mcast_writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/mcast_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_mcast_cores,
            compile_time_args=[
                FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
                number_of_senders,
                receiver_semaphore_descriptor.id,
                sender_semaphore_descriptor.id,
            ],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1,
                noc=ttnn.NOC.NOC_1,
                noc_mode=ttnn.NOC_MODE.DM_DEDICATED_NOC,
            ),
        )
        return mcast_reader_kernel_descriptor, mcast_writer_kernel_descriptor, mcast_cb_descriptor

    @staticmethod
    def validate_tensor_types(
        input_tensor: ttnn.Tensor, weight0: ttnn.Tensor, weight1: ttnn.Tensor, output_tensor: ttnn.Tensor
    ) -> None:
        """Validate inputs are ttnn.Tensor and on same device."""
        assert isinstance(input_tensor, ttnn.Tensor), f"Input tensor must be ttnn.Tensor, got {type(input_tensor)}"
        assert isinstance(weight0, ttnn.Tensor), f"Weight must be ttnn.Tensor, got {type(weight0)}"
        assert isinstance(weight1, ttnn.Tensor), f"Weight must be ttnn.Tensor, got {type(weight1)}"
        assert isinstance(output_tensor, ttnn.Tensor), f"Output tensor must be ttnn.Tensor, got {type(output_tensor)}"
        assert (
            input_tensor.device() == weight0.device() == weight1.device() == output_tensor.device()
        ), "All tensors must be on the same device"

    @staticmethod
    def validate_tensor_ranks(
        input_shape: Tuple[int, ...],
        weight0_shape: Tuple[int, ...],
        weight1_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
    ) -> None:
        """Validate rank and matmul shape compatibility."""
        assert (
            len(input_shape) >= FusedResblock.MIN_TENSOR_RANK
        ), f"Input tensor must have rank >= {FusedResblock.MIN_TENSOR_RANK}, got {len(input_shape)}"
        assert (
            len(weight0_shape) >= FusedResblock.MIN_TENSOR_RANK
        ), f"Weight0 must have rank >= {FusedResblock.MIN_TENSOR_RANK}, got {len(weight0_shape)}"
        assert (
            len(weight1_shape) >= FusedResblock.MIN_TENSOR_RANK
        ), f"Weight1 must have rank >= {FusedResblock.MIN_TENSOR_RANK}, got {len(weight1_shape)}"
        assert (
            len(output_shape) >= FusedResblock.MIN_TENSOR_RANK
        ), f"Output tensor must have rank >= {FusedResblock.MIN_TENSOR_RANK}, got {len(output_shape)}"

    @staticmethod
    def validate_weight_shapes(
        input_k: int,
        weight0_k_in: int,
        weight0_k_out: int,
        weight1_k_in: int,
        weight1_k_out: int,
        weight0_shape: Tuple[int, ...],
        weight1_shape: Tuple[int, ...],
        num_layers: int,
    ) -> None:
        """Validate stacked weight shapes: weights should be [num_layers * K, K]."""
        assert (
            weight0_k_in == num_layers * input_k
        ), f"Weight0 K_in ({weight0_k_in}) must equal num_layers * input K ({num_layers * input_k})"
        assert (
            weight1_k_in == num_layers * input_k
        ), f"Weight1 K_in ({weight1_k_in}) must equal num_layers * input K ({num_layers * input_k})"
        assert weight0_k_out == input_k, f"Weight0 K_out ({weight0_k_out}) must match input K ({input_k})"
        assert weight1_k_out == input_k, f"Weight1 K_out ({weight1_k_out}) must match input K ({input_k})"
        assert (
            weight0_shape[-2] == weight1_shape[-2]
        ), f"Weight0 and weight1 must have same stacked height: {weight0_shape[-2]} vs {weight1_shape[-2]}"

    @staticmethod
    def validate_tile_compatibility(
        input_shape: Tuple[int, ...], input_tile: ttnn.Tile, output_k: int, input_k: int
    ) -> int:
        """Validate tile compatibility and K dimension divisibility."""
        assert (
            input_shape[1] % input_tile.tile_shape[1] == 0
        ), f"Input K ({input_shape[1]}) must be divisible by tile width ({input_tile.tile_shape[1]})"
        num_tiles_k = input_shape[1] // input_tile.tile_shape[1]
        assert output_k == input_k, f"Output K dimension ({output_k}) must match input K ({input_k})"
        return num_tiles_k

    @staticmethod
    def validate_dtype_compatibility(weight0_dtype: ttnn.DataType, weight1_dtype: ttnn.DataType) -> None:
        """Validate dtype compatibility."""
        assert (
            weight0_dtype == weight1_dtype
        ), f"Weight0 dtype ({weight0_dtype}) must match weight1 dtype ({weight1_dtype})"

    @staticmethod
    def validate_sharding(
        input_tensor: ttnn.Tensor, weight0: ttnn.Tensor, weight1: ttnn.Tensor, output_tensor: ttnn.Tensor
    ) -> None:
        """Validate sharding expectations."""
        assert (
            input_tensor.memory_config().shard_spec is not None
        ), "Input tensor must have shard_spec (must be sharded)"
        assert weight0.memory_config().shard_spec is not None, "Weight0 must have shard_spec (must be sharded)"
        assert weight1.memory_config().shard_spec is not None, "Weight1 must have shard_spec (must be sharded)"
        assert (
            output_tensor.memory_config().shard_spec is not None
        ), "Output tensor must have shard_spec (must be sharded)"

    @staticmethod
    def validate_core_allocation(all_matmul_cores: ttnn.CoreRangeSet) -> None:
        """Validate matmul cores don't overlap with mcast core."""
        assert not all_matmul_cores.contains(
            FusedResblock.MCAST_CORE
        ), f"Matmul cores {all_matmul_cores} must not contain mcast core {FusedResblock.MCAST_CORE}"

    @staticmethod
    def validate_weight_tiles(
        weight0_tile: ttnn.Tile,
        weight1_tile: ttnn.Tile,
        weight0_shard_shape: Tuple[int, ...],
        weight1_shard_shape: Tuple[int, ...],
    ) -> None:
        """Validate weight tile is 32x32 and weight shard width is 1 tile."""
        weight0_shard_tiles_w = weight0_shard_shape[1] // weight0_tile.tile_shape[1]
        weight1_shard_tiles_w = weight1_shard_shape[1] // weight1_tile.tile_shape[1]

        assert (
            weight0_tile.tile_shape[0] == FusedResblock.WEIGHT_TILE_HEIGHT
            and weight0_tile.tile_shape[1] == FusedResblock.WEIGHT_TILE_WIDTH
        ), f"Weight0 tile must be exactly {FusedResblock.WEIGHT_TILE_HEIGHT}x{FusedResblock.WEIGHT_TILE_WIDTH}, got {weight0_tile.tile_shape}"
        assert (
            weight1_tile.tile_shape[0] == FusedResblock.WEIGHT_TILE_HEIGHT
            and weight1_tile.tile_shape[1] == FusedResblock.WEIGHT_TILE_WIDTH
        ), f"Weight1 tile must be exactly {FusedResblock.WEIGHT_TILE_HEIGHT}x{FusedResblock.WEIGHT_TILE_WIDTH}, got {weight1_tile.tile_shape}"
        assert (
            weight0_shard_tiles_w == FusedResblock.WEIGHT_SHARD_TILES_W
        ), f"Weight0 shard width must be exactly {FusedResblock.WEIGHT_SHARD_TILES_W} tile (got {weight0_shard_tiles_w} tiles). This op only iterates over K dimension and does not support multiple output tiles in N dimension. Shard shape: {weight0_shard_shape} elements, tile: {weight0_tile.tile_shape}"
        assert (
            weight1_shard_tiles_w == FusedResblock.WEIGHT_SHARD_TILES_W
        ), f"Weight1 shard width must be exactly {FusedResblock.WEIGHT_SHARD_TILES_W} tile (got {weight1_shard_tiles_w} tiles). This op only iterates over K dimension and does not support multiple output tiles in N dimension. Shard shape: {weight1_shard_shape} elements, tile: {weight1_tile.tile_shape}"

    @staticmethod
    def validate_output_tiles(out_shard_shape: Tuple[int, ...], out_tile: ttnn.Tile) -> None:
        """Validate that we only support single output tile per core (M=1 tile, N=1 tile per shard)."""
        out_shard_tiles_m = out_shard_shape[0] // out_tile.tile_shape[0]
        out_shard_tiles_n = out_shard_shape[1] // out_tile.tile_shape[1]
        assert (
            out_shard_tiles_m == FusedResblock.OUTPUT_SHARD_TILES_M
        ), f"Output shard M dimension must be exactly {FusedResblock.OUTPUT_SHARD_TILES_M} tile (got {out_shard_tiles_m} tiles). This op only supports single output tile in M dimension per core and iterates over K dimension only. Shard shape: {out_shard_shape} elements, tile: {out_tile.tile_shape}"
        assert (
            out_shard_tiles_n == FusedResblock.OUTPUT_SHARD_TILES_N
        ), f"Output shard N dimension must be exactly {FusedResblock.OUTPUT_SHARD_TILES_N} tile (got {out_shard_tiles_n} tiles). This op only supports single output tile in N dimension per core and iterates over K dimension only. Shard shape: {out_shard_shape} elements, tile: {out_tile.tile_shape}"

    @staticmethod
    def extract_tensor_properties(
        input_tensor: ttnn.Tensor, weight0: ttnn.Tensor, weight1: ttnn.Tensor, output_tensor: ttnn.Tensor
    ) -> Tuple[
        Tuple[int, ...],
        ttnn.Tile,
        Tuple[int, ...],
        ttnn.Tile,
        Tuple[int, ...],
        ttnn.Tile,
        Tuple[int, ...],
        ttnn.Tile,
        ttnn.DataType,
        ttnn.DataType,
        ttnn.DataType,
        ttnn.DataType,
    ]:
        """Extract shapes, tiles, and dtypes from tensors."""
        input_shape = input_tensor.shape
        input_tile = input_tensor.get_tile()
        weight0_shape = weight0.shape
        weight0_tile = weight0.get_tile()
        weight1_shape = weight1.shape
        weight1_tile = weight1.get_tile()
        output_shape = output_tensor.shape
        output_tile = output_tensor.get_tile()

        input_dtype = input_tensor.dtype
        weight0_dtype = weight0.dtype
        weight1_dtype = weight1.dtype
        out_dtype = output_tensor.dtype

        return (
            input_shape,
            input_tile,
            weight0_shape,
            weight0_tile,
            weight1_shape,
            weight1_tile,
            output_shape,
            output_tile,
            input_dtype,
            weight0_dtype,
            weight1_dtype,
            out_dtype,
        )

    @staticmethod
    def create_cb_descriptors(
        input_tensor: ttnn.Tensor,
        weight0: ttnn.Tensor,
        weight1: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        all_matmul_cores: ttnn.CoreRangeSet,
        all_mcast_cores: ttnn.CoreRangeSet,
        num_tiles_k: int,
        out_dtype: ttnn.DataType,
        out_tile: ttnn.Tile,
    ) -> Tuple[
        ttnn.CBDescriptor,
        ttnn.CBDescriptor,
        ttnn.CBDescriptor,
        ttnn.CBDescriptor,
        ttnn.CBDescriptor,
        ttnn.CBDescriptor,
    ]:
        """Create all circular buffer descriptors."""
        mm1_full_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            FusedResblock.MatmulCoreCBIndex.MM1_FULL_CB, input_tensor
        )
        weight0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            FusedResblock.MatmulCoreCBIndex.WEIGHT0_CB, weight0
        )
        weight1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            FusedResblock.MatmulCoreCBIndex.WEIGHT1_CB, weight1
        )
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            FusedResblock.MatmulCoreCBIndex.OUT_CB, output_tensor
        )

        out_tile_size = out_tile.get_tile_size(out_dtype)
        out_tile_descriptor = ttnn.TileDescriptor(out_tile)

        intermediate_pregather_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=FusedResblock.MatmulCoreCBIndex.INTERMEDIATE_PREGATHER_CB,
            data_format=out_dtype,
            page_size=out_tile_size,
            tile=out_tile_descriptor,
        )
        intermediate_pregather_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_tile_size,
            core_ranges=all_matmul_cores,
            format_descriptors=[intermediate_pregather_cb_format],
        )
        mm2_full_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=FusedResblock.MatmulCoreCBIndex.MM2_FULL_CB,
            data_format=out_dtype,
            page_size=out_tile_size,
            tile=out_tile_descriptor,
        )
        mm2_full_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_tile_size * num_tiles_k,
            core_ranges=all_matmul_cores.merge(
                all_mcast_cores
            ),  # Include mcast cores to ensure all cores can get the correct CB address with get_write_ptr
            format_descriptors=[mm2_full_cb_format],
        )

        return (
            mm1_full_cb_descriptor,
            weight0_cb_descriptor,
            weight1_cb_descriptor,
            out_cb_descriptor,
            intermediate_pregather_cb_descriptor,
            mm2_full_cb_descriptor,
        )

    @staticmethod
    def create_semaphore_descriptors(
        all_cores: ttnn.CoreRangeSet,
    ) -> Tuple[ttnn.SemaphoreDescriptor, ttnn.SemaphoreDescriptor]:
        """Create semaphore descriptors."""
        mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=FusedResblock.SEMAPHORE_RECEIVER_ID,
            core_ranges=all_cores,
            initial_value=0,
        )
        mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=FusedResblock.SEMAPHORE_SENDER_ID,
            core_ranges=all_cores,
            initial_value=0,
        )
        return mcast_receiver_semaphore_descriptor, mcast_sender_semaphore_descriptor

    @staticmethod
    def partition_cores_by_noc(
        all_matmul_cores: ttnn.CoreRangeSet,
        gather_destination_core: ttnn.CoreCoord,
        device: ttnn.Device,
        debug: bool = False,
    ) -> Tuple[ttnn.CoreRangeSet, ttnn.CoreRangeSet, List[ttnn.CoreCoord], List[ttnn.CoreCoord]]:
        """
        Partition sender cores into NOC0 and NOC1 groups based on hop distance to gather destination.

        Args:
            all_matmul_cores: CoreRangeSet containing all matmul sender cores
            gather_destination_core: Destination core for gather operations
            device: TTNN device instance
            debug: Enable debug logging

        Returns:
            Tuple of (noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores)
        """
        # Convert CoreRangeSet to list of cores
        input_cores_list = ttnn.corerange_to_cores(all_matmul_cores, row_wise=True)
        noc0_cores = []
        noc1_cores = []

        # Partition cores based on hop distance
        for core in input_cores_list:
            noc0_hop = device.get_worker_noc_hop_distance(core, gather_destination_core, ttnn.NOC.NOC_0)
            noc1_hop = device.get_worker_noc_hop_distance(core, gather_destination_core, ttnn.NOC.NOC_1)
            if noc0_hop <= noc1_hop:
                noc0_cores.append(core)
            else:
                noc1_cores.append(core)

        # Create CoreRangeSets for NOC0 and NOC1 cores
        noc0_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in noc0_cores])
        noc1_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in noc1_cores])

        # Validate that all cores are assigned
        assert (
            len(noc0_cores) + len(noc1_cores) == all_matmul_cores.num_cores()
        ), f"Core partition mismatch: noc0={len(noc0_cores)}, noc1={len(noc1_cores)}, total={all_matmul_cores.num_cores()}"

        return noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores

    @staticmethod
    def generate_runtime_args(all_matmul_cores: ttnn.CoreRangeSet) -> List[Tuple[ttnn.CoreCoord, List[int]]]:
        """Generate runtime args for kernels."""
        sender_core_range = all_matmul_cores.ranges()[0]
        sender_logical_x_start = sender_core_range.start.x
        sender_logical_y_start = sender_core_range.start.y
        sender_logical_x_end = sender_core_range.end.x
        sender_logical_y_end = sender_core_range.end.y
        sender_grid_width = sender_logical_x_end - sender_logical_x_start + 1

        return [
            (
                ttnn.CoreCoord(core.x, core.y),
                [(core.y - sender_logical_y_start) * sender_grid_width + (core.x - sender_logical_x_start)],
            )
            for core_range in all_matmul_cores.ranges()
            for core in [
                ttnn.CoreCoord(x, y)
                for y in range(core_range.start.y, core_range.end.y + 1)
                for x in range(core_range.start.x, core_range.end.x + 1)
            ]
        ]

    @staticmethod
    def create_kernel_descriptors(
        all_matmul_cores: ttnn.CoreRangeSet,
        all_mcast_cores: ttnn.CoreRangeSet,
        gather_destination_core: ttnn.CoreCoord,
        mcast_receiver_semaphore_descriptor: ttnn.SemaphoreDescriptor,
        mcast_sender_semaphore_descriptor: ttnn.SemaphoreDescriptor,
        num_tiles_k: int,
        num_layers: int,
        fp32_dest_acc_en: bool,
        use_custom_mm: bool,
        out_dtype: ttnn.DataType,
        out_tile_size: int,
        out_tile_descriptor: ttnn.TileDescriptor,
        device: ttnn.Device,
        debug: bool,
        input_buffer_address: int,
        runtime_args: List[Tuple[ttnn.CoreCoord, List[int]]],
    ) -> Tuple[
        List[ttnn.KernelDescriptor],
        ttnn.KernelDescriptor,
        ttnn.KernelDescriptor,
        ttnn.KernelDescriptor,
        ttnn.KernelDescriptor,
        ttnn.CBDescriptor,
    ]:
        """Create all kernel descriptors (reader, writer, compute, mcast_reader, mcast_writer) and mcast CB descriptor."""
        (
            mcast_reader_kernel_descriptor,
            mcast_writer_kernel_descriptor,
            mcast_cb_descriptor,
        ) = FusedResblock.create_mcast_kernel(
            all_mcast_cores,
            all_matmul_cores,
            out_dtype,
            out_tile_size,
            num_tiles_k,
            out_tile_descriptor,
            mcast_receiver_semaphore_descriptor,
            mcast_sender_semaphore_descriptor,
            device,
            debug,
            num_layers,
            input_buffer_address,
        )

        # Partition sender cores by NOC hop distance
        noc0_core_range_set, noc1_core_range_set, noc0_cores, noc1_cores = FusedResblock.partition_cores_by_noc(
            all_matmul_cores, gather_destination_core, device, debug
        )

        # Create reader kernel descriptors for NOC0 and NOC1
        reader_kernel_descriptors = []
        common_compile_time_args = [
            FusedResblock.MatmulCoreCBIndex.MM1_FULL_CB,
            FusedResblock.MatmulCoreCBIndex.WEIGHT0_CB,
            FusedResblock.MatmulCoreCBIndex.WEIGHT1_CB,
            FusedResblock.MatmulCoreCBIndex.INTERMEDIATE_PREGATHER_CB,
            FusedResblock.MatmulCoreCBIndex.MM2_FULL_CB,
            FusedResblock.MatmulCoreCBIndex.OUT_CB,
            num_tiles_k,
            gather_destination_core.x,
            gather_destination_core.y,
            mcast_receiver_semaphore_descriptor.id,
            FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
            mcast_sender_semaphore_descriptor.id,
            num_layers,
        ]

        # Filter runtime_args for each NOC group
        noc0_runtime_args = [(core, args) for core, args in runtime_args if core in noc0_cores]
        noc1_runtime_args = [(core, args) for core, args in runtime_args if core in noc1_cores]

        # NOC0 reader kernel
        if not noc0_core_range_set.empty():
            noc0_reader_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source="models/demos/resblock/kernels/reader.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc0_core_range_set,
                compile_time_args=common_compile_time_args,
                runtime_args=noc0_runtime_args,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_0,
                ),
            )
            reader_kernel_descriptors.append(noc0_reader_kernel_descriptor)

        # NOC1 reader kernel
        if not noc1_core_range_set.empty():
            noc1_reader_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source="models/demos/resblock/kernels/reader.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc1_core_range_set,
                compile_time_args=common_compile_time_args,
                runtime_args=noc1_runtime_args,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.NOC_1,
                ),
            )
            reader_kernel_descriptors.append(noc1_reader_kernel_descriptor)

        # Create writer kernel descriptors for NOC0 and NOC1
        writer_kernel_descriptors = []

        # NOC0 writer kernel (opposite NOC and processor from noc0_reader)
        if not noc0_core_range_set.empty():
            noc0_writer_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source="models/demos/resblock/kernels/writer.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc0_core_range_set,
                compile_time_args=[
                    FusedResblock.MatmulCoreCBIndex.OUT_CB,
                ],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.NOC_1,
                ),
            )
            writer_kernel_descriptors.append(noc0_writer_kernel_descriptor)

        # NOC1 writer kernel (opposite NOC and processor from noc1_reader)
        if not noc1_core_range_set.empty():
            noc1_writer_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source="models/demos/resblock/kernels/writer.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc1_core_range_set,
                compile_time_args=[
                    FusedResblock.MatmulCoreCBIndex.OUT_CB,
                ],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,
                    noc=ttnn.NOC.NOC_0,
                ),
            )
            writer_kernel_descriptors.append(noc1_writer_kernel_descriptor)

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_matmul_cores,
            compile_time_args=[
                FusedResblock.MatmulCoreCBIndex.MM1_FULL_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT0_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT1_CB,
                FusedResblock.MatmulCoreCBIndex.OUT_CB,
                FusedResblock.MatmulCoreCBIndex.INTERMEDIATE_PREGATHER_CB,
                FusedResblock.MatmulCoreCBIndex.MM2_FULL_CB,
                num_tiles_k,
                1 if fp32_dest_acc_en else 0,
                num_layers,
                1 if use_custom_mm else 0,
            ],
            runtime_args=runtime_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=FusedResblock.DEFAULT_MATH_FIDELITY,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
        )

        return (
            reader_kernel_descriptors,
            writer_kernel_descriptors,
            compute_kernel_descriptor,
            mcast_reader_kernel_descriptor,
            mcast_writer_kernel_descriptor,
            mcast_cb_descriptor,
        )

    @staticmethod
    def op(
        input_tensor: ttnn.Tensor,
        weight0: ttnn.Tensor,
        weight1: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        num_layers: int = 1,
        fp32_dest_acc_en: bool = False,
        use_custom_mm: bool = True,
        debug: bool = False,
    ) -> ttnn.Tensor:
        """
        Fused ResBlock operation with per-layer weights.

        Args:
            input_tensor: Input tensor of shape [M, K], sharded across matmul cores
            weight0: Stacked weight0 tensor of shape [num_layers * K, K], width-sharded across cores.
                     Contains weights for all layers stacked along the row dimension.
            weight1: Stacked weight1 tensor of shape [num_layers * K, K], width-sharded across cores.
                     Contains weights for all layers stacked along the row dimension.
            output_tensor: Output tensor of shape [M, K], width-sharded across cores
            num_layers: Number of ResBlock layers. Must equal weight0.shape[-2] / K (and weight1.shape[-2] / K)
            fp32_dest_acc_en: Enable FP32 destination accumulation
            use_custom_mm: Use custom_mm API (MOP-based K-dimension reduction) instead of generic matmul.
                          Defaults to True for better performance.
            debug: Enable debug logging

        Returns:
            Output tensor (same as output_tensor parameter)
        """
        logger.info(
            f"Running ResBlock operation with shape {input_tensor.shape} x {weight0.shape} x {weight1.shape} -> {output_tensor.shape}, num_layers={num_layers}"
        )
        logger.debug(f"Input tensor sharding: {input_tensor.memory_config().shard_spec}")
        logger.debug(f"Weight0 tensor sharding: {weight0.memory_config().shard_spec}")
        logger.debug(f"Weight1 tensor sharding: {weight1.memory_config().shard_spec}")
        logger.debug(f"Output tensor sharding: {output_tensor.memory_config().shard_spec}")

        device = input_tensor.device()

        (
            input_shape,
            input_tile,
            weight0_shape,
            weight0_tile,
            weight1_shape,
            weight1_tile,
            output_shape,
            output_tile,
            input_dtype,
            weight0_dtype,
            weight1_dtype,
            out_dtype,
        ) = FusedResblock.extract_tensor_properties(input_tensor, weight0, weight1, output_tensor)

        FusedResblock.validate_tensor_types(input_tensor, weight0, weight1, output_tensor)
        FusedResblock.validate_tensor_ranks(input_shape, weight0_shape, weight1_shape, output_shape)

        input_m, input_k = input_shape[-2], input_shape[-1]
        weight0_k_in, weight0_k_out = weight0_shape[-2], weight0_shape[-1]
        weight1_k_in, weight1_k_out = weight1_shape[-2], weight1_shape[-1]
        output_m, output_k = output_shape[-2], output_shape[-1]

        FusedResblock.validate_weight_shapes(
            input_k, weight0_k_in, weight0_k_out, weight1_k_in, weight1_k_out, weight0_shape, weight1_shape, num_layers
        )

        num_tiles_k = FusedResblock.validate_tile_compatibility(input_shape, input_tile, output_k, input_k)
        FusedResblock.validate_dtype_compatibility(weight0_dtype, weight1_dtype)
        FusedResblock.validate_sharding(input_tensor, weight0, weight1, output_tensor)

        all_matmul_cores = weight0.memory_config().shard_spec.grid
        FusedResblock.validate_core_allocation(all_matmul_cores)

        weight0_shard_shape = weight0.memory_config().shard_spec.shape
        weight1_shard_shape = weight1.memory_config().shard_spec.shape
        FusedResblock.validate_weight_tiles(weight0_tile, weight1_tile, weight0_shard_shape, weight1_shard_shape)

        out_shard_shape = output_tensor.memory_config().shard_spec.shape
        FusedResblock.validate_output_tiles(out_shard_shape, output_tile)

        logger.debug(f"Placing mcast core at: {FusedResblock.MCAST_CORE}")
        gather_destination_core = device.worker_core_from_logical_core(FusedResblock.MCAST_CORE)
        all_mcast_cores = ttnn.CoreRangeSet({ttnn.CoreRange(FusedResblock.MCAST_CORE, FusedResblock.MCAST_CORE)})

        (
            mm1_full_cb_descriptor,
            weight0_cb_descriptor,
            weight1_cb_descriptor,
            out_cb_descriptor,
            intermediate_pregather_cb_descriptor,
            mm2_full_cb_descriptor,
        ) = FusedResblock.create_cb_descriptors(
            input_tensor,
            weight0,
            weight1,
            output_tensor,
            all_matmul_cores,
            all_mcast_cores,
            num_tiles_k,
            out_dtype,
            output_tile,
        )

        all_cores = all_matmul_cores.merge(all_mcast_cores)
        assert (
            all_cores.size() == all_matmul_cores.size() + all_mcast_cores.size()
        ), "All cores must be the same size as all matmul cores plus one for the mcast core"
        logger.debug(f"All cores: {all_cores}")

        (
            mcast_receiver_semaphore_descriptor,
            mcast_sender_semaphore_descriptor,
        ) = FusedResblock.create_semaphore_descriptors(all_cores)

        runtime_args = FusedResblock.generate_runtime_args(all_matmul_cores)

        input_buffer_address = input_tensor.buffer_address()
        out_tile_size = output_tile.get_tile_size(out_dtype)
        out_tile_descriptor = ttnn.TileDescriptor(output_tile)

        (
            reader_kernel_descriptors,
            writer_kernel_descriptors,
            compute_kernel_descriptor,
            mcast_reader_kernel_descriptor,
            mcast_writer_kernel_descriptor,
            mcast_cb_descriptor,
        ) = FusedResblock.create_kernel_descriptors(
            all_matmul_cores,
            all_mcast_cores,
            gather_destination_core,
            mcast_receiver_semaphore_descriptor,
            mcast_sender_semaphore_descriptor,
            num_tiles_k,
            num_layers,
            fp32_dest_acc_en,
            use_custom_mm,
            out_dtype,
            out_tile_size,
            out_tile_descriptor,
            device,
            debug,
            input_buffer_address,
            runtime_args,
        )

        return ttnn.generic_op(
            [input_tensor, weight0, weight1, output_tensor],
            ttnn.ProgramDescriptor(
                kernels=[
                    *reader_kernel_descriptors,
                    *writer_kernel_descriptors,
                    compute_kernel_descriptor,
                    mcast_reader_kernel_descriptor,
                    mcast_writer_kernel_descriptor,
                ],
                cbs=[
                    mm1_full_cb_descriptor,
                    weight0_cb_descriptor,
                    weight1_cb_descriptor,
                    out_cb_descriptor,
                    intermediate_pregather_cb_descriptor,
                    mm2_full_cb_descriptor,
                    mcast_cb_descriptor,
                ],
                semaphores=[mcast_receiver_semaphore_descriptor, mcast_sender_semaphore_descriptor],
            ),
        )
