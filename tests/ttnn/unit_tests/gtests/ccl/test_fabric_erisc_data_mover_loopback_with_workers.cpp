
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/ttnn/unit_tests/gtests/ccl/test_fabric_edm_common.hpp"

////////////////////////////////////////////////////////////////////
///  MESSAGE COUNT TERMINATION MODE
////////////////////////////////////////////////////////////////////

// Disabled non persistent fabric tests - non-persistent fabric mode not supported
TEST(WorkerFabricEdmDatapath, DISABLED_FabricEDMLoopback_With_Workers_SingleMessage) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender but not receiver buffers
// Disabled non persistent fabric tests - non-persistent fabric mode not supported
TEST(WorkerFabricEdmDatapath, DISABLED_FabricEDMLoopback_With_Workers_2_messages) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 2;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}
// Will wrapp sender but not receiver buffers
// Disabled non persistent fabric tests - non-persistent fabric mode not supported
TEST(WorkerFabricEdmDatapath, DISABLED_FabricEDMLoopback_With_Workers_10_messages) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender and receiver buffers
// Disabled non persistent fabric tests - non-persistent fabric mode not supported
TEST(WorkerFabricEdmDatapath, DISABLED_FabricEDMLoopback_With_Workers_20_messages) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 20;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}

// Disabled non persistent fabric tests - non-persistent fabric mode not supported
TEST(WorkerFabricEdmDatapath, DISABLED_FabricEDMLoopback_With_Workers) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, false);
    ASSERT_EQ(result, 0);
}

// -------------------------
// Persistent Fabric
// -------------------------

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_SingleMessage_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_2_messages_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 2;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}
// Will wrapp sender but not receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_10_messages_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}

// Will wrapp sender and receiver buffers
TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_20_messages_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 20;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}

TEST(WorkerFabricEdmDatapath, FabricEDMLoopback_With_Workers_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;

    auto result = TestLoopbackEntrypoint(page_size, num_pages_total, src_is_dram, dest_is_dram, true);
    ASSERT_EQ(result, 0);
}

////////////////////////////////

// Disabled non persistent fabric tests - non-persistent fabric mode not supported
TEST(WorkerFabricEdmDatapath, DISABLED_LineFabricMcast_SingleMessage_SingleSource) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram, false);

    ASSERT_EQ(result, 0);
}

// Non-functional on harvested parts. Needs testing on unharvested parts.
// Disabled non persistent fabric tests - non-persistent fabric mode not supported
TEST(WorkerFabricEdmDatapath, DISABLED_LineFabricMcast_ManyMessages_SingleSource) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram, false);

    ASSERT_EQ(result, 0);
}

TEST(WorkerFabricEdmDatapath, LineFabricMcast_SingleMessage_SingleSource_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 1;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram, true);

    ASSERT_EQ(result, 0);
}

// Non-functional on harvested parts. Needs testing on unharvested parts.
TEST(WorkerFabricEdmDatapath, LineFabricMcast_ManyMessages_SingleSource_PersistentFabric) {
    const uint32_t page_size = 2048;
    const uint32_t num_pages_total = 10000;
    const bool src_is_dram = true;
    const bool dest_is_dram = true;
    const size_t mcast_first_chip = 1;
    const size_t mcast_last_chip = 3;

    auto result = TestLineFabricEntrypoint(
        mcast_first_chip, mcast_last_chip, page_size, num_pages_total, src_is_dram, dest_is_dram, true);

    ASSERT_EQ(result, 0);
}

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
////               LOCAL CHIP TENSOR READ?WRITE (2 INPUT)
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_SinglePageTile) {
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        ttnn::Shape({1, 1, 32, 32}),
        Layout::TILE,
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0) {
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        ttnn::Shape({1, 1, 32, 64}),
        Layout::TILE,
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded) {
    ttnn::Shape tensor_shape({1, 1, 32, 64});
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}}},
            {tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3]},
            ShardOrientation::ROW_MAJOR,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded1) {
    ttnn::Shape tensor_shape({1, 1, 32, 128});
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}}},
            {tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3]},
            ShardOrientation::ROW_MAJOR,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded2) {
    ttnn::Shape tensor_shape({1, 1, 32, 128});
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}}},
            {tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3] / 4},
            ShardOrientation::ROW_MAJOR,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded3) {
    ttnn::Shape tensor_shape({1, 1, 32, 8192});
    size_t ncores_x = 8;
    size_t ncores_y = 4;
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{ncores_x - 1, ncores_y - 1}}}},
            {tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3] / (ncores_x * ncores_y)},
            ShardOrientation::ROW_MAJOR,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded4) {
    ttnn::Shape tensor_shape({1, 1, 32, 1024});
    size_t ncores_x = 8;
    size_t ncores_y = 4;
    auto mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{ncores_x - 1, ncores_y - 1}}}},
            {tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3] / (ncores_x * ncores_y)},
            ShardOrientation::ROW_MAJOR,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config,
        mem_config,
        mem_config,
        mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded_WithReshard0) {
    ttnn::Shape tensor_shape({1, 1, 32, 128});
    Layout const layout = Layout::TILE;
    auto input_mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{0, 0}}}},
            {tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3]},
            ShardOrientation::ROW_MAJOR,
            ShardMode::LOGICAL));
    auto output_mem_config = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{3, 0}}}},
            {tensor_shape[0] * tensor_shape[1] * tensor_shape[2], tensor_shape[3] / 4},
            ShardOrientation::ROW_MAJOR,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        input_mem_config,
        input_mem_config,
        output_mem_config,
        output_mem_config,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage0_Sharded_WithReshard0_UniquePerStream) {
    ttnn::Shape tensor_shape({1, 1, 32, 128});
    Layout const layout = Layout::TILE;
    size_t in_shard_grid_x = 1;
    size_t in_shard_grid_y = 1;
    size_t out_shard_grid_x = 4;
    size_t out_shard_grid_y = 1;
    auto mem_config0 = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{
                std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{in_shard_grid_x - 1, in_shard_grid_y - 1}}}},
            {tensor_shape[0] * tensor_shape[1] * tensor_shape[2],
             tensor_shape[3] / (in_shard_grid_x * in_shard_grid_y)},
            ShardOrientation::ROW_MAJOR,
            ShardMode::LOGICAL));
    auto mem_config1 = MemoryConfig(
        TensorMemoryLayout::WIDTH_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet{
                std::set<CoreRange>{CoreRange{CoreCoord{0, 0}, CoreCoord{out_shard_grid_x - 1, out_shard_grid_y - 1}}}},
            {tensor_shape[0] * tensor_shape[1] * tensor_shape[2],
             tensor_shape[3] / (out_shard_grid_x * out_shard_grid_y)},
            ShardOrientation::ROW_MAJOR,
            ShardMode::LOGICAL));
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        mem_config0,
        mem_config1,
        mem_config1,
        mem_config0,
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

// Copying even slightly large tensors exposes issues in underlying tensor code
// that isn't under test here
TEST(WorkerCclCommandProcessingKernelLocalMode, MultiInputReader_MultiPage1) {
    ttnn::Shape tensor_shape({1, 1, 256, 256});
    auto pass = RunMultiInputReaderTestPropagateFullTensorIn(
        tensor_shape,
        Layout::TILE,
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        TwoInputReaderKernelWriteMode::LOCAL_WRITEBACK);
    ASSERT_TRUE(pass);
}

// TODO: update the test infra to be able to properly compare tensors if we are only
// doing a slice of the larger tensor

// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////
// ////               FABRIC UNICAST TENSOR WRITE (2 INPUT)
// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////

TEST(WorkerCclCommandProcessingKernelFabricUnicastMode, MultiInputReader_SinglePageTile_OneHop_PersistentFabric) {
    ttnn::Shape tensor_shape({1, 1, 32, 32});
    constexpr size_t distance_dest_device = 1;
    constexpr size_t num_devices = 4;
    Layout const layout = Layout::TILE;
    MemoryConfig const in0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const in1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const out0_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    MemoryConfig const out1_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);

    auto num_elems = std::reduce(tensor_shape.cbegin(), tensor_shape.cend(), 1, std::multiplies<uint32_t>());
    Tensor input_tensor0 =
        ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::UINT32), tensor_shape).to_layout(layout);
    Tensor input_tensor1 =
        ttnn::experimental::view(ttnn::arange(num_elems, 2 * num_elems, 1, DataType::UINT32), tensor_shape)
            .to_layout(layout);
    Tensor output_tensor0 = ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape);
    Tensor output_tensor1 = ttnn::experimental::view(ttnn::ones(tensor_shape, DataType::UINT32, layout), tensor_shape);

    input_tensor0.set_tensor_spec(TensorSpec(
        tensor_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in0_memory_config)));
    input_tensor1.set_tensor_spec(TensorSpec(
        tensor_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), in1_memory_config)));
    output_tensor0.set_tensor_spec(TensorSpec(
        tensor_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out0_memory_config)));
    output_tensor1.set_tensor_spec(TensorSpec(
        tensor_shape, TensorLayout(DataType::UINT32, PageConfig(layout, tt_metal::Tile()), out1_memory_config)));

    size_t page_size = tile_size(DataFormat::RawUInt32);

    ttnn::ccl::Shape4D<uint32_t> tensor_shape_in_pages = shape_to_shape_in_tiles(tensor_shape);
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_shape_in_pages = tensor_shape_in_pages;
    ttnn::ccl::Shape4D<uint32_t> tensor_slice_offset = {0, 0, 0, 0};
    ttnn::ccl::Shape4D<uint32_t> worker_slice_shape = tensor_shape_in_pages;
    ttnn::ccl::Shape4D<uint32_t> worker_slice_offset = {0, 0, 0, 0};

    ttnn::ccl::v2::TensorSlice tensor_slice{
        tensor_shape_in_pages,
        tensor_slice_shape_in_pages,
        tensor_slice_offset,
        worker_slice_shape,
        worker_slice_offset};

    auto const in0_tensor_slice = tensor_slice;
    auto const in1_tensor_slice = tensor_slice;
    auto const out0_tensor_slice = tensor_slice;
    auto const out1_tensor_slice = tensor_slice;

    ttnn::ccl::cmd::CclCommandDestArgs dest_args = ttnn::ccl::cmd::UnicastCommandDestArgs{distance_dest_device, true};
    auto pass = TestMultiInputReaderKernel(
        num_devices,
        input_tensor0,
        in0_memory_config,
        input_tensor1,
        in1_memory_config,
        output_tensor0,
        out0_memory_config,
        output_tensor1,
        out1_memory_config,

        in0_tensor_slice,
        in1_tensor_slice,
        out0_tensor_slice,
        out1_tensor_slice,

        page_size,
        TwoInputReaderKernelWriteMode::FABRIC_UNICAST,
        dest_args,
        true);

    ASSERT_TRUE(pass);
}

// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////
// ////               FABRIC MCAST TENSOR WRITE (2 INPUT)
// ////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////

TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_SinglePageTile_SingleHop_PersistentFabric) {
    ttnn::Shape tensor_shape({1, 1, 32, 32});
    constexpr size_t distance_dest_device = 1;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}

TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_SinglePageTile_TwoHop_PersistentFabric) {
    ttnn::Shape tensor_shape({1, 1, 32, 32});
    constexpr size_t distance_dest_device = 2;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_SinglePageTile_ThreeHop_PersistentFabric) {
    ttnn::Shape tensor_shape({1, 1, 32, 32});
    constexpr size_t distance_dest_device = 3;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}

TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_4PageTile_SingleHop_PersistentFabric) {
    ttnn::Shape tensor_shape({1, 1, 32, 128});
    constexpr size_t distance_dest_device = 1;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, DMultiInputReader_4PageTile_TwoHop_PersistentFabric) {
    ttnn::Shape tensor_shape({1, 1, 128, 32});
    constexpr size_t distance_dest_device = 2;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_4PageTile_ThreeHop_PersistentFabric) {
    ttnn::Shape tensor_shape({1, 1, 64, 64});
    constexpr size_t distance_dest_device = 3;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}
TEST(WorkerCclCommandProcessingKernelFabricMulticastMode, MultiInputReader_lotsPageTile_ThreeHop_PersistentFabric) {
    ttnn::Shape tensor_shape({1, 1, 64, 16384});
    constexpr size_t distance_dest_device = 3;
    constexpr size_t num_devices = 4;
    RunFabricMcastFullTensorPropagateTest(tensor_shape, distance_dest_device, num_devices, true);
}

TEST(WorkerCclCommandProcessingKernels, ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly0) {
    ttnn::Shape tensor_shape({1, 1, 64, 16384});
    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const size_t slices_per_stage = 4;
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<size_t> num_workers_per_stage = {1, 1, 1, 1};

    std::vector<std::vector<size_t>> worker_chunk_read_order = {
        {0, 1, 2, 3},  // first input
        {3, 2, 1, 0},  // read in reverse order
        {2, 0, 3, 1},  // read in non-sequential order
        {1, 2, 3, 0}   // read in non-sequential order
    };
    std::vector<MemoryConfig> mem_configs{
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)};

    auto pass = RunPipelinedWorkersTest(

        tensor_shape,
        split_dim,

        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
        num_stages,
        num_workers_per_stage,
        slices_per_stage,
        data_format,
        page_size_bytes,
        cb_packet_size_in_pages,
        num_packets_per_cb,
        layout,

        worker_chunk_read_order,
        mem_configs);

    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernels, ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly1) {
    ttnn::Shape tensor_shape({1, 1, 64, 128});
    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const size_t slices_per_stage = 4;
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<size_t> num_workers_per_stage = {1, 1, 1, 1};

    std::vector<std::vector<size_t>> worker_chunk_read_order = {
        {0, 1, 2, 3},  // first input
        {3, 2, 1, 0},  // read in reverse order
        {2, 0, 3, 1},  // read in non-sequential order
        {1, 2, 3, 0}   // read in non-sequential order
    };
    std::vector<MemoryConfig> mem_configs{
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)};

    auto pass = RunPipelinedWorkersTest(

        tensor_shape,
        split_dim,

        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
        num_stages,
        num_workers_per_stage,
        slices_per_stage,
        data_format,
        page_size_bytes,
        cb_packet_size_in_pages,
        num_packets_per_cb,
        layout,

        worker_chunk_read_order,
        mem_configs);

    ASSERT_TRUE(pass);
}
TEST(WorkerCclCommandProcessingKernels, ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly2) {
    ttnn::Shape tensor_shape({1, 1, 64, 8192});
    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const size_t slices_per_stage = 2;
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<size_t> num_workers_per_stage = {1, 1, 1, 1};

    std::vector<std::vector<size_t>> worker_chunk_read_order = {
        {0, 1},  // first input
        {1, 0},  // read in reverse order
        {1, 0},  // read in non-sequential order
        {0, 1}   // read in non-sequential order
    };
    std::vector<MemoryConfig> mem_configs{
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)};

    auto pass = RunPipelinedWorkersTest(

        tensor_shape,
        split_dim,

        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
        num_stages,
        num_workers_per_stage,
        slices_per_stage,
        data_format,
        page_size_bytes,
        cb_packet_size_in_pages,
        num_packets_per_cb,
        layout,

        worker_chunk_read_order,
        mem_configs);

    ASSERT_TRUE(pass);
}

// Hits issues with input tensor copy-back
TEST(
    WorkerCclCommandProcessingKernels,
    DISABLED_ChainOfCommandProcessorsWithVaryingDataReadOrders_LocalOnly_SmallSweep) {
    std::vector<ttnn::Shape> tensor_shapes = {
        ttnn::Shape({1, 1, 64, 8192}),
        ttnn::Shape({1, 4, 64, 768}),
        ttnn::Shape({4, 1, 64, 768}),
        ttnn::Shape({4, 4, 64, 768}),
        ttnn::Shape({1, 1, 64, 768}),
        ttnn::Shape({5, 3, 64, 768})};

    const size_t split_dim = 3;

    // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be configurable)
    constexpr size_t num_stages = 4;
    const std::vector<size_t> slices_per_stage_sweep = {2, 3, 4};
    const size_t cb_packet_size_in_pages = 4;
    const size_t num_packets_per_cb = 4;
    auto layout = Layout::TILE;
    const tt::DataFormat data_format = tt::DataFormat::RawUInt32;
    const size_t page_size_bytes = tile_size(DataFormat::RawUInt32);
    std::vector<std::vector<size_t>> num_workers_per_stage_sweep = {
        {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};

    std::vector<std::vector<std::vector<size_t>>> worker_chunk_read_order = {
        {{}},
        {
            {0},
            {0},
            {0},
            {0},
        },
        {
            {0, 1},
            {1, 0},
            {1, 0},
            {0, 1},
        },
        {
            {2, 0, 1},
            {1, 0, 2},
            {0, 1, 2},
            {2, 1, 0},
        },
        {
            {0, 1, 2, 3},  // first input
            {3, 2, 1, 0},  // read in reverse order
            {2, 0, 3, 1},  // read in non-sequential order
            {1, 2, 3, 0}   // read in non-sequential order
        }};
    std::vector<std::vector<MemoryConfig>> mem_configs_sweep = {
        {
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
            MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
        },
        {MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1)},
        {MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)},
        {MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::L1),
         MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)},
    };

    for (auto& tensor_shape : tensor_shapes) {
        for (auto& num_workers_per_stage : num_workers_per_stage_sweep) {
            for (size_t slices_per_stage : slices_per_stage_sweep) {
                for (auto& mem_configs : mem_configs_sweep) {
                    log_info(
                        tt::LogTest,
                        "tensor shape {} and workers stage {} slices_per_stage {}",
                        tensor_shape,
                        num_workers_per_stage,
                        slices_per_stage);
                    auto pass = RunPipelinedWorkersTest(

                        tensor_shape,
                        split_dim,

                        // In this test we will have n stages with anywhere from 1 to 8 workers per stage (this will be
                        // configurable)
                        num_stages,
                        num_workers_per_stage,
                        slices_per_stage,
                        data_format,
                        page_size_bytes,
                        cb_packet_size_in_pages,
                        num_packets_per_cb,
                        layout,

                        worker_chunk_read_order[slices_per_stage],
                        mem_configs);

                    ASSERT_TRUE(pass);
                }
            }
        }
    }
}

TEST(CclAsyncOp, ReduceScatterSmall_PersistentFabric) {
    const size_t dim = 3;
    const size_t num_links = 1;
    constexpr auto layout = Layout::TILE;
    // DEVICES setup
    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    constexpr size_t test_expected_num_devices = 4;
    if (tt::tt_metal::GetNumAvailableDevices() < test_expected_num_devices) {
        log_info("This test can only be run on T3000 devices");
        return;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info("Test must be run on WH");
        return;
    }
    T3000TestDevice test_fixture;
    auto view = test_fixture.mesh_device_->get_view();

    // build a line of devices
    std::vector<IDevice*> devices = {
        view.get_device(MeshCoordinate(0, 1)),
        view.get_device(MeshCoordinate(1, 1)),
        view.get_device(MeshCoordinate(1, 2)),
        view.get_device(MeshCoordinate(0, 2))};
    const size_t num_devices = devices.size();
    TT_FATAL(
        test_expected_num_devices == num_devices,
        "Expected {} devices but got {}",
        test_expected_num_devices,
        num_devices);
    const ttnn::Shape input_shape({1, 1, 32, 32 * num_devices});
    const MemoryConfig in_memory_config = MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM);
    const auto num_elems = input_shape.volume();

    // INPUT TENSOR setup
    size_t page_size = tile_size(DataFormat::Float16);
    std::vector<Tensor> device_input_tensors;
    for (size_t i = 0; i < num_devices; i++) {
        // host_input_tensors.push_back(ttnn::numpy::random::uniform(bfloat16(-1.0f), bfloat16(1.0f) ,
        // {input_shape[0],input_shape[1],input_shape[2],input_shape[3]}, layout).to_device(devices[i]));
        auto t =
            ttnn::experimental::view(ttnn::arange(0, num_elems, 1, DataType::BFLOAT16), input_shape).to_layout(layout);
        t.set_tensor_spec(TensorSpec(
            input_shape, TensorLayout(DataType::BFLOAT16, PageConfig(layout, tt_metal::Tile()), in_memory_config)));

        device_input_tensors.push_back(t.to_device(devices[i]));
    }
    // Need to make it a mesh tensor for use with the op
    const Tensor input_mesh_tensor = ttnn::distributed::aggregate_as_tensor(device_input_tensors, AllGatherTensor{});

    // FABRIC setup
    const bool enable_persistent_fabric = true;

    std::vector<Program> dummy_worker_programs;
    std::optional<SubdeviceInfo> subdevice_managers = std::nullopt;
    std::optional<std::vector<Program>> fabric_programs;
    std::vector<Program*> fabric_program_ptrs;
    std::optional<ttnn::ccl::EdmLineFabricOpInterface> fabric_handle;
    setup_test_with_persistent_fabric(
        devices,
        dummy_worker_programs,
        subdevice_managers,
        fabric_programs,
        fabric_program_ptrs,
        fabric_handle,
        enable_persistent_fabric,
        num_links);

    ttnn::global_semaphore::MultiDeviceGlobalSemaphore from_remote_multi_device_global_semaphore =
        ttnn::global_semaphore::create_global_semaphore_with_same_address(
            test_fixture.mesh_device_.get(),
            devices[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            0,                             // initial value
            tt::tt_metal::BufferType::L1,  // buffer type
            10                             // attempts
        );

    ttnn::global_semaphore::MultiDeviceGlobalSemaphore to_remote_multi_device_global_semaphore =
        ttnn::global_semaphore::create_global_semaphore_with_same_address(
            test_fixture.mesh_device_.get(),
            devices[0]->worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}),
            0,                             // initial value
            tt::tt_metal::BufferType::L1,  // buffer type
            10                             // attempts
        );

    auto output_tensor = ttnn::operations::experimental::ccl::reduce_scatter(
        input_mesh_tensor,
        dim,
        from_remote_multi_device_global_semaphore,
        to_remote_multi_device_global_semaphore,
        ttnn::operations::reduction::ReduceType::Sum,
        tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        ttnn::ccl::Topology::Linear,
        num_links,
        subdevice_managers->worker_subdevice_id.at(devices[0]->id()),
        fabric_handle);

    // wait for op completion
    log_info(tt::LogTest, "Waiting for Op finish");
    std::ranges::for_each(devices, [&](IDevice* d) {
        tt_metal::Finish(d->command_queue(), {subdevice_managers->worker_subdevice_id.at(d->id())});
    });
    log_info(tt::LogTest, "Main op done");

    log_info(tt::LogTest, "Fabric teardown");
    persistent_fabric_teardown_sequence(
        devices, subdevice_managers, fabric_handle.value(), tt::tt_fabric::TerminationSignal::GRACEFULLY_TERMINATE);

    log_info(tt::LogTest, "Waiting for teardown completion");
    for (auto d : devices) {
        tt_metal::Synchronize(d, *ttnn::DefaultQueueId);
    }
    log_info(tt::LogTest, "Finished");
}

TEST(CclAsyncOp, AllGather_PersistentFabric_Dim3_Links1_Shape1_1_32_128) {
    run_all_gather_with_persistent_fabric(3, 1, ttnn::Shape({1, 1, 32, 128}));
}
TEST(CclAsyncOp, AllGather_PersistentFabric_Dim3_Links1_Shape1_1_32_8192) {
    run_all_gather_with_persistent_fabric(3, 1, ttnn::Shape({1, 1, 32, 8192}));
}
// Mesh device setup seems to not provide the correct configuration for multi-link? To be investigated
TEST(CclAsyncOp, DISABLED_AllGather_PersistentFabric_Dim3_Links2_Shape1_1_32_128) {
    run_all_gather_with_persistent_fabric(3, 2, ttnn::Shape({1, 1, 32, 128}));
}
// Mesh device setup seems to not provide the correct configuration for multi-link? To be investigated
TEST(CclAsyncOp, DISABLED_AllGather_PersistentFabric_Dim3_Links2_Shape1_1_32_8192) {
    run_all_gather_with_persistent_fabric(3, 2, ttnn::Shape({1, 1, 32, 8192}));
}

TEST(CclAsyncOp, RingAllGather_PersistentFabric_Dim3_Links1_Shape1_256_32_8192) {
    run_ring_all_gather_with_persistent_fabric(3, 1, ttnn::Shape({1, 256, 32, 8192}));
}

TEST(EdmFabric, BasicMcastThroughputTest_SingleLink_LineSize2_SingleMcast) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    params.line_size = 2;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SingleMcast) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_SingleWorker_2Device) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_2Device) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_SingleWorker_4Device) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 4;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_TwoWorkers_4Device) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 4;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 2;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap_SingleWorker_2Device) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap_2Device) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwiceFilled_ReceiverOnceFilled_2Device) {
    const size_t num_mcasts = 18;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwiceFilled_ReceiverOnceFilled) {
    const size_t num_mcasts = 18;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwoWrap_ReceiverOneWrap) {
    const size_t num_mcasts = 19;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SingleLink_LineSize2_SingleMcast_LineSync) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SingleMcast_LineSync) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFullNoWrap_ReceiverNoWrap_LineSync) {
    const size_t num_mcasts = 9;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap_2Device_LineSync) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderOneElemWrap_ReceiverNoWrap_LineSync) {
    const size_t num_mcasts = 10;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwiceFilled_ReceiverOnceFilled_2Device_LineSync) {
    const size_t num_mcasts = 18;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwiceFilled_ReceiverOnceFilled_LineSync) {
    const size_t num_mcasts = 18;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_SenderFourTImesFilled_ReceiverTwiceFilled_2Device_1Worker) {
    const size_t num_mcasts = 36;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    params.num_devices_with_workers = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFourTImesFilled_ReceiverTwiceFilled_2Device_LineSync) {
    const size_t num_mcasts = 36;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = line_size;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderFourTImesFilled_ReceiverTwiceFilled_LineSync) {
    const size_t num_mcasts = 36;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SenderTwoWrap_ReceiverOneWrap_LineSync) {
    const size_t num_mcasts = 19;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SmallPerf_2Device) {
    const size_t num_mcasts = 70;
    const size_t num_unicasts = 0;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const size_t line_size = 2;
    const bool report_performance = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = report_performance;
    params.line_size = line_size;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, BasicMcastThroughputTest_SmallPerf0) {
    const size_t num_mcasts = 70;
    const size_t num_unicasts = 0;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = true;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_SmallPerf1) {
    const size_t num_mcasts = 70;
    const size_t num_unicasts = 0;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = true;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_0) {
    const size_t num_mcasts = 100;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_size = 2;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_1) {
    const size_t num_mcasts = 1000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = false;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_2) {
    const size_t num_mcasts = 50000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;

    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_3_SingleLink) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 0;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_3) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_3_onehop) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 1;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    params.line_size = 2;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_4) {
    const size_t num_mcasts = 800000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}

TEST(EdmFabric, BasicMcastThroughputTest_5) {
    const size_t num_mcasts = 1;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 20000;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_6) {
    const size_t num_mcasts = 100;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 8000;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_7) {
    const size_t num_mcasts = 1000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1000;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_8) {
    const size_t num_mcasts = 50000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 200;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_9) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 150;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
// DISABLED due to long runtime
TEST(EdmFabric, DISABLED_BasicMcastThroughputTest_10) {
    const size_t num_mcasts = 800000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 50;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_6_Short) {
    const size_t num_mcasts = 100;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 100;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_7_Short) {
    const size_t num_mcasts = 1000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 50;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_8_Short) {
    const size_t num_mcasts = 50000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 20;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_9_Short) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 10;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}
TEST(EdmFabric, BasicMcastThroughputTest_10_Short) {
    const size_t num_mcasts = 800000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 5;
    RunWriteThroughputStabilityTestWithPersistentFabric(num_mcasts, num_unicasts, num_links, num_op_invocations);
}

TEST(EdmFabric, BasicMcastThroughputTest_0_WithLineSync) {
    const size_t num_mcasts = 100;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_1_WithLineSync) {
    const size_t num_mcasts = 1000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_2_WithLineSync) {
    const size_t num_mcasts = 50000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_3_WithLineSync) {
    const size_t num_mcasts = 200000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}
TEST(EdmFabric, BasicMcastThroughputTest_4_WithLineSync) {
    const size_t num_mcasts = 800000;
    const size_t num_unicasts = 2;
    const size_t num_links = 2;
    const size_t num_op_invocations = 1;
    const bool line_sync = true;
    WriteThroughputStabilityTestWithPersistentFabricParams params;
    params.line_sync = line_sync;
    RunWriteThroughputStabilityTestWithPersistentFabric(
        num_mcasts, num_unicasts, num_links, num_op_invocations, params);
}

TEST(EdmFabric, RingDeadlockStabilityTest) {
    const size_t num_mcasts = 200000;
    const size_t num_links = 1;
    const size_t num_op_invocations = 5;
    const bool line_sync = true;
    log_trace(tt::LogTest, "Running RingDeadlockStabilityTest with forward mcast only");
    RunRingDeadlockStabilityTestWithPersistentFabric(num_mcasts, num_links, num_op_invocations, true, false);
    log_trace(tt::LogTest, "Running RingDeadlockStabilityTest with backward mcast only");
    RunRingDeadlockStabilityTestWithPersistentFabric(num_mcasts, num_links, num_op_invocations, false, true);
    log_trace(tt::LogTest, "Running RingDeadlockStabilityTest with forward and backward mcast");
    RunRingDeadlockStabilityTestWithPersistentFabric(num_mcasts, num_links, num_op_invocations, true, true);
}
