// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::dram {
// Test config, i.e. test parameters
struct DramConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreRangeSet cores = CoreRangeSet();
    array<uint32_t, 2> tensor_shape_in_pages = {0, 0};
    array<uint32_t, 2> num_dram_banks = {0, 0};
};

/// @brief Does Dram --> Reader --> L1 CB --> Writer --> Dram.
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const DramConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // DRAM Buffers
    const size_t total_size_bytes =
        test_config.num_of_transactions * test_config.transaction_size_pages * test_config.page_size_bytes;

    InterleavedBufferConfig interleaved_dram_config{
        .device = device, .size = total_size_bytes, .page_size = total_size_bytes, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> input_dram_buffer;
    if (!test_config.num_dram_banks[0]) {
        input_dram_buffer = CreateBuffer(interleaved_dram_config);
    } else {
        ShardSpecBuffer shard_spec = ShardSpecBuffer(
            test_config.cores,
            {test_config.tensor_shape_in_pages[0] * 4 / test_config.num_dram_banks[0],
             test_config.tensor_shape_in_pages[1] * (device->arch() == ARCH::BLACKHOLE ? 8 : 4) /
                 test_config.num_dram_banks[1]},
            ShardOrientation::ROW_MAJOR,
            {4, (device->arch() == ARCH::BLACKHOLE) ? 8 : 4},
            test_config.tensor_shape_in_pages);
        ShardedBufferConfig sharded_dram_config{
            .device = device,
            .size = total_size_bytes,
            .page_size = test_config.page_size_bytes,
            .buffer_type = BufferType::DRAM,
            .buffer_layout = TensorMemoryLayout::BLOCK_SHARDED,
            .shard_parameters = shard_spec};
        input_dram_buffer = CreateBuffer(sharded_dram_config);
    }
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto output_dram_buffer = CreateBuffer(interleaved_dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / bfloat16::SIZEOF, chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    uint8_t l1_cb_index = CBIndex::c_0;

    // Compile-time arguments for kernels
    vector<uint32_t> reader_compile_args = {
        (uint32_t)input_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id};

    vector<uint32_t> writer_compile_args = {
        (uint32_t)output_dram_byte_address,
        (uint32_t)0,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)l1_cb_index,
        (uint32_t)test_config.test_id};

    // Create circular buffers
    CircularBufferConfig l1_cb_config =
        CircularBufferConfig(total_size_bytes, {{l1_cb_index, test_config.l1_data_format}})
            .set_page_size(l1_cb_index, total_size_bytes);
    auto l1_cb = CreateCircularBuffer(program, test_config.cores, l1_cb_config);

    // Kernels
    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/dram_unary/kernels/reader_unary.cpp",
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/dram_unary/kernels/writer_unary.cpp",
        test_config.cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program and record outputs
    vector<uint32_t> packed_output;
    detail::WriteToBuffer(input_dram_buffer, packed_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());
    detail::LaunchProgram(device, program);
    detail::ReadFromBuffer(output_dram_buffer, packed_output);

    // Results comparison
    bool pcc = is_close_packed_vectors<bfloat16, uint32_t>(
        packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });

    if (!pcc) {
        log_error(tt::LogTest, "PCC Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(tt::LogTest, "Output vector");
        print_vector<uint32_t>(packed_output);
    }

    return pcc;
}
}  // namespace unit_tests::dm::dram

/* ========== Test case for varying transaction numbers and sizes; Test id = 0 ========== */
TEST_F(DeviceFixture, TensixDataMovementDRAMInterleavedPacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;           // Bound for testing different number of transactions
    uint32_t max_transaction_size_pages = 64;  // Bound for testing different transaction sizes
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            if (num_of_transactions * transaction_size_pages * page_size_bytes >= 1024 * 1024) {
                continue;
            }

            // Test config
            unit_tests::dm::dram::DramConfig test_config = {
                .test_id = 0,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set};

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for varying core locations; Test id = 1 ========== */
TEST_F(DeviceFixture, TensixDataMovementDRAMInterleavedCoreLocations) {
    uint32_t num_of_transactions = 128;     // Bound for testing different number of transactions
    uint32_t transaction_size_pages = 128;  // Bound for testing different transaction sizes
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH

    for (unsigned int id = 0; id < num_devices_; id++) {
        // Cores
        auto grid_size = devices_.at(id)->compute_with_storage_grid_size();
        log_info(tt::LogTest, "Grid size x: {}, y: {}", grid_size.x, grid_size.y);

        for (unsigned int x = 0; x < grid_size.x; x++) {
            for (unsigned int y = 0; y < grid_size.y; y++) {
                CoreRangeSet core_range_set(CoreRange({x, y}, {x, y}));

                // Test config
                unit_tests::dm::dram::DramConfig test_config = {
                    .test_id = 1,
                    .num_of_transactions = num_of_transactions,
                    .transaction_size_pages = transaction_size_pages,
                    .page_size_bytes = page_size_bytes,
                    .l1_data_format = DataFormat::Float16_b,
                    .cores = core_range_set};

                // Run
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Sharded dram buffer test; Test id = 2 ========== */
TEST_F(DeviceFixture, TensixDataMovementDRAMSharded) {
    // Parameters
    uint32_t max_tensor_dim_pages = 1;  // Arbitrary tensor for sharding
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH

    // 2 * 1024 * 1024 * 1024   = dram bank size / max shard size
    // x * x * 64        = shard size where x is one dim of tensor_shape_in_pages
    // x * x <= 1024 * 1024 * 32 this many pages should be able to fit on one dram bank
    // max x => 1024 * 4 = 4096
    // Fails when one dram bank size is set to 4GB (DRAM error)

    // L1 capacity is 1 MB, error when shard size cannot exceed that (L1 error)
    // So 1024 * 1024 / 64 = x * x = 128 * 128 => x = 128

    // Fails: (when num dram banks isnt 1, 1), possibly due to the noc_addr_from_bank_id function in kernel
    // TODO: Expand test case to cover multiple dram banks

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    uint32_t transaction_size_pages = 1;
    for (uint32_t tensor_dim_size = 1; tensor_dim_size <= max_tensor_dim_pages; tensor_dim_size *= 2) {
        array<uint32_t, 2> tensor_shape_in_pages = {tensor_dim_size, tensor_dim_size};
        uint32_t num_of_transactions = tensor_dim_size * tensor_dim_size / transaction_size_pages;
        for (uint32_t dram_banks_dim_ratio = 1; dram_banks_dim_ratio <= tensor_dim_size; dram_banks_dim_ratio *= 2) {
            uint32_t dram_banks_dim_size = tensor_dim_size / dram_banks_dim_ratio;
            array<uint32_t, 2> num_dram_banks = {dram_banks_dim_size, dram_banks_dim_size};

            // Test config
            unit_tests::dm::dram::DramConfig test_config = {
                .test_id = 2,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .cores = core_range_set,
                .tensor_shape_in_pages = tensor_shape_in_pages,
                .num_dram_banks = num_dram_banks};

            log_info(tt::LogTest, "Tensor shape in pages: {}", tensor_shape_in_pages);
            log_info(tt::LogTest, "Number of dram banks: {}", num_dram_banks);

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Directed ideal test case; Test id = 3 ========== */
TEST_F(DeviceFixture, TensixDataMovementDRAMDirectedIdeal) {
    // Parameters
    uint32_t num_of_transactions = 180;
    uint32_t transaction_size_pages = 4 * 32;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    // Max transaction size = 4 * 32 pages = 128 * 32 bytes = 4096 bytes for WH; 8192 bytes for BH
    // Max total transaction size = 180 * 8192 bytes = 1474560 bytes = 1.4 MB = L1 capacity

    // Cores
    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});

    // Test config
    unit_tests::dm::dram::DramConfig test_config = {
        .test_id = 3,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = transaction_size_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .cores = core_range_set};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
