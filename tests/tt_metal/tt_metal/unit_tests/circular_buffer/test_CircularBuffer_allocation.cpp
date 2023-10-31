// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include "circular_buffer_test_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"

using namespace tt::tt_metal;

namespace basic_tests::circular_buffer {

void validate_cb_address(Program &program, Device *device, const CoreRangeSet &cr_set, const std::map<CoreCoord, std::map<uint8_t, uint32_t>> &core_to_address_per_buffer_index) {
    LaunchProgram(device, program);

    vector<u32> cb_config_vector;
    u32 cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32);

    for (const CoreRange &core_range : cr_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord core_coord{.x = x, .y = y};
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device, core_coord, CIRCULAR_BUFFER_CONFIG_BASE, cb_config_buffer_size, cb_config_vector);

                std::map<uint8_t, uint32_t> address_per_buffer_index = core_to_address_per_buffer_index.at(core_coord);

                for (const auto &[buffer_index, expected_address] : address_per_buffer_index) {
                    auto base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * buffer_index;
                    EXPECT_EQ(expected_address >> 4, cb_config_vector.at(base_index));
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TestCircularBuffersSequentiallyPlaced) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    CBConfig cb_config;
    CoreCoord core({.x = 0, .y = 0});
    CoreRange cr = {.start = core, .end = core};
    CoreRangeSet cr_set({cr});

    std::map<uint8_t, uint32_t> expected_addresses;
    auto expected_cb_addr = L1_UNRESERVED_BASE;
    for (uint8_t cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{cb_id, cb_config.data_format}}).set_page_size(cb_id, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core, config1);
        expected_addresses[cb_id] = expected_cb_addr;
        expected_cb_addr += cb_config.page_size;
    }

    initialize_program(program, cr_set);

    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses = {
        {core, expected_addresses}
    };

    validate_cb_address(program, this->devices_.at(id), cr_set, golden_addresses);
}
}

TEST_F(DeviceFixture, TestCircularBufferSequentialAcrossAllCores) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    CBConfig cb_config;

    CoreCoord core0{.x = 0, .y = 0};
    CoreCoord core1{.x = 0, .y = 1};
    CoreCoord core2{.x = 0, .y = 2};

    const static std::map<CoreCoord, u32> core_to_num_cbs = {{core0, 3}, {core1, 0}, {core2, 5}};
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;

    u32 max_num_cbs = 0;
    for (const auto &[core, num_cbs] : core_to_num_cbs) {
        auto expected_cb_addr = L1_UNRESERVED_BASE;
        max_num_cbs = std::max(max_num_cbs, num_cbs);
        std::map<uint8_t, uint32_t> expected_addresses;
        for (u32 buffer_id = 0; buffer_id < num_cbs; buffer_id++) {
            CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_id, cb_config.data_format}}).set_page_size(buffer_id, cb_config.page_size);
            auto cb = CreateCircularBuffer(program, core, config1);
            expected_addresses[buffer_id] = expected_cb_addr;
            expected_cb_addr += cb_config.page_size;
        }
        golden_addresses_per_core[core] = expected_addresses;
    }

    CoreRange cr = {.start = core0, .end = core2};
    CoreRangeSet cr_set({cr});

    auto expected_multi_core_address = L1_UNRESERVED_BASE + (max_num_cbs * cb_config.page_size);
    uint8_t multicore_buffer_idx = NUM_CIRCULAR_BUFFERS - 1;
    CircularBufferConfig config2 = CircularBufferConfig(cb_config.page_size, {{multicore_buffer_idx, cb_config.data_format}}).set_page_size(multicore_buffer_idx, cb_config.page_size);
    auto multi_core_cb = CreateCircularBuffer(program, cr_set, config2);
    golden_addresses_per_core[core0][multicore_buffer_idx] = expected_multi_core_address;
    golden_addresses_per_core[core1][multicore_buffer_idx] = expected_multi_core_address;
    golden_addresses_per_core[core2][multicore_buffer_idx] = expected_multi_core_address;

    initialize_program(program, cr_set);
    validate_cb_address(program, this->devices_.at(id), cr_set, golden_addresses_per_core);
}
}

TEST_F(DeviceFixture, TestValidCircularBufferAddress) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    CBConfig cb_config;

    auto buffer_size = cb_config.page_size;
    auto l1_buffer = CreateBuffer(this->devices_.at(id), buffer_size, buffer_size, BufferStorage::L1);

    CoreRange cr = {.start = {0, 0}, .end = {0, 2}};
    CoreRangeSet cr_set({cr});
    std::vector<uint8_t> buffer_indices = {16, 24};

    u32 expected_cb_addr = l1_buffer.address();
    CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_indices[0], cb_config.data_format}, {buffer_indices[1], cb_config.data_format}}, expected_cb_addr)
        .set_page_size(buffer_indices[0], cb_config.page_size)
        .set_page_size(buffer_indices[1], cb_config.page_size);
    auto multi_core_cb = CreateCircularBuffer(program, cr_set, config1);

    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    for (const CoreRange &core_range : cr_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord core_coord{.x = x, .y = y};
                for (uint8_t buffer_index : buffer_indices) {
                    golden_addresses_per_core[core_coord][buffer_index] = expected_cb_addr;
                }
            }
        }
    }

    initialize_program(program, cr_set);
    validate_cb_address(program, this->devices_.at(id), cr_set, golden_addresses_per_core);
}
}

TEST_F(DeviceFixture, TestInvalidCircularBufferAddress) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    CBConfig cb_config;

    CoreCoord core0{.x = 0, .y = 0};
    const static u32 core0_num_cbs = 3;
    for (u32 buffer_id = 0; buffer_id < core0_num_cbs; buffer_id++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_id, cb_config.data_format}}).set_page_size(buffer_id, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core0, config1);
    }

    CoreRange cr = {.start = {0, 0}, .end = {0, 1}};
    CoreRangeSet cr_set({cr});

    constexpr u32 multi_core_cb_index = core0_num_cbs + 1;
    uint32_t invalid_requested_address = L1_UNRESERVED_BASE;

    CircularBufferConfig config2 = CircularBufferConfig(cb_config.page_size, {{multi_core_cb_index, cb_config.data_format}}, invalid_requested_address).set_page_size(multi_core_cb_index, cb_config.page_size);
    auto cb2 = CreateCircularBuffer(program, cr_set, config2);

    initialize_program(program, cr_set);
    EXPECT_ANY_THROW(LaunchProgram(this->devices_.at(id), program));
}
}

TEST_F(DeviceFixture, TestCircularBuffersAndL1BuffersCollision) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    uint32_t page_size = TileSize(tt::DataFormat::Float16_b);

    auto buffer_size = page_size * 128;
    auto l1_buffer = CreateBuffer(this->devices_.at(id), buffer_size, buffer_size, BufferStorage::L1);

    // L1 buffer is entirely in bank 0
    auto core = l1_buffer.logical_core_from_bank_id(0);
    CoreRange cr = {.start = core, .end = core};
    CoreRangeSet cr_set({cr});
    initialize_program(program, cr_set);

    uint32_t num_pages = (l1_buffer.address() - L1_UNRESERVED_BASE) / NUM_CIRCULAR_BUFFERS / page_size + 1;
    CBConfig cb_config = {.num_pages=num_pages};
    for (u32 buffer_id = 0; buffer_id < NUM_CIRCULAR_BUFFERS; buffer_id++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size * cb_config.num_pages, {{buffer_id, cb_config.data_format}}).set_page_size(buffer_id, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core, config1);
    }

    detail::CompileProgram(this->devices_.at(id), program);
    EXPECT_ANY_THROW(detail::ConfigureDeviceWithProgram(this->devices_.at(id), program));
}
}

TEST_F(DeviceFixture, TestValidUpdateCircularBufferSize) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    CBConfig cb_config;
    CoreCoord core0{.x = 0, .y = 0};
    CoreRange cr = {.start = core0, .end = core0};
    CoreRangeSet cr_set({cr});

    initialize_program(program, cr_set);

    const u32 core0_num_cbs = 2;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    std::vector<CircularBufferID> cb_ids;
    auto expected_cb_addr = L1_UNRESERVED_BASE;
    for (u32 buffer_idx = 0; buffer_idx < core0_num_cbs; buffer_idx++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_idx, cb_config.data_format}}).set_page_size(buffer_idx, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core0, config1);
        golden_addresses_per_core[core0][buffer_idx] = expected_cb_addr;
        cb_ids.push_back(cb);
        expected_cb_addr += cb_config.page_size;
    }

    validate_cb_address(program, this->devices_.at(id), cr_set, golden_addresses_per_core);

    // Update size of the first CB
    GetCircularBufferConfig(program, cb_ids[0]).set_total_size(cb_config.page_size * 2);
    golden_addresses_per_core[core0][0] = L1_UNRESERVED_BASE;
    golden_addresses_per_core[core0][1] = (L1_UNRESERVED_BASE + (cb_config.page_size * 2));

    validate_cb_address(program, this->devices_.at(id), cr_set, golden_addresses_per_core);
}
}

TEST_F(DeviceFixture, TestInvalidUpdateCircularBufferSize) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    CBConfig cb_config;
    CoreCoord core0{.x = 0, .y = 0};
    CoreRange cr = {.start = core0, .end = core0};
    CoreRangeSet cr_set({cr});

    initialize_program(program, cr_set);

    const u32 core0_num_cbs = 2;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    std::vector<CircularBufferID> cb_ids;
    auto expected_cb_addr = L1_UNRESERVED_BASE;
    for (u32 buffer_idx = 0; buffer_idx < core0_num_cbs; buffer_idx++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_idx, cb_config.data_format}}).set_page_size(buffer_idx, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core0, config1);
        golden_addresses_per_core[core0][buffer_idx] = expected_cb_addr;
        cb_ids.push_back(cb);
        expected_cb_addr += cb_config.page_size;
    }

    validate_cb_address(program, this->devices_.at(id), cr_set, golden_addresses_per_core);

    // Update size of the first CB
    GetCircularBufferConfig(program, cb_ids[0]).set_total_size(cb_config.page_size / 2);
    EXPECT_ANY_THROW(LaunchProgram(this->devices_.at(id), program));
}
}

TEST_F(DeviceFixture, TestUpdateCircularBufferAddress) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    CBConfig cb_config;
    CoreCoord core0{.x = 0, .y = 0};
    CoreRange cr = {.start = core0, .end = core0};
    CoreRangeSet cr_set({cr});

    auto buffer_size = cb_config.page_size;
    auto l1_buffer = CreateBuffer(this->devices_.at(id), buffer_size, buffer_size, BufferStorage::L1);

    initialize_program(program, cr_set);

    const u32 core0_num_cbs = 2;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    std::vector<CircularBufferID> cb_ids;
    auto expected_cb_addr = L1_UNRESERVED_BASE;
    for (u32 buffer_idx = 0; buffer_idx < core0_num_cbs; buffer_idx++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_idx, cb_config.data_format}}).set_page_size(buffer_idx, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core0, config1);
        golden_addresses_per_core[core0][buffer_idx] = expected_cb_addr;
        cb_ids.push_back(cb);
        expected_cb_addr += cb_config.page_size;
    }

    validate_cb_address(program, this->devices_.at(id), cr_set, golden_addresses_per_core);
    // Update address of the first CB
    GetCircularBufferConfig(program, cb_ids[0]).set_globally_allocated_address(l1_buffer.address());
    golden_addresses_per_core[core0][0] = l1_buffer.address();
    golden_addresses_per_core[core0][1] = (L1_UNRESERVED_BASE);
    validate_cb_address(program, this->devices_.at(id), cr_set, golden_addresses_per_core);
}
}

TEST_F(DeviceFixture, TestUpdateCircularBufferPageSize) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    CBConfig cb_config;
    CoreCoord core0{.x = 0, .y = 0};
    CoreRange cr = {.start = core0, .end = core0};
    CoreRangeSet cr_set({cr});

    initialize_program(program, cr_set);

    const u32 core0_num_cbs = 2;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_num_pages_per_core;
    std::vector<CircularBufferID> cb_ids;
    auto expected_cb_addr = L1_UNRESERVED_BASE;
    for (u32 buffer_idx = 0; buffer_idx < core0_num_cbs; buffer_idx++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_idx, cb_config.data_format}}).set_page_size(buffer_idx, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core0, config1);
        golden_addresses_per_core[core0][buffer_idx] = expected_cb_addr;
        golden_num_pages_per_core[core0][buffer_idx] = 1;
        cb_ids.push_back(cb);
        expected_cb_addr += cb_config.page_size;
    }

    LaunchProgram(this->devices_.at(id), program);

    vector<u32> cb_config_vector;
    u32 cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32);

    for (const CoreRange &core_range : cr_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord core_coord{.x = x, .y = y};
                tt::tt_metal::detail::ReadFromDeviceL1(
                    this->devices_.at(id), core_coord, CIRCULAR_BUFFER_CONFIG_BASE, cb_config_buffer_size, cb_config_vector);

                std::map<uint8_t, uint32_t> address_per_buffer_index = golden_addresses_per_core.at(core_coord);
                std::map<uint8_t, uint32_t> num_pages_per_buffer_index = golden_num_pages_per_core.at(core_coord);

                for (const auto &[buffer_index, expected_address] : address_per_buffer_index) {
                    auto base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * buffer_index;
                    EXPECT_EQ(expected_address >> 4, cb_config_vector.at(base_index)); // address validation
                    EXPECT_EQ(num_pages_per_buffer_index.at(buffer_index), cb_config_vector.at(base_index + 2)); // num pages validation
                }
            }
        }
    }

    GetCircularBufferConfig(program, cb_ids[1]).set_page_size(1, cb_config.page_size / 2);
    golden_num_pages_per_core[core0][1] = 2;

    LaunchProgram(this->devices_.at(id), program);

    // addresses should not be changed
    for (const CoreRange &core_range : cr_set.ranges()) {
        for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
            for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                CoreCoord core_coord{.x = x, .y = y};
                tt::tt_metal::detail::ReadFromDeviceL1(
                    this->devices_.at(id), core_coord, CIRCULAR_BUFFER_CONFIG_BASE, cb_config_buffer_size, cb_config_vector);

                std::map<uint8_t, uint32_t> address_per_buffer_index = golden_addresses_per_core.at(core_coord);
                std::map<uint8_t, uint32_t> num_pages_per_buffer_index = golden_num_pages_per_core.at(core_coord);

                for (const auto &[buffer_index, expected_address] : address_per_buffer_index) {
                    auto base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * buffer_index;
                    EXPECT_EQ(expected_address >> 4, cb_config_vector.at(base_index)); // address validation
                    EXPECT_EQ(num_pages_per_buffer_index.at(buffer_index), cb_config_vector.at(base_index + 2)); // num pages validation
                }
            }
        }
    }
}
}

TEST_F(DeviceFixture, TestDataCopyWithUpdatedCircularBufferConfig) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Program program;
    CoreCoord core{.x = 0, .y = 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2;
    uint32_t buffer_size = single_tile_size * num_tiles;

    auto src_dram_buffer = CreateBuffer(this->devices_.at(id), buffer_size, buffer_size, BufferStorage::DRAM);
    auto dst_dram_buffer = CreateBuffer(this->devices_.at(id), buffer_size, buffer_size, BufferStorage::DRAM);
    auto global_cb_buffer = CreateBuffer(this->devices_.at(id), buffer_size, buffer_size, BufferStorage::L1);

    uint32_t cb_index = 0;
    uint32_t num_input_tiles = num_tiles;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(buffer_size, {{cb_index, tt::DataFormat::Float16_b}}).set_page_size(cb_index, single_tile_size);
    auto cb_src0 = CreateCircularBuffer(program, core, cb_src0_config);

    auto reader_kernel = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {cb_index}});

    auto writer_kernel = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {cb_index}});

    SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)src_dram_buffer.address(),
            (uint32_t)src_dram_buffer.noc_coordinates().x,
            (uint32_t)src_dram_buffer.noc_coordinates().y,
            (uint32_t)num_tiles,
        });
    SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)dst_dram_buffer.address(),
            (uint32_t)dst_dram_buffer.noc_coordinates().x,
            (uint32_t)dst_dram_buffer.noc_coordinates().y,
            (uint32_t)num_tiles,
        });

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    WriteToBuffer(src_dram_buffer, src_vec);

    LaunchProgram(this->devices_.at(id), program);

    std::vector<uint32_t> result_vec;
    ReadFromBuffer(dst_dram_buffer, result_vec);
    EXPECT_EQ(src_vec, result_vec);

    std::vector<uint32_t> input_cb_data;
    uint32_t cb_address = L1_UNRESERVED_BASE;
    detail::ReadFromDeviceL1(this->devices_.at(id), core, L1_UNRESERVED_BASE, buffer_size, input_cb_data);
    EXPECT_EQ(src_vec, input_cb_data);

    // update cb address
    GetCircularBufferConfig(program, cb_src0).set_globally_allocated_address(global_cb_buffer.address());

    // zero out dst buffer
    std::vector<uint32_t> zero_vec = create_constant_vector_of_bfloat16(buffer_size, 0);
    WriteToBuffer(dst_dram_buffer, zero_vec);

    // relaunch program
    LaunchProgram(this->devices_.at(id), program);

    std::vector<uint32_t> second_result_vec;
    ReadFromBuffer(dst_dram_buffer, second_result_vec);
    EXPECT_EQ(src_vec, second_result_vec);

    std::vector<uint32_t> second_cb_data;
    detail::ReadFromDeviceL1(this->devices_.at(id), core, global_cb_buffer.address(), buffer_size, second_cb_data);
    EXPECT_EQ(src_vec, second_cb_data);
}
}

}   // end namespace basic_tests::circular_buffer
