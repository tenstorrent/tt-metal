// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include "circular_buffer_test_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/program/program_pool.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"

using namespace tt::tt_metal;

namespace basic_tests::circular_buffer {

void validate_cb_address(Program &program, Device *device, const CoreRangeSet &cr_set, const std::map<CoreCoord, std::map<uint8_t, uint32_t>> &core_to_address_per_buffer_index) {
    detail::LaunchProgram(device, program);

    vector<uint32_t> cb_config_vector;
    uint32_t cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

    for (const CoreRange &core_range : cr_set.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core_coord(x, y);
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device, core_coord, program.get_cb_base_addr(device, core_coord, CoreType::WORKER), cb_config_buffer_size, cb_config_vector);

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
    auto program = tt::tt_metal::CreateScopedProgram();
    CBConfig cb_config;
    CoreCoord core(0, 0);
    CoreRange cr(core, core);
    CoreRangeSet cr_set({cr});

    std::map<uint8_t, uint32_t> expected_addresses;
    auto expected_cb_addr = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
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

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    validate_cb_address(*program_ptr, this->devices_.at(id), cr_set, golden_addresses);
}
}

TEST_F(DeviceFixture, TestCircularBufferSequentialAcrossAllCores) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    auto program = tt::tt_metal::CreateScopedProgram();
    CBConfig cb_config;

    CoreCoord core0(0, 0);
    CoreCoord core1(0, 1);
    CoreCoord core2(0, 2);

    const static std::map<CoreCoord, uint32_t> core_to_num_cbs = {{core0, 3}, {core1, 0}, {core2, 5}};
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;

    uint32_t max_num_cbs = 0;
    for (const auto &[core, num_cbs] : core_to_num_cbs) {
        auto expected_cb_addr = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
        max_num_cbs = std::max(max_num_cbs, num_cbs);
        std::map<uint8_t, uint32_t> expected_addresses;
        for (uint32_t buffer_id = 0; buffer_id < num_cbs; buffer_id++) {
            CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_id, cb_config.data_format}}).set_page_size(buffer_id, cb_config.page_size);
            auto cb = CreateCircularBuffer(program, core, config1);
            expected_addresses[buffer_id] = expected_cb_addr;
            expected_cb_addr += cb_config.page_size;
        }
        golden_addresses_per_core[core] = expected_addresses;
    }

    CoreRange cr(core0, core2);
    CoreRangeSet cr_set({cr});

    auto expected_multi_core_address = devices_.at(id)->get_base_allocator_addr(HalMemType::L1) + (max_num_cbs * cb_config.page_size);
    uint8_t multicore_buffer_idx = NUM_CIRCULAR_BUFFERS - 1;
    CircularBufferConfig config2 = CircularBufferConfig(cb_config.page_size, {{multicore_buffer_idx, cb_config.data_format}}).set_page_size(multicore_buffer_idx, cb_config.page_size);
    auto multi_core_cb = CreateCircularBuffer(program, cr_set, config2);
    golden_addresses_per_core[core0][multicore_buffer_idx] = expected_multi_core_address;
    golden_addresses_per_core[core1][multicore_buffer_idx] = expected_multi_core_address;
    golden_addresses_per_core[core2][multicore_buffer_idx] = expected_multi_core_address;

    initialize_program(program, cr_set);
    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    validate_cb_address(*program_ptr, this->devices_.at(id), cr_set, golden_addresses_per_core);
}
}

TEST_F(DeviceFixture, TestValidCircularBufferAddress) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    auto program = tt::tt_metal::CreateScopedProgram();
    CBConfig cb_config;

    auto buffer_size = cb_config.page_size;
    tt::tt_metal::InterleavedBufferConfig buff_config{
                    .device=this->devices_.at(id),
                    .size = buffer_size,
                    .page_size = buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::L1
        };
    auto l1_buffer = CreateBuffer(buff_config);

    CoreRange cr({0, 0}, {0, 2});
    CoreRangeSet cr_set({cr});
    std::vector<uint8_t> buffer_indices = {16, 24};

    uint32_t expected_cb_addr = l1_buffer->address();
    CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_indices[0], cb_config.data_format}, {buffer_indices[1], cb_config.data_format}}, *l1_buffer)
        .set_page_size(buffer_indices[0], cb_config.page_size)
        .set_page_size(buffer_indices[1], cb_config.page_size);
    auto multi_core_cb = CreateCircularBuffer(program, cr_set, config1);

    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    for (const CoreRange &core_range : cr_set.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core_coord(x, y);
                for (uint8_t buffer_index : buffer_indices) {
                    golden_addresses_per_core[core_coord][buffer_index] = expected_cb_addr;
                }
            }
        }
    }

    initialize_program(program, cr_set);
    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    validate_cb_address(*program_ptr, this->devices_.at(id), cr_set, golden_addresses_per_core);
}
}

TEST_F(DeviceFixture, TestCircularBuffersAndL1BuffersCollision) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    auto program = tt::tt_metal::CreateScopedProgram();
    uint32_t page_size = TileSize(tt::DataFormat::Float16_b);

    auto buffer_size = page_size * 128;
    tt::tt_metal::InterleavedBufferConfig buff_config{
                    .device=this->devices_.at(id),
                    .size = buffer_size,
                    .page_size = buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::L1
        };
    auto l1_buffer = CreateBuffer(buff_config);

    // L1 buffer is entirely in bank 0
    auto core = l1_buffer->logical_core_from_bank_id(0);
    CoreRange cr(core, core);
    CoreRangeSet cr_set({cr});
    initialize_program(program, cr_set);

    uint32_t num_pages = (l1_buffer->address() - devices_.at(id)->get_base_allocator_addr(HalMemType::L1)) / NUM_CIRCULAR_BUFFERS / page_size + 1;
    CBConfig cb_config = {.num_pages=num_pages};
    for (uint32_t buffer_id = 0; buffer_id < NUM_CIRCULAR_BUFFERS; buffer_id++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size * cb_config.num_pages, {{buffer_id, cb_config.data_format}}).set_page_size(buffer_id, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core, config1);
    }

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    detail::CompileProgram(this->devices_.at(id), *program_ptr);
    EXPECT_ANY_THROW(detail::ConfigureDeviceWithProgram(this->devices_.at(id), *program_ptr));
}
}

TEST_F(DeviceFixture, TestValidUpdateCircularBufferSize) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    auto program = tt::tt_metal::CreateScopedProgram();
    CBConfig cb_config;
    CoreCoord core0(0, 0);
    CoreRange cr(core0, core0);
    CoreRangeSet cr_set({cr});

    initialize_program(program, cr_set);

    const uint32_t core0_num_cbs = 2;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    std::vector<CBHandle> cb_ids;
    uint32_t l1_unreserved_base = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
    auto expected_cb_addr = l1_unreserved_base;
    for (uint32_t buffer_idx = 0; buffer_idx < core0_num_cbs; buffer_idx++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_idx, cb_config.data_format}}).set_page_size(buffer_idx, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core0, config1);
        golden_addresses_per_core[core0][buffer_idx] = expected_cb_addr;
        cb_ids.push_back(cb);
        expected_cb_addr += cb_config.page_size;
    }

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    validate_cb_address(*program_ptr, this->devices_.at(id), cr_set, golden_addresses_per_core);

    // Update size of the first CB
    UpdateCircularBufferTotalSize(program, cb_ids[0], cb_config.page_size * 2);
    golden_addresses_per_core[core0][0] = l1_unreserved_base;
    golden_addresses_per_core[core0][1] = (l1_unreserved_base + (cb_config.page_size * 2));

    validate_cb_address(*program_ptr, this->devices_.at(id), cr_set, golden_addresses_per_core);
}
}

TEST_F(DeviceFixture, TestInvalidUpdateCircularBufferSize) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    auto program = tt::tt_metal::CreateScopedProgram();
    CBConfig cb_config;
    CoreCoord core0(0, 0);
    CoreRange cr(core0, core0);
    CoreRangeSet cr_set({cr});

    initialize_program(program, cr_set);

    const uint32_t core0_num_cbs = 2;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    std::vector<CBHandle> cb_ids;
    auto expected_cb_addr = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
    for (uint32_t buffer_idx = 0; buffer_idx < core0_num_cbs; buffer_idx++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_idx, cb_config.data_format}}).set_page_size(buffer_idx, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core0, config1);
        golden_addresses_per_core[core0][buffer_idx] = expected_cb_addr;
        cb_ids.push_back(cb);
        expected_cb_addr += cb_config.page_size;
    }

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    validate_cb_address(*program_ptr, this->devices_.at(id), cr_set, golden_addresses_per_core);

    // Update size of the first CB
    UpdateCircularBufferTotalSize(program, cb_ids[0], cb_config.page_size / 2);
    EXPECT_ANY_THROW(detail::LaunchProgram(this->devices_.at(id), *program_ptr));
}
}

TEST_F(DeviceFixture, TestUpdateCircularBufferAddress) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    auto program = tt::tt_metal::CreateScopedProgram();
    CBConfig cb_config;
    CoreCoord core0(0, 0);
    CoreRange cr(core0, core0);
    CoreRangeSet cr_set({cr});

    auto buffer_size = cb_config.page_size;
    tt::tt_metal::InterleavedBufferConfig buff_config{
                    .device=this->devices_.at(id),
                    .size = buffer_size,
                    .page_size = buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::L1
        };
    auto l1_buffer = CreateBuffer(buff_config);

    initialize_program(program, cr_set);

    const uint32_t core0_num_cbs = 2;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    std::vector<CBHandle> cb_ids;
    auto expected_cb_addr = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
    for (uint32_t buffer_idx = 0; buffer_idx < core0_num_cbs; buffer_idx++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_idx, cb_config.data_format}}).set_page_size(buffer_idx, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core0, config1);
        golden_addresses_per_core[core0][buffer_idx] = expected_cb_addr;
        cb_ids.push_back(cb);
        expected_cb_addr += cb_config.page_size;
    }

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    validate_cb_address(*program_ptr, this->devices_.at(id), cr_set, golden_addresses_per_core);
    // Update address of the first CB
    UpdateDynamicCircularBufferAddress(program, cb_ids[0], *l1_buffer);
    golden_addresses_per_core[core0][0] = l1_buffer->address();
    validate_cb_address(*program_ptr, this->devices_.at(id), cr_set, golden_addresses_per_core);
}
}

TEST_F(DeviceFixture, TestUpdateCircularBufferPageSize) {
  for (unsigned int id = 0; id < num_devices_; id++) {
    Device *device = this->devices_.at(id);
    auto program = tt::tt_metal::CreateScopedProgram();
    CBConfig cb_config;
    CoreCoord core0(0, 0);
    CoreRange cr(core0, core0);
    CoreRangeSet cr_set({cr});

    initialize_program(program, cr_set);

    const uint32_t core0_num_cbs = 2;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_addresses_per_core;
    std::map<CoreCoord, std::map<uint8_t, uint32_t>> golden_num_pages_per_core;
    std::vector<CBHandle> cb_ids;
    auto expected_cb_addr = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
    for (uint32_t buffer_idx = 0; buffer_idx < core0_num_cbs; buffer_idx++) {
        CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, {{buffer_idx, cb_config.data_format}}).set_page_size(buffer_idx, cb_config.page_size);
        auto cb = CreateCircularBuffer(program, core0, config1);
        golden_addresses_per_core[core0][buffer_idx] = expected_cb_addr;
        golden_num_pages_per_core[core0][buffer_idx] = 1;
        cb_ids.push_back(cb);
        expected_cb_addr += cb_config.page_size;
    }

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    detail::LaunchProgram(device, *program_ptr);

    vector<uint32_t> cb_config_vector;
    uint32_t cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(uint32_t);

    for (const CoreRange &core_range : cr_set.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core_coord(x, y);
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device, core_coord, program_ptr->get_cb_base_addr(device, core_coord, CoreType::WORKER), cb_config_buffer_size, cb_config_vector);

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

    UpdateCircularBufferPageSize(program, cb_ids[1], 1, cb_config.page_size / 2);
    golden_num_pages_per_core[core0][1] = 2;

    detail::LaunchProgram(device, *program_ptr);

    // addresses should not be changed
    for (const CoreRange &core_range : cr_set.ranges()) {
        for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
            for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                CoreCoord core_coord(x, y);
                tt::tt_metal::detail::ReadFromDeviceL1(
                    device, core_coord, program_ptr->get_cb_base_addr(device, core_coord, CoreType::WORKER), cb_config_buffer_size, cb_config_vector);

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
    auto program = tt::tt_metal::CreateScopedProgram();
    CoreCoord core(0, 0);

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2;
    uint32_t buffer_size = single_tile_size * num_tiles;

    tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device=this->devices_.at(id),
                    .size = buffer_size,
                    .page_size = buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    tt::tt_metal::InterleavedBufferConfig l1_config{
                    .device=this->devices_.at(id),
                    .size = buffer_size,
                    .page_size = buffer_size,
                    .buffer_type = tt::tt_metal::BufferType::L1
        };

    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);
    auto global_cb_buffer = CreateBuffer(l1_config);

    uint32_t cb_index = 0;
    uint32_t num_input_tiles = num_tiles;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(buffer_size, {{cb_index, tt::DataFormat::Float16_b}}).set_page_size(cb_index, single_tile_size);
    auto cb_src0 = CreateCircularBuffer(program, core, cb_src0_config);

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {cb_index}});

    auto writer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
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
            (uint32_t)src_dram_buffer->address(),
            (uint32_t)src_dram_buffer->noc_coordinates().x,
            (uint32_t)src_dram_buffer->noc_coordinates().y,
            (uint32_t)num_tiles,
        });
    SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)dst_dram_buffer->address(),
            (uint32_t)dst_dram_buffer->noc_coordinates().x,
            (uint32_t)dst_dram_buffer->noc_coordinates().y,
            (uint32_t)num_tiles,
        });

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    detail::LaunchProgram(this->devices_.at(id), *program_ptr);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);
    EXPECT_EQ(src_vec, result_vec);

    std::vector<uint32_t> input_cb_data;
    uint32_t cb_address = devices_.at(id)->get_base_allocator_addr(HalMemType::L1);
    detail::ReadFromDeviceL1(this->devices_.at(id), core, devices_.at(id)->get_base_allocator_addr(HalMemType::L1), buffer_size, input_cb_data);
    EXPECT_EQ(src_vec, input_cb_data);

    // update cb address
    UpdateDynamicCircularBufferAddress(program, cb_src0, *global_cb_buffer);

    // zero out dst buffer
    std::vector<uint32_t> zero_vec = create_constant_vector_of_bfloat16(buffer_size, 0);
    detail::WriteToBuffer(dst_dram_buffer, zero_vec);

    // relaunch program
    detail::LaunchProgram(this->devices_.at(id), *program_ptr);

    std::vector<uint32_t> second_result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, second_result_vec);
    EXPECT_EQ(src_vec, second_result_vec);

    std::vector<uint32_t> second_cb_data;
    detail::ReadFromDeviceL1(this->devices_.at(id), core, global_cb_buffer->address(), buffer_size, second_cb_data);
    EXPECT_EQ(src_vec, second_cb_data);
}
}

}   // end namespace basic_tests::circular_buffer
