// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "circular_buffer_test_utils.hpp"
#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"
#include "tt_metal/impl/program/program.hpp"
#include "tt_metal/impl/program/program_pool.hpp"


using namespace tt::tt_metal;

namespace basic_tests::circular_buffer {

bool test_cb_config_written_to_core(Program &program, Device *device, const CoreRangeSet &cr_set, const std::map<uint8_t, std::vector<uint32_t>> &cb_config_per_buffer_index) {
    bool pass = true;

    detail::CompileProgram(device, program);
    detail::ConfigureDeviceWithProgram(device, program);

    vector<uint32_t> cb_config_vector;

    for (const auto cb: program.circular_buffers()) {
        for (const CoreRange &core_range : cb->core_ranges().ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord core_coord(x, y);
                    uint32_t cb_config_buffer_size = program.get_cb_size(device, core_coord, CoreType::WORKER);

                    tt::tt_metal::detail::ReadFromDeviceL1(
                        device, core_coord, program.get_sem_base_addr(device, core_coord, CoreType::WORKER), cb_config_buffer_size, cb_config_vector);

                    for (const auto &[buffer_index, golden_cb_config] : cb_config_per_buffer_index) {
                        auto base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * buffer_index;
                        pass &= (golden_cb_config.at(0) == cb_config_vector.at(base_index));    // address
                        pass &= (golden_cb_config.at(1) == cb_config_vector.at(base_index + 1)); // size
                        pass &= (golden_cb_config.at(2) == cb_config_vector.at(base_index + 2)); // num pages
                    }
                }
            }
        }
    }

    return pass;
}

TEST_F(DeviceFixture, TestCreateCircularBufferAtValidIndices) {
    CBConfig cb_config;

    CoreRange cr({0, 0}, {0, 1});
    CoreRangeSet cr_set({cr});

    auto program = tt::tt_metal::CreateScopedProgram();
    initialize_program(program, cr_set);

    uint32_t l1_unreserved_base = devices_.at(0)->get_base_allocator_addr(HalMemType::L1);
    std::map<uint8_t, std::vector<uint32_t>> golden_cb_config = {
        {0, {l1_unreserved_base >> 4, cb_config.page_size >> 4, cb_config.num_pages}},
        {2, {l1_unreserved_base >> 4, cb_config.page_size >> 4, cb_config.num_pages}},
        {16, {l1_unreserved_base >> 4, cb_config.page_size >> 4, cb_config.num_pages}},
        {24, {l1_unreserved_base >> 4, cb_config.page_size >> 4, cb_config.num_pages}}
    };
    std::map<uint8_t, tt::DataFormat> data_format_spec = {
        {0, cb_config.data_format},
        {2, cb_config.data_format},
        {16, cb_config.data_format},
        {24, cb_config.data_format}
    };
    CircularBufferConfig config = CircularBufferConfig(cb_config.page_size, data_format_spec)
        .set_page_size(0, cb_config.page_size)
        .set_page_size(2, cb_config.page_size)
        .set_page_size(16, cb_config.page_size)
        .set_page_size(24, cb_config.page_size);
    auto cb = CreateCircularBuffer(program, cr_set, config);

    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    for (unsigned int id = 0; id < num_devices_; id++) {
        program_ptr->finalize(devices_.at(id));
        EXPECT_TRUE(test_cb_config_written_to_core(*program_ptr, this->devices_.at(id), cr_set, golden_cb_config));
    }
}

TEST_F(DeviceFixture, TestCreateCircularBufferAtInvalidIndex) {
    CBConfig cb_config;

    EXPECT_ANY_THROW(CircularBufferConfig(cb_config.page_size, {{NUM_CIRCULAR_BUFFERS, cb_config.data_format}}));
}

TEST_F(DeviceFixture, TestCreateCircularBufferWithMismatchingConfig) {
    auto program = tt::tt_metal::CreateScopedProgram();
    CBConfig cb_config;

    EXPECT_ANY_THROW(CircularBufferConfig(cb_config.page_size, {{0, cb_config.data_format}}).set_page_size(1, cb_config.page_size));
}

TEST_F(DeviceFixture, TestCreateCircularBufferAtOverlappingIndex) {
    auto program = tt::tt_metal::CreateScopedProgram();
    CBConfig cb_config;

    CoreRange cr({0, 0}, {1, 1});
    CoreRangeSet cr_set({cr});

    std::map<uint8_t, tt::DataFormat> data_format_spec1 = {
        {0, cb_config.data_format},
        {16, cb_config.data_format}
    };
    CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, data_format_spec1)
        .set_page_size(0, cb_config.page_size)
        .set_page_size(16, cb_config.page_size);

    std::map<uint8_t, tt::DataFormat> data_format_spec2 = {
        {1, cb_config.data_format},
        {2, cb_config.data_format},
        {16, cb_config.data_format}
    };
    CircularBufferConfig config2 = CircularBufferConfig(cb_config.page_size, data_format_spec2)
        .set_page_size(1, cb_config.page_size)
        .set_page_size(2, cb_config.page_size)
        .set_page_size(16, cb_config.page_size);

    auto valid_cb = CreateCircularBuffer(program, cr_set, config1);

    EXPECT_ANY_THROW(CreateCircularBuffer(program, cr_set, config2));
}

}   // end namespace basic_tests::circular_buffer
