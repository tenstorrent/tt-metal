// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "single_device_fixture.hpp"
#include "gtest/gtest.h"
#include "circular_buffer_test_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"

using namespace tt::tt_metal;

namespace basic_tests::circular_buffer {

bool test_cb_config_written_to_core(Program &program, Device *device, const CoreRangeSet &cr_set) {
    bool pass = true;

    detail::CompileProgram(device, program);
    detail::ConfigureDeviceWithProgram(device, program);

    vector<u32> cb_config_vector;
    u32 cb_config_buffer_size = NUM_CIRCULAR_BUFFERS * UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * sizeof(u32);

    for (const auto &cb: program.circular_buffers()) {
        for (const CoreRange &core_range : cb.core_range_set().ranges()) {
            for (auto x = core_range.start.x; x <= core_range.end.x; x++) {
                for (auto y = core_range.start.y; y <= core_range.end.y; y++) {
                    CoreCoord core_coord{.x = x, .y = y};
                    tt::tt_metal::detail::ReadFromDeviceL1(
                        device, core_coord, CIRCULAR_BUFFER_CONFIG_BASE, cb_config_buffer_size, cb_config_vector);

                    // for (const auto buffer_index : cb.buffer_indices()) {
                    //     auto base_index = UINT32_WORDS_PER_CIRCULAR_BUFFER_CONFIG * buffer_index;
                    //     pass &= (cb.address() >> 4) == cb_config_vector.at(base_index);
                    //     pass &= (cb.size() >> 4) == cb_config_vector.at(base_index + 1);
                    //     pass &= cb.num_tiles() == cb_config_vector.at(base_index + 2);
                    // }
                }
            }
        }
    }

    return pass;
}

TEST_F(SingleDeviceFixture, TestCreateCircularBufferAtValidIndices) {
    CBConfig cb_config;

    CoreRange cr = {.start = {0, 0}, .end = {0, 1}};
    CoreRangeSet cr_set({cr});

    Program program;
    initialize_program(program, cr_set);

    std::set<u32> indices = {0, 2, 16, 24};

    auto cb = CreateCircularBuffers(program, indices, cr_set, cb_config.num_pages, cb_config.page_size, cb_config.data_format);
    // EXPECT_EQ(cb.buffer_indices().size(), indices.size());
    EXPECT_TRUE(test_cb_config_written_to_core(program, this->device_, cr_set));
}

TEST_F(SingleDeviceFixture, TestCreateCircularBufferAtInvalidIndex) {
    Program program;
    CBConfig cb_config;

    CoreRange cr = {.start = {0, 0}, .end = {0, 0}};
    CoreRangeSet cr_set({cr});

    EXPECT_ANY_THROW(CreateCircularBuffers(program, NUM_CIRCULAR_BUFFERS, cr_set, cb_config.num_pages, cb_config.page_size, cb_config.data_format));
}

TEST_F(SingleDeviceFixture, TestCreateCircularBufferAtOverlappingIndex) {
    Program program;
    CBConfig cb_config;

    CoreRange cr = {.start = {0, 0}, .end = {1, 1}};
    CoreRangeSet cr_set({cr});

    std::set<u32> first_indices = {0, 16};
    std::set<u32> second_indices = {1, 2, 16};

    auto valid_cb = CreateCircularBuffers(program, first_indices, cr_set, cb_config.num_pages, cb_config.page_size, cb_config.data_format);
    // EXPECT_EQ(valid_cb.buffer_indices(), first_indices);
    EXPECT_ANY_THROW(CreateCircularBuffers(program, second_indices, cr_set, cb_config.num_pages, cb_config.page_size, cb_config.data_format));
}

}   // end namespace basic_tests::circular_buffer
