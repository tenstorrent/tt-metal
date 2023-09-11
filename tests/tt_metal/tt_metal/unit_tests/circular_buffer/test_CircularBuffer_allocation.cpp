// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "single_device_fixture.hpp"
#include "gtest/gtest.h"
#include "circular_buffer_test_utils.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/buffers/circular_buffer.hpp"

using namespace tt::tt_metal;

namespace basic_tests::circular_buffer {

TEST_F(SingleDeviceFixture, TestCircularBuffersSequentiallyPlaced) {
    Program program;
    CBConfig cb_config;
    CoreRange cr = {.start = {0, 0}, .end = {0, 0}};
    CoreRangeSet cr_set({cr});

    auto expected_cb_addr = L1_UNRESERVED_BASE;
    for (auto cb_id = 0; cb_id < NUM_CIRCULAR_BUFFERS; cb_id++) {
        auto cb = CreateCircularBuffers(program, cb_id, cr_set, cb_config.num_pages, cb_config.page_size, cb_config.data_format);
        EXPECT_EQ(cb.address(), expected_cb_addr);
        expected_cb_addr += cb_config.page_size;
    }
}

TEST_F(SingleDeviceFixture, TestCircularBufferSequentialAcrossAllCores) {
    Program program;
    CBConfig cb_config;

    CoreCoord core0{.x = 0, .y = 0};
    CoreCoord core1{.x = 0, .y = 1};
    CoreCoord core2{.x = 0, .y = 2};

    const static std::map<CoreCoord, u32> core_to_num_cbs = {{core0, 3}, {core1, 0}, {core2, 5}};

    u32 max_num_cbs = 0;
    for (const auto &[core, num_cbs] : core_to_num_cbs) {
        auto expected_cb_addr = L1_UNRESERVED_BASE;
        max_num_cbs = std::max(max_num_cbs, num_cbs);
        for (u32 buffer_id = 0; buffer_id < num_cbs; buffer_id++) {
            auto cb = CreateCircularBuffer(program, buffer_id, core, cb_config.num_pages, cb_config.page_size, cb_config.data_format);
            EXPECT_EQ(cb.address(), expected_cb_addr);
            expected_cb_addr += cb_config.page_size;
        }
    }

    CoreRange cr = {.start = core0, .end = core2};
    CoreRangeSet cr_set({cr});

    auto expected_address = L1_UNRESERVED_BASE + (max_num_cbs * cb_config.page_size);
    auto multi_core_cb = CreateCircularBuffers(program, NUM_CIRCULAR_BUFFERS - 1, cr_set, cb_config.num_pages, cb_config.page_size, cb_config.data_format);
    EXPECT_EQ(multi_core_cb.address(), expected_address);
}

TEST_F(SingleDeviceFixture, TestValidCircularBufferAddress) {
    Program program;
    CBConfig cb_config;

    CoreRange cr = {.start = {0, 0}, .end = {0, 2}};
    CoreRangeSet cr_set({cr});

    u32 expected_cb_addr = L1_UNRESERVED_BASE + (NUM_CIRCULAR_BUFFERS * cb_config.page_size);
    auto multi_core_cb = CreateCircularBuffers(program, {16, 24}, cr_set, cb_config.num_pages, cb_config.page_size, cb_config.data_format, expected_cb_addr);
    EXPECT_EQ(multi_core_cb.address(), expected_cb_addr);
}

TEST_F(SingleDeviceFixture, TestInvalidCircularBufferAddress) {
    Program program;
    CBConfig cb_config;

    CoreCoord core0{.x = 0, .y = 0};
    const static u32 core0_num_cbs = 3;
    auto expected_core0_cb_addr = L1_UNRESERVED_BASE;
    for (u32 buffer_id = 0; buffer_id < core0_num_cbs; buffer_id++) {
        auto cb = CreateCircularBuffer(program, buffer_id, core0, cb_config.num_pages, cb_config.page_size, cb_config.data_format);
        EXPECT_EQ(cb.address(), expected_core0_cb_addr);
        expected_core0_cb_addr += cb_config.page_size;
    }

    CoreRange cr = {.start = {0, 0}, .end = {0, 1}};
    CoreRangeSet cr_set({cr});

    constexpr u32 multi_core_cb_index = core0_num_cbs + 1;
    EXPECT_ANY_THROW(CreateCircularBuffers(program, multi_core_cb_index, cr_set, cb_config.num_pages, cb_config.page_size, cb_config.data_format, L1_UNRESERVED_BASE));
}

TEST_F(SingleDeviceFixture, TestCircularBuffersAndL1BuffersCollision) {
    Program program;
    CBConfig cb_config {.num_pages = 5};

    auto buffer_size = cb_config.page_size * 128;
    auto l1_buffer = Buffer(this->device_, buffer_size, buffer_size, BufferType::L1);

    // L1 buffer is entirely in bank 0
    auto core = l1_buffer.logical_core_from_bank_id(0);
    CoreRange cr = {.start = core, .end = core};
    CoreRangeSet cr_set({cr});
    initialize_program(program, cr_set);

    auto cb_buffer_size = cb_config.page_size * cb_config.num_pages;
    auto cb_addr = l1_buffer.address() - (cb_buffer_size * (NUM_CIRCULAR_BUFFERS - 1));
    for (u32 buffer_id = 0; buffer_id < NUM_CIRCULAR_BUFFERS; buffer_id++) {
        auto cb = CreateCircularBuffer(program, buffer_id, core, cb_config.num_pages, cb_buffer_size, cb_config.data_format, cb_addr);
        EXPECT_EQ(cb.address(), cb_addr);
        cb_addr += cb_buffer_size;
    }

    CompileProgram(this->device_, program);
    EXPECT_ANY_THROW(ConfigureDeviceWithProgram(this->device_, program));
}

}   // end namespace basic_tests::circular_buffer
