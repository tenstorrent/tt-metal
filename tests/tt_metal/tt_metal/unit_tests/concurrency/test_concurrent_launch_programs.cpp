// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <unistd.h>

#include "concurrent_fixture.hpp"
#include "concurrent_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/executor.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"

using namespace tt::tt_metal;
using namespace tt::test_utils;

TEST_F(ConcurrentFixture, TestMultiThreadLaunchProgram) {
    const chip_id_t device_id = 0;
    Device *device = CreateDevice(device_id);

    concurrent_tests::DatacopyProgramConfig test_config;
    uint32_t num_banks = device->num_banks(BufferType::L1);
    test_config.num_tiles = num_banks * 2;
    test_config.output_buffer_type = BufferType::L1;

    std::vector<std::future<void>> events;

    const int num_threads = 3;
    for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
        events.emplace_back(
            detail::async (
                [&] {
                    bool pass = concurrent_tests::reader_datacopy_writer(device, test_config);
                    ASSERT_TRUE(pass);
                }
            )
        );
    }

    for (auto &f : events) {
        f.wait();
    }

    CloseDevice(device);
}

TEST_F(ConcurrentFixture, DISABLED_TestMultiThreadLaunchProgramWriteL1Buffers) {

}
