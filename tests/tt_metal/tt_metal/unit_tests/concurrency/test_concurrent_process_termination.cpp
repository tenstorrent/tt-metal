// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <unistd.h>
#include <boost/process.hpp>

#include "concurrent_fixture.hpp"
#include "concurrent_test_utils.hpp"
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/executor.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"

using namespace tt::tt_metal;
using namespace tt::test_utils;

TEST_F(ConcurrentFixture, TestSingleDeviceRandomKillSignal) {
    const chip_id_t device_id = 0;
    Device *device = CreateDevice(device_id);

    concurrent_tests::DatacopyProgramConfig test_config;
    uint32_t num_banks = device->num_banks(BufferType::L1);
    test_config.num_tiles = num_banks * 2;
    test_config.output_buffer_type = BufferType::L1;

    bool pass = concurrent_tests::reader_datacopy_writer(device, test_config);
    ASSERT_TRUE(pass);

    CloseDevice(device);
}
