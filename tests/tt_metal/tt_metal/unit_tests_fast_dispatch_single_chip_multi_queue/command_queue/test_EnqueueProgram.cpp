// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "command_queue_fixture.hpp"
#include "command_queue_test_utils.hpp"
#include "gtest/gtest.h"
#include "impl/buffers/buffer.hpp"
#include "impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/common/scoped_timer.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

using std::vector;
using namespace tt::tt_metal;

struct CBConfig {
    uint32_t cb_id;
    uint32_t num_pages;
    uint32_t page_size;
    tt::DataFormat data_format;
};

struct DummyProgramConfig {
    CoreRangeSet cr_set;
    CBConfig cb_config;
    uint32_t num_cbs;
    uint32_t num_sems;
};

struct DummyProgramMultiCBConfig {
    CoreRangeSet cr_set;
    std::vector<CBConfig> cb_config_vector;
    uint32_t num_sems;
};


namespace local_test_functions {

// Create randomly sized pair of unique and common runtime args vectors, with careful not to exceed max between the two.
// Optionally force the max size for one of the vectors.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> create_runtime_args(bool force_max_size = false, uint32_t unique_base = 0, uint32_t common_base = 100){

    constexpr uint32_t MAX_RUNTIME_ARGS = 255;

    // Generate Unique Runtime Args. Common RT args starting address must be L1 Aligned, so account for that here via padding
    uint32_t num_rt_args_unique = num_rt_args_unique = rand() % (MAX_RUNTIME_ARGS + 1);
    uint32_t num_rt_args_unique_padded = align(num_rt_args_unique, hal.get_alignment(HalMemType::L1) / sizeof(uint32_t));
    uint32_t num_rt_args_common = num_rt_args_unique_padded < MAX_RUNTIME_ARGS ? rand() % (MAX_RUNTIME_ARGS - num_rt_args_unique_padded + 1) : 0;

    if (force_max_size) {
        if (rand() % 2) {
            num_rt_args_unique = MAX_RUNTIME_ARGS;
            num_rt_args_common = 0;
        } else {
            num_rt_args_common = MAX_RUNTIME_ARGS;
            num_rt_args_unique = 0;
        }
    }

    vector<uint32_t> rt_args_common;
    for (uint32_t i = 0; i < num_rt_args_common; i++) {
        rt_args_common.push_back(common_base + i);
    }

    vector<uint32_t> rt_args_unique;
    for (uint32_t i = 0; i < num_rt_args_unique; i++) {
        rt_args_unique.push_back(unique_base + i);
    }

    log_trace(tt::LogTest, "{} - num_rt_args_unique: {} num_rt_args_common: {} force_max_size: {}", __FUNCTION__, num_rt_args_unique, num_rt_args_common, force_max_size);
    return std::make_pair(rt_args_unique, rt_args_common);
}


}  // namespace local_test_functions

namespace stress_tests {



}  // namespace stress_tests
