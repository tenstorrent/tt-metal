// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <map>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer_constants.h>
#include "circular_buffer_test_utils.hpp"
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include "device_fixture.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/hal_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/program.hpp>
#include "umd/device/tt_core_coordinates.h"

namespace tt {
enum class DataFormat : uint8_t;
}  // namespace tt

using std::vector;
using namespace tt::tt_metal;

namespace basic_tests::circular_buffer {

bool test_cb_config_written_to_core(
    Program& program,
    IDevice* device,
    const CoreRangeSet& cr_set,
    const std::map<uint8_t, std::vector<uint32_t>>& cb_config_per_buffer_index) {
    bool pass = true;

    detail::CompileProgram(device, program);
    detail::ConfigureDeviceWithProgram(device, program);

    vector<uint32_t> cb_config_vector;

    for (const auto& cb : program.circular_buffers()) {
        for (const CoreRange& core_range : cb->core_ranges().ranges()) {
            for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
                for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
                    CoreCoord core_coord(x, y);
                    uint32_t cb_config_buffer_size = program.get_cb_size(device, core_coord, CoreType::WORKER);

                    tt::tt_metal::detail::ReadFromDeviceL1(
                        device,
                        core_coord,
                        program.get_sem_base_addr(device, core_coord, CoreType::WORKER),
                        cb_config_buffer_size,
                        cb_config_vector);

                    for (const auto& [buffer_index, golden_cb_config] : cb_config_per_buffer_index) {
                        auto base_index = UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG * buffer_index;
                        pass &= (golden_cb_config.at(0) == cb_config_vector.at(base_index));      // address
                        pass &= (golden_cb_config.at(1) == cb_config_vector.at(base_index + 1));  // size
                        pass &= (golden_cb_config.at(2) == cb_config_vector.at(base_index + 2));  // num pages
                    }
                }
            }
        }
    }

    return pass;
}

TEST_F(DeviceFixture, TensixTestCreateCircularBufferAtValidIndices) {
    CBConfig cb_config;

    CoreRange cr({0, 0}, {0, 1});
    CoreRangeSet cr_set({cr});

    Program program;
    initialize_program(program, cr_set);

    uint32_t l1_unreserved_base = devices_.at(0)->allocator()->get_base_allocator_addr(HalMemType::L1);
    std::map<uint8_t, std::vector<uint32_t>> golden_cb_config = {
        {0, {l1_unreserved_base, cb_config.page_size, cb_config.num_pages}},
        {2, {l1_unreserved_base, cb_config.page_size, cb_config.num_pages}},
        {16, {l1_unreserved_base, cb_config.page_size, cb_config.num_pages}},
        {24, {l1_unreserved_base, cb_config.page_size, cb_config.num_pages}}};
    std::map<uint8_t, tt::DataFormat> data_format_spec = {
        {0, cb_config.data_format},
        {2, cb_config.data_format},
        {16, cb_config.data_format},
        {24, cb_config.data_format}};
    CircularBufferConfig expected_config = CircularBufferConfig(cb_config.page_size, data_format_spec)
                                               .set_page_size(tt::CBIndex::c_0, cb_config.page_size)
                                               .set_page_size(tt::CBIndex::c_2, cb_config.page_size)
                                               .set_page_size(tt::CBIndex::c_16, cb_config.page_size)
                                               .set_page_size(tt::CBIndex::c_24, cb_config.page_size);

    CircularBufferConfig actual_config = CircularBufferConfig(cb_config.page_size);
    actual_config.index(tt::CBIndex::c_0).set_page_size(cb_config.page_size).set_data_format(cb_config.data_format);
    actual_config.index(tt::CBIndex::c_2).set_page_size(cb_config.page_size).set_data_format(cb_config.data_format);
    actual_config.index(tt::CBIndex::c_16).set_page_size(cb_config.page_size).set_data_format(cb_config.data_format);
    actual_config.index(tt::CBIndex::c_24).set_page_size(cb_config.page_size).set_data_format(cb_config.data_format);

    EXPECT_TRUE(actual_config == expected_config);

    auto cb = CreateCircularBuffer(program, cr_set, actual_config);

    for (unsigned int id = 0; id < num_devices_; id++) {
        detail::CompileProgram(devices_.at(id), program);
        program.finalize_offsets(devices_.at(id));
        EXPECT_TRUE(test_cb_config_written_to_core(program, this->devices_.at(id), cr_set, golden_cb_config));
    }
}

TEST_F(DeviceFixture, TestCreateCircularBufferAtInvalidIndex) {
    CBConfig cb_config;

    EXPECT_ANY_THROW(CircularBufferConfig(cb_config.page_size, {{NUM_CIRCULAR_BUFFERS, cb_config.data_format}}));
}

TEST_F(DeviceFixture, TestCreateCircularBufferWithMismatchingConfig) {
    Program program;
    CBConfig cb_config;

    EXPECT_ANY_THROW(
        CircularBufferConfig(cb_config.page_size, {{0, cb_config.data_format}}).set_page_size(1, cb_config.page_size));
}

TEST_F(DeviceFixture, TensixTestCreateCircularBufferAtOverlappingIndex) {
    Program program;
    CBConfig cb_config;

    CoreRange cr({0, 0}, {1, 1});
    CoreRangeSet cr_set({cr});

    std::map<uint8_t, tt::DataFormat> data_format_spec1 = {{0, cb_config.data_format}, {16, cb_config.data_format}};
    CircularBufferConfig config1 = CircularBufferConfig(cb_config.page_size, data_format_spec1)
                                       .set_page_size(0, cb_config.page_size)
                                       .set_page_size(16, cb_config.page_size);

    std::map<uint8_t, tt::DataFormat> data_format_spec2 = {
        {1, cb_config.data_format}, {2, cb_config.data_format}, {16, cb_config.data_format}};
    CircularBufferConfig config2 = CircularBufferConfig(cb_config.page_size, data_format_spec2)
                                       .set_page_size(1, cb_config.page_size)
                                       .set_page_size(2, cb_config.page_size)
                                       .set_page_size(16, cb_config.page_size);

    auto valid_cb = CreateCircularBuffer(program, cr_set, config1);

    EXPECT_ANY_THROW(CreateCircularBuffer(program, cr_set, config2));
}

}  // end namespace basic_tests::circular_buffer
