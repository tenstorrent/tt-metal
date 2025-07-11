// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <stdint.h>
#include <tt-logger/tt-logger.hpp>
#include <array>
#include <functional>
#include <memory>
#include <stdexcept>

#include "gtest/gtest.h"
#include <tt-metalium/hal_types.hpp>
#include "impl/context/metal_context.hpp"
#include "umd/device/tt_core_coordinates.h"
#include "impl/dispatch/dispatch_settings.hpp"

namespace tt::tt_metal {

// Loop through test_func for WORKER, ETH X 1, 2 CQs
void ForEachCoreTypeXHWCQs(const std::function<void(const CoreType& core_type, const uint32_t num_hw_cqs)>& test_func) {
    const auto core_types_to_test = std::array<CoreType, 2>{CoreType::WORKER, CoreType::ETH};
    const auto num_hw_cqs_to_test = std::array<uint32_t, 2>{1, 2};

    for (const auto& core_type : core_types_to_test) {
        if (core_type == CoreType::ETH && MetalContext::instance().hal().get_programmable_core_type_index(
                                              tt::tt_metal::HalProgrammableCoreType::IDLE_ETH) == -1) {
            // This device does not have the eth core
            log_info(tt::LogTest, "IDLE_ETH core type is not on this device");
            continue;
        }
        for (const auto& num_hw_cqs : num_hw_cqs_to_test) {
            test_func(core_type, num_hw_cqs);
        }
    }
}

TEST(DispatchSettingsTest, TestDispatchSettingsDefaultUnsupportedCoreType) {
    DispatchSettings::initialize(tt::tt_metal::MetalContext::instance().get_cluster());
    const auto unsupported_core = CoreType::ARC;
    EXPECT_THROW(DispatchSettings::get(unsupported_core, 1), std::runtime_error);
}

TEST(DispatchSettingsTest, TestDispatchSettingsMissingArgs) {
    DispatchSettings settings;
    EXPECT_THROW(settings.build(), std::runtime_error);
}

TEST(DispatchSettingsTest, TestDispatchSettingsEq) {
    const uint32_t hw_cqs = 2;
    DispatchSettings::initialize(tt::tt_metal::MetalContext::instance().get_cluster());
    auto settings = DispatchSettings::get(CoreType::WORKER, hw_cqs);
    auto settings_2 = settings; // Copy
    EXPECT_EQ(settings, settings_2);
    settings_2.dispatch_size_ += 1;
    EXPECT_NE(settings, settings_2);
}

TEST(DispatchSettingsTest, TestDispatchSettingsSetPrefetchDBuffer) {
    const uint32_t hw_cqs = 2;
    const uint32_t expected_buffer_bytes = 0xcafe;
    const uint32_t expected_page_count =
        expected_buffer_bytes / (1 << DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE);
    DispatchSettings::initialize(tt::tt_metal::MetalContext::instance().get_cluster());
    auto settings = DispatchSettings::get(CoreType::WORKER, hw_cqs);
    settings.prefetch_d_buffer_size(expected_buffer_bytes);
    EXPECT_EQ(settings.prefetch_d_buffer_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.prefetch_d_pages_, expected_page_count);
}

TEST(DispatchSettingsTest, TestDispatchSettingsSetPrefetchQBuffer) {
    const uint32_t hw_cqs = 2;
    const uint32_t expected_buffer_entries = 0x1000;
    const uint32_t expected_buffer_bytes = expected_buffer_entries * sizeof(DispatchSettings::prefetch_q_entry_type);
    DispatchSettings::initialize(tt::tt_metal::MetalContext::instance().get_cluster());
    auto settings = DispatchSettings::get(CoreType::WORKER, hw_cqs);
    settings.prefetch_q_entries(expected_buffer_entries);
    EXPECT_EQ(settings.prefetch_q_entries_, expected_buffer_entries);
    EXPECT_EQ(settings.prefetch_q_size_, expected_buffer_bytes);
}

TEST(DispatchSettingsTest, TestDispatchSettingsSetDispatchBuffer) {
    const uint32_t hw_cqs = 2;
    const uint32_t expected_buffer_bytes = 0x2000;
    const uint32_t expected_page_count = expected_buffer_bytes / (1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);
    DispatchSettings::initialize(tt::tt_metal::MetalContext::instance().get_cluster());
    auto settings = DispatchSettings::get(CoreType::WORKER, hw_cqs);
    settings.dispatch_size(expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_pages_, expected_page_count);
}

TEST(DispatchSettingsTest, TestDispatchSettingsSetDispatchSBuffer) {
    const uint32_t hw_cqs = 2;
    const uint32_t expected_buffer_bytes = 0x2000;
    const uint32_t expected_page_count =
        expected_buffer_bytes / (1 << DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
    DispatchSettings::initialize(tt::tt_metal::MetalContext::instance().get_cluster());
    auto settings = DispatchSettings::get(CoreType::WORKER, hw_cqs);
    settings.dispatch_s_buffer_size(expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_s_buffer_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_s_buffer_pages_, expected_page_count);
}

TEST(DispatchSettingsTest, TestDispatchSettingsMutations) {
    if (MetalContext::instance().hal().get_programmable_core_type_index(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH) == -1) {
        // This device does not have the eth core
        log_info(tt::LogTest, "Test not supported on this device");
        return;
    }
    const auto core_type = CoreType::WORKER;
    const uint32_t hw_cqs = 1;
    const uint32_t prefetch_d_size = 0x1000;
    const uint32_t mux_size = 0x2000;
    const uint32_t cmddat_size = 0x2000;
    const uint32_t dispatch_s_size = 32;
    const uint32_t dispatch_size = 4096;
    const uint32_t max_cmd_size = 1024;
    const uint32_t prefetch_q_entries = 512;
    const uint32_t scratch_db_size = 5120;

    DispatchSettings::initialize(tt::tt_metal::MetalContext::instance().get_cluster());
    auto& settings = DispatchSettings::get(core_type, hw_cqs);
    DispatchSettings original_settings = settings;  // Copy the original to be restored later

    EXPECT_EQ(settings.core_type_, core_type);

    // Modify settings
    settings.prefetch_q_entries(prefetch_q_entries);
    settings.prefetch_d_buffer_size(prefetch_d_size);
    settings.prefetch_cmddat_q_size(cmddat_size);
    settings.dispatch_s_buffer_size(dispatch_s_size);
    settings.prefetch_max_cmd_size(max_cmd_size);
    settings.prefetch_scratch_db_size(scratch_db_size);
    settings.dispatch_size(dispatch_size);

    // Change instance
    // Check they are not the same
    auto& settings_2 = DispatchSettings::get(CoreType::ETH, hw_cqs);
    EXPECT_NE(settings_2.prefetch_q_entries_, prefetch_q_entries);
    EXPECT_NE(settings_2.prefetch_d_buffer_size_, prefetch_d_size);
    EXPECT_NE(settings_2.prefetch_cmddat_q_size_, cmddat_size);
    EXPECT_NE(settings_2.dispatch_s_buffer_size_, dispatch_s_size);
    EXPECT_NE(settings_2.prefetch_max_cmd_size_, max_cmd_size);
    EXPECT_NE(settings_2.prefetch_scratch_db_size_, scratch_db_size);
    EXPECT_NE(settings_2.dispatch_size_, dispatch_size);

    // Change back to the instance that we modified
    auto& settings_3 = DispatchSettings::get(core_type, hw_cqs);
    EXPECT_EQ(settings_3.prefetch_q_entries_, prefetch_q_entries);
    EXPECT_EQ(settings_3.prefetch_d_buffer_size_, prefetch_d_size);
    EXPECT_EQ(settings_3.prefetch_cmddat_q_size_, cmddat_size);
    EXPECT_EQ(settings_3.dispatch_s_buffer_size_, dispatch_s_size);
    EXPECT_EQ(settings_3.prefetch_max_cmd_size_, max_cmd_size);
    EXPECT_EQ(settings_3.prefetch_scratch_db_size_, scratch_db_size);
    EXPECT_EQ(settings_3.dispatch_size_, dispatch_size);

    DispatchSettings::initialize(original_settings);
}

}  // namespace tt::tt_metal
