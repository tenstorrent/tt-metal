// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <cstdint>
#include <tt-logger/tt-logger.hpp>
#include <array>
#include <functional>
#include <memory>
#include <stdexcept>

#include "gtest/gtest.h"
#include <tt-metalium/hal_types.hpp>
#include "impl/context/metal_context.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include "impl/dispatch/dispatch_settings.hpp"

namespace tt::tt_metal {

static constexpr uint32_t default_l1_alignment = 16;

TEST(DispatchSettingsTest, TestDispatchSettingsDefaultUnsupportedCoreType) {
    const auto unsupported_core = CoreType::ARC;
    EXPECT_THROW(DispatchSettings(1, unsupported_core, false, false, default_l1_alignment, 4), std::runtime_error);
}

TEST(DispatchSettingsTest, TestDispatchSettingsInvalidPrefetchQEntrySize) {
    EXPECT_THROW(DispatchSettings(1, CoreType::WORKER, false, false, default_l1_alignment, 3), std::runtime_error);
}

TEST(DispatchSettingsTest, TestDispatchSettingsEq) {
    const uint32_t hw_cqs = 2;
    DispatchSettings settings(hw_cqs, CoreType::WORKER, false, false, default_l1_alignment, 4);
    DispatchSettings settings_2 = settings;  // Copy
    EXPECT_EQ(settings, settings_2);
    settings_2.dispatch_size_ += 1;
    EXPECT_NE(settings, settings_2);
}

TEST(DispatchSettingsTest, TestDispatchSettingsSetPrefetchDBuffer) {
    const uint32_t hw_cqs = 2;
    const uint32_t expected_buffer_bytes = 0xcafe;
    const uint32_t expected_page_count =
        expected_buffer_bytes / (1 << DispatchSettings::PREFETCH_D_BUFFER_LOG_PAGE_SIZE);
    DispatchSettings settings(hw_cqs, CoreType::WORKER, false, false, default_l1_alignment, 4);
    settings.prefetch_d_buffer_size(expected_buffer_bytes);
    EXPECT_EQ(settings.prefetch_d_buffer_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.prefetch_d_pages_, expected_page_count);
}

TEST(DispatchSettingsTest, TestDispatchSettingsSetPrefetchQBuffer) {
    const uint32_t hw_cqs = 2;
    const uint32_t expected_buffer_entries = 0x1000;
    const uint32_t expected_buffer_bytes = expected_buffer_entries * 4;
    DispatchSettings settings(hw_cqs, CoreType::WORKER, false, false, default_l1_alignment, 4);
    settings.prefetch_q_entries(expected_buffer_entries);
    EXPECT_EQ(settings.prefetch_q_entries_, expected_buffer_entries);
    EXPECT_EQ(settings.prefetch_q_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.prefetch_q_entry_size_bytes_, 4);
}

TEST(DispatchSettingsTest, TestDispatchSettingsSetPrefetchQBufferWith2ByteEntries) {
    const uint32_t hw_cqs = 2;
    const uint32_t expected_buffer_entries = 0x1000;
    const uint32_t expected_buffer_bytes = expected_buffer_entries * 2;
    DispatchSettings settings(hw_cqs, CoreType::ETH, false, false, default_l1_alignment, 2);
    settings.prefetch_q_entries(expected_buffer_entries);
    EXPECT_EQ(settings.prefetch_q_entries_, expected_buffer_entries);
    EXPECT_EQ(settings.prefetch_q_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.prefetch_q_entry_size_bytes_, 2);
}

TEST(DispatchSettingsTest, TestDispatchSettingsSetDispatchBuffer) {
    const uint32_t hw_cqs = 2;
    const uint32_t expected_buffer_bytes = 0x2000;
    const uint32_t expected_page_count = expected_buffer_bytes / (1 << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);
    DispatchSettings settings(hw_cqs, CoreType::WORKER, false, false, default_l1_alignment, 4);
    settings.dispatch_size(expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_pages_, expected_page_count);
}

TEST(DispatchSettingsTest, TestDispatchSettingsSetDispatchSBuffer) {
    const uint32_t hw_cqs = 2;
    const uint32_t expected_buffer_bytes = 0x2000;
    const uint32_t expected_page_count =
        expected_buffer_bytes / (1 << DispatchSettings::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
    DispatchSettings settings(hw_cqs, CoreType::WORKER, false, false, default_l1_alignment, 4);
    settings.dispatch_s_buffer_size(expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_s_buffer_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_s_buffer_pages_, expected_page_count);
}

}  // namespace tt::tt_metal
