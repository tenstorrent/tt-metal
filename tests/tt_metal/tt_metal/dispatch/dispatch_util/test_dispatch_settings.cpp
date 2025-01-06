// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <stdexcept>
#include "command_queue_fixture.hpp"
#include "common/logger.hpp"
#include "dispatch/dispatch_constants.hpp"
#include "gtest/gtest.h"
#include "llrt/hal.hpp"
#include "tt_metal/impl/dispatch/util/include/dispatch_settings.hpp"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"
#include "umd/device/tt_core_coordinates.h"

using namespace tt::tt_metal::dispatch;

// Loop through test_func for WORKER, ETH X 1, 2 CQs
void ForEachCoreTypeXHWCQs(const std::function<void(const CoreType& core_type, const uint32_t num_hw_cqs)>& test_func) {
    static constexpr auto core_types_to_test = std::array<CoreType, 2>{CoreType::WORKER, CoreType::ETH};
    static constexpr auto num_hw_cqs_to_test = std::array<uint32_t, 2>{1, 2};

    for (const auto& core_type : core_types_to_test) {
        if (core_type == CoreType::ETH &&
            hal.get_programmable_core_type_index(tt::tt_metal::HalProgrammableCoreType::IDLE_ETH) == -1) {
            // This device does not have the eth core
            tt::log_info(tt::LogTest, "IDLE_ETH core type is not on this device");
            continue;
        }
        for (const auto& num_hw_cqs : num_hw_cqs_to_test) {
            test_func(core_type, num_hw_cqs);
        }
    }
}

TEST_F(CommandQueueSingleCardFixture, TestDispatchSettingsDefaultParity) {
    ForEachCoreTypeXHWCQs([&](const CoreType& core_type, uint32_t num_hw_cqs) {
        auto settings = DispatchSettings::defaults(core_type, tt::Cluster::instance(), num_hw_cqs);

        const auto& old_constants = dispatch_constants::get(core_type, num_hw_cqs);

        ASSERT_EQ(settings.num_hw_cqs_, num_hw_cqs);

        ASSERT_EQ(settings.prefetch_q_entries_, old_constants.prefetch_q_entries());
        ASSERT_EQ(settings.prefetch_q_size_, old_constants.prefetch_q_size());
        ASSERT_EQ(settings.prefetch_max_cmd_size_, old_constants.max_prefetch_command_size());
        ASSERT_EQ(settings.prefetch_cmddat_q_size_, old_constants.cmddat_q_size());
        ASSERT_EQ(settings.prefetch_scratch_db_size_, old_constants.scratch_db_size());

        ASSERT_EQ(settings.prefetch_d_buffer_size_, old_constants.prefetch_d_buffer_size());
        ASSERT_EQ(settings.prefetch_d_pages_, old_constants.prefetch_d_buffer_pages());
        ASSERT_EQ(settings.prefetch_d_blocks_, dispatch_constants::PREFETCH_D_BUFFER_BLOCKS);

        ASSERT_EQ(settings.tunneling_buffer_size_ / num_hw_cqs, old_constants.mux_buffer_size(num_hw_cqs));
        ASSERT_EQ(settings.tunneling_buffer_pages_ / num_hw_cqs, old_constants.mux_buffer_pages(num_hw_cqs));

        ASSERT_EQ(settings.dispatch_pages_, old_constants.dispatch_buffer_pages());
        ASSERT_EQ(settings.dispatch_pages_per_block_, dispatch_constants::DISPATCH_BUFFER_SIZE_BLOCKS);

        ASSERT_EQ(settings.dispatch_s_buffer_size_, old_constants.dispatch_s_buffer_size());
        ASSERT_EQ(settings.dispatch_s_buffer_pages_, old_constants.dispatch_s_buffer_pages());

        EXPECT_NO_THROW(settings.build());
    });
}

TEST_F(CommandQueueSingleCardFixture, TestDispatchSettingsDefaultUnsupportedCoreType) {
    const auto unsupported_core = CoreType::ARC;
    EXPECT_THROW(DispatchSettings::defaults(unsupported_core, tt::Cluster::instance(), 1), std::runtime_error);
}

TEST_F(CommandQueueSingleCardFixture, TestDispatchSettingsMissingArgs) {
    DispatchSettings settings;
    EXPECT_THROW(settings.build(), std::runtime_error);
}

TEST_F(CommandQueueSingleCardFixture, TestDispatchSettingsEq) {
    static constexpr uint32_t hw_cqs = 2;
    auto settings = DispatchSettings::worker_defaults(tt::Cluster::instance(), hw_cqs);
    auto settings_2 = settings; // Copy
    EXPECT_EQ(settings, settings_2);
    settings_2.dispatch_size_ += 1;
    EXPECT_NE(settings, settings_2);
}

TEST_F(CommandQueueSingleCardFixture, TestDispatchSettingsSetPrefetchDBuffer) {
    static constexpr uint32_t hw_cqs = 2;
    static constexpr uint32_t expected_buffer_bytes = 0xcafe;
    static constexpr uint32_t expected_page_count = expected_buffer_bytes / (1 << DispatchConstants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE);
    auto settings = DispatchSettings::worker_defaults(tt::Cluster::instance(), hw_cqs);
    settings.prefetch_d_buffer_size(expected_buffer_bytes);
    EXPECT_EQ(settings.prefetch_d_buffer_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.prefetch_d_pages_, expected_page_count);
}

TEST_F(CommandQueueSingleCardFixture, TestDispatchSettingsSetPrefetchQBuffer) {
    static constexpr uint32_t hw_cqs = 2;
    static constexpr uint32_t expected_buffer_entries = 0x1000;
    static constexpr uint32_t expected_buffer_bytes = expected_buffer_entries * sizeof(DispatchConstants::prefetch_q_entry_type);
    auto settings = DispatchSettings::worker_defaults(tt::Cluster::instance(), hw_cqs);
    settings.prefetch_q_entries(expected_buffer_entries);
    EXPECT_EQ(settings.prefetch_q_entries_, expected_buffer_entries);
    EXPECT_EQ(settings.prefetch_q_size_, expected_buffer_bytes);
}

TEST_F(CommandQueueSingleCardFixture, TestDispatchSettingsSetDispatchBuffer) {
    static constexpr uint32_t hw_cqs = 2;
    static constexpr uint32_t expected_buffer_bytes = 0x2000;
    static constexpr uint32_t expected_page_count = expected_buffer_bytes / (1 << DispatchConstants::DISPATCH_BUFFER_LOG_PAGE_SIZE);
    auto settings = DispatchSettings::worker_defaults(tt::Cluster::instance(), hw_cqs);
    settings.dispatch_size(expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_pages_, expected_page_count);
}

TEST_F(CommandQueueSingleCardFixture, TestDispatchSettingsSetDispatchSBuffer) {
    static constexpr uint32_t hw_cqs = 2;
    static constexpr uint32_t expected_buffer_bytes = 0x2000;
    static constexpr uint32_t expected_page_count = expected_buffer_bytes / (1 << DispatchConstants::DISPATCH_S_BUFFER_LOG_PAGE_SIZE);
    auto settings = DispatchSettings::worker_defaults(tt::Cluster::instance(), hw_cqs);
    settings.dispatch_s_buffer_size(expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_s_buffer_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.dispatch_s_buffer_pages_, expected_page_count);
}

TEST_F(CommandQueueSingleCardFixture, TestDispatchSettingsSetTunnelerBuffer) {
    static constexpr uint32_t hw_cqs = 2;
    static constexpr uint32_t expected_buffer_bytes = 0x2000;
    static constexpr uint32_t expected_page_count = expected_buffer_bytes / (1 << DispatchConstants::PREFETCH_D_BUFFER_LOG_PAGE_SIZE);
    auto settings = DispatchSettings::worker_defaults(tt::Cluster::instance(), hw_cqs);
    settings.tunneling_buffer_size(expected_buffer_bytes);
    EXPECT_EQ(settings.tunneling_buffer_size_, expected_buffer_bytes);
    EXPECT_EQ(settings.tunneling_buffer_pages_, expected_page_count);
}
