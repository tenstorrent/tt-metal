// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/hw/inc/internal/dataflow_buffer_interface.h"
#include "tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"

namespace tt::tt_metal {

// These tests create a DFB and validate that the tile counter and remapper config is correct

// Validation structure for DFB tile counter configuration
struct DFBTileCounterExpectation {
    uint8_t expected_producer_tc_count;  // TCs per producer
    uint8_t expected_consumer_tc_count;  // TCs per consumer

    // Map of producer risc_id -> vector of (consumer risc_id, producer_tc_slot, consumer_tc_slot)
    std::map<uint8_t, std::vector<std::tuple<uint8_t, uint8_t, uint8_t>>> producer_to_consumer_pairings;
};

// Validates DFB tile counter configuration against expected pairings
void validate_dfb_tile_counters(
    Program& program,
    const CoreCoord& logical_core,
    const experimental::dfb::DataflowBufferConfig& config,
    const DFBTileCounterExpectation& expectation) {
    auto dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1) << "Expected exactly 1 DFB on core";

    const auto& dfb = dfbs[0];

    ASSERT_EQ(dfb->risc_mask, config.producer_risc_mask | config.consumer_risc_mask);
    ASSERT_EQ(dfb->risc_configs.size(), config.num_producers + config.num_consumers);

    // risc ID to risc config maps
    std::map<uint8_t, const experimental::dfb::detail::DFBRiscConfig*> producer_configs;
    std::map<uint8_t, const experimental::dfb::detail::DFBRiscConfig*> consumer_configs;

    for (const auto& rc : dfb->risc_configs) {
        if (rc.is_producer) {
            producer_configs[rc.risc_id] = &rc;
        } else {
            consumer_configs[rc.risc_id] = &rc;
        }
    }

    for (const auto& [risc_id, rc] : producer_configs) {
        EXPECT_EQ(rc->config.num_tcs_to_rr, expectation.expected_producer_tc_count)
            << "Producer RISC " << (int)risc_id << " has wrong TC count";
    }

    for (const auto& [risc_id, rc] : consumer_configs) {
        EXPECT_EQ(rc->config.num_tcs_to_rr, expectation.expected_consumer_tc_count)
            << "Consumer RISC " << (int)risc_id << " has wrong TC count";
    }

    // Validate TC tensix_id for Tensix RISCs
    // Key constraint: Tensix RISCs can only access TCs from their own tensix_id
    for (const auto& [risc_id, rc] : producer_configs) {
        bool is_tensix_risc = risc_id >= 8;
        if (is_tensix_risc) {
            uint8_t expected_tensix_id = (risc_id - 8) % 4;
            for (uint8_t tc = 0; tc < rc->config.num_tcs_to_rr; tc++) {
                auto ptc = rc->config.packed_tile_counter[tc];
                uint8_t actual_tensix_id = ::experimental::get_tensix_id(ptc);
                EXPECT_EQ(actual_tensix_id, expected_tensix_id)
                    << "Tensix producer RISC " << (int)risc_id << " TC[" << (int)tc
                    << "] must use tensix_id=" << (int)expected_tensix_id << " but has " << (int)actual_tensix_id;
            }
        }
    }

    for (const auto& [risc_id, rc] : consumer_configs) {
        bool is_tensix_risc = risc_id >= 8;
        if (is_tensix_risc) {
            uint8_t expected_tensix_id = (risc_id - 8) % 4;
            for (uint8_t tc = 0; tc < rc->config.num_tcs_to_rr; tc++) {
                auto ptc = rc->config.packed_tile_counter[tc];
                uint8_t actual_tensix_id = ::experimental::get_tensix_id(ptc);
                EXPECT_EQ(actual_tensix_id, expected_tensix_id)
                    << "Tensix consumer RISC " << (int)risc_id << " TC[" << (int)tc
                    << "] must use tensix_id=" << (int)expected_tensix_id << " but has " << (int)actual_tensix_id;
            }
        }
    }

    // For BLOCKED mode, validate remapper pair indices
    if (config.cap == ::experimental::AccessPattern::BLOCKED) {
        std::set<uint8_t> seen_remapper_indices;
        for (const auto& [risc_id, rc] : producer_configs) {
            uint8_t remapper_idx = rc->config.remapper_pair_index;

            // Check valid range (0-63)
            EXPECT_LT(remapper_idx, 64) << "BLOCKED: Producer RISC " << (int)risc_id
                                        << " has invalid remapper_pair_index " << (int)remapper_idx
                                        << " (must be 0-63)";

            // Check uniqueness among producers
            EXPECT_EQ(seen_remapper_indices.count(remapper_idx), 0)
                << "BLOCKED: Producer RISC " << (int)risc_id << " has duplicate remapper_pair_index "
                << (int)remapper_idx;
            seen_remapper_indices.insert(remapper_idx);

            log_info(tt::LogTest, "BLOCKED: Producer {} has remapper_pair_index {}", risc_id, remapper_idx);
        }
    }

    for (const auto& [producer_risc_id, pairings] : expectation.producer_to_consumer_pairings) {
        auto producer_it = producer_configs.find(producer_risc_id);
        ASSERT_NE(producer_it, producer_configs.end());

        const auto* producer_rc = producer_it->second;

        // For BLOCKED mode, accumulate expected_consumer_tcs across all pairings for this producer
        uint32_t expected_consumer_tcs = 0;
        size_t consumer_idx = 0;

        for (const auto& [consumer_risc_id, producer_tc_slot, consumer_tc_slot] : pairings) {
            auto consumer_it = consumer_configs.find(consumer_risc_id);
            ASSERT_NE(consumer_it, consumer_configs.end());

            const auto* consumer_rc = consumer_it->second;

            ASSERT_LT(producer_tc_slot, 4) << "Max of 4 TCs allowed per producer";
            ASSERT_LT(consumer_tc_slot, 4) << "Max of 4 TCs allowed per consumer";

            auto producer_ptc = producer_rc->config.packed_tile_counter[producer_tc_slot];
            auto consumer_ptc = consumer_rc->config.packed_tile_counter[consumer_tc_slot];

            if (config.cap == ::experimental::AccessPattern::BLOCKED) {
                // For BLOCKED mode, consumer TCs are different from producer TC (remapper-based)
                // Accumulate the consumer TC IDs into expected_consumer_tcs
                if (consumer_idx < 4) {
                    uint8_t consumer_tc_id = ::experimental::get_counter_id(consumer_ptc);
                    expected_consumer_tcs |= (consumer_tc_id & 0x1F) << (consumer_idx * 5);
                    consumer_idx++;
                }

                log_info(
                    tt::LogTest,
                    "BLOCKED: Producer {} TC[{}]=(tensix:{}, tc:{}) -> Consumer {} TC[{}]=(tensix:{}, tc:{})",
                    producer_risc_id,
                    producer_tc_slot,
                    ::experimental::get_tensix_id(producer_ptc),
                    ::experimental::get_counter_id(producer_ptc),
                    consumer_risc_id,
                    consumer_tc_slot,
                    ::experimental::get_tensix_id(consumer_ptc),
                    ::experimental::get_counter_id(consumer_ptc));
            } else {
                // For STRIDED mode, producer and consumer should share the exact same TC
                EXPECT_EQ(producer_ptc, consumer_ptc)
                    << "STRIDED: Producer " << (int)producer_risc_id << " TC[" << (int)producer_tc_slot
                    << "] should share TC with Consumer " << (int)consumer_risc_id << " TC[" << (int)consumer_tc_slot
                    << "]. Producer has (tensix:" << (int)::experimental::get_tensix_id(producer_ptc)
                    << ", tc:" << (int)::experimental::get_counter_id(producer_ptc)
                    << "), Consumer has (tensix:" << (int)::experimental::get_tensix_id(consumer_ptc)
                    << ", tc:" << (int)::experimental::get_counter_id(consumer_ptc) << ")";

                log_info(
                    tt::LogTest,
                    "STRIDED: Producer {} TC[{}] and Consumer {} TC[{}] share (tensix:{}, tc:{})",
                    producer_risc_id,
                    producer_tc_slot,
                    consumer_risc_id,
                    consumer_tc_slot,
                    ::experimental::get_tensix_id(producer_ptc),
                    ::experimental::get_counter_id(producer_ptc));
            }
        }

        if (config.cap == ::experimental::AccessPattern::BLOCKED) {
            uint32_t actual_consumer_tcs = producer_rc->config.consumer_tcs;
            ASSERT_EQ(actual_consumer_tcs, expected_consumer_tcs)
                << "BLOCKED: Producer " << (int)producer_risc_id << " consumer_tcs mismatch. "
                << "Expected: 0x" << std::hex << expected_consumer_tcs << ", Actual: 0x" << actual_consumer_tcs
                << std::dec;

            log_info(
                tt::LogTest,
                "BLOCKED: Producer {} consumer_tcs validated: 0x{:x}",
                producer_risc_id,
                actual_consumer_tcs);
        }
    }
}

TEST_F(MeshDeviceFixture, DMTensixTest1xDFB1Sx1SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // 1 producer with 1 consumer -> 1 TC per producer
        .expected_consumer_tc_count = 1,  // 1 consumer with 1 producer -> 1 TC per consumer
        .producer_to_consumer_pairings = {
            {0, {{0, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 0 TC[0]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB1Sx4SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x1E,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 4,  // 1 producer with 4 consumers -> 4 TCs per producer
        .expected_consumer_tc_count = 1,  // Each consumer pairs with 1 producer -> 1 TC per consumer
        .producer_to_consumer_pairings = {
            {0,
             {
                 {1, 0, 0},  // Producer 0 TC[0] pairs with Consumer risc 1 TC[0]
                 {2, 1, 0},  // Producer 0 TC[1] pairs with Consumer risc 2 TC[0]
                 {3, 2, 0},  // Producer 0 TC[2] pairs with Consumer risc 3 TC[0]
                 {4, 3, 0},  // Producer 0 TC[3] pairs with Consumer risc 4 TC[0]
             }}}};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTensixTest1xDFB4Sx1SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // Each producer pairs with 1 consumer -> 1 TC per producer
        .expected_consumer_tc_count = 4,  // 1 consumer with 4 producers -> 4 TCs per consumer
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 4 TC[0]
            {1, {{4, 0, 1}}},  // Producer 1 TC[0] pairs with Consumer risc 4 TC[1]
            {2, {{4, 0, 2}}},  // Producer 2 TC[0] pairs with Consumer risc 4 TC[2]
            {3, {{4, 0, 3}}},  // Producer 3 TC[0] pairs with Consumer risc 4 TC[3]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx1SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // Each producer pairs with 1 consumer -> 1 TC per producer
        .expected_consumer_tc_count = 4,  // 1 consumer with 4 producers -> 4 TCs per consumer
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 4 TC[0]
            {1, {{4, 0, 1}}},  // Producer 1 TC[0] pairs with Consumer risc 4 TC[1]
            {2, {{4, 0, 2}}},  // Producer 2 TC[0] pairs with Consumer risc 4 TC[2]
            {3, {{4, 0, 3}}},  // Producer 3 TC[0] pairs with Consumer risc 4 TC[3]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx4SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0xF0,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // Equal producers and consumers -> 1 TC per producer
        .expected_consumer_tc_count = 1,  // Equal producers and consumers -> 1 TC per consumer
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 4 TC[0]
            {1, {{5, 0, 0}}},  // Producer 1 TC[0] pairs with Consumer risc 5 TC[0]
            {2, {{6, 0, 0}}},  // Producer 2 TC[0] pairs with Consumer risc 6 TC[0]
            {3, {{7, 0, 0}}},  // Producer 3 TC[0] pairs with Consumer risc 7 TC[0]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB2Sx4SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x3,
        .num_producers = 2,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x3C,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 2,  // 4 consumers / 2 producers = 2 TCs per producer
        .expected_consumer_tc_count = 1,  // 2 producers / 4 consumers = 1 TC per consumer (min 1)
        .producer_to_consumer_pairings = {
            {0,
             {
                 {2, 0, 0},  // Producer 0 TC[0] pairs with Consumer risc 2 TC[0]
                 {4, 1, 0},  // Producer 0 TC[1] pairs with Consumer risc 4 TC[0]
             }},
            {1,
             {
                 {3, 0, 0},  // Producer 1 TC[0] pairs with Consumer risc 3 TC[0]
                 {5, 1, 0},  // Producer 1 TC[1] pairs with Consumer risc 5 TC[0]
             }},
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx2SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x30,
        .num_consumers = 2,
        .cap = ::experimental::AccessPattern::STRIDED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // 2 consumers / 4 producers = 1 TC per producer (min 1)
        .expected_consumer_tc_count = 2,  // 4 producers / 2 consumers = 2 TCs per consumer
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 4 TC[0]
            {1, {{5, 0, 0}}},  // Producer 1 TC[0] pairs with Consumer risc 5 TC[0]
            {2, {{4, 0, 1}}},  // Producer 2 TC[0] pairs with Consumer risc 4 TC[1]
            {3, {{5, 0, 1}}},  // Producer 3 TC[0] pairs with Consumer risc 5 TC[1]
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB1Sx1BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // BLOCKED: each producer has 1 TC
        .expected_consumer_tc_count = 1,  // BLOCKED: each consumer has num_producers TCs = 1
        .producer_to_consumer_pairings = {
            {0, {{1, 0, 0}}},  // Producer 0 TC[0] maps to Consumer risc 1 TC[0] via remapper
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB1Sx4BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x1E,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // BLOCKED: each producer has 1 TC
        .expected_consumer_tc_count = 1,  // BLOCKED: each consumer has num_producers TCs = 1
        .producer_to_consumer_pairings = {
            {0,
             {
                 {1, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 1 TC[0] via remapper
                 {2, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 2 TC[0] via remapper
                 {3, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 3 TC[0] via remapper
                 {4, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0] via remapper
             }}}};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx1BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // BLOCKED: each producer has 1 TC
        .expected_consumer_tc_count = 4,  // BLOCKED: each consumer has num_producers TCs = 4
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0] via remapper
            {1, {{4, 0, 1}}},  // Producer 1 TC[0] maps to Consumer risc 4 TC[1] via remapper
            {2, {{4, 0, 2}}},  // Producer 2 TC[0] maps to Consumer risc 4 TC[2] via remapper
            {3, {{4, 0, 3}}},  // Producer 3 TC[0] maps to Consumer risc 4 TC[3] via remapper
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx4BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0xF0,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // BLOCKED: each producer has 1 TC
        .expected_consumer_tc_count = 4,  // BLOCKED: each consumer has num_producers TCs = 4
        .producer_to_consumer_pairings = {
            {0,
             {
                 {4, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0]
                 {5, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 5 TC[0]
                 {6, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 6 TC[0]
                 {7, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 7 TC[0]
             }},
            {1,
             {
                 {4, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 4 TC[1]
                 {5, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 5 TC[1]
                 {6, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 6 TC[1]
                 {7, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 7 TC[1]
             }},
            {2,
             {
                 {4, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 4 TC[2]
                 {5, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 5 TC[2]
                 {6, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 6 TC[2]
                 {7, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 7 TC[2]
             }},
            {3,
             {
                 {4, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 4 TC[3]
                 {5, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 5 TC[3]
                 {6, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 6 TC[3]
                 {7, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 7 TC[3]
             }},
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

TEST_F(MeshDeviceFixture, DMTest1xDFB4Sx2BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0xF,
        .num_producers = 4,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x30,
        .num_consumers = 2,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // BLOCKED: each producer has 1 TC
        .expected_consumer_tc_count = 4,  // BLOCKED: each consumer has num_producers TCs = 4
        .producer_to_consumer_pairings = {
            {0,
             {
                 {4, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0]
                 {5, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 5 TC[0]
             }},
            {1,
             {
                 {4, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 4 TC[1]
                 {5, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 5 TC[1]
             }},
            {2,
             {
                 {4, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 4 TC[2]
                 {5, 0, 2},  // Producer 2 TC[0] maps to Consumer risc 5 TC[2]
             }},
            {3,
             {
                 {4, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 4 TC[3]
                 {5, 0, 3},  // Producer 3 TC[0] maps to Consumer risc 5 TC[3]
             }},
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

// 2S x 4B: 2 producers (riscs 0,1) with 4 blocked consumers (riscs 2,3,4,5)
// Each producer has 1 TC, each consumer has 2 TCs (num_producers TCs)
// BLOCKED: Each consumer's TC[i] pairs with producer[i]
TEST_F(MeshDeviceFixture, DMTest1xDFB2Sx4BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x3,
        .num_producers = 2,
        .pap = ::experimental::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x3C,
        .num_consumers = 4,
        .cap = ::experimental::AccessPattern::BLOCKED,
        .enable_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    // consumer_risc_mask 0x3C = riscs 2,3,4,5
    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // BLOCKED: each producer has 1 TC
        .expected_consumer_tc_count = 2,  // BLOCKED: each consumer has num_producers TCs = 2
        .producer_to_consumer_pairings = {
            {0,
             {
                 {2, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 2 TC[0]
                 {3, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 3 TC[0]
                 {4, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 4 TC[0]
                 {5, 0, 0},  // Producer 0 TC[0] maps to Consumer risc 5 TC[0]
             }},
            {1,
             {
                 {2, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 2 TC[1]
                 {3, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 3 TC[1]
                 {4, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 4 TC[1]
                 {5, 0, 1},  // Producer 1 TC[0] maps to Consumer risc 5 TC[1]
             }},
        }};

    validate_dfb_tile_counters(program, logical_core, config, expectation);
}

}  // end namespace tt::tt_metal
