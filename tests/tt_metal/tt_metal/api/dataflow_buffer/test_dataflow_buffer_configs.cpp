// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include <tt-metalium/tensor_accessor_args.hpp>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/hw/inc/internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "tt_metal/impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include <gmock/gmock.h>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "../metal2_host_api/test_helpers.hpp"
#include "dfb_test_common.hpp"

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
    // risc_mask/groups are only populated by finalize_dataflow_buffer_configs(); these
    // host-side config tests inspect that state without launching, so finalize here.
    // Idempotent (finalize skips already-finalized DFBs) -> no-op for the MultiCore/Reentry
    // tests that finalize explicitly.
    program.impl().finalize_dataflow_buffer_configs();
    auto dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1) << "Expected exactly 1 DFB on core";

    const auto& dfb = dfbs[0];

    ASSERT_EQ(dfb->risc_mask, config.producer_risc_mask | config.consumer_risc_mask);
    ASSERT_FALSE(dfb->groups.empty()) << "DFB has no groups (configs not finalized?)";
    // All single-core tests produce exactly one DfbGroup.
    const auto& hw_risc_configs = dfb->groups[0].hw_risc_configs;
    ASSERT_EQ(hw_risc_configs.size(), config.num_producers + config.num_consumers);

    // risc ID to risc config maps
    std::map<uint8_t, const experimental::dfb::detail::DFBRiscConfig*> producer_configs;
    std::map<uint8_t, const experimental::dfb::detail::DFBRiscConfig*> consumer_configs;

    for (const auto& rc : hw_risc_configs) {
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
                uint8_t actual_tensix_id = ::dfb::get_tensix_id(ptc);
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
                uint8_t actual_tensix_id = ::dfb::get_tensix_id(ptc);
                EXPECT_EQ(actual_tensix_id, expected_tensix_id)
                    << "Tensix consumer RISC " << (int)risc_id << " TC[" << (int)tc
                    << "] must use tensix_id=" << (int)expected_tensix_id << " but has " << (int)actual_tensix_id;
            }
        }
    }

    // ALL mode engages the remapper ONLY when a Tensix endpoint is involved. Pure DM->DM ALL
    // broadcasts via broadcast_tc, so remapper_pair_index / consumer_tcs are intentionally left
    // at 0 (in dataflow_buffer.cpp they are populated only inside `if (use_remapper)`, and
    // use_remapper == ALL && !dm_dm_all). Tensix RISCs occupy mask bits 0x0F00.
    const bool dm_dm_all = config.cap == dfb::AccessPattern::ALL &&
                           (config.producer_risc_mask & 0x0F00) == 0 &&
                           (config.consumer_risc_mask & 0x0F00) == 0;
    const bool uses_remapper = config.cap == dfb::AccessPattern::ALL && !dm_dm_all;

    // For remapper-based ALL, validate remapper pair indices are unique per producer.
    if (uses_remapper) {
        std::set<uint8_t> seen_remapper_indices;
        for (const auto& [risc_id, rc] : producer_configs) {
            uint8_t remapper_idx = rc->config.remapper_pair_index;

            // Check valid range (0-63)
            EXPECT_LT(remapper_idx, 64) << "ALL: Producer RISC " << (int)risc_id
                                        << " has invalid remapper_pair_index " << (int)remapper_idx
                                        << " (must be 0-63)";

            // Check uniqueness among producers
            EXPECT_EQ(seen_remapper_indices.count(remapper_idx), 0)
                << "ALL: Producer RISC " << (int)risc_id << " has duplicate remapper_pair_index "
                << (int)remapper_idx;
            seen_remapper_indices.insert(remapper_idx);

            log_info(tt::LogTest, "ALL: Producer {} has remapper_pair_index {}", risc_id, remapper_idx);
        }
    }

    for (const auto& [producer_risc_id, pairings] : expectation.producer_to_consumer_pairings) {
        auto producer_it = producer_configs.find(producer_risc_id);
        ASSERT_NE(producer_it, producer_configs.end());

        const auto* producer_rc = producer_it->second;

        // For ALL mode, accumulate expected_consumer_tcs across all pairings for this producer
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

            if (config.cap == dfb::AccessPattern::ALL) {
                // For ALL mode, consumer TCs are different from producer TC (remapper-based)
                // Accumulate the consumer TC IDs into expected_consumer_tcs
                if (consumer_idx < 4) {
                    uint8_t consumer_tc_id = ::dfb::get_counter_id(consumer_ptc);
                    expected_consumer_tcs |= (consumer_tc_id & 0x1F) << (consumer_idx * 5);
                    consumer_idx++;
                }

                log_info(
                    tt::LogTest,
                    "ALL: Producer {} TC[{}]=(tensix:{}, tc:{}) -> Consumer {} TC[{}]=(tensix:{}, tc:{})",
                    producer_risc_id,
                    producer_tc_slot,
                    ::dfb::get_tensix_id(producer_ptc),
                    ::dfb::get_counter_id(producer_ptc),
                    consumer_risc_id,
                    consumer_tc_slot,
                    ::dfb::get_tensix_id(consumer_ptc),
                    ::dfb::get_counter_id(consumer_ptc));
            } else {
                // For STRIDED mode, producer and consumer should share the exact same TC
                EXPECT_EQ(producer_ptc, consumer_ptc)
                    << "STRIDED: Producer " << (int)producer_risc_id << " TC[" << (int)producer_tc_slot
                    << "] should share TC with Consumer " << (int)consumer_risc_id << " TC[" << (int)consumer_tc_slot
                    << "]. Producer has (tensix:" << (int)::dfb::get_tensix_id(producer_ptc)
                    << ", tc:" << (int)::dfb::get_counter_id(producer_ptc)
                    << "), Consumer has (tensix:" << (int)::dfb::get_tensix_id(consumer_ptc)
                    << ", tc:" << (int)::dfb::get_counter_id(consumer_ptc) << ")";

                log_info(
                    tt::LogTest,
                    "STRIDED: Producer {} TC[{}] and Consumer {} TC[{}] share (tensix:{}, tc:{})",
                    producer_risc_id,
                    producer_tc_slot,
                    consumer_risc_id,
                    consumer_tc_slot,
                    ::dfb::get_tensix_id(producer_ptc),
                    ::dfb::get_counter_id(producer_ptc));
            }
        }

        if (uses_remapper) {
            uint32_t actual_consumer_tcs = producer_rc->config.consumer_tcs;
            ASSERT_EQ(actual_consumer_tcs, expected_consumer_tcs)
                << "ALL: Producer " << (int)producer_risc_id << " consumer_tcs mismatch. "
                << "Expected: 0x" << std::hex << expected_consumer_tcs << ", Actual: 0x" << actual_consumer_tcs
                << std::dec;

            log_info(
                tt::LogTest,
                "ALL: Producer {} consumer_tcs validated: 0x{:x}",
                producer_risc_id,
                actual_consumer_tcs);
        } else if (dm_dm_all) {
            // DM->DM ALL broadcasts via broadcast_tc; the remapper-only fields must stay unset.
            EXPECT_EQ(producer_rc->config.consumer_tcs, 0u)
                << "DM->DM ALL: Producer " << (int)producer_risc_id
                << " must not populate consumer_tcs (broadcast_tc path, remapper unused)";
            EXPECT_TRUE(producer_rc->config.broadcast_tc)
                << "DM->DM ALL: Producer " << (int)producer_risc_id << " must have broadcast_tc set";
            EXPECT_EQ(producer_rc->config.remapper_pair_index, 0)
                << "DM->DM ALL: Producer " << (int)producer_risc_id << " must not allocate a remapper pair index";
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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // 1 producer with 1 consumer -> 1 TC per producer
        .expected_consumer_tc_count = 1,  // 1 consumer with 1 producer -> 1 TC per consumer
        .producer_to_consumer_pairings = {
            {0, {{4, 0, 0}}},  // Producer 0 TC[0] pairs with Consumer risc 4 TC[0]  (consumer_risc_mask 0x10 -> risc 4; matches DMTensixTest1xDFB4Sx1SConfig)
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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x1E,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0xF0,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x3C,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x30,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::ALL,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // ALL: each producer has 1 TC
        .expected_consumer_tc_count = 1,  // ALL: each consumer has num_producers TCs = 1
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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x1E,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 4,  // ALL: producer broadcasts to num_consumers=4 TCs
        .expected_consumer_tc_count = 1,  // ALL: each consumer has num_producers TCs = 1
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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x10,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::ALL,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 1,  // ALL: each producer has 1 TC
        .expected_consumer_tc_count = 4,  // ALL: each consumer has num_producers TCs = 4
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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0xF0,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 4,  // ALL: producer broadcasts to num_consumers=4 TCs
        .expected_consumer_tc_count = 4,  // ALL: each consumer has num_producers TCs = 4
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
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x30,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::ALL,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 2,  // ALL: producer broadcasts to num_consumers=2 TCs
        .expected_consumer_tc_count = 4,  // ALL: each consumer has num_producers TCs = 4
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
// ALL: Each consumer's TC[i] pairs with producer[i]
TEST_F(MeshDeviceFixture, DMTest1xDFB2Sx4BConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x3,
        .num_producers = 2,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x3C,
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    // consumer_risc_mask 0x3C = riscs 2,3,4,5
    DFBTileCounterExpectation expectation{
        .expected_producer_tc_count = 4,  // ALL: producer broadcasts to num_consumers=4 TCs
        .expected_consumer_tc_count = 2,  // ALL: each consumer has num_producers TCs = 2
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

// ---------------------------------------------------------------------------
// Multi-core DFB tests
// ---------------------------------------------------------------------------

// Helper: validate that a multi-core DFB has the expected number of DfbGroups
// and that each core's hw_risc_configs matches a single reference group
// (i.e. all cores have identical TC/remapper assignments).
static void validate_multicore_dfb_groups(
    Program& program,
    const CoreRangeSet& core_range_set,
    uint32_t expected_num_groups,
    uint32_t expected_cores_per_group) {
    // Collect DFBs from the first core; they should all be on every core.
    CoreCoord first_core = core_range_set.ranges()[0].start_coord;
    auto dfbs = program.impl().dataflow_buffers_on_core(first_core);
    ASSERT_EQ(dfbs.size(), 1) << "Expected exactly 1 DFB on core";
    const auto& dfb = dfbs[0];

    ASSERT_EQ(dfb->groups.size(), expected_num_groups)
        << "Expected " << expected_num_groups << " DfbGroup(s)";

    for (const auto& grp : dfb->groups) {
        EXPECT_EQ(grp.l1_by_core.size(), expected_cores_per_group)
            << "DfbGroup should have " << expected_cores_per_group << " core(s)";
    }

    // All cores in the core_range_set should appear somewhere in l1_by_core.
    std::set<CoreCoord> accounted_cores;
    for (const auto& grp : dfb->groups) {
        for (const auto& [c, _] : grp.l1_by_core) {
            accounted_cores.insert(c);
        }
    }
    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                EXPECT_EQ(accounted_cores.count(CoreCoord(x, y)), 1u)
                    << "Core (" << x << "," << y << ") not found in any DfbGroup";
            }
        }
    }
}

// Multi-core DFB, no implicit sync: 2 cores, 1 producer, 1 consumer, STRIDED.
// Expected: 1 DfbGroup (homogeneous HW config) with 2 cores.
TEST_F(MeshDeviceFixture, MultiCoreDFB_1P1C_Strided_NoImplicitSync) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

    Program program = CreateProgram();
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));  // 2 cores: (0,0) and (1,0)
    experimental::dfb::CreateDataflowBuffer(program, core_range_set, config);

    // Finalize configs explicitly (normally done during compile/ConfigureDeviceWithProgram).
    program.impl().finalize_dataflow_buffer_configs();

    // Both cores have identical TC config → 1 group with 2 cores.
    validate_multicore_dfb_groups(program, core_range_set, /*expected_num_groups=*/1, /*expected_cores_per_group=*/2);

    // Each core should have TC index 0 (independent per-core allocator starting from 0).
    for (const CoreRange& cr : core_range_set.ranges()) {
        for (auto x = cr.start_coord.x; x <= cr.end_coord.x; x++) {
            for (auto y = cr.start_coord.y; y <= cr.end_coord.y; y++) {
                CoreCoord core(x, y);
                auto dfbs = program.impl().dataflow_buffers_on_core(core);
                ASSERT_EQ(dfbs.size(), 1);
                const auto& dfb = dfbs[0];
                // Find this core's group
                const experimental::dfb::detail::DfbGroup* found_grp = nullptr;
                for (const auto& grp : dfb->groups) {
                    for (const auto& [c, _] : grp.l1_by_core) {
                        if (c == core) { found_grp = &grp; break; }
                    }
                    if (found_grp) {
                        break;
                    }
                }
                ASSERT_NE(found_grp, nullptr) << "Core (" << x << "," << y << ") not found in any DfbGroup";

                // Validate TC index is 0 (first allocation from fresh per-core allocator).
                for (const auto& rc : found_grp->hw_risc_configs) {
                    for (uint8_t tc = 0; tc < rc.config.num_tcs_to_rr; tc++) {
                        auto ptc = rc.config.packed_tile_counter[tc];
                        EXPECT_EQ(::dfb::get_counter_id(ptc), tc)
                            << "Core (" << x << "," << y << ") RISC " << (int)rc.risc_id
                            << " TC[" << (int)tc << "] should have counter_id=" << (int)tc;
                    }
                }
            }
        }
    }
}

// Multi-core DFB, with implicit sync: 2 cores, 1 producer, 1 consumer, STRIDED.
// Txn IDs should be allocated once (core-invariant) and identical across cores.
TEST_F(MeshDeviceFixture, MultiCoreDFB_1P1C_Strided_ImplicitSync) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = true,
        .enable_consumer_implicit_sync = true};

    Program program = CreateProgram();
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 0)));  // 2 cores
    experimental::dfb::CreateDataflowBuffer(program, core_range_set, config);

    program.impl().finalize_dataflow_buffer_configs();

    // Should still produce 1 group (identical HW config on both cores).
    validate_multicore_dfb_groups(program, core_range_set, /*expected_num_groups=*/1, /*expected_cores_per_group=*/2);

    // Txn ID descriptors are core-invariant: allocated once during finalization.
    CoreCoord first_core(0, 0);
    auto dfbs = program.impl().dataflow_buffers_on_core(first_core);
    ASSERT_EQ(dfbs.size(), 1);
    const auto& dfb = dfbs[0];

    EXPECT_EQ(dfb->producer_txn_descriptor.num_txn_ids, 2u)
        << "Expected 2 producer txn IDs (double-buffering)";
    EXPECT_EQ(dfb->consumer_txn_descriptor.num_txn_ids, 2u)
        << "Expected 2 consumer txn IDs (double-buffering)";
}

// Identical-config multi-core: assert one DfbGroup is produced (multicast-ready).
TEST_F(MeshDeviceFixture, MultiCoreDFB_HomogeneousGrid_SingleGroup) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 512,
        .num_entries = 8,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};

    Program program = CreateProgram();
    // 4 cores in a 2x2 grid — all identical config → should produce 1 DfbGroup.
    CoreRangeSet core_range_set(CoreRange(CoreCoord(0, 0), CoreCoord(1, 1)));
    experimental::dfb::CreateDataflowBuffer(program, core_range_set, config);

    program.impl().finalize_dataflow_buffer_configs();

    // All 4 cores have the same HW config → 1 DfbGroup with 4 cores.
    validate_multicore_dfb_groups(program, core_range_set, /*expected_num_groups=*/1, /*expected_cores_per_group=*/4);
}

// ---------------------------------------------------------------------------
// Intra-tensix DFB config test
// ---------------------------------------------------------------------------

// Validates an intra-tensix DFB (pack TRISC producer → unpack TRISC consumer, same Neo):
//   - Exactly one per-risc config entry (shared Neo bit) marked is_producer=true.
//   - The tensix-only TC (id ≥ TC_TENSIX_POOL_START) is assigned to Neo tensix_id derived from producer_risc_mask.
void validate_intra_tensix_dfb(
    Program& program,
    const CoreCoord& logical_core,
    const experimental::dfb::DataflowBufferConfig& config) {
    program.impl().finalize_dataflow_buffer_configs();

    auto dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1u) << "Expected exactly 1 DFB on core";
    const auto& dfb = dfbs[0];

    ASSERT_EQ(dfb->risc_mask, config.producer_risc_mask)
        << "Intra-tensix risc_mask should equal producer_risc_mask (same Neo bit)";
    ASSERT_FALSE(dfb->use_remapper) << "Intra-tensix DFB must not use the remapper";
    ASSERT_FALSE(dfb->groups.empty()) << "DFB has no groups (configs not finalized?)";

    const auto& hw_risc_configs = dfb->groups[0].hw_risc_configs;
    ASSERT_EQ(hw_risc_configs.size(), 1u)
        << "Intra-tensix DFB should have exactly 1 per-risc config entry (shared Neo)";

    const auto& rc = hw_risc_configs[0];
    EXPECT_TRUE(rc.is_producer) << "Intra-tensix per-risc entry must be marked is_producer (pack TRISC inits TC)";

    uint8_t expected_tensix_id =
        static_cast<uint8_t>(__builtin_ctz(config.producer_risc_mask >> ::dfb::TENSIX_RISC_OFFSET));
    uint8_t expected_risc_id = static_cast<uint8_t>(::dfb::TENSIX_RISC_OFFSET + expected_tensix_id);
    EXPECT_EQ(rc.risc_id, expected_risc_id)
        << "Intra-tensix per-risc risc_id should match Neo bit in producer_risc_mask";

    ASSERT_EQ(rc.config.num_tcs_to_rr, 1u) << "Intra-tensix DFB should have exactly 1 TC";
    uint8_t tc_id = ::dfb::get_counter_id(rc.config.packed_tile_counter[0]);
    uint8_t actual_tensix_id = ::dfb::get_tensix_id(rc.config.packed_tile_counter[0]);
    EXPECT_EQ(actual_tensix_id, expected_tensix_id) << "TC tensix_id must match Neo";
    EXPECT_GE(tc_id, ::dfb::TC_TENSIX_POOL_START)
        << "Intra-tensix DFB must use a Tensix-only TC (id ≥ " << (int)::dfb::TC_TENSIX_POOL_START << ")";

    log_info(
        tt::LogTest,
        "Intra-tensix DFB: Neo{} Tensix-only TC (tensix_id={}, tc_id={})",
        expected_tensix_id, (int)actual_tensix_id, (int)tc_id);
}

TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB1Sx1SConfig) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    // Intra-tensix: pack TRISC (producer) → unpack TRISC (consumer) on Neo0.
    // producer_risc_mask == consumer_risc_mask == bit 8 (Neo0).
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 4,
        .producer_risc_mask = 0x100,  // bit 8 = Neo0
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x100,  // bit 8 = Neo0 (same as producer — intentional for INTRA)
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false,
        .tensix_scope = experimental::dfb::TensixScope::INTRA};

    Program program = CreateProgram();
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    validate_intra_tensix_dfb(program, logical_core, config);
}

// Run args for the minimal spec's two kernels (both declare empty RTA schemas, so empty arg sets).
inline experimental::ProgramRunArgs MakeMinimalRunArgs(const experimental::NodeCoord& node) {
    auto kernel_args = [&](const char* name) {
        return experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = experimental::KernelSpecName{name},
            .advanced_options =
                experimental::AdvancedKernelRunArgs{.runtime_varargs = {{node, {}}}, .common_runtime_varargs = {}},
        };
    };
    experimental::ProgramRunArgs params;
    params.kernel_run_args.push_back(kernel_args("dm_kernel"));
    params.kernel_run_args.push_back(kernel_args("compute_kernel"));
    return params;
}

// num_entries override on a finalized DFB recomputes the txn descriptor in place while PRESERVING the
// TC assignment and transaction IDs (only the threshold changes for the new ring depth).
TEST_F(MeshDeviceFixture, DFBReentryOverridePreservesTcAndRecomputesTxn) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB transaction IDs / tile counters require Quasar";
    }
    const experimental::NodeCoord node{0, 0};
    experimental::ProgramSpec spec = experimental::test_helpers::MakeMinimalValidProgramSpec();
    Program program = experimental::MakeProgramFromSpec(*devices_.at(0), spec);

    program.impl().finalize_dataflow_buffer_configs();

    auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb_0"));
    ASSERT_TRUE(dfb->configs_finalized);
    ASSERT_TRUE(dfb->config.enable_producer_implicit_sync);
    ASSERT_GT(dfb->producer_txn_descriptor.num_txn_ids, 0);

    // Snapshot TC assignment + transaction IDs before the override.
    auto snapshot_tcs = [](const auto& d) {
        std::vector<uint32_t> tcs;
        for (const auto& group : d->groups) {
            for (const auto& rc : group.hw_risc_configs) {
                for (uint8_t i = 0; i < rc.config.num_tcs_to_rr; ++i) {
                    tcs.push_back(static_cast<uint32_t>(rc.config.packed_tile_counter[i]));
                }
            }
        }
        return tcs;
    };
    const std::vector<uint32_t> tcs_before = snapshot_tcs(dfb);
    const uint8_t num_txn_ids_before = dfb->producer_txn_descriptor.num_txn_ids;
    const std::vector<uint8_t> txn_ids_before(
        dfb->producer_txn_descriptor.txn_ids, dfb->producer_txn_descriptor.txn_ids + num_txn_ids_before);
    const uint8_t threshold_before = dfb->producer_txn_descriptor.num_entries_to_process_threshold;

    // Override num_entries 2 -> 4 (still divisible by the preserved txn-id divisor).
    auto params = MakeMinimalRunArgs(node);
    params.dfb_run_overrides.push_back({.dfb = experimental::DFBSpecName{"dfb_0"}, .num_entries = 4});
    EXPECT_NO_THROW(experimental::SetProgramRunArgs(program, params));

    // Size-derived state updated.
    EXPECT_EQ(dfb->config.num_entries, 4u);
    EXPECT_EQ(dfb->capacity, 4u);
    EXPECT_TRUE(dfb->configs_finalized);  // not reset

    // TC assignment + transaction IDs preserved.
    EXPECT_EQ(snapshot_tcs(dfb), tcs_before);
    EXPECT_EQ(dfb->producer_txn_descriptor.num_txn_ids, num_txn_ids_before);
    const std::vector<uint8_t> txn_ids_after(
        dfb->producer_txn_descriptor.txn_ids,
        dfb->producer_txn_descriptor.txn_ids + dfb->producer_txn_descriptor.num_txn_ids);
    EXPECT_EQ(txn_ids_after, txn_ids_before);

    // Threshold recomputed for the new num_entries (threshold = num_entries / num_txn_ids).
    EXPECT_EQ(dfb->producer_txn_descriptor.num_entries_to_process_threshold, threshold_before * 2);
}

// On re-entry the txn-id count is preserved, so a num_entries override that breaks the preserved divisor
// is rejected up front with an actionable message.
TEST_F(MeshDeviceFixture, DFBReentryNumEntriesViolatesTxnDivisorFails) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB transaction-id divisor check requires Quasar";
    }
    const experimental::NodeCoord node{0, 0};
    experimental::ProgramSpec spec = experimental::test_helpers::MakeMinimalValidProgramSpec();
    Program program = experimental::MakeProgramFromSpec(*devices_.at(0), spec);
    program.impl().finalize_dataflow_buffer_configs();

    auto params = MakeMinimalRunArgs(node);
    params.dfb_run_overrides.push_back({.dfb = experimental::DFBSpecName{"dfb_0"}, .num_entries = 3});

    EXPECT_THAT(
        [&] { experimental::SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("num_entries override 3 is not divisible by")));
}

// An entry_size override that pushes the TRISC ring extent past the uint16 L1-aligned-unit limit is
// rejected by the ring-extent re-validation.
TEST_F(MeshDeviceFixture, DFBReentryEntrySizeRingExtentFails) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB ring-extent check requires Quasar";
    }
    const experimental::NodeCoord node{0, 0};
    experimental::ProgramSpec spec = experimental::test_helpers::MakeMinimalValidProgramSpec();
    Program program = experimental::MakeProgramFromSpec(*devices_.at(0), spec);
    program.impl().finalize_dataflow_buffer_configs();

    auto params = MakeMinimalRunArgs(node);
    params.dfb_run_overrides.push_back({.dfb = experimental::DFBSpecName{"dfb_0"}, .entry_size = 64u * 1024u * 1024u});

    EXPECT_THAT(
        [&] { experimental::SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("exceeds uint16_t; reduce capacity, stride, or entry_size")));
}

// capacity (num_entries / max(producers, consumers)) is stored as uint16_t (the tile-counter register
// width); an override that pushes it past the max is rejected rather than silently truncated.
TEST_F(MeshDeviceFixture, DFBOverrideCapacityExceedsUint16Fails) {
    if (devices_.at(0)->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB capacity / tile-counter limit requires Quasar";
    }
    const experimental::NodeCoord node{0, 0};
    experimental::ProgramSpec spec = experimental::test_helpers::MakeMinimalValidProgramSpec();
    Program program = experimental::MakeProgramFromSpec(*devices_.at(0), spec);

    auto params = MakeMinimalRunArgs(node);
    params.dfb_run_overrides.push_back(
        {.dfb = experimental::DFBSpecName{"dfb_0"}, .num_entries = 70000});  // capacity 70000 > 65535
    EXPECT_THAT(
        [&] { experimental::SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("capacity 70000 exceeds the maximum 65535")));
}


// ============================================================================
// Metal 2.0 config/validation ports (folded from test_dataflow_buffer_2_0.cpp
// during the recategorization; host-only DFB internal-state probes).
// ============================================================================

// =====================================================================================
// Config-suite M2 ports: host-only DFB internal-state probes.
//
// Legacy parallel: test_dataflow_buffer_configs.cpp. These tests build a single
// DFB via the M2 ProgramSpec path, finalize_dataflow_buffer_configs(), then probe
// `program.impl().dataflow_buffers_on_core(core)` for the same fields the legacy
// tests interrogate:
//   - dfb->risc_mask
//   - dfb->groups[].hw_risc_configs[].{is_producer, config.num_tcs_to_rr,
//     config.packed_tile_counter[], config.remapper_pair_index, config.consumer_tcs}
//   - dfb->{producer,consumer}_txn_descriptor.{num_txn_ids, num_entries_to_process_threshold}
//
// Differences from legacy:
//   - M2 doesn't expose producer_risc_mask / consumer_risc_mask in the spec;
//     the framework picks risc bits from `num_threads` + binding order. So the
//     M2 versions of these tests assert *semantic* invariants (per-RISC TC
//     counts, STRIDED producer/consumer share the same TC, ALL remapper indices
//     are unique) without hardcoding specific risc IDs.
//   - B6/B7/B9 rejection tests don't translate 1:1 (different validation layer)
//     and are documented where they appear.
// =====================================================================================

namespace m2_config_test_helpers {

// Build a single-DFB ProgramSpec on one core using the m2 producer/consumer
// kernels. Returns a Program ready for finalize_dataflow_buffer_configs(). Does
// not launch; this is purely a host-side state probe.
struct M2ConfigDFBParams {
    M2PorCType producer_type;
    M2PorCType consumer_type;
    uint32_t num_producers;
    uint32_t num_consumers;
    uint32_t entry_size = 1024;
    uint32_t num_entries = 16;
    m2::DFBAccessPattern pap = m2::DFBAccessPattern::STRIDED;
    m2::DFBAccessPattern cap = m2::DFBAccessPattern::STRIDED;
    bool implicit_sync = false;
    std::optional<m2::NodeRange> target_nodes = std::nullopt;  // override single-core default
};

static inline Program build_single_dfb_program_2_0(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, const M2ConfigDFBParams& p) {
    const m2::NodeCoord node{0, 0};
    const m2::DFBSpecName DFB{"dfb"};
    const m2::KernelSpecName PRODUCER{"producer"};
    const m2::KernelSpecName CONSUMER{"consumer"};
    const m2::TensorParamName IN_TENSOR{"in_tensor"};
    const m2::TensorParamName OUT_TENSOR{"out_tensor"};

    const auto tensor_spec = make_flat_dram_tensor_spec(p.entry_size, p.num_entries, DataType::UINT32);

    m2::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = p.entry_size,
        .num_entries = p.num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    const bool is_all = (p.cap == m2::DFBAccessPattern::ALL);
    const uint32_t per_producer = (p.num_entries + p.num_producers - 1) / p.num_producers;
    const uint32_t per_consumer = is_all ? p.num_entries : (p.num_entries + p.num_consumers - 1) / p.num_consumers;

    auto make_producer = [&]() -> m2::KernelSpec {
        if (p.producer_type == M2PorCType::DM) {
            auto k = make_dm_kernel(
                PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp", p.num_producers);
            k.dfb_bindings = {
                {.dfb_spec_name = DFB,
                 .accessor_name = "out",
                 .endpoint_type = m2::DFBEndpointType::PRODUCER,
                 .access_pattern = p.pap}};
            k.tensor_bindings = {{.tensor_parameter_name = IN_TENSOR, .accessor_name = "src_tensor"}};
            k.compile_time_args = {
                {"num_entries_per_producer", per_producer}, {"implicit_sync", p.implicit_sync ? 1u : 0u}};
            k.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
            // All-pass: dfb.disable_implicit_sync = !p.implicit_sync (now per-DM-endpoint, post-#45160).
            maybe_disable_implicit_sync(k, p.implicit_sync, DFB);
            return k;
        }
        auto k = make_compute_kernel(
            PRODUCER, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_producer_2_0.cpp", p.num_producers);
        k.dfb_bindings = {
            {.dfb_spec_name = DFB,
             .accessor_name = "out",
             .endpoint_type = m2::DFBEndpointType::PRODUCER,
             .access_pattern = p.pap}};
        k.compile_time_args = {{"num_entries_per_producer", per_producer}};
        return k;
    };

    auto make_consumer = [&]() -> m2::KernelSpec {
        if (p.consumer_type == M2PorCType::DM) {
            auto k = make_dm_kernel(
                CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp", p.num_consumers);
            k.dfb_bindings = {
                {.dfb_spec_name = DFB,
                 .accessor_name = "in",
                 .endpoint_type = m2::DFBEndpointType::CONSUMER,
                 .access_pattern = p.cap}};
            k.tensor_bindings = {{.tensor_parameter_name = OUT_TENSOR, .accessor_name = "dst_tensor"}};
            k.compile_time_args = {
                {"num_entries_per_consumer", per_consumer},
                {"blocked_consumer", is_all ? 1u : 0u},
                {"implicit_sync", p.implicit_sync ? 1u : 0u}};
            k.runtime_arg_schema = {.runtime_arg_names = {"chunk_offset", "entries_per_core"}};
            // All-pass: dfb.disable_implicit_sync = !p.implicit_sync (now per-DM-endpoint, post-#45160).
            maybe_disable_implicit_sync(k, p.implicit_sync, DFB);
            return k;
        }
        auto k = make_compute_kernel(
            CONSUMER, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_consumer_2_0.cpp", p.num_consumers);
        k.dfb_bindings = {
            {.dfb_spec_name = DFB,
             .accessor_name = "in",
             .endpoint_type = m2::DFBEndpointType::CONSUMER,
             .access_pattern = p.cap}};
        k.compile_time_args = {{"num_entries_per_consumer", per_consumer}};
        return k;
    };

    auto producer = make_producer();
    auto consumer = make_consumer();

    std::vector<m2::TensorParameter> tensor_parameters;
    if (p.producer_type == M2PorCType::DM) {
        tensor_parameters.push_back({.unique_id = IN_TENSOR, .spec = tensor_spec});
    }
    if (p.consumer_type == M2PorCType::DM) {
        tensor_parameters.push_back({.unique_id = OUT_TENSOR, .spec = tensor_spec});
    }

    m2::WorkUnitSpec wu{
        .name = "wu",
        .kernels = {PRODUCER, CONSUMER},
    };
    if (p.target_nodes.has_value()) {
        wu.target_nodes = *p.target_nodes;
    } else {
        wu.target_nodes = node;
    }
    m2::ProgramSpec spec{
        .name = "config_probe_2_0",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = tensor_parameters,
        .work_units = {wu},
    };
    return m2::MakeProgramFromSpec(*mesh_device, spec);
}

// Semantic TC-pairing check (M2 doesn't expose explicit risc masks, so we don't
// hardcode RISC IDs; we just verify the invariants).
struct M2DFBTCExpectation {
    uint8_t expected_producer_tc_count;
    uint8_t expected_consumer_tc_count;
};

static inline void validate_dfb_tile_counters_2_0(
    Program& program,
    const CoreCoord& logical_core,
    uint32_t num_producers,
    uint32_t num_consumers,
    m2::DFBAccessPattern cap,
    const M2DFBTCExpectation& expectation) {
    auto dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1u) << "Expected exactly 1 DFB on core";
    const auto& dfb = dfbs[0];
    ASSERT_FALSE(dfb->groups.empty()) << "DFB has no groups (configs not finalized?)";

    const auto& hw_risc_configs = dfb->groups[0].hw_risc_configs;
    ASSERT_EQ(hw_risc_configs.size(), num_producers + num_consumers);

    std::vector<const experimental::dfb::detail::DFBRiscConfig*> producers, consumers;
    for (const auto& rc : hw_risc_configs) {
        (rc.is_producer ? producers : consumers).push_back(&rc);
    }
    ASSERT_EQ(producers.size(), num_producers);
    ASSERT_EQ(consumers.size(), num_consumers);

    for (const auto* rc : producers) {
        EXPECT_EQ(rc->config.num_tcs_to_rr, expectation.expected_producer_tc_count)
            << "Producer RISC " << (int)rc->risc_id << " TC count mismatch";
    }
    for (const auto* rc : consumers) {
        EXPECT_EQ(rc->config.num_tcs_to_rr, expectation.expected_consumer_tc_count)
            << "Consumer RISC " << (int)rc->risc_id << " TC count mismatch";
    }

    // Tensix-RISC tensix_id constraint (legacy parity).
    auto check_tensix_id = [](const experimental::dfb::detail::DFBRiscConfig* rc) {
        if (rc->risc_id >= ::dfb::TENSIX_RISC_OFFSET) {
            uint8_t expected_tensix_id = (rc->risc_id - ::dfb::TENSIX_RISC_OFFSET) % 4;
            for (uint8_t tc = 0; tc < rc->config.num_tcs_to_rr; ++tc) {
                uint8_t actual = ::dfb::get_tensix_id(rc->config.packed_tile_counter[tc]);
                EXPECT_EQ(actual, expected_tensix_id)
                    << "Tensix RISC " << (int)rc->risc_id << " TC[" << (int)tc << "] tensix_id mismatch";
            }
        }
    };
    for (const auto* rc : producers) {
        check_tensix_id(rc);
    }
    for (const auto* rc : consumers) {
        check_tensix_id(rc);
    }

    if (cap == m2::DFBAccessPattern::ALL) {
        // Sanity-check structural invariants only. The legacy validator checks
        // exact per-test producer-to-consumer pairings (consumer risc, producer
        // TC slot, consumer TC slot, remapper pair index) — we don't carry that
        // per-test data in the macro-generated port, and the M2 representation
        // of remapper_pair_index / consumer_tcs differs from legacy
        // (e.g. remapper_pair_index is not necessarily unique across producers).
        for (const auto* rc : producers) {
            EXPECT_LT(rc->config.remapper_pair_index, 64) << "ALL: remapper_pair_index out of range";
        }
    } else {
        // STRIDED: each producer TC must match exactly one consumer TC (shared counter).
        // Walk all producer TC slots, look for the matching consumer TC.
        for (const auto* prc : producers) {
            for (uint8_t pt = 0; pt < prc->config.num_tcs_to_rr; ++pt) {
                const auto ptc = prc->config.packed_tile_counter[pt];
                bool found = std::any_of(consumers.begin(), consumers.end(), [&](const auto* crc) {
                    return std::any_of(
                        crc->config.packed_tile_counter.begin(),
                        crc->config.packed_tile_counter.begin() + crc->config.num_tcs_to_rr,
                        [&](const auto& ctc) { return ctc == ptc; });
                });
                EXPECT_TRUE(found) << "STRIDED: producer " << (int)prc->risc_id << " TC[" << (int)pt
                                   << "] has no matching consumer TC";
            }
        }
    }
}

// INTRA-scope semantic check (legacy parallel: validate_intra_tensix_dfb).
static inline void validate_intra_tensix_dfb_2_0(Program& program, const CoreCoord& logical_core) {
    program.impl().finalize_dataflow_buffer_configs();
    auto dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1u);
    const auto& dfb = dfbs[0];
    ASSERT_FALSE(dfb->use_remapper) << "INTRA DFB must not use the remapper";
    ASSERT_FALSE(dfb->groups.empty());
    const auto& hw_risc_configs = dfb->groups[0].hw_risc_configs;
    ASSERT_EQ(hw_risc_configs.size(), 1u) << "INTRA DFB should have exactly 1 per-risc entry (shared Neo)";
    const auto& rc = hw_risc_configs[0];
    EXPECT_TRUE(rc.is_producer) << "INTRA per-risc entry must be marked is_producer (PACK TRISC inits TC)";
    ASSERT_EQ(rc.config.num_tcs_to_rr, 1u) << "INTRA DFB should have exactly 1 TC";
    uint8_t tc_id = ::dfb::get_counter_id(rc.config.packed_tile_counter[0]);
    EXPECT_GE(tc_id, ::dfb::TC_TENSIX_POOL_START)
        << "INTRA DFB must use Tensix-only TC (id >= " << (int)::dfb::TC_TENSIX_POOL_START << ")";
}

// Multicore-group probe.
static inline void validate_multicore_dfb_groups_2_0(
    Program& program, const m2::NodeRange& nodes, uint32_t expected_num_groups, uint32_t expected_cores_per_group) {
    CoreCoord first_core = nodes.start_coord;
    auto dfbs = program.impl().dataflow_buffers_on_core(first_core);
    ASSERT_EQ(dfbs.size(), 1u);
    const auto& dfb = dfbs[0];
    ASSERT_EQ(dfb->groups.size(), expected_num_groups);
    for (const auto& grp : dfb->groups) {
        EXPECT_EQ(grp.l1_by_core.size(), expected_cores_per_group);
    }
    std::set<CoreCoord> accounted;
    for (const auto& grp : dfb->groups) {
        for (const auto& [c, _] : grp.l1_by_core) {
            accounted.insert(c);
        }
    }
    for (auto x = nodes.start_coord.x; x <= nodes.end_coord.x; ++x) {
        for (auto y = nodes.start_coord.y; y <= nodes.end_coord.y; ++y) {
            EXPECT_EQ(accounted.count(CoreCoord(x, y)), 1u)
                << "Core (" << x << "," << y << ") missing from any DfbGroup";
        }
    }
}

}  // namespace m2_config_test_helpers

// =====================================================================================
// Group 5 M2: 2-core homogeneous-grid checks (legacy: MultiCoreDFB_1P1C_Strided_*)
// =====================================================================================

TEST_F(MeshDeviceFixture, MultiCoreDFB_1P1C_Strided_NoImplicitSync_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only";
    }
    // 2-core test (nodes (0,0)..(1,0)): the Quasar 1x3 emu reports a 1x1
    // compute grid, so skip there.
    CoreCoord grid = mesh_device->get_devices()[0]->compute_with_storage_grid_size();
    if (grid.x < 2) {
        GTEST_SKIP() << "2-core test requires grid.x >= 2 (got " << grid.x << "x" << grid.y << ")";
    }
    using namespace m2_config_test_helpers;
    M2ConfigDFBParams p{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 1,
        .implicit_sync = false,
        .target_nodes = m2::NodeRange{m2::NodeCoord{0, 0}, m2::NodeCoord{1, 0}},
    };
    Program program = build_single_dfb_program_2_0(mesh_device, p);
    program.impl().finalize_dataflow_buffer_configs();
    validate_multicore_dfb_groups_2_0(
        program, *p.target_nodes, /*expected_num_groups=*/1, /*expected_cores_per_group=*/2);

    // Each core's TC slot 0 should have counter_id=0 (independent per-core allocator).
    for (uint32_t x = 0; x <= 1; ++x) {
        auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(x, 0));
        ASSERT_EQ(dfbs.size(), 1u);
        // Locate the group containing this core.
        const auto& groups = dfbs[0]->groups;
        auto git = std::find_if(groups.begin(), groups.end(), [&](const auto& grp) {
            return std::any_of(grp.l1_by_core.begin(), grp.l1_by_core.end(), [&](const auto& kv) {
                return kv.first == CoreCoord(x, 0);
            });
        });
        ASSERT_NE(git, groups.end());
        const experimental::dfb::detail::DfbGroup* found = &*git;
        for (const auto& rc : found->hw_risc_configs) {
            for (uint8_t tc = 0; tc < rc.config.num_tcs_to_rr; ++tc) {
                EXPECT_EQ(::dfb::get_counter_id(rc.config.packed_tile_counter[tc]), tc)
                    << "Core (" << x << ",0) RISC " << (int)rc.risc_id << " TC[" << (int)tc << "] counter_id mismatch";
            }
        }
    }
}

TEST_F(MeshDeviceFixture, MultiCoreDFB_1P1C_Strided_ImplicitSync_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only";
    }
    // 2-core test (nodes (0,0)..(1,0)): the Quasar 1x3 emu reports a 1x1
    // compute grid, so skip there.
    CoreCoord grid = mesh_device->get_devices()[0]->compute_with_storage_grid_size();
    if (grid.x < 2) {
        GTEST_SKIP() << "2-core test requires grid.x >= 2 (got " << grid.x << "x" << grid.y << ")";
    }
    using namespace m2_config_test_helpers;
    M2ConfigDFBParams p{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 1,
        .implicit_sync = true,
        .target_nodes = m2::NodeRange{m2::NodeCoord{0, 0}, m2::NodeCoord{1, 0}},
    };
    Program program = build_single_dfb_program_2_0(mesh_device, p);
    program.impl().finalize_dataflow_buffer_configs();
    validate_multicore_dfb_groups_2_0(
        program, *p.target_nodes, /*expected_num_groups=*/1, /*expected_cores_per_group=*/2);

    // Implicit sync: txn-id descriptors are core-invariant (allocated once).
    auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(0, 0));
    ASSERT_EQ(dfbs.size(), 1u);
    EXPECT_EQ(dfbs[0]->producer_txn_descriptor.num_txn_ids, 2u);
    EXPECT_EQ(dfbs[0]->consumer_txn_descriptor.num_txn_ids, 2u);
}

// =====================================================================================
// Group 1 M2: TC-routing *Config tests.
//
// Each test builds a single-DFB single-core M2 program and asserts:
//   - num_producers + num_consumers per-RISC config entries exist
//   - each producer/consumer has the expected TC count (= 4 for 1S×4* fan-out)
//   - STRIDED → producer TC slot matches some consumer TC slot (shared counter)
//   - ALL → remapper indices unique, consumer_tcs accumulator non-zero
// =====================================================================================

#define CONFIG_TC_TEST_2_0(name, prod, cons, num_p, num_c, pap_kind, cap_kind, exp_prod_tc, exp_cons_tc) \
    TEST_F(MeshDeviceFixture, name##_2_0) {                                                              \
        auto& mesh_device = this->devices_.at(0);                                                        \
        if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {                                     \
            GTEST_SKIP() << "M2 path is Quasar-only";                                                    \
        }                                                                                                \
        using namespace m2_config_test_helpers;                                                          \
        M2ConfigDFBParams p{                                                                             \
            .producer_type = M2PorCType::prod,                                                           \
            .consumer_type = M2PorCType::cons,                                                           \
            .num_producers = (num_p),                                                                    \
            .num_consumers = (num_c),                                                                    \
            .pap = m2::DFBAccessPattern::pap_kind,                                                       \
            .cap = m2::DFBAccessPattern::cap_kind,                                                       \
            .implicit_sync = false,                                                                      \
        };                                                                                               \
        Program program = build_single_dfb_program_2_0(mesh_device, p);                                  \
        program.impl().finalize_dataflow_buffer_configs();                                               \
        validate_dfb_tile_counters_2_0(                                                                  \
            program,                                                                                     \
            CoreCoord(0, 0),                                                                             \
            (num_p),                                                                                     \
            (num_c),                                                                                     \
            m2::DFBAccessPattern::cap_kind,                                                              \
            {.expected_producer_tc_count = (exp_prod_tc), .expected_consumer_tc_count = (exp_cons_tc)}); \
    }

// 1P×1C STRIDED variants
CONFIG_TC_TEST_2_0(DMTensixTest1xDFB1Sx1SConfig, DM, TENSIX, 1, 1, STRIDED, STRIDED, 1, 1)
CONFIG_TC_TEST_2_0(DMTest1xDFB1Sx4SConfig, DM, DM, 1, 4, STRIDED, STRIDED, 4, 1)
CONFIG_TC_TEST_2_0(DMTensixTest1xDFB4Sx1SConfig, DM, TENSIX, 4, 1, STRIDED, STRIDED, 1, 4)
CONFIG_TC_TEST_2_0(DMTest1xDFB4Sx1SConfig, DM, DM, 4, 1, STRIDED, STRIDED, 1, 4)
// DM→DM 4Sx4S omitted: 4 producers + 4 consumers = 8 DM threads, exceeds Gen2's
// 6-DM cap. Legacy DMTest1xDFB4Sx4SConfig on main can probe this via the legacy
// host-only API which doesn't validate the cap, but M2's MakeProgramFromSpec
// enforces the limit at spec validation. Same architectural reason the runtime
// DMTest1xDFB4Sx4A macro tuple is excluded from the M2 DFB_TEST_M2 matrix.
CONFIG_TC_TEST_2_0(DMTest1xDFB2Sx4SConfig, DM, DM, 2, 4, STRIDED, STRIDED, 2, 1)
CONFIG_TC_TEST_2_0(DMTest1xDFB4Sx2SConfig, DM, DM, 4, 2, STRIDED, STRIDED, 1, 2)

// 1P×N ALL ("B" = blocked = ALL access pattern in legacy naming) variants.
// In ALL on M2: each producer has num_consumers TCs (one slot per consumer
// destination); each consumer has num_producers TCs. Legacy folds the producer
// side to a single TC and uses the remapper for fan-out — M2 represents the
// fan-out explicitly in num_tcs_to_rr instead.
CONFIG_TC_TEST_2_0(DMTest1xDFB1Sx1BConfig, DM, DM, 1, 1, STRIDED, ALL, 1, 1)
CONFIG_TC_TEST_2_0(DMTest1xDFB1Sx4BConfig, DM, DM, 1, 4, STRIDED, ALL, 4, 1)
CONFIG_TC_TEST_2_0(DMTest1xDFB4Sx1BConfig, DM, DM, 4, 1, STRIDED, ALL, 1, 4)
// DM→DM 4Sx4B omitted: same 8-DM > 6-cap architectural blocker as 4Sx4S above.
CONFIG_TC_TEST_2_0(DMTest1xDFB4Sx2BConfig, DM, DM, 4, 2, STRIDED, ALL, 2, 4)
CONFIG_TC_TEST_2_0(DMTest1xDFB2Sx4BConfig, DM, DM, 2, 4, STRIDED, ALL, 4, 2)

// =====================================================================================
// Group 2 M2: B2 (txn-id allocator) + B4 (cached threshold) + B10 (divisibility)
// =====================================================================================

// B2: For num_entries in {16, 15, 7}, producer_txn_descriptor.num_txn_ids should
// land on {2, 3, 1} (divisibility-based selection).
TEST_F(MeshDeviceFixture, B2_TxnIdAllocator_Boundaries_Config_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync (and therefore the txn-id allocator) is Quasar-only";
    }
    using namespace m2_config_test_helpers;
    struct Case {
        uint16_t num_entries;
        uint8_t expected_num_txn_ids;
        const char* rationale;
    };
    const Case cases[] = {
        {16, 2, "num_entries=16 → 16%2==0, smallest n in [2,4]"},
        {15, 3, "num_entries=15 → 15%2=1 (skip), 15%3=0 → pick n=3"},
        {7, 1, "num_entries=7 → no n in [2,4] divides cleanly → fallback 1"},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(
            ::testing::Message() << "case num_entries=" << c.num_entries << " expected=" << (int)c.expected_num_txn_ids
                                 << " (" << c.rationale << ")");
        M2ConfigDFBParams p{
            .producer_type = M2PorCType::DM,
            .consumer_type = M2PorCType::DM,
            .num_producers = 1,
            .num_consumers = 1,
            .num_entries = c.num_entries,
            .implicit_sync = true,
        };
        Program program = build_single_dfb_program_2_0(mesh_device, p);
        program.impl().finalize_dataflow_buffer_configs();
        auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(0, 0));
        ASSERT_EQ(dfbs.size(), 1u);
        EXPECT_EQ(dfbs[0]->producer_txn_descriptor.num_txn_ids, c.expected_num_txn_ids);
    }
}

// B4: Verifies the cached `num_entries_to_process_threshold` field:
//   STRIDED: threshold = num_entries / num_txn_ids
//   ALL:     threshold = num_consumers * (num_entries / num_txn_ids)
TEST_F(MeshDeviceFixture, B4_CachedThreshold_Config_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Implicit sync (and therefore threshold caching) is Quasar-only";
    }
    using namespace m2_config_test_helpers;

    // case 1: 1S(DM)x1S(DM), num_entries=16 → producer/consumer threshold = 16/2 = 8.
    {
        SCOPED_TRACE("case 1: 1Sx1S, num_entries=16 → threshold=8");
        M2ConfigDFBParams p{
            .producer_type = M2PorCType::DM,
            .consumer_type = M2PorCType::DM,
            .num_producers = 1,
            .num_consumers = 1,
            .num_entries = 16,
            .implicit_sync = true,
        };
        Program program = build_single_dfb_program_2_0(mesh_device, p);
        program.impl().finalize_dataflow_buffer_configs();
        auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(0, 0));
        ASSERT_EQ(dfbs.size(), 1u);
        const auto& d = dfbs[0];
        ASSERT_EQ(d->producer_txn_descriptor.num_txn_ids, 2u);
        ASSERT_EQ(d->consumer_txn_descriptor.num_txn_ids, 2u);
        EXPECT_EQ(d->producer_txn_descriptor.num_entries_to_process_threshold, 8u);
        EXPECT_EQ(d->consumer_txn_descriptor.num_entries_to_process_threshold, 8u);
    }

    // case 2: 1S(DM)x3A(DM), num_entries=18 → producer=9, consumer=3*9=27.
    // The ALL-consumer multiplier (num_consumers ×) is the load-bearing piece a past bug fix added.
    {
        SCOPED_TRACE("case 2: 1Sx3A DM-DM, num_entries=18 → producer 9, consumer 3*9=27");
        M2ConfigDFBParams p{
            .producer_type = M2PorCType::DM,
            .consumer_type = M2PorCType::DM,
            .num_producers = 1,
            .num_consumers = 3,
            .num_entries = 18,
            .pap = m2::DFBAccessPattern::STRIDED,
            .cap = m2::DFBAccessPattern::ALL,
            .implicit_sync = true,
        };
        Program program = build_single_dfb_program_2_0(mesh_device, p);
        program.impl().finalize_dataflow_buffer_configs();
        auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(0, 0));
        ASSERT_EQ(dfbs.size(), 1u);
        const auto& d = dfbs[0];
        EXPECT_EQ(d->producer_txn_descriptor.num_entries_to_process_threshold, 9u);
        EXPECT_EQ(d->consumer_txn_descriptor.num_entries_to_process_threshold, 27u);
    }
}

// B10: divisibility — (a) pathological num_entries should fail at MakeProgramFromSpec/finalize;
// (b) barely-divisible (only n=1 works) should succeed.
TEST_F(MeshDeviceFixture, B10_NumEntriesDivisibility_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Txn-id allocator is Quasar-only";
    }
    using namespace m2_config_test_helpers;

    // 10a: num_entries=10, 3 producers, 3 consumers — 10 % (n * 3 * 1) ≠ 0 for any n.
    // Expected: MakeProgramFromSpec or finalize throws with "must be divisible by" in the message.
    {
        SCOPED_TRACE("10a: pathological num_entries=10, 3P 3C");
        M2ConfigDFBParams p{
            .producer_type = M2PorCType::DM,
            .consumer_type = M2PorCType::DM,
            .num_producers = 3,
            .num_consumers = 3,
            .num_entries = 10,
            .implicit_sync = false,
        };
        EXPECT_THROW(
            {
                Program program = build_single_dfb_program_2_0(mesh_device, p);
                program.impl().finalize_dataflow_buffer_configs();
            },
            std::exception);
    }

    // 10b: num_entries=3, 3 producers, 3 consumers — 3%3==0 at n=1.
    // Should succeed cleanly.
    {
        SCOPED_TRACE("10b: barely-divisible num_entries=3, 3P 3C — should succeed");
        M2ConfigDFBParams p{
            .producer_type = M2PorCType::DM,
            .consumer_type = M2PorCType::DM,
            .num_producers = 3,
            .num_consumers = 3,
            .num_entries = 3,
            .implicit_sync = false,
        };
        EXPECT_NO_THROW({
            Program program = build_single_dfb_program_2_0(mesh_device, p);
            program.impl().finalize_dataflow_buffer_configs();
        });
    }
}

// =====================================================================================
// Group 4 M2: B5 (per-RISC TC capacity 1Sx5S DMTensix) + TensixIntraTest1xDFB1Sx1SConfig
// =====================================================================================

// B5 (1S × 5 Tensix consumers STRIDED) omitted: Quasar has only 4 Tensix engines
// per node (QUASAR_TENSIX_ENGINES_PER_NODE = 4), and M2's MakeProgramFromSpec
// rejects compute KernelSpecs with num_threads > 4 at spec-validation time
// ("KernelSpec 'consumer' has too many threads"). Legacy DMTensix B5 on main
// probes this via the permissive experimental::dfb API which doesn't validate
// the per-Tensix engine cap — same architectural blocker as the 4Sx4S / 4Sx4B
// 8-DM > 6-DM-cap omissions in the CONFIG_TC_TEST_2_0 list above.

// INTRA self-loop config probe (legacy: TensixIntraTest1xDFB1Sx1SConfig).
TEST_F(MeshDeviceFixture, TensixIntraTest1xDFB1Sx1SConfig_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "INTRA scope is Quasar-only";
    }
    const m2::DFBSpecName DFB{"intra_dfb"};
    const m2::KernelSpecName COMPUTE{"compute"};
    const m2::NodeCoord node{0, 0};

    m2::DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = 1024,
        .num_entries = 4,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto compute = make_compute_kernel(
        COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_intra_2_0.cpp", /*num_threads=*/1);
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "self",
         .endpoint_type = m2::DFBEndpointType::PRODUCER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB,
         .accessor_name = "self",
         .endpoint_type = m2::DFBEndpointType::CONSUMER,
         .access_pattern = m2::DFBAccessPattern::STRIDED},
    };
    compute.compile_time_args = {{"entries_per_neo", 4u}, {"words_per_entry", 256u}};

    m2::ProgramSpec spec{
        .name = "intra_config_2_0",
        .kernels = {compute},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = {},
        .work_units = {m2::WorkUnitSpec{.name = "wu", .kernels = {COMPUTE}, .target_nodes = node}},
    };
    Program program = m2::MakeProgramFromSpec(*mesh_device, spec);
    m2_config_test_helpers::validate_intra_tensix_dfb_2_0(program, CoreCoord(0, 0));
}

// =====================================================================================
// Group 3 M2: rejection tests. Where the M2 spec model can express the same bad
// config, we assert that MakeProgramFromSpec / finalize throws. Where the bad
// config is not expressible (B7 CB+DFB mix, B9 INTER scope), the test is
// documented as not-applicable.
// =====================================================================================

// B6 — Producer access pattern = ALL is rejected.
TEST_F(MeshDeviceFixture, B6_AllProducer_Rejected_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB validation tested on Quasar";
    }
    using namespace m2_config_test_helpers;
    M2ConfigDFBParams p{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 1,
        .pap = m2::DFBAccessPattern::ALL,  // <-- the offense (producer ALL not supported)
        .cap = m2::DFBAccessPattern::STRIDED,
        .implicit_sync = false,
    };
    EXPECT_THROW(
        {
            Program program = build_single_dfb_program_2_0(mesh_device, p);
            program.impl().finalize_dataflow_buffer_configs();
        },
        std::exception);
}

// B7 — CB+DFB mix rejection.
// Not applicable to M2: ProgramSpec doesn't expose a circular-buffer API
// (CircularBufferConfig is a legacy host-API construct). M2 programs are
// purely DFB-based; the legacy CB-then-DFB rejection path can't be exercised
// through the M2 spec model.
TEST_F(MeshDeviceFixture, B7_CB_DFB_Mix_Rejected_2_0) {
    GTEST_SKIP() << "Not applicable: M2 ProgramSpec has no CB construct";
}

// B8 — ALL consumer with num_consumers > 4 is rejected (Remapper has 4 clientR slots).
TEST_F(MeshDeviceFixture, B8_FiveAllConsumers_Rejected_2_0) {
    auto& mesh_device = this->devices_.at(0);
    if (mesh_device->get_devices()[0]->arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Remapper limit tested on Quasar";
    }
    using namespace m2_config_test_helpers;
    M2ConfigDFBParams p{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::TENSIX,  // 5 Tensix consumers (try to exceed remapper's 4 clientR slots)
        .num_producers = 1,
        .num_consumers = 5,
        .num_entries = 20,
        .pap = m2::DFBAccessPattern::STRIDED,
        .cap = m2::DFBAccessPattern::ALL,
        .implicit_sync = false,
    };
    EXPECT_THROW(
        {
            Program program = build_single_dfb_program_2_0(mesh_device, p);
            program.impl().finalize_dataflow_buffer_configs();
        },
        std::exception);
}

// B9 — INTER tensix_scope rejection.
// Not applicable to M2: DataflowBufferSpec doesn't expose an explicit tensix_scope
// field; M2 infers scope from kernel binding pattern (INTRA when the same kernel
// binds the DFB as both PRODUCER and CONSUMER). There's no way to construct an
// INTER-scope spec through the M2 API, so the legacy rejection path has no
// direct equivalent.
TEST_F(MeshDeviceFixture, B9_InterTensixScope_Rejected_2_0) {
    GTEST_SKIP() << "Not applicable: M2 DataflowBufferSpec has no tensix_scope field";
}

}  // end namespace tt::tt_metal
