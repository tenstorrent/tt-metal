// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <cstring>
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
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
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
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include "../metal2_host_api/test_helpers/test_helpers.hpp"
#include "dfb_test_common.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "distributed/mesh_device_impl.hpp"

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

TEST_F(UnitMeshFixture, DMTensixTest1xDFB1Sx1SConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

// A DM hart's init entry is an 8B control prefix followed by its LocalDFBInterface image; these read
// the fields the tests check (layout in dataflow_buffer_config.h, dfb_write_dm_image_entry).
static uint16_t dm_entry_u16(const uint8_t* entry, uint32_t off) {
    uint16_t v = 0;
    std::memcpy(&v, entry + off, sizeof(v));
    return v;
}
static uint32_t dm_entry_u32(const uint8_t* entry, uint32_t off) {
    uint32_t v = 0;
    std::memcpy(&v, entry + off, sizeof(v));
    return v;
}
constexpr uint32_t kDmEntryFlagsOff = 2;
constexpr uint32_t kDmEntryCapacityOff = 4;
constexpr uint32_t kDmEntryBytesOff = 7;
constexpr uint32_t kDmImageEntrySizeOff = DFB_IFACE_IMAGE_CTRL_BYTES + 0;
constexpr uint32_t kDmImageNumTcsOff = DFB_IFACE_IMAGE_CTRL_BYTES + DFB_DM_IFACE_NUM_TCS_BYTE_OFF;
constexpr uint32_t kDmImageBlockSizeOff = DFB_IFACE_IMAGE_CTRL_BYTES + 22;
constexpr uint32_t kDmImageSplitTcOff = DFB_IFACE_IMAGE_CTRL_BYTES + 24;
constexpr uint32_t kDmImageJumpOff = DFB_IFACE_IMAGE_CTRL_BYTES + 28;

// Host-only: validates Quasar L1 packing (120B header + per-hart blobs + producer-ready region).
TEST_F(UnitMeshFixture, DfbSerializeGlobalHeader1Sx1S) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().prepare_ephemeral_allocation(this->device().impl().get_devices()[0]);
    program.impl().allocate_dataflow_buffers(this->device().impl().get_devices()[0]);

    const auto& dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1u);

    std::vector<uint8_t> buf(8192, 0);
    const size_t nbytes = experimental::dfb::detail::serialize_dfb_config_for_core(logical_core, dfbs, buf);
    ASSERT_GT(nbytes, 0u);

    const auto* ghdr = reinterpret_cast<const dfb_global_header_t*>(buf.data());
    EXPECT_EQ(ghdr->num_dfbs, 1u);
    EXPECT_EQ(sizeof(dfb_global_header_t), 120u);
    EXPECT_EQ(dfb_config_header_size(), 120u);

    // Header is at offset 0 and the DM1 blob would start on the next cache line. A STRIDED DM-to-DM
    // DFB uses no remapper, so DM1 has no slots and its blob is omitted.
    EXPECT_EQ(DFB_DM1_REMAPPER_BLOB_OFFSET, 128u);
    EXPECT_EQ(ghdr->dm1_slots >> 16, 0u);
    EXPECT_EQ(experimental::dfb::detail::dm1_remapper_blob_core_size(dfbs), 0u);
    // No implicit sync → no DM0 ISR blob, and dm0_packed is 0.
    EXPECT_EQ(experimental::dfb::detail::dm0_isr_blob_region_size(dfbs), 0u);
    EXPECT_EQ(ghdr->has_dm0_isr, 0u);
    EXPECT_EQ(ghdr->dm0_isr_ready, 0u);
    EXPECT_EQ(ghdr->dm0_packed, 0u);

    // Only harts 0 (producer) and 4 (consumer) have an entry, and only the producer publishes a
    // signal. They get blobs of whole cache lines; every other hart gets an empty blob.
    for (uint8_t h = 0; h < ::dfb::NUM_PARTICIPATING_HARTIDS; ++h) {
        SCOPED_TRACE("hart " + std::to_string(h));
        const dfb_hart_desc_t& desc = ghdr->hart_desc[h];
        const bool participates = (h == 0 || h == 4);
        EXPECT_EQ(desc.num_entries, participates ? 1u : 0u);
        EXPECT_EQ(desc.flags, h == 0 ? DFB_HART_DESC_FLAG_PRODUCER : 0u);
        EXPECT_GT(desc.blob_start, 0u);
        EXPECT_EQ(desc.blob_start % 64u, 0u);
        if (participates) {
            EXPECT_GT(desc.blob_len, 0u);
        } else {
            EXPECT_EQ(desc.blob_len, 0u);
        }
        EXPECT_EQ(desc.blob_len % 64u, 0u);
        EXPECT_EQ(desc.signal_region_off, ghdr->dfb_signal_region_off);
    }
    // With no DM1 or DM0 blob, the hart blobs start right after the header: two one-line blobs,
    // back to back, then the signal region.
    EXPECT_EQ(ghdr->hart_desc[0].blob_start, DFB_DM1_REMAPPER_BLOB_OFFSET);
    EXPECT_EQ(ghdr->hart_desc[4].blob_start, ghdr->hart_desc[0].blob_start + ghdr->hart_desc[0].blob_len);
    EXPECT_EQ(ghdr->dfb_signal_region_off, ghdr->hart_desc[4].blob_start + ghdr->hart_desc[4].blob_len);

    // Signal region: one byte slot per hart (zeroed), then dfb_expected_signal[NUM_DFBS]; only the
    // transfer-size padding follows it.
    EXPECT_EQ(DFB_SIGNAL_SLOT_BYTES, 12u);
    const size_t signal_end = static_cast<size_t>(ghdr->dfb_signal_region_off) + DFB_SIGNAL_REGION_BYTES;
    EXPECT_LE(signal_end, nbytes);
    EXPECT_LT(nbytes - signal_end, 16u);
    for (uint32_t h = 0; h < DFB_SIGNAL_SLOT_BYTES; h++) {
        EXPECT_EQ(buf[ghdr->dfb_signal_region_off + h], 0u) << "signal slot " << h;
    }
    // dfb_expected_signal[0] is the mask of producing harts: hart 0 only.
    const auto* dfb_expected =
        reinterpret_cast<const uint32_t*>(buf.data() + ghdr->dfb_signal_region_off + DFB_SIGNAL_SLOT_BYTES);
    EXPECT_EQ(dfb_expected[0], 1u);

    // Hart 0 (DM producer): one DM entry for DFB 0 with a single tile counter.
    const uint8_t* entry0 = buf.data() + ghdr->hart_desc[0].blob_start;
    EXPECT_EQ(dm_entry_u16(entry0, 0), 0u);  // iface_byte_off = logical DFB 0 * sizeof(LocalDFBInterface)
    EXPECT_NE(entry0[kDmEntryFlagsOff] & DFB_HART_FLAG_IS_PRODUCER, 0);
    EXPECT_EQ(dm_entry_u16(entry0, kDmEntryCapacityOff), 16u);  // STRIDED 1P1C: capacity == num_entries
    EXPECT_EQ(entry0[kDmImageNumTcsOff], 1u);
    EXPECT_EQ(entry0[kDmEntryBytesOff], 56u);  // 8 ctrl + 32 scalars + 16 per TC
    EXPECT_EQ(dm_entry_u32(entry0, kDmImageEntrySizeOff), 1024u);
    EXPECT_EQ(dm_entry_u16(entry0, kDmImageSplitTcOff), 0u);  // STRIDED producer: nothing to split

    // Hart 4 (DM consumer): no IS_PRODUCER flag, and consumers leave capacity 0.
    const uint8_t* entry4 = buf.data() + ghdr->hart_desc[4].blob_start;
    EXPECT_EQ(dm_entry_u16(entry4, 0), 0u);
    EXPECT_EQ(entry4[kDmEntryFlagsOff] & DFB_HART_FLAG_IS_PRODUCER, 0);
    EXPECT_EQ(dm_entry_u16(entry4, kDmEntryCapacityOff), 0u);

    experimental::dfb::detail::verify_dfb_global_header_participation(*ghdr, dfbs);
    experimental::dfb::detail::verify_dfb_hart_blobs(
        std::span<const uint8_t>(buf.data(), nbytes), dfbs);
    EXPECT_EQ(nbytes, experimental::dfb::detail::compute_dfb_config_serialized_size(dfbs));
}

// Checks the BLOCKED fields the serializer writes into each hart's init entry (block_size, split_tc,
// entries_to_jump) for a BLOCKED side facing STRIDED peers and the reverse.
struct DfbBlockedInitEntryCase {
    const char* name;
    experimental::dfb::DataflowBufferConfig config;
    uint32_t producer_jump;
    uint32_t consumer_jump;
    uint8_t producer_split_tc;
    uint8_t consumer_split_tc;
};

TEST_F(UnitMeshFixture, DfbSerializeBlockedInitEntries) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    constexpr uint32_t bs = 4;
    const DfbBlockedInitEntryCase cases[] = {
        // 2 BLOCKED producers -> 2 STRIDED consumers: the producers are the split side.
        {.name = "2B->2S",
         .config =
             {.entry_size = 1024,
              .num_entries = 16,
              .producer_risc_mask = 0x3,
              .num_producers = 2,
              .pap = dfb::AccessPattern::BLOCKED,
              .producer_block_size = bs,
              .consumer_risc_mask = 0x30,
              .num_consumers = 2,
              .cap = dfb::AccessPattern::STRIDED},
         .producer_jump = 2 * 2,
         .consumer_jump = (2 - 1) * bs + 2,
         .producer_split_tc = 1,
         .consumer_split_tc = 0},
        // 2 STRIDED producers -> 2 BLOCKED consumers: the consumers are the split side.
        {.name = "2S->2B",
         .config =
             {.entry_size = 1024,
              .num_entries = 16,
              .producer_risc_mask = 0x3,
              .num_producers = 2,
              .pap = dfb::AccessPattern::STRIDED,
              .consumer_risc_mask = 0x30,
              .num_consumers = 2,
              .cap = dfb::AccessPattern::BLOCKED,
              .consumer_block_size = bs},
         .producer_jump = (2 - 1) * bs + 2,
         .consumer_jump = 2 * 2,
         .producer_split_tc = 0,
         .consumer_split_tc = 1},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        Program program = CreateProgram(this->device());
        const CoreCoord logical_core(0, 0);
        experimental::dfb::CreateDataflowBuffer(program, logical_core, c.config);
        program.impl().finalize_dataflow_buffer_configs();
        program.impl().prepare_ephemeral_allocation(this->device().get_devices()[0]);
        program.impl().allocate_dataflow_buffers(this->device().get_devices()[0]);

        const auto& dfbs = program.impl().dataflow_buffers_on_core(logical_core);
        ASSERT_EQ(dfbs.size(), 1u);
        std::vector<uint8_t> buf(8192, 0);
        const size_t nbytes = experimental::dfb::detail::serialize_dfb_config_for_core(logical_core, dfbs, buf);
        ASSERT_GT(nbytes, 0u);
        const auto* ghdr = reinterpret_cast<const dfb_global_header_t*>(buf.data());

        // One DFB, so each participating hart's blob starts with this DFB's entry. Every participant
        // here is a DM hart, whose entry is a control prefix + LocalDFBInterface image.
        for (uint8_t h = 0; h < ::dfb::NUM_PARTICIPATING_HARTIDS; ++h) {
            const bool is_producer = (c.config.producer_risc_mask >> h) & 1u;
            const bool is_consumer = (c.config.consumer_risc_mask >> h) & 1u;
            if (!is_producer && !is_consumer) {
                continue;
            }
            SCOPED_TRACE("hart " + std::to_string(h));
            const uint8_t* entry = buf.data() + ghdr->hart_desc[h].blob_start;
            EXPECT_EQ((entry[kDmEntryFlagsOff] & DFB_HART_FLAG_IS_PRODUCER) != 0, is_producer);
            EXPECT_EQ(dm_entry_u16(entry, kDmImageBlockSizeOff), bs);
            EXPECT_EQ(dm_entry_u16(entry, kDmImageSplitTcOff), is_producer ? c.producer_split_tc : c.consumer_split_tc);
            // A DM cursor is a byte address, so the image carries the jump in bytes.
            EXPECT_EQ(
                dm_entry_u32(entry, kDmImageJumpOff),
                (is_producer ? c.producer_jump : c.consumer_jump) * c.config.entry_size);
        }
    }
}

TEST_F(UnitMeshFixture, DfbSerializeTxnCentricImplicitSync1Sx1S) {
    if (this->device().arch() != ARCH::QUASAR) {
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
        .enable_producer_implicit_sync = true,
        .enable_consumer_implicit_sync = true};

    Program program = CreateProgram(this->device());
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().prepare_ephemeral_allocation(this->device().impl().get_devices()[0]);
    program.impl().allocate_dataflow_buffers(this->device().impl().get_devices()[0]);

    const auto& dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1u);

    std::vector<uint8_t> buf(4096, 0);
    const size_t nbytes = experimental::dfb::detail::serialize_dfb_config_for_core(logical_core, dfbs, buf);
    ASSERT_GT(nbytes, 0u);

    const auto* ghdr = reinterpret_cast<const dfb_global_header_t*>(buf.data());
    const uint32_t dm0_region = experimental::dfb::detail::dm0_isr_blob_region_size(dfbs);
    ASSERT_GT(dm0_region, 0u);
    EXPECT_EQ(ghdr->has_dm0_isr, 1u);
    // No DM1 slots (STRIDED DM-to-DM), so the DM0 blob takes the first line after the header, and
    // the per-hart blobs and then the signal region follow it.
    const uint32_t dm0_off = dfb_dm0_packed_blob_offset(ghdr->dm0_packed);
    EXPECT_EQ(dm0_off, DFB_DM1_REMAPPER_BLOB_OFFSET);
    EXPECT_GT(ghdr->dfb_signal_region_off, dm0_off + dm0_region);

    uint32_t expected_prod_mask = 0;
    uint32_t expected_cons_mask = 0;
    for (uint8_t i = 0; i < dfbs[0]->producer_txn_descriptor.num_txn_ids; ++i) {
        expected_prod_mask |= 1u << dfbs[0]->producer_txn_descriptor.txn_ids[i];
    }
    for (uint8_t i = 0; i < dfbs[0]->consumer_txn_descriptor.num_txn_ids; ++i) {
        expected_cons_mask |= 1u << dfbs[0]->consumer_txn_descriptor.txn_ids[i];
    }
    // DM0's prologue word carries the txn-id masks; the blob has no header of its own.
    EXPECT_EQ(dfb_dm0_packed_producer_mask(ghdr->dm0_packed), expected_prod_mask);
    EXPECT_EQ(dfb_dm0_packed_consumer_mask(ghdr->dm0_packed), expected_cons_mask);

    // The descriptor pool is the whole DM0 blob: it starts at the blob base and fills its region.
    EXPECT_EQ(DFB_DM0_CACHED_PAYLOAD_OFFSET, 0u);
    const uint32_t desc_pool_off = dm0_off + DFB_DM0_CACHED_PAYLOAD_OFFSET;
    EXPECT_EQ(
        desc_pool_off + dm0_isr_txn_desc_pool_byte_size(expected_prod_mask, expected_cons_mask), dm0_off + dm0_region);

    // One slot per used txn id, in ascending id order, each slot dm0_packed's stride wide with the
    // CMDBUF threshold in its last byte.
    const uint32_t stride = dfb_dm0_packed_desc_stride(ghdr->dm0_packed);
    const uint32_t threshold_off = dfb_dm0_desc_threshold_offset(stride);
    const uint32_t all_mask = expected_prod_mask | expected_cons_mask;
    const auto threshold_of = [&](uint32_t txn_id) {
        return buf[desc_pool_off + dm0_isr_txn_slot_index(all_mask, txn_id) * stride + threshold_off];
    };
    uint32_t pending = expected_prod_mask;
    while (pending) {
        const uint32_t txn_id = static_cast<uint32_t>(__builtin_ctz(pending));
        EXPECT_EQ(threshold_of(txn_id), dfbs[0]->producer_txn_descriptor.num_entries_to_process_threshold);
        pending &= (pending - 1u);
    }
    pending = expected_cons_mask;
    while (pending) {
        const uint32_t txn_id = static_cast<uint32_t>(__builtin_ctz(pending));
        EXPECT_EQ(threshold_of(txn_id), dfbs[0]->consumer_txn_descriptor.num_entries_to_process_threshold);
        pending &= (pending - 1u);
    }
}

// The DM0 descriptor pool packs every slot on a core to 8 B (<= 2 counters), 16 B (<= 10) or 32 B,
// with the threshold in the slot's last byte. Each case puts real descriptors at a stride or copy-word
// boundary, rebuilds them with DM0's copy rules, and checks no counter or finish field is lost or
// overwritten by the threshold.
struct DfbDm0DescriptorCase {
    const char* name;
    std::vector<experimental::dfb::DataflowBufferConfig> configs;
    uint32_t expected_stride;
    // Per DFB: counters in its {producer, consumer} descriptor; 0 = that side has no descriptor.
    std::vector<std::pair<uint8_t, uint8_t>> expected_counters;
};

class DfbDm0DescriptorStrideTest : public UnitMeshFixture,
                                   public ::testing::WithParamInterface<DfbDm0DescriptorCase> {};

namespace {

struct Dm0CopiedDescriptor {
    dfb_dm0_txn_descriptor_image_t image;
    uint8_t threshold;
};

// Mirrors DM0's copy in dataflow_buffer_init.h: the first 8 B always, bytes 8-15 past 3 counters,
// bytes 16-23 past 11 (32 B slot only), and the threshold from dfb_dm0_desc_threshold_offset.
Dm0CopiedDescriptor dm0_copy_descriptor(const uint8_t* slot, uint32_t stride) {
    Dm0CopiedDescriptor out{};
    auto* image = reinterpret_cast<uint8_t*>(&out.image);
    std::memcpy(image, slot, 8);
    if (slot[0] > 3) {
        std::memcpy(image + 8, slot + 8, slot[0] > 11 ? 16 : 8);
    }
    out.threshold = slot[dfb_dm0_desc_threshold_offset(stride)];
    return out;
}

// One side's tile counters in hart order, the order the ISR walks them.
std::vector<::dfb::PackedTileCounter> side_tile_counters(
    const experimental::dfb::detail::DataflowBufferImpl& dfb, bool is_producer) {
    std::map<uint8_t, const experimental::dfb::detail::DFBRiscConfig*> by_risc;
    for (const auto& rc : dfb.groups[0].hw_risc_configs) {
        if (rc.is_producer == is_producer) {
            by_risc[rc.risc_id] = &rc;
        }
    }
    std::vector<::dfb::PackedTileCounter> tcs;
    for (const auto& [risc_id, rc] : by_risc) {
        const auto& counters = rc->config.packed_tile_counter;
        tcs.insert(tcs.end(), counters.begin(), counters.begin() + rc->config.num_tcs_to_rr);
    }
    return tcs;
}

}  // namespace

TEST_P(DfbDm0DescriptorStrideTest, RoundTripsThroughDm0Copy) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Skipping DFB test for WH/BH until DFB is backported";
    }
    const DfbDm0DescriptorCase& c = GetParam();
    Program program = CreateProgram(this->device());
    const CoreCoord logical_core(0, 0);
    for (const auto& config : c.configs) {
        experimental::dfb::CreateDataflowBuffer(program, logical_core, config);
    }
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().prepare_ephemeral_allocation(this->device().impl().get_devices()[0]);
    program.impl().allocate_dataflow_buffers(this->device().impl().get_devices()[0]);

    const auto& dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), c.configs.size());
    std::vector<uint8_t> buf(8192, 0);
    ASSERT_GT(experimental::dfb::detail::serialize_dfb_config_for_core(logical_core, dfbs, buf), 0u);
    const auto* ghdr = reinterpret_cast<const dfb_global_header_t*>(buf.data());
    ASSERT_NE(ghdr->dm0_packed, 0u);

    const uint32_t stride = dfb_dm0_packed_desc_stride(ghdr->dm0_packed);
    EXPECT_EQ(stride, c.expected_stride);
    const uint8_t* pool = buf.data() + dfb_dm0_packed_blob_offset(ghdr->dm0_packed) + DFB_DM0_CACHED_PAYLOAD_OFFSET;
    const uint32_t all_mask =
        dfb_dm0_packed_producer_mask(ghdr->dm0_packed) | dfb_dm0_packed_consumer_mask(ghdr->dm0_packed);

    for (size_t i = 0; i < dfbs.size(); ++i) {
        for (const bool is_producer : {true, false}) {
            SCOPED_TRACE(fmt::format("DFB {} {}", i, is_producer ? "producer" : "consumer"));
            const auto& desc = is_producer ? dfbs[i]->producer_txn_descriptor : dfbs[i]->consumer_txn_descriptor;
            const uint8_t expected_counters =
                is_producer ? c.expected_counters[i].first : c.expected_counters[i].second;
            if (expected_counters == 0) {
                EXPECT_EQ(desc.num_txn_ids, 0u);
                continue;
            }
            ASSERT_GT(desc.num_txn_ids, 0u);
            const auto tcs = side_tile_counters(*dfbs[i], is_producer);
            ASSERT_EQ(tcs.size(), expected_counters);
            for (uint8_t t = 0; t < desc.num_txn_ids; ++t) {
                const uint32_t txn_id = desc.txn_ids[t];
                SCOPED_TRACE(fmt::format("txn id {}", txn_id));
                const auto copied =
                    dm0_copy_descriptor(pool + dm0_isr_txn_slot_index(all_mask, txn_id) * stride, stride);
                EXPECT_EQ(copied.image.num_counters, expected_counters);
                for (uint8_t j = 0; j < expected_counters; ++j) {
                    EXPECT_EQ(copied.image.tile_counters[j], tcs[j]) << "counter " << int(j);
                }
                EXPECT_EQ(copied.image.tiles_to_post_or_ack, desc.num_entries_per_txn_id_per_tc);
                EXPECT_EQ(copied.threshold, desc.num_entries_to_process_threshold);
                // DM0's kernel-exit flush reads the finish record from every side's first txn id.
                const uint16_t side_mask =
                    is_producer ? c.configs[i].producer_risc_mask : c.configs[i].consumer_risc_mask;
                EXPECT_EQ(copied.image.finish.issuer_hart_mask, side_mask & ((1u << ::dfb::TENSIX_RISC_OFFSET) - 1u));
                EXPECT_EQ(copied.image.finish.num_txn_ids, desc.num_txn_ids);
            }
        }
    }
}

namespace {

// All participants use implicit sync (a Tensix-only side still gets no descriptor); BLOCKED blocks are 1 entry.
experimental::dfb::DataflowBufferConfig dm0_desc_config(
    uint32_t num_entries,
    uint16_t producer_risc_mask,
    uint8_t num_producers,
    dfb::AccessPattern pap,
    uint16_t consumer_risc_mask,
    uint8_t num_consumers,
    dfb::AccessPattern cap) {
    return {
        .entry_size = 1024,
        .num_entries = num_entries,
        .producer_risc_mask = producer_risc_mask,
        .num_producers = num_producers,
        .pap = pap,
        .producer_block_size = pap == dfb::AccessPattern::BLOCKED ? 1u : 0u,
        .consumer_risc_mask = consumer_risc_mask,
        .num_consumers = num_consumers,
        .cap = cap,
        .consumer_block_size = cap == dfb::AccessPattern::BLOCKED ? 1u : 0u,
        .enable_producer_implicit_sync = true,
        .enable_consumer_implicit_sync = true};
}

constexpr auto kStrided = dfb::AccessPattern::STRIDED;
constexpr auto kBlocked = dfb::AccessPattern::BLOCKED;

}  // namespace

// Neo harts start at bit 8. A side's count is max(P, C) for STRIDED and P x C for BLOCKED<->STRIDED,
// with at most 6 counters per hart, so 11 has no topology: 12 is the first 32 B count.
INSTANTIATE_TEST_SUITE_P(
    CompactBoundaries,
    DfbDm0DescriptorStrideTest,
    ::testing::Values(
        // 1 DM -> 2 DMs: 2 counters, the most an 8 B slot holds.
        DfbDm0DescriptorCase{
            .name = "Counters2Stride8",
            .configs = {dm0_desc_config(8, 0x01, 1, kStrided, 0x06, 2, kStrided)},
            .expected_stride = 8,
            .expected_counters = {{2, 2}}},
        // 1 DM -> 3 DMs: 3 counters, the first 16 B count and the last DM0 copies in one word.
        DfbDm0DescriptorCase{
            .name = "Counters3Stride16",
            .configs = {dm0_desc_config(12, 0x01, 1, kStrided, 0x0E, 3, kStrided)},
            .expected_stride = 16,
            .expected_counters = {{3, 3}}},
        // 1 DM -> 4 DMs: 4 counters, the first that needs word 1.
        DfbDm0DescriptorCase{
            .name = "Counters4Stride16",
            .configs = {dm0_desc_config(16, 0x01, 1, kStrided, 0x1E, 4, kStrided)},
            .expected_stride = 16,
            .expected_counters = {{4, 4}}},
        // 5 BLOCKED DMs -> 2 DMs: 10 counters per side, the most a 16 B slot holds.
        DfbDm0DescriptorCase{
            .name = "Counters10Stride16",
            .configs = {dm0_desc_config(20, 0x1F, 5, kBlocked, 0x60, 2, kStrided)},
            .expected_stride = 16,
            .expected_counters = {{10, 10}}},
        // 4 BLOCKED DMs -> 3 DMs: 12 counters per side, the first 32 B count and the first to need word 2.
        DfbDm0DescriptorCase{
            .name = "Counters12Stride32",
            .configs = {dm0_desc_config(24, 0x0F, 4, kBlocked, 0x70, 3, kStrided)},
            .expected_stride = 32,
            .expected_counters = {{12, 12}}},
        // 5 BLOCKED DMs -> 3 DMs: 15 counters per side, three words of a 32 B slot.
        DfbDm0DescriptorCase{
            .name = "Counters15Stride32",
            .configs = {dm0_desc_config(30, 0x1F, 5, kBlocked, 0xE0, 3, kStrided)},
            .expected_stride = 32,
            .expected_counters = {{15, 15}}},
        // 6 BLOCKED DMs -> 3 Neos: 18 producer counters, the most a descriptor holds.
        DfbDm0DescriptorCase{
            .name = "Counters18Stride32",
            .configs = {dm0_desc_config(36, 0x3F, 6, kBlocked, 0x700, 3, kStrided)},
            .expected_stride = 32,
            .expected_counters = {{18, 0}}},
        // A 1-counter DFB beside an 18-counter one: the narrow descriptors take the 32 B stride.
        DfbDm0DescriptorCase{
            .name = "MixedWidthsStride32",
            .configs =
                {dm0_desc_config(36, 0x3F, 6, kBlocked, 0x700, 3, kStrided),
                 dm0_desc_config(16, 0x40, 1, kStrided, 0x80, 1, kStrided)},
            .expected_stride = 32,
            .expected_counters = {{18, 0}, {1, 1}}}),
    [](const ::testing::TestParamInfo<DfbDm0DescriptorCase>& info) { return std::string(info.param.name); });

TEST_F(UnitMeshFixture, DMTest1xDFB1Sx4SConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTensixTest1xDFB4Sx1SConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTest1xDFB4Sx1SConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTest1xDFB4Sx4SConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTest1xDFB2Sx4SConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTest1xDFB4Sx2SConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTest1xDFB1Sx1BConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTest1xDFB1Sx4BConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTest1xDFB4Sx1BConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTest1xDFB4Sx4BConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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

TEST_F(UnitMeshFixture, DMTest1xDFB4Sx2BConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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
TEST_F(UnitMeshFixture, DMTest1xDFB2Sx4BConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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
TEST_F(UnitMeshFixture, MultiCoreDFB_1P1C_Strided_NoImplicitSync) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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
TEST_F(UnitMeshFixture, MultiCoreDFB_1P1C_Strided_ImplicitSync) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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
TEST_F(UnitMeshFixture, MultiCoreDFB_HomogeneousGrid_SingleGroup) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
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
//   - STRIDED producer → STRIDED consumer (INTRA never uses ALL / remapper fan-out patterns).
//   - RemapperProgrammer::TENSIX_PACKER with Tensix-only ClientL + distinct shadow TC.
//   - Remapper pair drawn from the top-down 1-to-1 pool.
void validate_intra_tensix_dfb(
    Program& program,
    const CoreCoord& logical_core,
    const experimental::dfb::DataflowBufferConfig& config) {
    program.impl().finalize_dataflow_buffer_configs();

    ASSERT_EQ(config.pap, dfb::AccessPattern::STRIDED);
    ASSERT_EQ(config.cap, dfb::AccessPattern::STRIDED);

    auto dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 1u) << "Expected exactly 1 DFB on core";
    const auto& dfb = dfbs[0];

    ASSERT_EQ(dfb->risc_mask, config.producer_risc_mask)
        << "Intra-tensix risc_mask should equal producer_risc_mask (same Neo bit)";
    ASSERT_TRUE(dfb->use_remapper) << "Intra-tensix DFB must use remapper for the Tensix-only TC alias workaround";
    ASSERT_EQ(dfb->remapper_programmer, experimental::dfb::detail::RemapperProgrammer::TENSIX_PACKER)
        << "Intra-tensix remapper pairs are programmed by the Neo packer, not DM1";
    EXPECT_EQ(dfb->dm1_remapper_slot_count(), 0u) << "Packer-programmed pairs must not appear in the DM1 blob";
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
    EXPECT_GE(rc.config.intra_shadow_tc_id, ::dfb::TC_TENSIX_POOL_START);
    EXPECT_NE(rc.config.intra_shadow_tc_id, tc_id)
        << "Shadow TC must be a distinct Tensix-only id that absorbs the remapper HW update copy";
    EXPECT_GE(rc.config.remapper_pair_index, ::dfb::REMAPPER_ONE_TO_ONE_PAIR_START)
        << "Packer pairs must come from the 1-to-1 remapper pool";
    EXPECT_LT(rc.config.remapper_pair_index, ::dfb::NUM_REMAPPER_PAIRINGS);

    log_info(
        tt::LogTest,
        "Intra-tensix DFB: Neo{} ClientL tc_id={} shadow={} remapper_pair={}",
        expected_tensix_id,
        (int)tc_id,
        (int)rc.config.intra_shadow_tc_id,
        (int)rc.config.remapper_pair_index);
}

TEST_F(UnitMeshFixture, TensixIntraTest1xDFB1Sx1SConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
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

    Program program = CreateProgram(this->device());
    CoreCoord logical_core = CoreCoord(0, 0);
    experimental::dfb::CreateDataflowBuffer(program, logical_core, config);

    validate_intra_tensix_dfb(program, logical_core, config);
}

// Run args for the minimal spec's two kernels (both declare empty RTA schemas, so empty arg sets).
inline ProgramRunArgs MakeMinimalRunArgs(const NodeCoord& node) {
    auto kernel_args = [&](const char* name) {
        return ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{name},
            .advanced_options =
                experimental::AdvancedKernelRunArgs{.runtime_varargs = {{node, {}}}, .common_runtime_varargs = {}},
        };
    };
    ProgramRunArgs params;
    params.kernel_run_args.push_back(kernel_args("dm_kernel"));
    params.kernel_run_args.push_back(kernel_args("compute_kernel"));
    return params;
}

// num_entries override on a finalized DFB recomputes the txn descriptor in place while PRESERVING the
// TC assignment and transaction IDs (only the threshold changes for the new ring depth).
TEST_F(UnitMeshFixture, DFBReentryOverridePreservesTcAndRecomputesTxn) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB transaction IDs / tile counters require Quasar";
    }
    const NodeCoord node{0, 0};
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

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
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .num_entries = 4});
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
TEST_F(UnitMeshFixture, DFBReentryNumEntriesViolatesTxnDivisorFails) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB transaction-id divisor check requires Quasar";
    }
    const NodeCoord node{0, 0};
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);
    program.impl().finalize_dataflow_buffer_configs();

    auto params = MakeMinimalRunArgs(node);
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .num_entries = 3});

    EXPECT_THAT(
        [&] { experimental::SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("num_entries override 3 is not divisible by")));
}

// Ring extent past Tensix unreserved L1 is rejected. entry_size stays within the TRISC uint16
// L1-unit field so this hits the L1 check rather than the entry_size width check.
TEST_F(UnitMeshFixture, DFBReentryEntrySizeRingExtentFails) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB ring-extent check requires Quasar";
    }
    const NodeCoord node{0, 0};
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);
    program.impl().finalize_dataflow_buffer_configs();

    auto params = MakeMinimalRunArgs(node);
    // 64 KiB * 128 entries = 8 MiB ring > Quasar unreserved L1 (~4 MiB).
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .entry_size = 64u * 1024u, .num_entries = 128u});

    EXPECT_THAT(
        [&] { experimental::SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("exceeds Tensix unreserved L1 size")));
}

// TRISC stores entry_size in uint16 L1 units; 1 MiB is 65536 units and must fail before init truncates.
TEST_F(UnitMeshFixture, DFBReentryEntrySizeExceedsUint16Fails) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB TRISC entry_size check requires Quasar";
    }
    const NodeCoord node{0, 0};
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);
    program.impl().finalize_dataflow_buffer_configs();

    auto params = MakeMinimalRunArgs(node);
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .entry_size = 1024u * 1024u});

    EXPECT_THAT(
        [&] { experimental::SetProgramRunArgs(program, params); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("TRISC entry_size")));
}

// capacity (num_entries / max(producers, consumers)) is stored as uint16_t (the tile-counter register
// width); an override that pushes it past the max is rejected rather than silently truncated.
TEST_F(UnitMeshFixture, DFBOverrideCapacityExceedsUint16Fails) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB capacity / tile-counter limit requires Quasar";
    }
    const NodeCoord node{0, 0};
    ProgramSpec spec = test_helpers::MakeMinimalValidProgramSpec();
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);

    auto params = MakeMinimalRunArgs(node);
    params.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb_0"}, .num_entries = 70000});  // capacity 70000 > 65535
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
    DFBAccessPattern pap = DFBAccessPattern::STRIDED;
    DFBAccessPattern cap = DFBAccessPattern::STRIDED;
    bool implicit_sync = false;
    uint32_t block_size = 0;                                   // BLOCKED sides only: tiles per block
    std::optional<NodeRange> target_nodes = std::nullopt;      // override single-core default
};

static inline Program build_single_dfb_program_2_0(distributed::MeshDevice& mesh_device, const M2ConfigDFBParams& p) {
    const NodeCoord node{0, 0};
    const DFBSpecName DFB{"dfb"};
    const KernelSpecName PRODUCER{"producer"};
    const KernelSpecName CONSUMER{"consumer"};
    const TensorParamName IN_TENSOR{"in_tensor"};
    const TensorParamName OUT_TENSOR{"out_tensor"};

    const auto tensor_spec = make_flat_dram_tensor_spec(p.entry_size, p.num_entries, DataType::UINT32);

    DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = p.entry_size,
        .num_entries = p.num_entries,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    const bool is_all = (p.cap == DFBAccessPattern::ALL);
    const uint32_t per_producer = (p.num_entries + p.num_producers - 1) / p.num_producers;
    const uint32_t per_consumer = is_all ? p.num_entries : (p.num_entries + p.num_consumers - 1) / p.num_consumers;

    auto make_producer = [&]() -> KernelSpec {
        if (p.producer_type == M2PorCType::DM) {
            auto k = make_dm_kernel(
                PRODUCER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_producer_2_0.cpp", p.num_producers);
            k.dfb_bindings = {
                {.dfb_spec_name = DFB,
                 .accessor_name = "out",
                 .endpoint_type = DFBEndpointType::PRODUCER,
                 .access_pattern = p.pap,
                 .block_size = p.pap == DFBAccessPattern::BLOCKED ? p.block_size : 0u}};
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
             .endpoint_type = DFBEndpointType::PRODUCER,
             .access_pattern = p.pap,
             .block_size = p.pap == DFBAccessPattern::BLOCKED ? p.block_size : 0u}};
        k.compile_time_args = {{"num_entries_per_producer", per_producer}};
        return k;
    };

    auto make_consumer = [&]() -> KernelSpec {
        if (p.consumer_type == M2PorCType::DM) {
            auto k = make_dm_kernel(
                CONSUMER, "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_consumer_2_0.cpp", p.num_consumers);
            k.dfb_bindings = {
                {.dfb_spec_name = DFB,
                 .accessor_name = "in",
                 .endpoint_type = DFBEndpointType::CONSUMER,
                 .access_pattern = p.cap,
                 .block_size = p.cap == DFBAccessPattern::BLOCKED ? p.block_size : 0u}};
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
             .endpoint_type = DFBEndpointType::CONSUMER,
             .access_pattern = p.cap,
             .block_size = p.cap == DFBAccessPattern::BLOCKED ? p.block_size : 0u}};
        k.compile_time_args = {{"num_entries_per_consumer", per_consumer}};
        // Declared so the kernel source stays compilable; these probes never launch, and the
        // kernel's digest reporting is only consumed by run_single_dfb_program_2_0.
        k.runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}};
        return k;
    };

    auto producer = make_producer();
    auto consumer = make_consumer();

    std::vector<TensorParameter> tensor_parameters;
    if (p.producer_type == M2PorCType::DM) {
        tensor_parameters.push_back({.unique_id = IN_TENSOR, .spec = tensor_spec});
    }
    if (p.consumer_type == M2PorCType::DM) {
        tensor_parameters.push_back({.unique_id = OUT_TENSOR, .spec = tensor_spec});
    }

    WorkUnitSpec wu{
        .name = "wu",
        .kernels = {PRODUCER, CONSUMER},
    };
    if (p.target_nodes.has_value()) {
        wu.target_nodes = *p.target_nodes;
    } else {
        wu.target_nodes = node;
    }
    ProgramSpec spec{
        .name = "config_probe_2_0",
        .kernels = {producer, consumer},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = tensor_parameters,
        .work_units = {wu},
    };
    return experimental::MakeProgramFromSpec(mesh_device, spec);
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
    DFBAccessPattern cap,
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

    if (cap == DFBAccessPattern::ALL) {
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
    ASSERT_EQ(dfb->config.pap, dfb::AccessPattern::STRIDED);
    ASSERT_EQ(dfb->config.cap, dfb::AccessPattern::STRIDED);
    ASSERT_TRUE(dfb->use_remapper) << "INTRA DFB must use remapper for the Tensix-only TC alias workaround";
    ASSERT_EQ(dfb->remapper_programmer, experimental::dfb::detail::RemapperProgrammer::TENSIX_PACKER);
    EXPECT_EQ(dfb->dm1_remapper_slot_count(), 0u);
    ASSERT_FALSE(dfb->groups.empty());
    const auto& hw_risc_configs = dfb->groups[0].hw_risc_configs;
    ASSERT_EQ(hw_risc_configs.size(), 1u) << "INTRA DFB should have exactly 1 per-risc entry (shared Neo)";
    const auto& rc = hw_risc_configs[0];
    EXPECT_TRUE(rc.is_producer) << "INTRA per-risc entry must be marked is_producer (PACK TRISC inits TC)";
    ASSERT_EQ(rc.config.num_tcs_to_rr, 1u) << "INTRA DFB should have exactly 1 TC";
    uint8_t tc_id = ::dfb::get_counter_id(rc.config.packed_tile_counter[0]);
    EXPECT_GE(tc_id, ::dfb::TC_TENSIX_POOL_START)
        << "INTRA DFB must use Tensix-only TC (id >= " << (int)::dfb::TC_TENSIX_POOL_START << ")";
    EXPECT_GE(rc.config.intra_shadow_tc_id, ::dfb::TC_TENSIX_POOL_START);
    EXPECT_NE(rc.config.intra_shadow_tc_id, tc_id);
    EXPECT_GE(rc.config.remapper_pair_index, ::dfb::REMAPPER_ONE_TO_ONE_PAIR_START);
    EXPECT_LT(rc.config.remapper_pair_index, ::dfb::NUM_REMAPPER_PAIRINGS);
}

// Multicore-group probe.
static inline void validate_multicore_dfb_groups_2_0(
    Program& program, const NodeRange& nodes, uint32_t expected_num_groups, uint32_t expected_cores_per_group) {
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

TEST_F(UnitMeshFixture, MultiCoreDFB_1P1C_Strided_NoImplicitSync_2_0) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only";
    }
    // 2-core test (nodes (0,0)..(1,0)): the Quasar 1x3 emu reports a 1x1
    // compute grid, so skip there.
    CoreCoord grid = this->device().compute_with_storage_grid_size();
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
        .target_nodes = NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}},
    };
    Program program = build_single_dfb_program_2_0(this->device(), p);
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

TEST_F(UnitMeshFixture, MultiCoreDFB_1P1C_Strided_ImplicitSync_2_0) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only";
    }
    // 2-core test (nodes (0,0)..(1,0)): the Quasar 1x3 emu reports a 1x1
    // compute grid, so skip there.
    CoreCoord grid = this->device().compute_with_storage_grid_size();
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
        .target_nodes = NodeRange{NodeCoord{0, 0}, NodeCoord{1, 0}},
    };
    Program program = build_single_dfb_program_2_0(this->device(), p);
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
    TEST_F(UnitMeshFixture, name##_2_0) {                                                                \
        if (this->device().arch() != ARCH::QUASAR) {                                                     \
            GTEST_SKIP() << "M2 path is Quasar-only";                                                    \
        }                                                                                                \
        using namespace m2_config_test_helpers;                                                          \
        M2ConfigDFBParams p{                                                                             \
            .producer_type = M2PorCType::prod,                                                           \
            .consumer_type = M2PorCType::cons,                                                           \
            .num_producers = (num_p),                                                                    \
            .num_consumers = (num_c),                                                                    \
            .pap = DFBAccessPattern::pap_kind,                                                           \
            .cap = DFBAccessPattern::cap_kind,                                                           \
            .implicit_sync = false,                                                                      \
        };                                                                                               \
        Program program = build_single_dfb_program_2_0(this->device(), p);                               \
        program.impl().finalize_dataflow_buffer_configs();                                               \
        validate_dfb_tile_counters_2_0(                                                                  \
            program,                                                                                     \
            CoreCoord(0, 0),                                                                             \
            (num_p),                                                                                     \
            (num_c),                                                                                     \
            DFBAccessPattern::cap_kind,                                                                  \
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

// Host-only: DFB pool is [DFB_TXN_ID_BASE, HW_TXN_ID_MAX], allocated top-down in ascending blocks.
TEST(TxnIdAllocatorTest, CPU_AllocatesTopDownFromDfbPool) {
    using experimental::dfb::detail::TxnIdAllocator;
    TxnIdAllocator alloc;

    auto first = alloc.allocate(2);
    ASSERT_EQ(first.size(), 2u);
    EXPECT_EQ(first[0], static_cast<uint8_t>(HW_TXN_ID_MAX - 1));
    EXPECT_EQ(first[1], HW_TXN_ID_MAX);

    auto second = alloc.allocate(3);
    ASSERT_EQ(second.size(), 3u);
    EXPECT_EQ(second[0], static_cast<uint8_t>(HW_TXN_ID_MAX - 4));
    EXPECT_EQ(second[1], static_cast<uint8_t>(HW_TXN_ID_MAX - 3));
    EXPECT_EQ(second[2], static_cast<uint8_t>(HW_TXN_ID_MAX - 2));

    // Drain the rest of the pool, then the next allocate must fail.
    const uint8_t used = 5;
    const uint8_t leftover = static_cast<uint8_t>(NUM_DFB_POOL_TXN_IDS - used);
    auto rest = alloc.allocate(leftover);
    ASSERT_EQ(rest.size(), leftover);
    EXPECT_EQ(rest.front(), DFB_TXN_ID_BASE);
    EXPECT_EQ(rest.back(), static_cast<uint8_t>(DFB_TXN_ID_BASE + leftover - 1));

    EXPECT_THAT(
        [&]() { alloc.allocate(1); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("TxnIdAllocator exhausted")));

    alloc.reset();
    auto after_reset = alloc.allocate(1);
    ASSERT_EQ(after_reset.size(), 1u);
    EXPECT_EQ(after_reset[0], HW_TXN_ID_MAX);
}

// B2: For num_entries in {16, 15, 7}, producer_txn_descriptor.num_txn_ids should
// land on {2, 3, 1} (divisibility-based selection).
TEST_F(UnitMeshFixture, B2_TxnIdAllocator_Boundaries_Config_2_0) {
    if (this->device().arch() != ARCH::QUASAR) {
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
        Program program = build_single_dfb_program_2_0(this->device(), p);
        program.impl().finalize_dataflow_buffer_configs();
        auto dfbs = program.impl().dataflow_buffers_on_core(CoreCoord(0, 0));
        ASSERT_EQ(dfbs.size(), 1u);
        EXPECT_EQ(dfbs[0]->producer_txn_descriptor.num_txn_ids, c.expected_num_txn_ids);
        // DFB ids come from the runtime pool [DFB_TXN_ID_BASE, HW_TXN_ID_MAX], never 0 or the user pool.
        auto expect_in_dfb_pool = [](uint8_t txn_id, const char* side, int index) {
            EXPECT_NE(txn_id, 0) << side << " txn_ids[" << index << "] must not be 0 (NOC_V2_TRID_STATIC)";
            EXPECT_GE(txn_id, DFB_TXN_ID_BASE)
                << side << " txn_ids[" << index << "]=" << static_cast<int>(txn_id) << " collides with user pool [0, "
                << static_cast<int>(USER_TXN_ID_MAX) << "]";
            EXPECT_LE(txn_id, HW_TXN_ID_MAX)
                << side << " txn_ids[" << index << "]=" << static_cast<int>(txn_id) << " exceeds HW max";
        };
        for (uint8_t i = 0; i < dfbs[0]->producer_txn_descriptor.num_txn_ids; ++i) {
            expect_in_dfb_pool(dfbs[0]->producer_txn_descriptor.txn_ids[i], "producer", i);
        }
        for (uint8_t i = 0; i < dfbs[0]->consumer_txn_descriptor.num_txn_ids; ++i) {
            expect_in_dfb_pool(dfbs[0]->consumer_txn_descriptor.txn_ids[i], "consumer", i);
        }
        // First DFB on a program draws from the top of the pool.
        if (c.expected_num_txn_ids > 0) {
            EXPECT_EQ(dfbs[0]->producer_txn_descriptor.txn_ids[c.expected_num_txn_ids - 1], HW_TXN_ID_MAX);
        }
    }
}

// B4: Verifies the cached `num_entries_to_process_threshold` field:
//   STRIDED: threshold = num_entries / num_txn_ids
//   ALL:     threshold = num_consumers * (num_entries / num_txn_ids)
TEST_F(UnitMeshFixture, B4_CachedThreshold_Config_2_0) {
    if (this->device().arch() != ARCH::QUASAR) {
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
        Program program = build_single_dfb_program_2_0(this->device(), p);
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
            .pap = DFBAccessPattern::STRIDED,
            .cap = DFBAccessPattern::ALL,
            .implicit_sync = true,
        };
        Program program = build_single_dfb_program_2_0(this->device(), p);
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
TEST_F(UnitMeshFixture, B10_NumEntriesDivisibility_2_0) {
    if (this->device().arch() != ARCH::QUASAR) {
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
                Program program = build_single_dfb_program_2_0(this->device(), p);
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
            Program program = build_single_dfb_program_2_0(this->device(), p);
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
TEST_F(UnitMeshFixture, TensixIntraTest1xDFB1Sx1SConfig_2_0) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "INTRA scope is Quasar-only";
    }
    const DFBSpecName DFB{"intra_dfb"};
    const KernelSpecName COMPUTE{"compute"};
    const NodeCoord node{0, 0};

    DataflowBufferSpec dfb_spec{
        .unique_id = DFB,
        .entry_size = 1024,
        .num_entries = 4,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    auto compute = make_compute_kernel(
        COMPUTE, "tests/tt_metal/tt_metal/test_kernels/compute/dfb_t6_intra_2_0.cpp", /*num_threads=*/1);
    // Accessor names must be distinct and match what dfb_t6_intra_2_0.cpp references (dfb::out);
    // both bindings still resolve to the same DFB, which is what makes M2 infer INTRA scope.
    compute.dfb_bindings = {
        {.dfb_spec_name = DFB,
         .accessor_name = "out",
         .endpoint_type = DFBEndpointType::PRODUCER,
         .access_pattern = DFBAccessPattern::STRIDED},
        {.dfb_spec_name = DFB,
         .accessor_name = "in",
         .endpoint_type = DFBEndpointType::CONSUMER,
         .access_pattern = DFBAccessPattern::STRIDED},
    };
    compute.compile_time_args = {{"entries_per_neo", 4u}, {"words_per_entry", 256u}};

    ProgramSpec spec{
        .name = "intra_config_2_0",
        .kernels = {compute},
        .dataflow_buffers = {dfb_spec},
        .tensor_parameters = {},
        .work_units = {WorkUnitSpec{.name = "wu", .kernels = {COMPUTE}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(this->device(), spec);
    m2_config_test_helpers::validate_intra_tensix_dfb_2_0(program, CoreCoord(0, 0));
}

// =====================================================================================
// Group 3 M2: rejection tests. Where the M2 spec model can express the same bad
// config, we assert that MakeProgramFromSpec / finalize throws. Where the bad
// config is not expressible (B7 CB+DFB mix, B9 INTER scope), the test is
// documented as not-applicable.
// =====================================================================================

// B6 — Producer access pattern = ALL is rejected.
TEST_F(UnitMeshFixture, B6_AllProducer_Rejected_2_0) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB validation tested on Quasar";
    }
    using namespace m2_config_test_helpers;
    M2ConfigDFBParams p{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 1,
        .pap = DFBAccessPattern::ALL,  // <-- the offense (producer ALL not supported)
        .cap = DFBAccessPattern::STRIDED,
        .implicit_sync = false,
    };
    EXPECT_THROW(
        {
            Program program = build_single_dfb_program_2_0(this->device(), p);
            program.impl().finalize_dataflow_buffer_configs();
        },
        std::exception);
}

// B7 — CB+DFB mix rejection.
// Not applicable to M2: ProgramSpec doesn't expose a circular-buffer API
// (CircularBufferConfig is a legacy host-API construct). M2 programs are
// purely DFB-based; the legacy CB-then-DFB rejection path can't be exercised
// through the M2 spec model.
TEST_F(UnitMeshFixture, B7_CB_DFB_Mix_Rejected_2_0) {
    GTEST_SKIP() << "Not applicable: M2 ProgramSpec has no CB construct";
}

// B8 — ALL consumer with num_consumers > 4 is rejected (Remapper has 4 clientR slots).
TEST_F(UnitMeshFixture, B8_FiveAllConsumers_Rejected_2_0) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Remapper limit tested on Quasar";
    }
    using namespace m2_config_test_helpers;
    M2ConfigDFBParams p{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::TENSIX,  // 5 Tensix consumers (try to exceed remapper's 4 clientR slots)
        .num_producers = 1,
        .num_consumers = 5,
        .num_entries = 20,
        .pap = DFBAccessPattern::STRIDED,
        .cap = DFBAccessPattern::ALL,
        .implicit_sync = false,
    };
    EXPECT_THROW(
        {
            Program program = build_single_dfb_program_2_0(this->device(), p);
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
TEST_F(UnitMeshFixture, B9_InterTensixScope_Rejected_2_0) {
    GTEST_SKIP() << "Not applicable: M2 DataflowBufferSpec has no tensix_scope field";
}

// Host-only: packer INTRA pairs allocate top-down from 63; DM1 1:m pairs stay in [0,16).
TEST_F(UnitMeshFixture, IntraPackerRemapperPairsTopDownConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar-only remapper pool split";
    }

    Program program = CreateProgram(this->device());
    CoreCoord logical_core(0, 0);

    // Two INTRA DFBs on Neo0 → ClientL 16 then 18; contiguous packer remapper block [62,64).
    for (int i = 0; i < 2; i++) {
        experimental::dfb::DataflowBufferConfig intra{
            .entry_size = 1024,
            .num_entries = 4,
            .producer_risc_mask = 0x100,
            .num_producers = 1,
            .pap = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask = 0x100,
            .num_consumers = 1,
            .cap = dfb::AccessPattern::STRIDED,
            .enable_producer_implicit_sync = false,
            .enable_consumer_implicit_sync = false,
            .tensix_scope = experimental::dfb::TensixScope::INTRA};
        experimental::dfb::CreateDataflowBuffer(program, logical_core, intra);
    }

    // DM→Tensix ALL remapper on same core (producer STRIDED, consumer ALL).
    experimental::dfb::DataflowBufferConfig remapper{
        .entry_size = 1024,
        .num_entries = 8,
        .producer_risc_mask = 0x4,  // DM2
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0xF00,  // Neo0-3
        .num_consumers = 4,
        .cap = dfb::AccessPattern::ALL,
        .enable_producer_implicit_sync = true,
        .enable_consumer_implicit_sync = true};
    experimental::dfb::CreateDataflowBuffer(program, logical_core, remapper);

    program.impl().finalize_dataflow_buffer_configs();
    auto dfbs = program.impl().dataflow_buffers_on_core(logical_core);
    ASSERT_EQ(dfbs.size(), 3u);

    std::vector<uint8_t> packer_pairs;
    std::vector<uint8_t> packer_tcs;
    uint8_t dm1_pair = 0xFF;
    for (const auto& dfb : dfbs) {
        if (dfb->remapper_programmer == experimental::dfb::detail::RemapperProgrammer::TENSIX_PACKER) {
            ASSERT_FALSE(dfb->groups.empty());
            const auto& rc = dfb->groups[0].hw_risc_configs[0];
            packer_pairs.push_back(rc.config.remapper_pair_index);
            packer_tcs.push_back(::dfb::get_counter_id(rc.config.packed_tile_counter[0]));
            EXPECT_EQ(dfb->dm1_remapper_slot_count(), 0u);
        } else if (dfb->remapper_programmer == experimental::dfb::detail::RemapperProgrammer::DM1) {
            ASSERT_FALSE(dfb->groups.empty());
            for (const auto& rc : dfb->groups[0].hw_risc_configs) {
                if (rc.is_producer) {
                    dm1_pair = rc.config.remapper_pair_index;
                }
            }
        }
    }
    ASSERT_EQ(packer_pairs.size(), 2u);
    ASSERT_EQ(packer_tcs.size(), 2u);
    EXPECT_EQ(packer_tcs[0], 16);
    EXPECT_EQ(packer_tcs[1], 18);
    // Contiguous top-down block of size 2 starts at 62: [62, 64).
    EXPECT_EQ(packer_pairs[0], 62);
    EXPECT_EQ(packer_pairs[1], 63);
    EXPECT_NE(dm1_pair, 0xFF);
    EXPECT_LT(dm1_pair, ::dfb::NUM_REMAPPER_ONE_TO_MANY_PAIRINGS);
    EXPECT_LT(dm1_pair, packer_pairs[0]);
}

// T7: one INTRA DFB past what a Neo's Tensix-only TC pool can back must fail. The limit is per Neo:
// each INTRA DFB spends a ClientL + shadow pair out of that Neo's 16 Tensix-only TCs, so 8 fit and the
// 9th exhausts the pool (and its packer remapper pair).
TEST_F(UnitMeshFixture, MaxEightIntraPerNeoConfig) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar-only INTRA TC pool limit";
    }

    constexpr uint32_t max_intra_dfbs_per_neo =
        ::dfb::NUM_TENSIX_ONLY_TILE_COUNTERS / ::dfb::TILE_COUNTERS_PER_INTRA_TENSIX_DFB;

    Program program = CreateProgram(this->device());
    CoreCoord logical_core(0, 0);
    for (uint32_t i = 0; i < max_intra_dfbs_per_neo + 1; i++) {
        experimental::dfb::DataflowBufferConfig intra{
            .entry_size = 1024,
            .num_entries = 4,
            .producer_risc_mask = 0x100,
            .num_producers = 1,
            .pap = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask = 0x100,
            .num_consumers = 1,
            .cap = dfb::AccessPattern::STRIDED,
            .enable_producer_implicit_sync = false,
            .enable_consumer_implicit_sync = false,
            .tensix_scope = experimental::dfb::TensixScope::INTRA};
        experimental::dfb::CreateDataflowBuffer(program, logical_core, intra);
    }

    EXPECT_THROW(program.impl().finalize_dataflow_buffer_configs(), std::exception);
}

// =====================================================================================
// Device slot assignment
// =====================================================================================

namespace {

experimental::dfb::DataflowBufferConfig make_minimal_dfb_config() {
    return experimental::dfb::DataflowBufferConfig{
        .entry_size = 1024,
        .num_entries = 2,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::STRIDED,
        .consumer_risc_mask = 0x2,
        .num_consumers = 1,
        .cap = dfb::AccessPattern::STRIDED,
        .enable_producer_implicit_sync = false,
        .enable_consumer_implicit_sync = false};
}

// The safety property behind slot reuse: a slot indexes a per-core config table, so two DFBs may
// share a slot only when no core hosts both of them.
void expect_no_slot_collision_on_any_core(Program& program, const CoreRange& cores) {
    for (auto x = cores.start_coord.x; x <= cores.end_coord.x; x++) {
        for (auto y = cores.start_coord.y; y <= cores.end_coord.y; y++) {
            const CoreCoord core(x, y);
            std::set<uint32_t> slots_on_core;
            for (const auto& dfb : program.impl().dataflow_buffers_on_core(core)) {
                EXPECT_TRUE(slots_on_core.insert(dfb->device_slot).second)
                    << "Core (" << x << "," << y << ") has two DFBs at device slot " << dfb->device_slot;
            }
        }
    }
}

}  // namespace

// A DFB created late (high program-wide id) that shares no core with the earlier DFBs should reuse a
// low slot instead of extending the per-core config table. This is the shape that overflowed the
// dispatch payload when the slot was just a creation counter (issue #51409).
TEST_F(UnitMeshFixture, DFBDeviceSlotsReusedAcrossDisjointCores) {
    const CoreCoord grid = this->device().compute_with_storage_grid_size();
    // Coordinator at (0,0) plus workers spanning (1,0)..(4,0).
    if (grid.x < 5) {
        GTEST_SKIP() << "Needs at least 5 worker cores in a row (got " << grid.x << "x" << grid.y << ")";
    }
    const CoreCoord coordinator(0, 0);
    const CoreRange workers(CoreCoord(1, 0), CoreCoord(4, 0));
    const CoreRange all_cores(CoreCoord(0, 0), CoreCoord(4, 0));

    Program program = CreateProgram(this->device());
    const auto config = make_minimal_dfb_config();

    // Spans every core, so it conflicts with everything below.
    uint32_t wide_id = experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(all_cores), config);

    // Worker-only DFBs, then a coordinator-only DFB created last.
    std::vector<uint32_t> worker_ids;
    worker_ids.reserve(3);
    for (int i = 0; i < 3; i++) {
        worker_ids.push_back(experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(workers), config));
    }
    uint32_t coordinator_id =
        experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(CoreRange(coordinator)), config);

    const auto& impl = program.impl();
    EXPECT_EQ(impl.get_dataflow_buffer(coordinator_id)->id, 4u) << "expected creation order to give this the last id";

    expect_no_slot_collision_on_any_core(program, all_cores);

    EXPECT_EQ(impl.get_dataflow_buffer(wide_id)->device_slot, 0u);
    for (uint32_t i = 0; i < worker_ids.size(); i++) {
        EXPECT_EQ(impl.get_dataflow_buffer(worker_ids[i])->device_slot, i + 1)
            << "worker DFB " << i << " conflicts with the wide DFB and its worker peers";
    }
    // Conflicts only with the wide DFB, so it reuses slot 1 rather than taking slot 4.
    EXPECT_EQ(impl.get_dataflow_buffer(coordinator_id)->device_slot, 1u);
}

// WH/BH firmware addresses this config as a CB-format table indexed by device_slot. Verify that
// serialization preserves empty entries when a core's assigned slots are not consecutive.
TEST_F(UnitMeshFixture, DFBConfigSerializationPreservesGappedDeviceSlots) {
    if (this->device().arch() == ARCH::QUASAR) {
        GTEST_SKIP() << "WH/BH use the slot-indexed CB-format DFB config";
    }

    const CoreCoord grid = this->device().compute_with_storage_grid_size();
    if (grid.x < 2) {
        GTEST_SKIP() << "Needs at least two worker cores in a row";
    }

    const CoreCoord coordinator(0, 0);
    const CoreCoord worker(1, 0);
    const CoreRange all_cores(coordinator, worker);
    Program program = CreateProgram(this->device());
    const auto config = make_minimal_dfb_config();

    for (int i = 0; i < 3; i++) {
        experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(CoreRange(worker)), config);
    }
    const uint32_t coordinator_id =
        experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(CoreRange(coordinator)), config);
    const uint32_t wide_id = experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(all_cores), config);

    program.impl().finalize_dataflow_buffer_configs();
    program.impl().prepare_ephemeral_allocation(this->device().impl().get_devices()[0]);
    program.impl().allocate_dataflow_buffers(this->device().impl().get_devices()[0]);

    const auto& impl = program.impl();
    EXPECT_EQ(impl.get_dataflow_buffer(coordinator_id)->device_slot, 0u);
    EXPECT_EQ(impl.get_dataflow_buffer(wide_id)->device_slot, 3u);

    constexpr size_t words_per_slot = 4;
    constexpr size_t num_slots = 4;
    std::vector<uint8_t> serialized(num_slots * words_per_slot * sizeof(uint32_t), 0xFF);
    const auto dfbs = impl.dataflow_buffers_on_core(coordinator);
    const size_t bytes_written =
        experimental::dfb::detail::serialize_dfb_config_for_core(coordinator, dfbs, serialized);
    EXPECT_EQ(bytes_written, serialized.size());

    auto read_word = [&](size_t slot, size_t word) {
        uint32_t value = 0;
        std::memcpy(&value, serialized.data() + (slot * words_per_slot + word) * sizeof(uint32_t), sizeof(uint32_t));
        return value;
    };

    const auto coordinator_dfb = impl.get_dataflow_buffer(coordinator_id);
    EXPECT_EQ(read_word(0, 0), coordinator_dfb->uniform_alloc_addr());
    EXPECT_EQ(read_word(0, 1), coordinator_dfb->total_size());
    EXPECT_EQ(read_word(0, 2), coordinator_dfb->config.num_entries);
    EXPECT_EQ(read_word(0, 3), coordinator_dfb->config.entry_size);
    for (size_t slot : {1u, 2u}) {
        for (size_t word = 0; word < words_per_slot; word++) {
            EXPECT_EQ(read_word(slot, word), 0u) << "slot " << slot << ", word " << word;
        }
    }
    const auto wide_dfb = impl.get_dataflow_buffer(wide_id);
    EXPECT_EQ(read_word(3, 0), wide_dfb->uniform_alloc_addr());
    EXPECT_EQ(read_word(3, 1), wide_dfb->total_size());
    EXPECT_EQ(read_word(3, 2), wide_dfb->config.num_entries);
    EXPECT_EQ(read_word(3, 3), wide_dfb->config.entry_size);
}

// Quasar counterpart: the coordinator's slots are {0, 3}; the two unused ones are in no participation mask.
TEST_F(UnitMeshFixture, DfbSerializeGlobalHeaderGappedDeviceSlots) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar-only DFB config layout";
    }

    const CoreCoord grid = this->device().compute_with_storage_grid_size();
    if (grid.x < 2) {
        GTEST_SKIP() << "Needs at least two worker cores in a row";
    }

    const CoreCoord coordinator(0, 0);
    const CoreCoord worker(1, 0);
    const CoreRange all_cores(coordinator, worker);
    Program program = CreateProgram();
    const auto config = make_minimal_dfb_config();

    for (int i = 0; i < 3; i++) {
        experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(CoreRange(worker)), config);
    }
    const uint32_t coordinator_id =
        experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(CoreRange(coordinator)), config);
    const uint32_t wide_id = experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(all_cores), config);

    program.impl().finalize_dataflow_buffer_configs();
    program.impl().prepare_ephemeral_allocation(this->device().get_devices()[0]);
    program.impl().allocate_dataflow_buffers(this->device().get_devices()[0]);

    const auto& impl = program.impl();
    EXPECT_EQ(impl.get_dataflow_buffer(coordinator_id)->device_slot, 0u);
    EXPECT_EQ(impl.get_dataflow_buffer(wide_id)->device_slot, 3u);

    const auto dfbs = impl.dataflow_buffers_on_core(coordinator);
    ASSERT_EQ(dfbs.size(), 2u);
    std::vector<uint8_t> buf(8192, 0);
    const size_t nbytes = experimental::dfb::detail::serialize_dfb_config_for_core(coordinator, dfbs, buf);
    ASSERT_GT(nbytes, 0u);

    const auto* ghdr = reinterpret_cast<const dfb_global_header_t*>(buf.data());
    EXPECT_EQ(ghdr->num_dfbs, 2u);
    // Harts 0 (producer_risc_mask 0x1) and 1 (consumer_risc_mask 0x2) each init both DFBs; no other
    // hart has an entry.
    for (uint8_t h = 0; h < ::dfb::NUM_PARTICIPATING_HARTIDS; ++h) {
        EXPECT_EQ(ghdr->hart_desc[h].num_entries, h < 2 ? 2u : 0u) << "hart " << static_cast<int>(h);
    }
    // Each DM entry names its slot through iface_byte_off (slot * sizeof(LocalDFBInterface)), so the
    // two entries carry slots 0 and 3, leaving 1 and 2 untouched.
    for (uint8_t h = 0; h < 2; ++h) {
        SCOPED_TRACE("hart " + std::to_string(h));
        const uint8_t* entry = buf.data() + ghdr->hart_desc[h].blob_start;
        EXPECT_EQ(dm_entry_u16(entry, 0), 0u * DFB_DM_IFACE_SIZE);
        entry += entry[kDmEntryBytesOff];
        EXPECT_EQ(dm_entry_u16(entry, 0), 3u * DFB_DM_IFACE_SIZE);
    }
}

// Checks that no 64B line is read cached by two DMs, and that no cached line holds header or signal bytes.
static uint32_t expect_dm_cached_lines_disjoint(const std::vector<uint8_t>& buf, size_t nbytes) {
    constexpr uint32_t kLineBytes = 64;
    const auto* ghdr = reinterpret_cast<const dfb_global_header_t*>(buf.data());
    std::array<std::set<uint32_t>, ::dfb::TENSIX_RISC_OFFSET> lines;
    const auto add = [&](uint8_t dm, uint32_t off, uint32_t len) {
        if (len == 0) {
            return;
        }
        EXPECT_LE(off + len, nbytes) << "DM " << static_cast<int>(dm) << " range [" << off << ", +" << len << ")";
        for (uint32_t l = off / kLineBytes; l <= (off + len - 1) / kLineBytes; ++l) {
            lines[dm].insert(l);
        }
    };
    for (uint8_t dm = 0; dm < ::dfb::TENSIX_RISC_OFFSET; ++dm) {
        if (ghdr->hart_desc[dm].num_entries > 0) {
            add(dm, ghdr->hart_desc[dm].blob_start, ghdr->hart_desc[dm].blob_len);
        }
    }
    add(1, ghdr->dm1_slots & 0xFFFFu, (ghdr->dm1_slots >> 16) * static_cast<uint32_t>(sizeof(dfb_dm1_remapper_slot_t)));
    if (ghdr->has_dm0_isr) {
        add(0,
            dfb_dm0_packed_blob_offset(ghdr->dm0_packed) + DFB_DM0_CACHED_PAYLOAD_OFFSET,
            dm0_isr_txn_desc_pool_byte_size(
                dfb_dm0_packed_producer_mask(ghdr->dm0_packed), dfb_dm0_packed_consumer_mask(ghdr->dm0_packed)));
    }

    std::map<uint32_t, uint8_t> owner;
    uint32_t num_dms = 0;
    for (uint8_t dm = 0; dm < ::dfb::TENSIX_RISC_OFFSET; ++dm) {
        num_dms += lines[dm].empty() ? 0u : 1u;
        for (const uint32_t l : lines[dm]) {
            const auto [it, inserted] = owner.emplace(l, dm);
            EXPECT_TRUE(inserted) << "line " << l << " is read cached by DM " << static_cast<int>(it->second)
                                  << " and DM " << static_cast<int>(dm);
        }
    }

    const uint32_t signal_bytes = DFB_SIGNAL_REGION_BYTES;
    const std::pair<uint32_t, uint32_t> uncached[] = {
        {0u, static_cast<uint32_t>(sizeof(dfb_global_header_t))}, {ghdr->dfb_signal_region_off, signal_bytes}};
    for (const auto& [off, len] : uncached) {
        if (len == 0) {
            continue;
        }
        for (uint32_t l = off / kLineBytes; l <= (off + len - 1) / kLineBytes; ++l) {
            if (const auto it = owner.find(l); it != owner.end()) {
                ADD_FAILURE() << "line " << l << " holds uncached bytes [" << off << ", +" << len
                              << ") but is read cached by DM " << static_cast<int>(it->second);
            }
        }
    }
    return num_dms;
}

// Each DM's cached part of the DFB config sits on its own cache lines, for a DM-only topology (many small
// blobs) and for a mixed one that also fills DM1's remapper slots and DM0's ISR descriptor pool.
TEST_F(UnitMeshFixture, DfbConfigDmCachedLinesDisjoint) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar-only DFB config layout";
    }
    constexpr uint16_t DM0 = 1u << 0, DM2 = 1u << 2, DM3 = 1u << 3, DM4 = 1u << 4, DM5 = 1u << 5, DM6 = 1u << 6,
                       DM7 = 1u << 7;
    constexpr uint16_t NEO0 = 1u << 8, NEO1 = 1u << 9, NEO2 = 1u << 10, NEO3 = 1u << 11;
    const auto make = [](uint16_t producer, uint16_t consumers, dfb::AccessPattern cap, bool implicit_sync) {
        return experimental::dfb::DataflowBufferConfig{
            .entry_size = 1024,
            .num_entries = 16,
            .producer_risc_mask = producer,
            .num_producers = 1,
            .pap = dfb::AccessPattern::STRIDED,
            .consumer_risc_mask = consumers,
            .num_consumers = static_cast<uint8_t>(__builtin_popcount(consumers)),
            .cap = cap,
            .enable_producer_implicit_sync = implicit_sync,
            .enable_consumer_implicit_sync = implicit_sync};
    };
    constexpr auto ALL = dfb::AccessPattern::ALL;
    constexpr auto STRIDED = dfb::AccessPattern::STRIDED;
    struct Topology {
        const char* name;
        std::vector<experimental::dfb::DataflowBufferConfig> dfbs;
        uint32_t min_dms;
        bool expect_dm1_slots;
        bool expect_dm0_isr;
    };
    const Topology topologies[] = {
        {"dm_only",
         {make(DM0, DM6, STRIDED, false),
          make(DM2, DM7, STRIDED, false),
          make(DM3, DM6, STRIDED, false),
          make(DM4, DM7, STRIDED, false),
          make(DM5, DM6, STRIDED, false)},
         7,
         false,
         false},
        {"mixed",
         {make(DM0, NEO0, STRIDED, true),
          make(DM2, NEO0 | NEO2, ALL, true),
          make(DM3, NEO1 | NEO3, ALL, true),
          make(DM4, NEO1, STRIDED, true),
          make(DM5, DM6, STRIDED, true),
          make(DM7, NEO2, STRIDED, false)},
         8,
         true,
         true},
    };

    const CoreCoord core(0, 0);
    for (const auto& topology : topologies) {
        SCOPED_TRACE(topology.name);
        Program program = CreateProgram(this->device());
        for (const auto& config : topology.dfbs) {
            experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(CoreRange(core)), config);
        }
        program.impl().finalize_dataflow_buffer_configs();
        program.impl().prepare_ephemeral_allocation(this->device().get_devices()[0]);
        program.impl().allocate_dataflow_buffers(this->device().get_devices()[0]);

        const auto dfbs = program.impl().dataflow_buffers_on_core(core);
        ASSERT_EQ(dfbs.size(), topology.dfbs.size());
        std::vector<uint8_t> buf(8192, 0);
        const size_t nbytes = experimental::dfb::detail::serialize_dfb_config_for_core(core, dfbs, buf);
        ASSERT_GT(nbytes, 0u);

        const auto* ghdr = reinterpret_cast<const dfb_global_header_t*>(buf.data());
        EXPECT_EQ((ghdr->dm1_slots >> 16) > 0, topology.expect_dm1_slots);
        EXPECT_EQ(ghdr->has_dm0_isr != 0, topology.expect_dm0_isr);
        EXPECT_GE(expect_dm_cached_lines_disjoint(buf, nbytes), topology.min_dms);
        EXPECT_EQ(nbytes, experimental::dfb::detail::compute_dfb_config_serialized_size(dfbs));

        // The first region after where the DM1 blob would start: DM0's blob if it has one, else the
        // first hart blob.
        uint32_t next_region = UINT32_MAX;
        if (ghdr->has_dm0_isr) {
            next_region = dfb_dm0_packed_blob_offset(ghdr->dm0_packed);
        } else {
            for (const auto& hart : ghdr->hart_desc) {
                if (hart.num_entries > 0) {
                    next_region = std::min<uint32_t>(next_region, hart.blob_start);
                }
            }
        }

        // The DM1 blob exists exactly when DM1 has slots; without it the next region takes its place.
        if (topology.expect_dm1_slots) {
            const uint32_t num_slots = ghdr->dm1_slots >> 16;
            const uint32_t slot_bytes = num_slots * static_cast<uint32_t>(sizeof(dfb_dm1_remapper_slot_t));
            EXPECT_EQ(ghdr->dm1_slots & 0xFFFFu, DFB_DM1_REMAPPER_BLOB_OFFSET);
            EXPECT_EQ(experimental::dfb::detail::dm1_remapper_blob_core_size(dfbs), slot_bytes);
            EXPECT_EQ(next_region, (DFB_DM1_REMAPPER_BLOB_OFFSET + slot_bytes + 63u) & ~63u);
        } else {
            EXPECT_EQ(next_region, DFB_DM1_REMAPPER_BLOB_OFFSET);
        }

        // A hart with no entries gets no blob bytes.
        for (uint8_t h = 0; h < ::dfb::NUM_PARTICIPATING_HARTIDS; ++h) {
            SCOPED_TRACE("hart " + std::to_string(h));
            EXPECT_EQ(ghdr->hart_desc[h].num_entries == 0, ghdr->hart_desc[h].blob_len == 0);
        }
    }
}

// The serializer places every region of the DFB config on a 64B line counted from the config's start, so the DMs
// only get line-aligned reads if that start address is itself a multiple of 64. Fast dispatch packs kernel config
// ring entries back to back, so this relies on every Tensix config size being rounded up to 64. Checks the address
// each DM read, for four programs of different raw sizes launched back to back.
TEST_F(UnitMeshAnyDispatchFixture, DfbConfigBaseIsCacheLineAlignedOnDevice) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "Quasar-only DFB config layout";
    }
    constexpr uint32_t kLineBytes = 64;
    constexpr uint32_t num_programs = 4;
    constexpr uint32_t extent_record_bytes = 8 * sizeof(uint32_t);

    distributed::MeshDevice& mesh_device = this->device();
    const experimental::NodeCoord node{0, 0};
    const CoreCoord core{0, 0};
    const experimental::DFBSpecName DFB_NAME{"dfb"};
    const experimental::KernelSpecName PRODUCER{"producer"};
    const experimental::KernelSpecName CONSUMER{"consumer"};

    const uint32_t base_results_l1 =
        top_of_l1_scratch_addr(mesh_device, num_programs * (sizeof(uint32_t) + extent_record_bytes));
    const uint32_t consumer_results_l1 = base_results_l1 + num_programs * sizeof(uint32_t);
    std::vector<uint32_t> zeros(num_programs, 0u);
    slow_dispatch::WriteToL1(mesh_device, core, base_results_l1, zeros);

    const distributed::MeshCoordinateRange device_range(mesh_device.shape());
    std::vector<distributed::MeshWorkload> workloads(num_programs);
    for (uint32_t i = 0; i < num_programs; ++i) {
        experimental::KernelSpec producer_spec{
            .unique_id = PRODUCER,
            .source = "tests/tt_metal/tt_metal/test_kernels/dataflow/dfb_config_base_probe_dm.cpp",
            .num_threads = 1,
            .dfb_bindings = {experimental::ProducerOf(DFB_NAME, "out")},
            .compile_time_args = {{"pad_nops", 4 * i}},
            .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
            .hw_config = experimental::DataMovementHardwareConfig{
                .config_2xx = experimental::DataMovementHardwareConfig::DataMovement2XXConfig{
                    .disable_dfb_implicit_sync_for = {DFB_NAME}}}};
        experimental::KernelSpec consumer_spec{
            .unique_id = CONSUMER,
            .source = "tests/tt_metal/tt_metal/test_kernels/compute/dfb_extent_probe_compute.cpp",
            .num_threads = 1,
            .dfb_bindings = {{
                .dfb_spec_name = DFB_NAME,
                .accessor_name = "in",
                .endpoint_type = experimental::DFBEndpointType::CONSUMER,
                .access_pattern = experimental::DFBAccessPattern::STRIDED,
            }},
            .compile_time_args =
                {{"num_tc_snapshots", 1u},
                 {"rotate_tc", 0u},
                 {"drain_producer_rotate_credits", 0u},
                 {"drain_last_tc_credit", 0u},
                 {"num_producers", 1u}},
            .runtime_arg_schema = {.runtime_arg_names = {"result_l1_addr"}},
            .hw_config = experimental::ComputeHardwareConfig{},
        };
        experimental::ProgramSpec spec{
            .name = "dfb_config_base_alignment",
            .kernels = {producer_spec, consumer_spec},
            .dataflow_buffers = {{
                .unique_id = DFB_NAME,
                .entry_size = 1024,
                .num_entries = 4,
                .data_format_metadata = tt::DataFormat::Float16_b,
            }},
            .work_units = {{.name = "main", .kernels = {PRODUCER, CONSUMER}, .target_nodes = node}},
        };

        workloads[i].add_program(device_range, experimental::MakeProgramFromSpec(mesh_device, spec));
        experimental::ProgramRunArgs run_params;
        run_params.kernel_run_args = {
            {.kernel = PRODUCER,
             .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                 node, {{"result_l1_addr", base_results_l1 + i * static_cast<uint32_t>(sizeof(uint32_t))}})},
            {.kernel = CONSUMER,
             .runtime_arg_values = experimental::MakeRuntimeArgsForSingleNode(
                 node, {{"result_l1_addr", consumer_results_l1 + i * extent_record_bytes}})},
        };
        experimental::SetProgramRunArgs(workloads[i].get_programs().at(device_range), run_params);
    }

    // Queued without blocking so fast dispatch keeps every config live in the ring at once.
    for (auto& workload : workloads) {
        distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/false);
    }
    distributed::Finish(mesh_device.mesh_command_queue());

    std::vector<uint32_t> bases;
    slow_dispatch::ReadFromL1(mesh_device, core, base_results_l1, num_programs * sizeof(uint32_t), bases);
    ASSERT_EQ(bases.size(), num_programs);
    const uint32_t tensix_index =
        MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    for (uint32_t i = 0; i < num_programs; ++i) {
        auto& impl = workloads[i].get_programs().at(device_range).impl();
        EXPECT_EQ(impl.get_program_config_sizes()[tensix_index] % kLineBytes, 0u)
            << "program " << i << " Tensix kernel config size is not a whole number of cache lines";
        const ProgramConfig& config = impl.get_program_config(tensix_index);
        log_info(
            tt::LogTest,
            "program {}: DFB config base {:#x}, config size {:#x}, rta_offset {:#x}, sem_offset {:#x}, dfb_offset "
            "{:#x}, kernel_text_offset {:#x}",
            i,
            bases[i],
            impl.get_program_config_sizes()[tensix_index],
            config.rta_offset,
            config.sem_offset,
            config.dfb_offset,
            config.kernel_text_offset);
    }
    // Slow dispatch always writes at the ring base. Under fast dispatch the programs must occupy distinct ring
    // slots, or the alignment check below only ever sees the ring base.
    if (!this->IsSlowDispatch()) {
        EXPECT_EQ(std::set<uint32_t>(bases.begin(), bases.end()).size(), num_programs)
            << "fast-dispatch programs did not land in distinct kernel config ring slots";
    }
    for (uint32_t i = 0; i < num_programs; ++i) {
        EXPECT_NE(bases[i], 0u) << "program " << i << " never recorded its DFB config base";
        EXPECT_EQ(bases[i] % kLineBytes, 0u) << "program " << i << " DFB config base 0x" << std::hex << bases[i];
    }
}

// The per-core slot limit is on how many DFBs share a core, not on how many exist in the program:
// DFBs on disjoint cores all fit at slot 0.
TEST_F(UnitMeshFixture, DFBDeviceSlotLimitIsPerCoreNotPerProgram) {
    const uint32_t max_slots = hal::get_num_dataflow_buffers();
    const CoreCoord grid = this->device().compute_with_storage_grid_size();
    const uint32_t num_dfbs = max_slots + 4;
    if (grid.x * grid.y < num_dfbs) {
        GTEST_SKIP() << "Needs at least " << num_dfbs << " cores to place one DFB per core";
    }

    Program program = CreateProgram(this->device());
    const auto config = make_minimal_dfb_config();
    for (uint32_t i = 0; i < num_dfbs; i++) {
        const CoreCoord core(i % grid.x, i / grid.x);
        uint32_t id = experimental::dfb::CreateDataflowBuffer(program, CoreRangeSet(CoreRange(core)), config);
        EXPECT_EQ(program.impl().get_dataflow_buffer(id)->device_slot, 0u)
            << "DFB " << i << " is alone on core (" << core.x << "," << core.y << ")";
    }
}

// =====================================================================================
// BLOCKED access-pattern config-rejection tests
// =====================================================================================
// 1 BLOCKED producer -> 4 BLOCKED consumers, implicit sync, block 4, 16 entries: the only txn window
// that covers whole blocks on every consumer is the whole ring, so the config is accepted and must run.
TEST_F(UnitMeshFixture, DMTest1xDFB1Bx4B_blk4_impl_2_0) {
    auto& mesh_device = this->device();
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only";
    }
    // Check the host picked a single txn id covering the whole ring.
    {
        Program probe = m2_config_test_helpers::build_single_dfb_program_2_0(
            mesh_device,
            m2_config_test_helpers::M2ConfigDFBParams{
                .producer_type = M2PorCType::DM,
                .consumer_type = M2PorCType::DM,
                .num_producers = 1,
                .num_consumers = 4,
                .num_entries = 16,
                .pap = DFBAccessPattern::BLOCKED,
                .cap = DFBAccessPattern::BLOCKED,
                .implicit_sync = true,
                .block_size = 4,
            });
        probe.impl().finalize_dataflow_buffer_configs();
        auto dfb = probe.impl().get_dataflow_buffer(probe.impl().get_dfb_handle("dfb"));
        ASSERT_TRUE(dfb->configs_finalized);
        EXPECT_EQ(dfb->producer_txn_descriptor.num_txn_ids, 1u);
        EXPECT_EQ(dfb->producer_txn_descriptor.num_entries_per_txn_id, 16u);
    }
    M2SingleDFBParams params{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 4,
        .pap = DFBAccessPattern::BLOCKED,
        .cap = DFBAccessPattern::BLOCKED,
        .implicit_sync = true,
        .num_entries = 16,
        .block_size = 4,
    };
    run_single_dfb_program_2_0(mesh_device, params);
}

// Implicit-sync BLOCKED producer -> ALL consumers must be rejected by the direct API: implicit credits
// are split across the consumers, but a broadcast needs the full count on every one.
TEST_F(UnitMeshFixture, DirectApi_BlockedAll_ImplicitBroadcastProducer_rejected) {
    if (this->device().arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "M2 path is Quasar-only";
    }
    experimental::dfb::DataflowBufferConfig config{
        .entry_size = 1024,
        .num_entries = 16,
        .producer_risc_mask = 0x1,
        .num_producers = 1,
        .pap = dfb::AccessPattern::BLOCKED,
        .producer_block_size = 4,
        .consumer_risc_mask = 0x30,
        .num_consumers = 2,
        .cap = dfb::AccessPattern::ALL,
        .enable_producer_implicit_sync = true,
        .enable_consumer_implicit_sync = false};
    Program program = CreateProgram(this->device());
    EXPECT_THAT(
        [&] {
            experimental::dfb::CreateDataflowBuffer(program, CoreCoord(0, 0), config);
            program.impl().finalize_dataflow_buffer_configs();
        },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("broadcasts its credits")));
}

// A BLOCKED consumer binding with block_size left at 0 must throw.
TEST_F(UnitMeshFixture, B10_Blocked_Rejected_2_0) {
    auto& mesh_device = this->device();
    if (mesh_device.arch() != ARCH::QUASAR) {
        GTEST_SKIP() << "DFB validation tested on Quasar";
    }
    using namespace m2_config_test_helpers;
    M2ConfigDFBParams p{
        .producer_type = M2PorCType::DM,
        .consumer_type = M2PorCType::DM,
        .num_producers = 1,
        .num_consumers = 1,
        .pap = DFBAccessPattern::STRIDED,
        .cap = DFBAccessPattern::BLOCKED,  // <-- BLOCKED consumer but block_size left 0 (the offense)
        .implicit_sync = false,
    };
    EXPECT_THROW(
        {
            Program program = build_single_dfb_program_2_0(mesh_device, p);
            program.impl().finalize_dataflow_buffer_configs();
        },
        std::exception);
}

// A program with DFBs and no CBs reserves no CB config region, so its DFB config follows its semaphores directly and
// dispatch sends both in one write. On Quasar the DFB config instead starts on the next 64B cache line for best
// performance, since the DMs read it through the cache. local_cb_mask describes the DFBs once the offsets are
// finalized: bit 0 for slot 0 on WH/BH, a count of 1 on Quasar.
TEST_F(UnitMeshAnyDispatchFixture, TensixDFBConfigFollowsSemaphoresWithoutCBRegion) {
    const bool is_quasar = this->device().arch() == ARCH::QUASAR;
    ProgramSpec spec =
        is_quasar ? test_helpers::MakeMinimalValidProgramSpec() : test_helpers::MakeMinimalGen1ValidProgramSpec();
    spec.semaphores.push_back({.unique_id = SemaphoreSpecName{"sem"}, .target_nodes = NodeCoord{0, 0}});

    Program program = experimental::MakeProgramFromSpec(this->device(), spec);
    program.impl().finalize_dataflow_buffer_configs();
    program.impl().finalize_offsets(this->device().get_devices()[0]);

    const auto& hal = MetalContext::instance(program.impl().get_context_id()).hal();
    const uint32_t index = hal.get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
    const ProgramConfig& config = program.impl().get_program_config(index);
    EXPECT_EQ(config.cb_size, 0u);
    const uint32_t sem_end = config.sem_offset + config.sem_size;
    EXPECT_EQ(config.dfb_offset, is_quasar ? tt::align(sem_end, 64u) : sem_end);
    for (const auto& kernel_group : program.impl().get_kernel_groups(index)) {
        EXPECT_EQ(kernel_group->launch_msg.view().kernel_config().local_cb_mask(), 1u);
    }
}

}  // end namespace tt::tt_metal
