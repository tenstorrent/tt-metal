// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// tests/host/test_packet_injection.cpp -- pure host tests for ABI layouts,
// admissions, golden layouts, capacity limits, and wrapping distances.

#include <gtest/gtest.h>

#include <cstring>

#include "x280/device/x280_state.hpp"
#include "x280_address_map.h"
#include "x280_hart_protocol.h"
#include "kernels/packet_injection/packet_injection_abi.h"
#include "kernels/packet_injection/packet_injection_slot.hpp"
#include "kernels/shared/tensix_live_timer.h"

// Define test defaults for host-only unit tests so headers can be included
#ifndef PACKET_INJECTION_MODE
#define PACKET_INJECTION_MODE 1
#endif
#ifndef PACKET_INJECTION_INCLUDE_CREATION
#define PACKET_INJECTION_INCLUDE_CREATION 0
#endif

#include "kernels/packet_injection/tensix_producer.hpp"
#include "kernels/packet_injection/x280_consumer.hpp"
#include "rt/x280_noc_map.h"
#include "tests/device/suites/packet_injection/regions.hpp"
#include "harness/output/output.hpp"
#include "tests/host/translated_coord_factory.hpp"

// Define the global mock reader for host tests
uint32_t (*g_tensix_live_timer_mock_read)(uintptr_t addr) = nullptr;

using namespace x280::harness::regions::packet_injection;

namespace {

// The two payload sizes a per-worker rate is scaled by: the fabric header a
// request actually carries, and the ring slot it occupies.
constexpr uint64_t kUsefulHeaderBytes = sizeof(fs_hybrid_mesh_packet_header_t);

// Parameter set for one pi_compute_layout() call, so the admission boundaries
// can be tabulated instead of restated.
struct LayoutCase {
    const char* name;
    uint32_t P;
    uint32_t T;
    uint32_t S;
    uint32_t B;
    uint64_t N;
    uint32_t mode;
    uint32_t l1_tile_base;
    // The class pi_compute_layout() must refuse this with; PI_LAYOUT_OK for a
    // case that has to be accepted.
    pi_layout_code_t expect = PI_LAYOUT_OK;
};

// Succeeds only when pi_compute_layout() both rejects `c` and leaves every byte
// of its output untouched, so no address can be derived from rejected input.
::testing::AssertionResult RejectsLeavingOutputUntouched(const LayoutCase& c) {
    pi_layout_t out;
    pi_layout_t poison;
    std::memset(&out, 0xA5, sizeof(out));
    std::memset(&poison, 0xA5, sizeof(poison));

    const pi_layout_check_t check = pi_compute_layout(c.P, c.T, c.S, c.B, c.N, c.mode, c.l1_tile_base, &out);
    if (check.code == (uint32_t)PI_LAYOUT_OK) {
        return ::testing::AssertionFailure() << c.name << ": accepted, expected rejection";
    }
    if (check.code != (uint32_t)c.expect) {
        return ::testing::AssertionFailure() << c.name << ": refused as " << pi_layout_code_name(check.code)
                                             << ", expected " << pi_layout_code_name((uint32_t)c.expect);
    }
    if (std::memcmp(&out, &poison, sizeof(out)) != 0) {
        return ::testing::AssertionFailure() << c.name << ": wrote to *out on failure";
    }
    return ::testing::AssertionSuccess();
}

::testing::AssertionResult Accepts(const LayoutCase& c, pi_layout_t* out) {
    const pi_layout_check_t check = pi_compute_layout(c.P, c.T, c.S, c.B, c.N, c.mode, c.l1_tile_base, out);
    if (check.code != (uint32_t)PI_LAYOUT_OK) {
        return ::testing::AssertionFailure() << c.name << ": rejected as " << pi_layout_code_name(check.code)
                                             << ", value " << check.value << " against bound " << check.bound;
    }
    return ::testing::AssertionSuccess();
}

// Compares the two independent progress witnesses a run leaves behind: the
// X280's pi_consumer_slot_t in LIM and the producer's own pi_producer_result_t
// in Tensix L1. Wrap-safe; true only when every pairing agrees.
bool ProgressWitnessesAgree(const pi_consumer_slot_t& consumer, const pi_producer_result_t& producer) {
    return pi_counter_distance(consumer.consumed, producer.published_progress) == 0u &&
           pi_counter_distance(consumer.credited, producer.consumed_progress) == 0u &&
           consumer.phase_done == producer.measured_completed;
}

}  // namespace

// --------------------------------------------------------------------------
// Record sizes, alignments, and serialized offsets
// --------------------------------------------------------------------------

// Only an actor that reached DONE admits a measurement, and a failed one never
// reads as still in progress, whatever its error.
TEST(PacketInjectionAbiRecords, OnlyDoneIsAMeasurementAndEveryFailureIsTerminal) {
    for (uint32_t phase = PI_PHASE_INIT; phase <= PI_PHASE_FAILED; ++phase) {
        for (uint32_t error = PI_ERROR_NONE; error <= PI_ERROR_LAYOUT_INVARIANT; ++error) {
            if (error == 14u) {
                continue;  // unused
            }
            const ubench_outcome_t outcome = pi_outcome(static_cast<pi_phase_t>(phase), static_cast<pi_error_t>(error));
            EXPECT_EQ(ubench_outcome_is_measurement(outcome), phase == PI_PHASE_DONE)
                << "phase " << phase << " error " << error;
            EXPECT_EQ(outcome == UBENCH_OUTCOME_IN_PROGRESS, phase != PI_PHASE_DONE && phase != PI_PHASE_FAILED)
                << "phase " << phase << " error " << error;
        }
    }
}

// The consumer's cleared-window check covers exactly the slots the run opened:
// after an ASIC reset an unopened slot reads {index, 0, 0, 0} (observed; no
// document states it), so a throughput or backpressure run, which never opens the timer
// slot, must not check it. A tile count past the credit range would alias it.
TEST(PacketInjectionWindows, OnlyTheSlotsTheRunOpensAreReadBackCleared) {
    constexpr uint32_t c0 = X280_TLB_PACKET_INJECTION_CREDIT_BASE;
    EXPECT_EQ(consumer_window_slots(PI_MODE_THROUGHPUT, 2u), (std::vector<uint32_t>{c0, c0 + 1u}));
    EXPECT_EQ(consumer_window_slots(PI_MODE_BACKPRESSURE, 2u), (std::vector<uint32_t>{c0, c0 + 1u}));
    EXPECT_EQ(
        consumer_window_slots(PI_MODE_LATENCY, 2u),
        (std::vector<uint32_t>{c0, c0 + 1u, X280_TLB_PACKET_INJECTION_TIMER}));
    EXPECT_NO_THROW(consumer_window_slots(PI_MODE_LATENCY, X280_TLB_PACKET_INJECTION_CREDIT_COUNT));
    EXPECT_THROW(
        consumer_window_slots(PI_MODE_LATENCY, X280_TLB_PACKET_INJECTION_CREDIT_COUNT + 1u), std::invalid_argument);
}

TEST(PacketInjectionAbiRecords, SizesAndAlignments) {
    EXPECT_EQ(sizeof(pi_x280_args_t), 128u);
    EXPECT_EQ(alignof(pi_x280_args_t), 64u);

    EXPECT_EQ(sizeof(pi_producer_args_t), 72u);
    EXPECT_EQ(sizeof(pi_producer_args_t), PI_PRODUCER_ARG_WORDS * sizeof(uint32_t));
    EXPECT_EQ(alignof(pi_producer_args_t), 8u);

    EXPECT_EQ(sizeof(pi_producer_arg_words_t), 72u);
    EXPECT_EQ(alignof(pi_producer_arg_words_t), 8u);

    // A whole cache line each: the producer publishes these by a NoC write whose
    // source and destination must stay congruent, which a smaller slot breaks.
    EXPECT_EQ(sizeof(pi_counter_slot_t), 64u);
    EXPECT_EQ(alignof(pi_counter_slot_t), 64u);

    EXPECT_EQ(sizeof(pi_ready_slot_t), 64u);
    EXPECT_EQ(alignof(pi_ready_slot_t), 64u);

    EXPECT_EQ(sizeof(pi_host_control_t), 64u);
    EXPECT_EQ(alignof(pi_host_control_t), 64u);

    EXPECT_EQ(sizeof(pi_failure_t), 16u);
    EXPECT_EQ(alignof(pi_failure_t), 4u);

    EXPECT_EQ(sizeof(pi_consumer_slot_t), 24u);
    EXPECT_EQ(alignof(pi_consumer_slot_t), 8u);

    EXPECT_EQ(sizeof(pi_producer_result_t), 128u);
    EXPECT_EQ(alignof(pi_producer_result_t), 8u);

    EXPECT_EQ(sizeof(pi_run_result_t), 128u);
    EXPECT_EQ(alignof(pi_run_result_t), 64u);

    EXPECT_EQ(sizeof(pi_slot_t), 128u);
    EXPECT_EQ(alignof(pi_slot_t), 64u);
    EXPECT_EQ(sizeof(pi_slot_t), (size_t)PI_SLOT_BYTES);
}

TEST(PacketInjectionAbiRecords, X280ArgsSerializedOffsets) {
    EXPECT_EQ(offsetof(pi_x280_args_t, abi_tag), 0u);
    EXPECT_EQ(offsetof(pi_x280_args_t, worker_count), 4u);
    EXPECT_EQ(offsetof(pi_x280_args_t, tile_count), 6u);
    EXPECT_EQ(offsetof(pi_x280_args_t, S), 8u);
    EXPECT_EQ(offsetof(pi_x280_args_t, B), 12u);
    EXPECT_EQ(offsetof(pi_x280_args_t, C), 16u);
    EXPECT_EQ(offsetof(pi_x280_args_t, K), 20u);
    EXPECT_EQ(offsetof(pi_x280_args_t, poll_cadence), 24u);
    EXPECT_EQ(offsetof(pi_x280_args_t, l1_tile_base), 28u);
    EXPECT_EQ(offsetof(pi_x280_args_t, W), 32u);
    EXPECT_EQ(offsetof(pi_x280_args_t, N), 40u);
    EXPECT_EQ(offsetof(pi_x280_args_t, x280_timeout_ticks), 48u);
    EXPECT_EQ(offsetof(pi_x280_args_t, run_nonce), 56u);
    EXPECT_EQ(offsetof(pi_x280_args_t, release_num_dests), 60u);
    EXPECT_EQ(offsetof(pi_x280_args_t, release_rect), 64u);
}

TEST(PacketInjectionAbiRecords, TensixArgsSerializedOffsets) {
    EXPECT_EQ(offsetof(pi_producer_args_t, abi_tag), 0u);
    EXPECT_EQ(offsetof(pi_producer_args_t, run_nonce), 4u);
    EXPECT_EQ(offsetof(pi_producer_args_t, worker_index), 8u);
    EXPECT_EQ(offsetof(pi_producer_args_t, worker_count), 12u);
    EXPECT_EQ(offsetof(pi_producer_args_t, tile_count), 14u);
    EXPECT_EQ(offsetof(pi_producer_args_t, S), 16u);
    EXPECT_EQ(offsetof(pi_producer_args_t, B), 20u);
    EXPECT_EQ(offsetof(pi_producer_args_t, poll_cadence), 24u);
    EXPECT_EQ(offsetof(pi_producer_args_t, dst_chip), 28u);
    EXPECT_EQ(offsetof(pi_producer_args_t, dst_mesh), 30u);
    EXPECT_EQ(offsetof(pi_producer_args_t, l1_tile_base), 32u);
    EXPECT_EQ(offsetof(pi_producer_args_t, peer), 36u);
    EXPECT_EQ(offsetof(pi_producer_args_t, W), 48u);
    EXPECT_EQ(offsetof(pi_producer_args_t, N), 56u);
    EXPECT_EQ(offsetof(pi_producer_args_t, timeout_ticks), 64u);
}

TEST(PacketInjectionAbiRecords, SlotAndControlSerializedOffsets) {
    EXPECT_EQ(offsetof(pi_counter_slot_t, value), 0u);
    EXPECT_EQ(offsetof(pi_counter_slot_t, pad), 4u);

    EXPECT_EQ(offsetof(pi_ready_slot_t, value), 0u);
    EXPECT_EQ(offsetof(pi_ready_slot_t, pad), 4u);

    EXPECT_EQ(offsetof(pi_host_control_t, collect_ack_run_value), 0u);
    EXPECT_EQ(offsetof(pi_host_control_t, pad), 8u);

    EXPECT_EQ(offsetof(pi_consumer_slot_t, consumed), 0u);
    EXPECT_EQ(offsetof(pi_consumer_slot_t, credited), 4u);
    EXPECT_EQ(offsetof(pi_consumer_slot_t, phase_done), 8u);
    EXPECT_EQ(offsetof(pi_consumer_slot_t, warmup_done), 16u);
}

TEST(PacketInjectionAbiRecords, ResultSerializedOffsets) {
    EXPECT_EQ(offsetof(pi_producer_result_t, abi_tag), 0u);
    EXPECT_EQ(offsetof(pi_producer_result_t, worker_index), 4u);
    EXPECT_EQ(offsetof(pi_producer_result_t, phase), 8u);
    EXPECT_EQ(offsetof(pi_producer_result_t, quiescent), 12u);
    EXPECT_EQ(offsetof(pi_producer_result_t, run_nonce), 16u);
    EXPECT_EQ(offsetof(pi_producer_result_t, reserved_progress), 20u);
    EXPECT_EQ(offsetof(pi_producer_result_t, published_progress), 24u);
    EXPECT_EQ(offsetof(pi_producer_result_t, consumed_progress), 28u);
    EXPECT_EQ(offsetof(pi_producer_result_t, warmup_completed), 32u);
    EXPECT_EQ(offsetof(pi_producer_result_t, measured_completed), 40u);
    EXPECT_EQ(offsetof(pi_producer_result_t, latency_count), 48u);
    EXPECT_EQ(offsetof(pi_producer_result_t, stall_count), 56u);
    EXPECT_EQ(offsetof(pi_producer_result_t, rtt_count), 64u);
    EXPECT_EQ(offsetof(pi_producer_result_t, throughput_start), 72u);
    EXPECT_EQ(offsetof(pi_producer_result_t, throughput_end), 80u);
    EXPECT_EQ(offsetof(pi_producer_result_t, failure), 88u);
    EXPECT_EQ(offsetof(pi_producer_result_t, measured_counter_writes), 120u);

    EXPECT_EQ(offsetof(pi_run_result_t, abi_tag), 0u);
    EXPECT_EQ(offsetof(pi_run_result_t, state), 4u);
    EXPECT_EQ(offsetof(pi_run_result_t, run_value), 8u);
    EXPECT_EQ(offsetof(pi_run_result_t, measured_completed), 16u);
    EXPECT_EQ(offsetof(pi_run_result_t, export_valid_bytes), 24u);
    EXPECT_EQ(offsetof(pi_run_result_t, failure), 32u);
    EXPECT_EQ(offsetof(pi_run_result_t, data_quiescent), 48u);
    EXPECT_EQ(offsetof(pi_run_result_t, windows_cleared), 52u);
    EXPECT_EQ(offsetof(pi_run_result_t, cache_restored), 56u);
    EXPECT_EQ(offsetof(pi_run_result_t, worker_count), 60u);
    EXPECT_EQ(offsetof(pi_run_result_t, aggregate_start_mtime), 64u);
    EXPECT_EQ(offsetof(pi_run_result_t, aggregate_end_mtime), 72u);
    EXPECT_EQ(offsetof(pi_run_result_t, aggregate_start_rdcycle), 80u);
    EXPECT_EQ(offsetof(pi_run_result_t, aggregate_end_rdcycle), 88u);
    EXPECT_EQ(offsetof(pi_run_result_t, empty_visits), 96u);
    EXPECT_EQ(offsetof(pi_run_result_t, released_group_dests), 104u);
    EXPECT_EQ(offsetof(pi_run_result_t, released_endpoints), 108u);
    EXPECT_EQ(offsetof(pi_run_result_t, pad), 112u);
}

TEST(PacketInjectionAbiRecords, EveryTaggedRecordCarriesItsTagFirst) {
    EXPECT_EQ(offsetof(pi_x280_args_t, abi_tag), 0u);
    EXPECT_EQ(offsetof(pi_producer_args_t, abi_tag), 0u);
    EXPECT_EQ(offsetof(pi_producer_result_t, abi_tag), 0u);
    EXPECT_EQ(offsetof(pi_run_result_t, abi_tag), 0u);
}

TEST(PacketInjectionAbiRecords, FixedArenaRecordsEndWhereTileCoordinatesBegin) {
    EXPECT_EQ(sizeof(pi_x280_args_t) + sizeof(pi_host_control_t) + sizeof(pi_run_result_t), 320u);
    EXPECT_EQ(
        CORE_SYNC_SLOT_BYTES + sizeof(pi_consumer_slot_t) + sizeof(pi_counter_slot_t) + sizeof(pi_ready_slot_t), 160u);
    EXPECT_LE(PI_L1_RESULT_OFFSET + sizeof(pi_producer_result_t), (size_t)PI_L1_STAGING_OFFSET);
}

TEST(PacketInjectionAbiRecords, Align64AgreesWithTheCheckedAlignUsedByTheLayout) {
    EXPECT_EQ(pi_align64(0u), 0u);
    EXPECT_EQ(pi_align64(1u), 64u);
    EXPECT_EQ(pi_align64(63u), 64u);
    EXPECT_EQ(pi_align64(64u), 64u);
    EXPECT_EQ(pi_align64(65u), 128u);
    EXPECT_EQ(pi_align64(128u), 128u);

    // pi_compute_layout() rounds through pi_checked_align(); anything else that
    // aligns an ABI span uses pi_align64(), so the two must not drift.
    for (uint64_t x = 0; x <= 320u; ++x) {
        uint64_t checked = 0;
        ASSERT_EQ(pi_checked_align(x, 64u, &checked), 1) << x;
        EXPECT_EQ(pi_align64(x), checked) << x;
        EXPECT_EQ(pi_align64(x) % 64u, 0u) << x;
        EXPECT_GE(pi_align64(x), x) << x;
        EXPECT_LT(pi_align64(x) - x, 64u) << x;
    }
}

TEST(PacketInjectionAbiRecords, CounterDistanceWrapping) {
    EXPECT_EQ(pi_counter_distance(10u, 5u), 5u);
    EXPECT_EQ(pi_counter_distance(5u, 5u), 0u);
    // Wrap across 0
    EXPECT_EQ(pi_counter_distance(2u, 0xFFFFFFFEu), 4u);
    EXPECT_EQ(pi_counter_distance(0u, 0xFFFFFFFFu), 1u);
}

// --------------------------------------------------------------------------
// ABI tag
// --------------------------------------------------------------------------

TEST(PacketInjectionAbiTag, EncodesMagicModeAndCreation) {
    const uint32_t tag = PI_ABI_TAG(PI_MODE_THROUGHPUT, 0u);
    EXPECT_EQ(tag >> 16, (uint32_t)PI_ABI_MAGIC);
    EXPECT_EQ((tag >> 8) & 0xFFu, 0u);
    EXPECT_EQ((tag >> 4) & 0xFu, (uint32_t)PI_MODE_THROUGHPUT);
    EXPECT_EQ(tag & 0xFu, 0u);

    EXPECT_EQ(tag, 0x50490020u);
    EXPECT_EQ(PI_ABI_TAG(PI_MODE_LATENCY, 1u), 0x50490011u);
    EXPECT_EQ(PI_ABI_TAG(PI_MODE_BACKPRESSURE, 0u), 0x50490030u);
}

TEST(PacketInjectionAbiTag, MatchingRejectsWrongModeCreationAndMagic) {
    const uint32_t tag = PI_ABI_TAG(PI_MODE_THROUGHPUT, 0u);

    EXPECT_EQ(pi_abi_tag_matches(tag, PI_MODE_THROUGHPUT, 0u), 1);
    EXPECT_EQ(pi_abi_tag_matches(tag, PI_MODE_LATENCY, 0u), 0);
    EXPECT_EQ(pi_abi_tag_matches(tag, PI_MODE_BACKPRESSURE, 0u), 0);
    EXPECT_EQ(pi_abi_tag_matches(tag, PI_MODE_THROUGHPUT, 1u), 0);
    EXPECT_EQ(pi_abi_tag_matches(PI_ABI_TAG(PI_MODE_THROUGHPUT, 1u), PI_MODE_THROUGHPUT, 0u), 0);

    EXPECT_EQ(pi_abi_tag_matches(tag ^ (1u << 8), PI_MODE_THROUGHPUT, 0u), 0);   // bits [15:8]
    EXPECT_EQ(pi_abi_tag_matches(tag ^ (1u << 16), PI_MODE_THROUGHPUT, 0u), 0);  // magic
    EXPECT_EQ(pi_abi_tag_matches(0u, PI_MODE_THROUGHPUT, 0u), 0);
    EXPECT_EQ(pi_abi_tag_matches(UINT32_MAX, PI_MODE_THROUGHPUT, 0u), 0);
}

TEST(PacketInjectionAbiTag, OutOfRangeReaderStateDoesNotAliasAValidTag) {
    // The encoding masks mode to four bits and include_creation to one, so 18
    // builds the same word as mode 2 and 3 the same word as creation 1.
    EXPECT_EQ(PI_ABI_TAG(18u, 0u), PI_ABI_TAG(PI_MODE_THROUGHPUT, 0u));
    EXPECT_EQ(PI_ABI_TAG(PI_MODE_LATENCY, 3u), PI_ABI_TAG(PI_MODE_LATENCY, 1u));

    EXPECT_EQ(pi_abi_tag_matches(PI_ABI_TAG(PI_MODE_THROUGHPUT, 0u), 18u, 0u), 0);
    EXPECT_EQ(pi_abi_tag_matches(PI_ABI_TAG(PI_MODE_LATENCY, 1u), PI_MODE_LATENCY, 3u), 0);
    EXPECT_EQ(pi_abi_tag_matches(PI_ABI_TAG(PI_MODE_LATENCY, 0u), 0u, 0u), 0);
    EXPECT_EQ(pi_abi_tag_matches(PI_ABI_TAG(PI_MODE_LATENCY, 0u), 4u, 0u), 0);
    EXPECT_EQ(pi_abi_tag_matches(PI_ABI_TAG(PI_MODE_LATENCY, 0u), UINT32_MAX, 0u), 0);
}

// --------------------------------------------------------------------------
// Golden layouts
// --------------------------------------------------------------------------

TEST(PacketInjectionLayout, GoldenP280T140S16B8Throughput) {
    pi_layout_t l{};
    const LayoutCase c{"P=280 T=140 S=16 B=8 throughput", 280, 140, 16, 8, 4096, PI_MODE_THROUGHPUT, 0};
    ASSERT_TRUE(Accepts(c, &l));

    EXPECT_EQ(l.worker_count, 280u);
    EXPECT_EQ(l.tile_count, 140u);
    EXPECT_EQ(l.risc_per_tile, 2u);
    EXPECT_EQ(l.staging_depth, 8u);

    EXPECT_EQ(l.args_offset, 0u);
    EXPECT_EQ(l.host_control_offset, 128u);
    EXPECT_EQ(l.run_result_offset, 192u);
    EXPECT_EQ(l.tile_coords_offset, 320u);
    // 8*140 = 1120 bytes of coordinates, padded to 1152 before the 280
    // core-sync arrival slots.
    EXPECT_EQ(l.sync_arrivals_offset, 1472u);
    EXPECT_EQ(l.consumer_slots_offset, 3712u);
    EXPECT_EQ(l.produced_offset, 10432u);
    EXPECT_EQ(l.ready_offset, 28352u);
    EXPECT_EQ(l.rings_offset, 46272u);
    EXPECT_EQ(l.ring_stride, 2048u);
    EXPECT_EQ(l.rings_bytes, 573440u);
    EXPECT_EQ(l.arena_base_bytes, 619712u);
    EXPECT_EQ(l.arena_base_bytes, (uint64_t)PI_ARENA_BASE_GOLDEN_P280_T140_S16);
    EXPECT_EQ(l.arena_bytes, 619712u);

    EXPECT_EQ(l.l1_tile_base, 0u);
    EXPECT_EQ(l.l1_release_offset, 0u);
    EXPECT_EQ(l.l1_worker_base_offset, 64u);
    EXPECT_EQ(l.l1_worker_stride, 1408u);
    EXPECT_EQ(l.l1_tile_bytes, 2880u);

    // Throughput carries neither the export region nor the sample arrays.
    EXPECT_EQ(l.latency_export_offset, 0u);
    EXPECT_EQ(l.latency_export_bytes, 0u);
    EXPECT_EQ(l.latency_sample_stride, 0u);
    EXPECT_EQ(l.l1_starts_offset, 0u);
    EXPECT_EQ(l.l1_stalls_offset, 0u);
    EXPECT_EQ(l.l1_rtts_offset, 0u);
}

TEST(PacketInjectionLayout, GoldenP1T1S16B8Throughput) {
    pi_layout_t l{};
    const LayoutCase c{"P=1 T=1 S=16 B=8 throughput", 1, 1, 16, 8, 4096, PI_MODE_THROUGHPUT, 0};
    ASSERT_TRUE(Accepts(c, &l));

    EXPECT_EQ(l.worker_count, 1u);
    EXPECT_EQ(l.tile_count, 1u);
    EXPECT_EQ(l.risc_per_tile, 1u);
    EXPECT_EQ(l.staging_depth, 8u);

    // One coordinate is 8 bytes padded to a full 64-byte line, followed by one
    // eight-byte core-sync arrival slot and padding to the consumer array.
    EXPECT_EQ(l.tile_coords_offset, 320u);
    EXPECT_EQ(l.sync_arrivals_offset, 384u);
    EXPECT_EQ(l.consumer_slots_offset, 448u);
    EXPECT_EQ(l.produced_offset, 512u);
    EXPECT_EQ(l.ready_offset, 576u);
    EXPECT_EQ(l.rings_offset, 640u);
    EXPECT_EQ(l.ring_stride, 2048u);
    EXPECT_EQ(l.rings_bytes, 2048u);
    EXPECT_EQ(l.arena_base_bytes, 2688u);
    EXPECT_EQ(l.arena_bytes, 2688u);

    EXPECT_EQ(l.l1_worker_stride, 1408u);
    EXPECT_EQ(l.l1_tile_bytes, 1472u);
}

TEST(PacketInjectionLayout, GoldenP2T1S16B8Throughput) {
    pi_layout_t l{};
    const LayoutCase c{"P=2 T=1 S=16 B=8 throughput", 2, 1, 16, 8, 4096, PI_MODE_THROUGHPUT, 0};
    ASSERT_TRUE(Accepts(c, &l));

    EXPECT_EQ(l.risc_per_tile, 2u);
    EXPECT_EQ(l.sync_arrivals_offset, 384u);
    EXPECT_EQ(l.consumer_slots_offset, 448u);
    EXPECT_EQ(l.produced_offset, 512u);
    EXPECT_EQ(l.ready_offset, 640u);
    EXPECT_EQ(l.rings_offset, 768u);
    EXPECT_EQ(l.arena_base_bytes, 4864u);
    EXPECT_EQ(l.arena_bytes, 4864u);
}

TEST(PacketInjectionLayout, EveryCounterDestinationIsNocWriteCongruent) {
    // The producer publishes produced and ready with a length-carrying NoC
    // write out of its own L1, and Blackhole requires the source and
    // destination to agree modulo NOC_L1_WRITE_ALIGNMENT_BYTES, which is 16
    // (tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-376,
    // and LOG_BASE_2_OF_L1_ALIGNMENT 4 in that architecture's core_config.h).
    // Producer sources are congruent to 0, so every destination must be too.
    // kernels/transport_bandwidth/backends/tensix_to_x280.hpp:123-124 refuses a
    // run on the same rule.
    //
    // pi_compute_layout() also refuses a base that breaks this, so loosening
    // the alignment shows up below as a rejected layout rather than as a
    // congruent-address failure. What this case adds is the rule stated where a
    // reader meets the offsets, and the negative control at the end.
    constexpr uint64_t kNocWriteAlignment = 16u;

    struct Placement {
        uint32_t P;
        uint32_t T;
    };
    const Placement placements[] = {{1, 1}, {2, 1}, {4, 2}, {PI_MAX_WORKERS, PI_MAX_TILES}};

    for (const auto& placement : placements) {
        pi_layout_t l{};
        const LayoutCase c{"congruence", placement.P, placement.T, 16, 8, 4096, PI_MODE_THROUGHPUT, 0};
        ASSERT_TRUE(Accepts(c, &l)) << "P=" << placement.P;

        for (uint32_t w = 0; w < l.worker_count; ++w) {
            const uint64_t produced = l.produced_offset + (uint64_t)w * sizeof(pi_counter_slot_t);
            const uint64_t ready = l.ready_offset + (uint64_t)w * sizeof(pi_ready_slot_t);

            EXPECT_EQ(produced % kNocWriteAlignment, 0u) << "P=" << placement.P << " worker " << w;
            EXPECT_EQ(ready % kNocWriteAlignment, 0u) << "P=" << placement.P << " worker " << w;
        }
    }

    // Negative control: an 8-byte slot stride is what reached silicon, and the
    // P=1 offsets it produced explain the hang exactly. Worker 0's ready write
    // was congruent and landed, so the X280 passed WAIT_WARMUP, while worker
    // 0's produced write was not, so the consumer read zero and timed out.
    constexpr uint64_t kOldProducedOffset = 280u;
    constexpr uint64_t kOldReadyOffset = 288u;
    constexpr uint64_t kOldStride = 8u;
    EXPECT_NE(kOldProducedOffset % kNocWriteAlignment, 0u);
    EXPECT_EQ(kOldReadyOffset % kNocWriteAlignment, 0u);
    EXPECT_NE((kOldReadyOffset + kOldStride) % kNocWriteAlignment, 0u);

    // A 64-byte slot at a 64-aligned base cannot reproduce that for any worker,
    // whatever the modulus in play.
    EXPECT_EQ(sizeof(pi_counter_slot_t) % kNocWriteAlignment, 0u);
    EXPECT_EQ(sizeof(pi_ready_slot_t) % kNocWriteAlignment, 0u);
    EXPECT_EQ(alignof(pi_counter_slot_t) % kNocWriteAlignment, 0u);
    EXPECT_EQ(alignof(pi_ready_slot_t) % kNocWriteAlignment, 0u);
}

TEST(PacketInjectionLayout, GoldenP280LatencyAddsExportAtTheArenaBase) {
    pi_layout_t l{};
    const LayoutCase c{"P=280 T=140 S=16 B=8 N=58 latency", 280, 140, 16, 8, 58, PI_MODE_LATENCY, 0};
    ASSERT_TRUE(Accepts(c, &l));

    EXPECT_EQ(l.arena_base_bytes, (uint64_t)PI_ARENA_BASE_GOLDEN_P280_T140_S16);
    EXPECT_EQ(l.latency_sample_stride, 464u);
    EXPECT_EQ(l.latency_export_offset, 619712u);
    EXPECT_EQ(l.latency_export_bytes, 129920u);
    EXPECT_LE(l.latency_export_bytes, (uint64_t)PI_CACHE_LIMIT_BYTES);
    EXPECT_EQ(l.arena_bytes, 749632u);
    EXPECT_LE(l.arena_bytes, (uint64_t)PI_ARENA_BYTES);

    // The export region is the only mode-specific arena growth.
    EXPECT_EQ(l.l1_worker_stride, 1408u);
    EXPECT_EQ(l.l1_starts_offset, 0u);
}

TEST(PacketInjectionLayout, GoldenP280BackpressureAddsSampleArraysToTheL1Slice) {
    pi_layout_t l{};
    const LayoutCase c{"P=280 T=140 S=16 B=8 N=48 backpressure", 280, 140, 16, 8, 48, PI_MODE_BACKPRESSURE, 0};
    ASSERT_TRUE(Accepts(c, &l));

    EXPECT_EQ(l.arena_base_bytes, (uint64_t)PI_ARENA_BASE_GOLDEN_P280_T140_S16);
    EXPECT_EQ(l.arena_bytes, (uint64_t)PI_ARENA_BASE_GOLDEN_P280_T140_S16);
    EXPECT_EQ(l.latency_export_bytes, 0u);

    // Staging ends at 384 + 128*8 = 1408; starts is align64(16*16), rtts align64(8*48).
    EXPECT_EQ(l.l1_starts_offset, 1408u);
    EXPECT_EQ(l.l1_stalls_offset, 1664u);
    EXPECT_EQ(l.l1_rtts_offset, 2048u);
    EXPECT_EQ(l.l1_worker_stride, 2432u);
    EXPECT_EQ(l.l1_tile_bytes, 4928u);
}

TEST(PacketInjectionLayout, ArenaConstantMatchesTheReservedLimRange) {
    const uint64_t backed_bytes = kRequiredLimBackedBytes;
    const auto arena = x280::available_lim_for_layout(x280::LimLayout::PacketInjection, backed_bytes);

    EXPECT_EQ(arena.size(), (uint64_t)PI_ARENA_BYTES);
    EXPECT_EQ(arena.begin, (uint64_t)X280_LIM_PACKET_INJECTION_ARENA_BASE);
}

// --------------------------------------------------------------------------
// Admission boundaries
// --------------------------------------------------------------------------

TEST(PacketInjectionLayoutRejects, NullOutputIsRefused) {
    EXPECT_EQ(
        pi_compute_layout(280, 140, 16, 8, 4096, PI_MODE_THROUGHPUT, 0, nullptr).code, (uint32_t)PI_LAYOUT_BAD_PARAM);
}

TEST(PacketInjectionLayoutRejects, EveryInvalidParameterLeavesOutputUntouched) {
    const LayoutCase cases[] = {
        {"P=0", 0, 1, 16, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"P>PI_MAX_WORKERS", PI_MAX_WORKERS + 1u, 1, 16, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"T=0", 1, 0, 16, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"T>PI_MAX_TILES",
         PI_MAX_TILES + 1u,
         PI_MAX_TILES + 1u,
         16,
         8,
         1024,
         PI_MODE_THROUGHPUT,
         0,
         PI_LAYOUT_BAD_PARAM},
        {"T does not divide P", 5, 2, 16, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"P/T == 3", 3, 1, 16, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"P/T == 4", 4, 1, 16, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"S=0", 1, 1, 0, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"S not a power of two", 1, 1, 15, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"B=0", 1, 1, 16, 0, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"N=0 latency", 1, 1, 16, 8, 0, PI_MODE_LATENCY, 0, PI_LAYOUT_BAD_PARAM},
        {"N=0 throughput", 1, 1, 16, 8, 0, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_BAD_PARAM},
        {"mode=0", 1, 1, 16, 8, 1024, 0, 0, PI_LAYOUT_BAD_PARAM},
        {"mode=4", 1, 1, 16, 8, 1024, 4, 0, PI_LAYOUT_BAD_PARAM},
        {"mode=UINT32_MAX", 1, 1, 16, 8, 1024, UINT32_MAX, 0, PI_LAYOUT_BAD_PARAM},
        {"l1_tile_base misaligned by 1", 1, 1, 16, 8, 1024, PI_MODE_THROUGHPUT, 1, PI_LAYOUT_ALIGNMENT},
        {"l1_tile_base misaligned by 32", 1, 1, 16, 8, 1024, PI_MODE_THROUGHPUT, 32, PI_LAYOUT_ALIGNMENT},
        // rings_offset + P*S*128 = 11968 + 1146880 exceeds PI_ARENA_BYTES.
        {"arena exceeds PI_ARENA_BYTES", 280, 140, 32, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_LIM_CAPACITY},
        // align64(280 * 8 * 59) = 132160 exceeds PI_CACHE_LIMIT_BYTES.
        {"latency export exceeds PI_CACHE_LIMIT_BYTES",
         280,
         140,
         16,
         8,
         59,
         PI_MODE_LATENCY,
         0,
         PI_LAYOUT_EXPORT_CAPACITY},
        {"8*N overflows u64 in latency", 1, 1, 16, 8, UINT64_MAX, PI_MODE_LATENCY, 0, PI_LAYOUT_OVERFLOW},
        {"8*N align overflows u64 in latency", 1, 1, 16, 8, UINT64_MAX / 8u, PI_MODE_LATENCY, 0, PI_LAYOUT_OVERFLOW},
        {"8*N overflows u64 in backpressure", 1, 1, 16, 8, UINT64_MAX, PI_MODE_BACKPRESSURE, 0, PI_LAYOUT_OVERFLOW},
        {"8*N align overflows u64 in backpressure",
         1,
         1,
         16,
         8,
         UINT64_MAX / 8u,
         PI_MODE_BACKPRESSURE,
         0,
         PI_LAYOUT_OVERFLOW},
        // Two align64(8*N) sample arrays push the L1 slice past 32 bits.
        {"L1 slice leaves the 32-bit space", 1, 1, 16, 8, 1u << 29, PI_MODE_BACKPRESSURE, 0, PI_LAYOUT_L1_CAPACITY},
        {"l1_tile_base + tile bytes leaves the 32-bit space",
         1,
         1,
         16,
         8,
         1024,
         PI_MODE_THROUGHPUT,
         0xFFFFFF00u,
         PI_LAYOUT_L1_CAPACITY},
    };

    for (const auto& c : cases) {
        EXPECT_TRUE(RejectsLeavingOutputUntouched(c));
    }
}

TEST(PacketInjectionLayoutRejects, AcceptsTheValidSideOfEveryBoundary) {
    struct Accepted {
        LayoutCase c;
        uint32_t risc_per_tile;
        uint32_t staging_depth;
    };
    const Accepted cases[] = {
        {{"P=PI_MAX_WORKERS paired", PI_MAX_WORKERS, PI_MAX_TILES, 16, 8, 1024, PI_MODE_THROUGHPUT, 0}, 2, 8},
        {{"P=T=PI_MAX_TILES", PI_MAX_TILES, PI_MAX_TILES, 16, 8, 1024, PI_MODE_THROUGHPUT, 0}, 1, 8},
        {{"S=1", 1, 1, 1, 8, 1024, PI_MODE_THROUGHPUT, 0}, 1, 1},
        {{"B=1", 1, 1, 16, 1, 1024, PI_MODE_THROUGHPUT, 0}, 1, 1},
        {{"B=UINT32_MAX clamps staging to S", 1, 1, 16, UINT32_MAX, 1024, PI_MODE_THROUGHPUT, 0}, 1, 16},
        {{"N=1", 1, 1, 16, 8, 1, PI_MODE_LATENCY, 0}, 1, 8},
        {{"largest latency N at P=280", 280, 140, 16, 8, 58, PI_MODE_LATENCY, 0}, 2, 8},
        // N is not a layout term outside latency and backpressure, so it cannot
        // overflow there.
        {{"N=UINT64_MAX in throughput", 1, 1, 16, 8, UINT64_MAX, PI_MODE_THROUGHPUT, 0}, 1, 8},
        {{"l1_tile_base near the top of the 32-bit space", 1, 1, 16, 8, 1024, PI_MODE_THROUGHPUT, 0xFFFFF000u}, 1, 8},
    };

    for (const auto& a : cases) {
        pi_layout_t l{};
        ASSERT_TRUE(Accepts(a.c, &l));
        EXPECT_EQ(l.risc_per_tile, a.risc_per_tile) << a.c.name;
        EXPECT_EQ(l.staging_depth, a.staging_depth) << a.c.name;
        EXPECT_LE(l.arena_bytes, (uint64_t)PI_ARENA_BYTES) << a.c.name;
        EXPECT_LE(l.latency_export_bytes, (uint64_t)PI_CACHE_LIMIT_BYTES) << a.c.name;
        EXPECT_EQ((uint64_t)l.l1_tile_base + l.l1_tile_bytes <= (uint64_t)UINT32_MAX, true) << a.c.name;
    }
}

TEST(PacketInjectionLayoutRejects, LargeSIsRefusedByTheArenaLimit) {
    // S is a uint32_t, so S*PI_SLOT_BYTES and 8*S cannot overflow 64 bits; the
    // arena limit is what rejects an oversized ring.
    for (uint32_t shift = 20; shift <= 31; ++shift) {
        const LayoutCase c{
            "large power-of-two S", 1, 1, 1u << shift, 8, 1024, PI_MODE_THROUGHPUT, 0, PI_LAYOUT_LIM_CAPACITY};
        EXPECT_TRUE(RejectsLeavingOutputUntouched(c)) << "S = 1 << " << shift;
    }
}

// --------------------------------------------------------------------------
// Derivation agreement between host, X280, and producer
// --------------------------------------------------------------------------

TEST(PacketInjectionLayout, PerWorkerArenaAddressesTileTheirRegions) {
    pi_layout_t l{};
    const LayoutCase c{"P=280 T=140 S=16 B=8 N=58 latency", 280, 140, 16, 8, 58, PI_MODE_LATENCY, 0x1000};
    ASSERT_TRUE(Accepts(c, &l));

    const uint64_t base = (uint64_t)X280_LIM_PACKET_INJECTION_ARENA_BASE;

    for (uint32_t w = 0; w < l.worker_count; ++w) {
        const uint64_t consumer = base + l.consumer_slots_offset + (uint64_t)w * sizeof(pi_consumer_slot_t);
        const uint64_t produced = base + l.produced_offset + (uint64_t)w * sizeof(pi_counter_slot_t);
        const uint64_t ready = base + l.ready_offset + (uint64_t)w * sizeof(pi_ready_slot_t);
        const uint64_t ring = base + l.rings_offset + (uint64_t)w * l.ring_stride;
        const uint64_t sample = base + l.latency_export_offset + (uint64_t)w * l.latency_sample_stride;

        EXPECT_EQ(consumer % alignof(pi_consumer_slot_t), 0u) << "worker " << w;
        EXPECT_EQ(produced % alignof(pi_counter_slot_t), 0u) << "worker " << w;
        EXPECT_EQ(ready % alignof(pi_ready_slot_t), 0u) << "worker " << w;
        EXPECT_EQ(ring % 64u, 0u) << "worker " << w;
        EXPECT_EQ(sample % 8u, 0u) << "worker " << w;

        // Each element lies wholly inside its own region.
        EXPECT_LE(consumer + sizeof(pi_consumer_slot_t), base + l.produced_offset) << "worker " << w;
        EXPECT_LE(produced + sizeof(pi_counter_slot_t), base + l.ready_offset) << "worker " << w;
        EXPECT_LE(ready + sizeof(pi_ready_slot_t), base + l.rings_offset) << "worker " << w;
        EXPECT_LE(ring + l.ring_stride, base + l.arena_base_bytes) << "worker " << w;
        EXPECT_LE(sample + l.latency_sample_stride, base + l.arena_bytes) << "worker " << w;
    }

    // The per-worker arrays tile their regions exactly.
    // The 24-byte consumer slots are packed, then padded up so the cache-line
    // counter slots start aligned; at P=280 that padding happens to be zero.
    EXPECT_GE(l.produced_offset, l.consumer_slots_offset + 280u * sizeof(pi_consumer_slot_t));
    EXPECT_LT(l.produced_offset - (l.consumer_slots_offset + 280u * sizeof(pi_consumer_slot_t)), 64u);
    EXPECT_EQ(l.produced_offset % 64u, 0u);
    EXPECT_EQ(l.produced_offset + 280u * sizeof(pi_counter_slot_t), l.ready_offset);
    EXPECT_EQ(l.ready_offset % 64u, 0u);
    EXPECT_EQ(l.ready_offset + 280u * sizeof(pi_ready_slot_t), 46272u);
    EXPECT_LE(l.ready_offset + 280u * sizeof(pi_ready_slot_t), l.rings_offset);
    EXPECT_EQ(l.rings_offset + l.rings_bytes, l.arena_base_bytes);
    EXPECT_EQ(l.latency_export_offset + 280u * l.latency_sample_stride, 749632u);
    EXPECT_LE(l.latency_export_offset + 280u * l.latency_sample_stride, l.arena_bytes);

    // The tile-coordinate array and the three fixed records do not overlap it.
    EXPECT_EQ(l.args_offset + sizeof(pi_x280_args_t), l.host_control_offset);
    EXPECT_EQ(l.host_control_offset + sizeof(pi_host_control_t), l.run_result_offset);
    EXPECT_EQ(l.run_result_offset + sizeof(pi_run_result_t), l.tile_coords_offset);
    EXPECT_LE(l.tile_coords_offset + 140u * sizeof(x280_noc_coord_t), l.sync_arrivals_offset);
    EXPECT_LE(l.sync_arrivals_offset + 280u * CORE_SYNC_SLOT_BYTES, l.consumer_slots_offset);

    // The host writes the coordinate table with write_available_lim(), which
    // takes whole cache lines only, so its span is padded to 64 and the records
    // after it stay 64-aligned. The gap is padding, never a third record.
    EXPECT_EQ(l.tile_coords_offset % 64u, 0u);
    EXPECT_EQ(l.consumer_slots_offset % 64u, 0u);
    EXPECT_LT(l.sync_arrivals_offset - (l.tile_coords_offset + 140u * sizeof(x280_noc_coord_t)), 64u);
    EXPECT_LT(l.consumer_slots_offset - (l.sync_arrivals_offset + 280u * CORE_SYNC_SLOT_BYTES), 64u);
}

TEST(PacketInjectionLayout, PerWorkerL1SlicesTileTheTileAllocation) {
    pi_layout_t l{};
    const LayoutCase c{"P=280 T=140 S=16 B=8 N=48 backpressure", 280, 140, 16, 8, 48, PI_MODE_BACKPRESSURE, 0x1000};
    ASSERT_TRUE(Accepts(c, &l));

    EXPECT_EQ(l.l1_worker_base_offset + l.risc_per_tile * l.l1_worker_stride, l.l1_tile_bytes);

    for (uint32_t risc = 0; risc < l.risc_per_tile; ++risc) {
        const uint32_t slice = l.l1_tile_base + l.l1_worker_base_offset + risc * l.l1_worker_stride;
        EXPECT_EQ(slice % 64u, 0u) << "risc " << risc;
        EXPECT_GE(slice, l.l1_tile_base) << "risc " << risc;
        EXPECT_LE(slice + l.l1_worker_stride, l.l1_tile_base + l.l1_tile_bytes) << "risc " << risc;

        // The fixed records, staging, and the sample arrays all fit the slice.
        EXPECT_LE(PI_L1_STAGING_OFFSET + PI_SLOT_BYTES * l.staging_depth, l.l1_worker_stride);
        EXPECT_GE(l.l1_starts_offset, PI_L1_STAGING_OFFSET + PI_SLOT_BYTES * l.staging_depth);
        EXPECT_LT(l.l1_starts_offset, l.l1_stalls_offset);
        EXPECT_LT(l.l1_stalls_offset, l.l1_rtts_offset);
        EXPECT_LE(l.l1_rtts_offset + sizeof(pi_local_sample_t) * 48u, l.l1_worker_stride);
    }

    // Host, producer, and X280 all turn a worker index into a tile, a RISC, and
    // an L1 slice through these three accessors, so they are the agreement.
    std::vector<uint32_t> seen(l.tile_count * l.risc_per_tile, 0u);
    for (uint32_t w = 0; w < l.worker_count; ++w) {
        const uint32_t tile = pi_worker_tile_index(&l, w);
        const uint32_t risc = pi_worker_risc_index(&l, w);
        const uint32_t slice = pi_worker_l1_base(&l, w);

        EXPECT_EQ(tile, w / l.risc_per_tile) << "worker " << w;
        EXPECT_EQ(risc, w % l.risc_per_tile) << "worker " << w;
        EXPECT_EQ(slice, l.l1_tile_base + l.l1_worker_base_offset + risc * l.l1_worker_stride) << "worker " << w;
        ASSERT_LT(tile, l.tile_count) << "worker " << w;
        ASSERT_LT(risc, l.risc_per_tile) << "worker " << w;
        EXPECT_LE(slice + l.l1_worker_stride, l.l1_tile_base + l.l1_tile_bytes) << "worker " << w;

        ++seen[tile * l.risc_per_tile + risc];
    }
    for (size_t i = 0; i < seen.size(); ++i) {
        EXPECT_EQ(seen[i], 1u) << "slot " << i;
    }

    // Paired placement gives even workers to BRISC and odd ones to NCRISC,
    // which is the claim the producer checks its own image against.
    EXPECT_EQ(pi_worker_risc_index(&l, 0u), 0u);
    EXPECT_EQ(pi_worker_risc_index(&l, 1u), 1u);
    EXPECT_EQ(pi_worker_tile_index(&l, 0u), pi_worker_tile_index(&l, 1u));
    EXPECT_NE(pi_worker_tile_index(&l, 1u), pi_worker_tile_index(&l, 2u));
}

TEST(PacketInjectionLayout, UnpairedPlacementPutsEveryWorkerOnItsOwnTileAtSliceZero) {
    pi_layout_t l{};
    const LayoutCase c{"P=140 T=140 S=16 B=8 throughput", 140, 140, 16, 8, 1024, PI_MODE_THROUGHPUT, 0x1000};
    ASSERT_TRUE(Accepts(c, &l));

    ASSERT_EQ(l.risc_per_tile, 1u);
    EXPECT_EQ(l.l1_tile_bytes, l.l1_worker_base_offset + l.l1_worker_stride);
    for (uint32_t w = 0; w < l.worker_count; ++w) {
        EXPECT_EQ(pi_worker_tile_index(&l, w), w) << "worker " << w;
        EXPECT_EQ(pi_worker_risc_index(&l, w), 0u) << "worker " << w;
        EXPECT_EQ(pi_worker_l1_base(&l, w), l.l1_tile_base + l.l1_worker_base_offset) << "worker " << w;
    }
}

TEST(PacketInjectionLayout, FixedL1RecordOffsetsAreDisjointCacheLines) {
    EXPECT_EQ(PI_L1_CONSUMED_OFFSET, 0u);
    EXPECT_EQ(PI_L1_CONTROL_OFFSET, 64u);
    EXPECT_EQ(PI_L1_PRODUCED_SRC_OFFSET, 128u);
    EXPECT_EQ(PI_L1_READY_SRC_OFFSET, 192u);
    EXPECT_EQ(PI_L1_RESULT_OFFSET, 256u);
    EXPECT_EQ(PI_L1_STAGING_OFFSET, 384u);

    const uint32_t offsets[] = {
        PI_L1_CONSUMED_OFFSET,
        PI_L1_CONTROL_OFFSET,
        PI_L1_PRODUCED_SRC_OFFSET,
        PI_L1_READY_SRC_OFFSET,
        PI_L1_RESULT_OFFSET,
        PI_L1_STAGING_OFFSET};
    for (const uint32_t offset : offsets) {
        EXPECT_EQ(offset % 64u, 0u) << offset;
    }
}

// --------------------------------------------------------------------------
// Runtime-argument packing
// --------------------------------------------------------------------------

TEST(PacketInjectionArgPacking, ProducerArgWordsRoundTripAtEdgeValues) {
    pi_producer_args_t edge{};
    edge.abi_tag = UINT32_MAX;
    edge.run_nonce = UINT32_MAX;
    edge.worker_index = 0u;
    edge.worker_count = UINT16_MAX;
    edge.tile_count = UINT16_MAX;
    edge.S = UINT32_MAX;
    edge.B = UINT32_MAX;
    edge.poll_cadence = 0u;
    edge.dst_chip = UINT16_MAX;
    edge.dst_mesh = 0u;
    edge.peer = {UINT32_MAX, 0u};
    edge.pad = UINT32_MAX;
    edge.l1_tile_base = UINT32_MAX;
    edge.W = UINT64_MAX;
    edge.N = UINT64_MAX;
    edge.timeout_ticks = 0u;

    // The union's overlay IS the serialization: the host ships block.words and
    // the kernel reads straight back into them, so there is no copy helper to
    // keep in step with the field offsets.
    pi_producer_arg_words_t block;
    std::memset(&block, 0x5A, sizeof(block));
    block.args = edge;

    pi_producer_arg_words_t received;
    std::memset(&received, 0xC3, sizeof(received));
    for (size_t i = 0; i < PI_PRODUCER_ARG_WORDS; ++i) {
        received.words[i] = block.words[i];
    }

    const pi_producer_args_t& back = received.args;
    EXPECT_EQ(std::memcmp(&back, &edge, sizeof(edge)), 0);
    EXPECT_EQ(back.abi_tag, edge.abi_tag);
    EXPECT_EQ(back.run_nonce, edge.run_nonce);
    EXPECT_EQ(back.worker_index, edge.worker_index);
    EXPECT_EQ(back.worker_count, edge.worker_count);
    EXPECT_EQ(back.tile_count, edge.tile_count);
    EXPECT_EQ(back.S, edge.S);
    EXPECT_EQ(back.B, edge.B);
    EXPECT_EQ(back.poll_cadence, edge.poll_cadence);
    EXPECT_EQ(back.dst_chip, edge.dst_chip);
    EXPECT_EQ(back.dst_mesh, edge.dst_mesh);
    EXPECT_EQ(back.peer.x, edge.peer.x);
    EXPECT_EQ(back.peer.y, edge.peer.y);
    EXPECT_EQ(back.l1_tile_base, edge.l1_tile_base);
    EXPECT_EQ(back.W, edge.W);
    EXPECT_EQ(back.N, edge.N);
    EXPECT_EQ(back.timeout_ticks, edge.timeout_ticks);
}

TEST(PacketInjectionArgPacking, RoundTripIsIdentityForAllZeroAndAllOnes) {
    for (const uint8_t fill : {(uint8_t)0x00, (uint8_t)0xFF}) {
        SCOPED_TRACE((int)fill);
        pi_producer_args_t source;
        std::memset(&source, fill, sizeof(source));

        pi_producer_arg_words_t block;
        std::memset(&block, (uint8_t)~fill, sizeof(block));
        block.args = source;

        pi_producer_arg_words_t received;
        std::memset(&received, (uint8_t)~fill, sizeof(received));
        for (size_t i = 0; i < PI_PRODUCER_ARG_WORDS; ++i) {
            received.words[i] = block.words[i];
        }

        EXPECT_EQ(std::memcmp(&received.args, &source, sizeof(source)), 0);
    }
}

TEST(PacketInjectionArgPacking, EachFieldLandsInItsDocumentedRuntimeWord) {
    // The word order is the launch's wire format -- the host ships words 0..15
    // as runtime arguments and the kernel reads them back in that order -- and
    // the union is all that ties it to the field offsets. Little-endian, as
    // both the host and every RISC-V target here are.
    pi_producer_args_t a{};
    a.abi_tag = 0x11111111u;
    a.run_nonce = 0x22222222u;
    a.worker_index = 0x33333333u;
    a.worker_count = 0x4444u;
    a.tile_count = 0x5555u;
    a.S = 0x66666666u;
    a.B = 0x77777777u;
    a.poll_cadence = 0x88888888u;
    a.dst_chip = 0x9999u;
    a.dst_mesh = 0xAAAAu;
    a.l1_tile_base = 0xDDDDDDDDu;
    a.peer = {0xBBBBBBBBu, 0xCCCCCCCCu};
    a.pad = 0x12121212u;
    a.W = 0x0123456789ABCDEFULL;
    a.N = 0xFEDCBA9876543210ULL;
    a.timeout_ticks = 0x0F1E2D3C4B5A6978ULL;

    pi_producer_arg_words_t block{};
    block.args = a;

    EXPECT_EQ(block.words[0], a.abi_tag);
    EXPECT_EQ(block.words[1], a.run_nonce);
    EXPECT_EQ(block.words[2], a.worker_index);
    EXPECT_EQ(block.words[3], (uint32_t)a.worker_count | ((uint32_t)a.tile_count << 16));
    EXPECT_EQ(block.words[4], a.S);
    EXPECT_EQ(block.words[5], a.B);
    EXPECT_EQ(block.words[6], a.poll_cadence);
    EXPECT_EQ(block.words[7], (uint32_t)a.dst_chip | ((uint32_t)a.dst_mesh << 16));
    EXPECT_EQ(block.words[8], a.l1_tile_base);
    EXPECT_EQ(block.words[9], a.peer.x);
    EXPECT_EQ(block.words[10], a.peer.y);
    EXPECT_EQ(block.words[11], a.pad);
    EXPECT_EQ(block.words[12], (uint32_t)(a.W & 0xFFFFFFFFu));
    EXPECT_EQ(block.words[13], (uint32_t)(a.W >> 32));
    EXPECT_EQ(block.words[14], (uint32_t)(a.N & 0xFFFFFFFFu));
    EXPECT_EQ(block.words[15], (uint32_t)(a.N >> 32));
    EXPECT_EQ(block.words[16], (uint32_t)(a.timeout_ticks & 0xFFFFFFFFu));
    EXPECT_EQ(block.words[17], (uint32_t)(a.timeout_ticks >> 32));
}

TEST(PacketInjectionArgPacking, RealisticLaunchRoundTripsThroughTheWordBlock) {
    pi_layout_t l{};
    const LayoutCase c{"P=280 T=140 S=16 B=8 N=4096 throughput", 280, 140, 16, 8, 4096, PI_MODE_THROUGHPUT, 0x1000};
    ASSERT_TRUE(Accepts(c, &l));

    pi_producer_args_t source{};
    source.abi_tag = PI_ABI_TAG(PI_MODE_THROUGHPUT, 0u);
    source.run_nonce = 0x1234ABCDu;
    source.worker_index = 279u;
    source.worker_count = (uint16_t)l.worker_count;
    source.tile_count = (uint16_t)l.tile_count;
    source.S = 16u;
    source.B = 8u;
    source.poll_cadence = PI_DEFAULT_POLL_CADENCE;
    source.dst_chip = 12u;
    source.dst_mesh = 34u;
    source.peer = {8u, 3u};
    source.l1_tile_base = l.l1_tile_base;
    source.W = 1000u;
    source.N = 4096u;
    source.timeout_ticks = 15000000ULL;

    // Exactly what the suite ships and the kernel reconstructs.
    pi_producer_arg_words_t block{};
    block.args = source;
    const std::vector<uint32_t> rt_args(block.words, block.words + PI_PRODUCER_ARG_WORDS);
    ASSERT_EQ(rt_args.size(), 18u);

    pi_producer_arg_words_t received{};
    for (size_t i = 0; i < rt_args.size(); ++i) {
        received.words[i] = rt_args[i];
    }

    EXPECT_EQ(std::memcmp(&received.args, &source, sizeof(source)), 0);
    EXPECT_EQ(pi_abi_tag_matches(received.args.abi_tag, PI_MODE_THROUGHPUT, 0u), 1);
    EXPECT_EQ(pi_abi_tag_matches(received.args.abi_tag, PI_MODE_LATENCY, 0u), 0);
    EXPECT_EQ(received.args.worker_index, 279u);
    EXPECT_EQ(received.args.timeout_ticks, 15000000ULL);
}

// --------------------------------------------------------------------------
// Failure records
// --------------------------------------------------------------------------

TEST(PacketInjectionFailure, AllZeroIsTheNoFailureEncoding) {
    pi_failure_t none{};
    uint8_t zero[sizeof(pi_failure_t)];
    std::memset(zero, 0, sizeof(zero));

    EXPECT_EQ(std::memcmp(&none, zero, sizeof(none)), 0);
    EXPECT_EQ(none.code, (uint32_t)PI_ERROR_NONE);
    EXPECT_EQ(none.phase, (uint32_t)PI_PHASE_INIT);
    EXPECT_EQ(none.worker_index, 0u);
    EXPECT_EQ(none.actor, (uint32_t)PI_ACTOR_HOST);
}

TEST(PacketInjectionFailure, EncodesCodePhaseWorkerAndActorInFourWords) {
    pi_failure_t f{};
    f.code = PI_ERROR_TIMER_RETRY_EXHAUSTED;
    f.phase = PI_PHASE_MEASURE;
    f.worker_index = 279u;
    f.actor = PI_ACTOR_NCRISC;

    const auto* words = reinterpret_cast<const uint32_t*>(&f);
    EXPECT_EQ(words[0], 9u);
    EXPECT_EQ(words[1], 4u);
    EXPECT_EQ(words[2], 279u);
    EXPECT_EQ(words[3], 3u);

    // UINT32_MAX marks a failure that is not worker-specific.
    f.worker_index = UINT32_MAX;
    EXPECT_EQ(words[2], UINT32_MAX);
}

TEST(PacketInjectionFailure, EmbedsIdenticallyInBothResultRecords) {
    pi_failure_t f{};
    f.code = PI_ERROR_PEER_FAILED;
    f.phase = PI_PHASE_DRAIN;
    f.worker_index = 7u;
    f.actor = PI_ACTOR_X280;

    pi_producer_result_t producer{};
    producer.failure = f;
    EXPECT_EQ(std::memcmp(reinterpret_cast<const uint8_t*>(&producer) + 88, &f, sizeof(f)), 0);

    pi_run_result_t run{};
    run.failure = f;
    EXPECT_EQ(std::memcmp(reinterpret_cast<const uint8_t*>(&run) + 32, &f, sizeof(f)), 0);

    EXPECT_EQ(std::memcmp(&producer.failure, &run.failure, sizeof(f)), 0);
}

// --------------------------------------------------------------------------
// Producer and consumer progress cross-check
// --------------------------------------------------------------------------

TEST(PacketInjectionProgress, AgreeingWitnessesPassAndEveryMismatchIsDetected) {
    pi_consumer_slot_t consumer{};
    consumer.consumed = 4096u;
    consumer.credited = 4096u;
    consumer.phase_done = 4096u;
    consumer.warmup_done = 1000u;

    pi_producer_result_t producer{};
    producer.published_progress = 4096u;
    producer.consumed_progress = 4096u;
    producer.measured_completed = 4096u;
    producer.warmup_completed = 1000u;

    EXPECT_TRUE(ProgressWitnessesAgree(consumer, producer));

    {
        pi_producer_result_t behind = producer;
        behind.published_progress = 4095u;
        EXPECT_FALSE(ProgressWitnessesAgree(consumer, behind));
    }
    {
        pi_producer_result_t ahead = producer;
        ahead.published_progress = 4097u;
        EXPECT_FALSE(ProgressWitnessesAgree(consumer, ahead));
    }
    {
        pi_producer_result_t credits = producer;
        credits.consumed_progress = 4095u;
        EXPECT_FALSE(ProgressWitnessesAgree(consumer, credits));
    }
    {
        pi_producer_result_t completions = producer;
        completions.measured_completed = 4095u;
        EXPECT_FALSE(ProgressWitnessesAgree(consumer, completions));
    }
}

TEST(PacketInjectionProgress, CrossCheckIsWrapSafe) {
    pi_consumer_slot_t consumer{};
    consumer.consumed = 2u;
    consumer.credited = 2u;
    consumer.phase_done = 0x100000002ULL;

    pi_producer_result_t producer{};
    producer.published_progress = 2u;
    producer.consumed_progress = 2u;
    producer.measured_completed = 0x100000002ULL;

    // Both counters have wrapped past 2^32 by the same amount.
    EXPECT_TRUE(ProgressWitnessesAgree(consumer, producer));
    EXPECT_EQ(pi_counter_distance(consumer.consumed, 0xFFFFFFFEu), 4u);

    producer.published_progress = 0xFFFFFFFEu;
    EXPECT_FALSE(ProgressWitnessesAgree(consumer, producer));
    EXPECT_EQ(pi_counter_distance(consumer.consumed, producer.published_progress), 4u);
}

TEST(PacketInjectionProgress, TheTwoWitnessesAreSeparateStorage) {
    pi_consumer_slot_t consumer{};
    pi_producer_result_t producer{};

    consumer.consumed = 0xAAAAAAAAu;
    consumer.credited = 0xBBBBBBBBu;
    consumer.phase_done = 0xCCCCCCCCCCCCCCCCULL;
    producer.published_progress = 0x11111111u;
    producer.consumed_progress = 0x22222222u;
    producer.measured_completed = 0x3333333333333333ULL;

    // Writing either witness leaves the other's evidence intact, so a run that
    // agrees did so from two records rather than one shared field.
    EXPECT_EQ(consumer.consumed, 0xAAAAAAAAu);
    EXPECT_EQ(consumer.credited, 0xBBBBBBBBu);
    EXPECT_EQ(consumer.phase_done, 0xCCCCCCCCCCCCCCCCULL);
    EXPECT_EQ(producer.published_progress, 0x11111111u);
    EXPECT_EQ(producer.consumed_progress, 0x22222222u);
    EXPECT_EQ(producer.measured_completed, 0x3333333333333333ULL);
    EXPECT_FALSE(ProgressWitnessesAgree(consumer, producer));

    // The X280 witness lives in the arena and the producer witness in Tensix L1.
    pi_layout_t l{};
    const LayoutCase c{"P=2 T=1 S=16 B=8 throughput", 2, 1, 16, 8, 1024, PI_MODE_THROUGHPUT, 0};
    ASSERT_TRUE(Accepts(c, &l));
    EXPECT_GT(l.consumer_slots_offset, 0u);
    EXPECT_EQ(PI_L1_RESULT_OFFSET, 256u);
}

// --------------------------------------------------------------------------
// Per-worker throughput
// --------------------------------------------------------------------------

TEST(PacketInjectionThroughput, PerWorkerRateComesFromThatWorkersRawEndpoints) {
    using x280::harness::output::throughput_json;

    pi_producer_result_t w0{};
    w0.throughput_start = 0u;
    w0.throughput_end = 2000000000ULL;
    w0.measured_completed = 4000u;
    const uint64_t w0_elapsed = w0.throughput_end - w0.throughput_start;
    ASSERT_EQ(w0_elapsed, 2000000000ULL);
    const auto j0 = throughput_json(
        w0.throughput_start, w0.throughput_end, w0.measured_completed, true, kUsefulHeaderBytes, PI_SLOT_BYTES);
    EXPECT_NE(j0.find("\"start_ticks\":0"), std::string::npos) << j0;
    EXPECT_NE(j0.find("\"end_ticks\":2000000000"), std::string::npos) << j0;
    EXPECT_NE(j0.find("\"useful_bytes_per_request\":" + std::to_string(kUsefulHeaderBytes)), std::string::npos) << j0;

    // Worker 1: the same request count over twice the interval halves the rate,
    // so no worker's endpoints reach another worker's number.
    pi_producer_result_t w1{};
    w1.throughput_start = 1000000000ULL;
    w1.throughput_end = 5000000000ULL;
    w1.measured_completed = 4000u;
    const auto j1 = throughput_json(
        w1.throughput_start, w1.throughput_end, w1.measured_completed, true, kUsefulHeaderBytes, PI_SLOT_BYTES);
    EXPECT_NE(j1.find("\"start_ticks\":1000000000"), std::string::npos) << j1;
    EXPECT_NE(j1.find("\"end_ticks\":5000000000"), std::string::npos) << j1;
    EXPECT_EQ(j1.find("requests_per_second"), std::string::npos) << j1;
}

TEST(PacketInjectionThroughput, BandwidthScalesWithTheCallerSuppliedByteCounts) {
    using x280::harness::output::throughput_json;
    const auto raw = throughput_json(0, 2000000000ULL, 4000, true, kUsefulHeaderBytes, PI_SLOT_BYTES);
    EXPECT_NE(raw.find("\"useful_bytes_per_request\":" + std::to_string(kUsefulHeaderBytes)), std::string::npos) << raw;
    EXPECT_NE(raw.find("\"slot_bytes_per_request\":" + std::to_string(PI_SLOT_BYTES)), std::string::npos) << raw;
    EXPECT_EQ(raw.find("gb_s"), std::string::npos) << raw;
}

TEST(PacketInjectionThroughput, IntervalValidityIsOneSharedPredicate) {
    using x280::harness::output::throughput_interval_valid;
    using x280::harness::output::throughput_json;

    EXPECT_FALSE(throughput_interval_valid(10, 10));
    EXPECT_TRUE(throughput_interval_valid(10, 20));
    // A wrapped pair of endpoints is still a usable interval.
    EXPECT_TRUE(throughput_interval_valid(UINT64_MAX - 5, 4));
    // A backwards pair reads as most of the range, which is not.
    EXPECT_FALSE(throughput_interval_valid(20, 10));
    EXPECT_TRUE(throughput_interval_valid(0, (1ULL << 63) - 1u));
    EXPECT_FALSE(throughput_interval_valid(0, 1ULL << 63));

    // The renderer classifies "invalid_interval" with the same predicate, so a
    // run failed on the interval and a rate refused for it cannot disagree.
    for (const auto& pair : std::vector<std::pair<uint64_t, uint64_t>>{
             {10, 10}, {20, 10}, {0, 1ULL << 63}, {10, 20}, {UINT64_MAX - 5, 4}}) {
        const auto json = throughput_json(pair.first, pair.second, 4000, true, kUsefulHeaderBytes, PI_SLOT_BYTES);
        const bool rendered_valid = json.find("\"interval_error\":null") != std::string::npos;
        EXPECT_EQ(rendered_valid, throughput_interval_valid(pair.first, pair.second)) << json;
    }
}

TEST(PacketInjectionThroughput, AggregateIntervalRequiresBothPairsForward) {
    using x280::harness::regions::packet_injection::aggregate_interval_valid;

    // Endpoints from a silicon brisc.T1 run: the counters differ in magnitude,
    // so a start taken from one pair and an end from the other reads backwards.
    pi_run_result_t valid{};
    valid.aggregate_start_mtime = 3233413152ULL;
    valid.aggregate_end_mtime = 3233449089ULL;
    valid.aggregate_start_rdcycle = 45125544170ULL;
    valid.aggregate_end_rdcycle = 45126801895ULL;
    EXPECT_TRUE(aggregate_interval_valid(valid));

    pi_run_result_t mtime_inverted = valid;
    mtime_inverted.aggregate_start_mtime = valid.aggregate_end_mtime;
    mtime_inverted.aggregate_end_mtime = valid.aggregate_start_mtime;
    EXPECT_FALSE(aggregate_interval_valid(mtime_inverted));

    pi_run_result_t rdcycle_inverted = valid;
    rdcycle_inverted.aggregate_start_rdcycle = valid.aggregate_end_rdcycle;
    rdcycle_inverted.aggregate_end_rdcycle = valid.aggregate_start_rdcycle;
    EXPECT_FALSE(aggregate_interval_valid(rdcycle_inverted));

    pi_run_result_t mtime_unwritten = valid;
    mtime_unwritten.aggregate_start_mtime = 0;
    mtime_unwritten.aggregate_end_mtime = 0;
    EXPECT_FALSE(aggregate_interval_valid(mtime_unwritten));

    pi_run_result_t rdcycle_unwritten = valid;
    rdcycle_unwritten.aggregate_start_rdcycle = 0;
    rdcycle_unwritten.aggregate_end_rdcycle = 0;
    EXPECT_FALSE(aggregate_interval_valid(rdcycle_unwritten));
}

TEST(PacketInjectionArgs, X280ArgsCarryTheSelectedReleaseGroup) {
    // A builder that dropped the selector's group would release a rectangle by
    // unicast; one that invented a group would broadcast to tiles that drop it.
    using x280::harness::regions::ReleaseGroup;
    using x280::harness::regions::packet_injection::Config;
    using x280::harness::regions::packet_injection::make_x280_args;
    using x280::testing::TranslatedCoordFactory;
    const Config cfg{};
    const pi_layout_t layout{};
    const pi_x280_args_t none = make_x280_args(cfg, 4u, 4u, layout, 0u, 1u, 1u, std::nullopt);
    EXPECT_EQ(none.release_num_dests, 0u);
    EXPECT_EQ(none.release_rect.end.x, 0u);

    const ReleaseGroup group{TranslatedCoordFactory::make(1u, 2u), TranslatedCoordFactory::make(10u, 3u), 4u};
    const pi_x280_args_t args = make_x280_args(cfg, 4u, 4u, layout, 0u, 1u, 1u, group);
    EXPECT_EQ(args.release_num_dests, 4u);
    EXPECT_EQ(args.release_rect.start.x, 1u);
    EXPECT_EQ(args.release_rect.start.y, 2u);
    EXPECT_EQ(args.release_rect.end.x, 10u);
    EXPECT_EQ(args.release_rect.end.y, 3u);
}

TEST(PacketInjectionThroughput, RejectsZeroAndHalfRangeInvalidIntervals) {
    using x280::harness::output::throughput_json;
    const auto zero = throughput_json(7000, 7000, 4000, true, kUsefulHeaderBytes, PI_SLOT_BYTES);
    EXPECT_NE(zero.find("\"interval_error\":\"invalid_interval\""), std::string::npos) << zero;

    // A backwards pair wraps to at least half the u64 range.
    const auto backwards = throughput_json(7000, 6000, 4000, true, kUsefulHeaderBytes, PI_SLOT_BYTES);
    EXPECT_NE(backwards.find("\"interval_error\":\"invalid_interval\""), std::string::npos) << backwards;

    const auto half_range = throughput_json(0, 1ULL << 63, 4000, true, kUsefulHeaderBytes, PI_SLOT_BYTES);
    EXPECT_NE(half_range.find("\"interval_error\":\"invalid_interval\""), std::string::npos) << half_range;

    // One tick below the half-range bound is still a usable interval.
    const auto largest_valid = throughput_json(0, (1ULL << 63) - 1u, 4000, true, kUsefulHeaderBytes, PI_SLOT_BYTES);
    EXPECT_NE(largest_valid.find("\"interval_error\":null"), std::string::npos) << largest_valid;

    // Both endpoints are retained even when the rate is refused.
    EXPECT_NE(zero.find("\"start_ticks\":7000"), std::string::npos) << zero;
    EXPECT_NE(zero.find("\"end_ticks\":7000"), std::string::npos) << zero;
    EXPECT_NE(backwards.find("\"start_ticks\":7000"), std::string::npos) << backwards;
    EXPECT_NE(backwards.find("\"end_ticks\":6000"), std::string::npos) << backwards;
}

// --------------------------------------------------------------------------
// Host admission policy that is not layout arithmetic
// --------------------------------------------------------------------------

namespace {

// A capacity with every reading populated, as admission requires: a full LIM
// arena, a whole Tensix L1, and both clocks. Zeroes are rejected, so a test
// that wants an admitted Config has to supply real values.
Capacity FullCapacity() {
    Capacity cap{};
    cap.lim_backed_bytes = kRequiredLimBackedBytes;
    cap.l1_span_limit = 1499136u;
    cap.l1_window_base = 98304u;
    cap.producer_timer_hz = 1350000000ULL;
    cap.mtime_hz = x280::harness::measurement::kX280MtimeHz;
    return cap;
}

}  // namespace

TEST(PacketInjectionSamplePrefix, AnUncreditedSampleSurvivesTheRunThatFailedBeforeItsCredit) {
    // The case the prefix exists for. X280 services a request, writes its
    // sample and counts it in phase_done; the run then fails before the
    // producer observes the credit, so latency_count is still 0. Truncating by
    // latency_count would discard a sample that exists and is valid, and on the
    // failures seen on silicon it is the only measurement a run produces.
    pi_layout_t l{};
    const LayoutCase c{"P=2 T=2 S=16 B=8 N=4 latency", 2, 2, 16, 8, 4, PI_MODE_LATENCY, 0};
    ASSERT_TRUE(Accepts(c, &l));
    ASSERT_EQ(l.latency_sample_stride, 32u);  // 8 * N

    pi_consumer_slot_t slot{};
    slot.phase_done = 1;
    pi_producer_result_t result{};
    result.latency_count = 0;

    const SamplePrefix prefix = latency_sample_prefix(slot, result, l, 0u);
    EXPECT_TRUE(prefix.ok()) << prefix.describe();
    EXPECT_EQ(prefix.error, PI_ERROR_NONE);
    // The discriminating assertion: the prefix is phase_done, not latency_count.
    EXPECT_EQ(prefix.count, slot.phase_done);
    EXPECT_EQ(prefix.count, 1u);
    EXPECT_NE(prefix.count, result.latency_count);

    // A completed worker takes the whole share, and the two counts agree there.
    pi_consumer_slot_t done{};
    done.phase_done = 4;
    pi_producer_result_t retired{};
    retired.latency_count = 4;
    const SamplePrefix full = latency_sample_prefix(done, retired, l, 1u);
    EXPECT_TRUE(full.ok()) << full.describe();
    EXPECT_EQ(full.count, 4u);
}

TEST(PacketInjectionSamplePrefix, EachRejectionCarriesItsOwnCodeAndComparedValues) {
    pi_layout_t l{};
    const LayoutCase c{"P=2 T=2 S=16 B=8 N=4 latency", 2, 2, 16, 8, 4, PI_MODE_LATENCY, 0};
    ASSERT_TRUE(Accepts(c, &l));
    const uint64_t share = l.latency_sample_stride / sizeof(uint64_t);
    ASSERT_EQ(share, 4u);

    // The impossible direction. The producer retires only on the credit X280
    // returns after servicing, so latency_count <= phase_done holds by
    // construction and no run can reach this; it is a corruption check and is
    // asserted through the helper's inputs rather than via a state machine.
    {
        pi_consumer_slot_t slot{};
        slot.phase_done = 1;
        pi_producer_result_t result{};
        result.latency_count = 2;
        const SamplePrefix prefix = latency_sample_prefix(slot, result, l, 0u);
        EXPECT_FALSE(prefix.ok());
        EXPECT_EQ(prefix.error, PI_ERROR_COUNTER_DISTANCE);
        EXPECT_EQ(prefix.observed, 2u);
        EXPECT_EQ(prefix.bound, 1u);
        EXPECT_EQ(prefix.count, 0u) << "a rejected prefix must not hand back a length";
    }
    // Beyond the worker's export share.
    {
        pi_consumer_slot_t slot{};
        slot.phase_done = share + 1u;
        pi_producer_result_t result{};
        result.latency_count = 0;
        const SamplePrefix prefix = latency_sample_prefix(slot, result, l, 0u);
        EXPECT_FALSE(prefix.ok());
        EXPECT_EQ(prefix.error, PI_ERROR_PHASE_OVERRUN);
        EXPECT_EQ(prefix.observed, share + 1u);
        EXPECT_EQ(prefix.bound, share);
        EXPECT_EQ(prefix.count, 0u);
    }
    // Exactly the share is still admitted, so the bound is inclusive.
    {
        pi_consumer_slot_t slot{};
        slot.phase_done = share;
        pi_producer_result_t result{};
        const SamplePrefix prefix = latency_sample_prefix(slot, result, l, 0u);
        EXPECT_TRUE(prefix.ok()) << prefix.describe();
        EXPECT_EQ(prefix.count, share);
    }
    // A worker index with no share in the region.
    {
        pi_consumer_slot_t slot{};
        slot.phase_done = 1;
        pi_producer_result_t result{};
        const SamplePrefix prefix = latency_sample_prefix(slot, result, l, 2u);
        EXPECT_FALSE(prefix.ok());
        EXPECT_EQ(prefix.error, PI_ERROR_CAPACITY);
        EXPECT_EQ(prefix.observed, 2u);
        EXPECT_EQ(prefix.bound, 2u);  // two shares for two workers
        EXPECT_EQ(prefix.count, 0u);
    }
    // The containment check divides rather than multiplying index by share, so
    // a hostile index is rejected instead of wrapping into the region. Both of
    // these must be refused, and the second is the one that distinguishes the
    // two implementations: at a share of 4 entries, 1<<30 times the share is
    // 1<<32, which in 32-bit arithmetic is zero and would pass a
    // multiply-then-compare containment test while addressing far outside the
    // export. UINT32_MAX alone does not discriminate -- its truncated product
    // still exceeds the bound.
    for (const uint32_t hostile : {UINT32_MAX, 1u << 30}) {
        SCOPED_TRACE(hostile);
        pi_consumer_slot_t slot{};
        slot.phase_done = 1;
        pi_producer_result_t result{};
        const SamplePrefix prefix = latency_sample_prefix(slot, result, l, hostile);
        EXPECT_FALSE(prefix.ok());
        EXPECT_EQ(prefix.error, PI_ERROR_CAPACITY);
        EXPECT_EQ(prefix.observed, hostile);
        EXPECT_EQ(prefix.bound, 2u);
        EXPECT_EQ(prefix.count, 0u);
        EXPECT_EQ((uint32_t)(hostile * share) < l.latency_export_bytes / sizeof(uint64_t), hostile == (1u << 30))
            << "the discriminating index is the one whose product wraps below the bound";
    }
}

TEST(PacketInjectionSamplePrefix, OutsideLatencyThereIsNoExportAndNoPrefix) {
    for (const uint32_t mode : {(uint32_t)PI_MODE_THROUGHPUT, (uint32_t)PI_MODE_BACKPRESSURE}) {
        SCOPED_TRACE(mode);
        pi_layout_t l{};
        const LayoutCase c{"no export", 2, 2, 16, 8, 4, mode, 0};
        ASSERT_TRUE(Accepts(c, &l));
        ASSERT_EQ(l.latency_sample_stride, 0u);

        // Safe to call unconditionally: a zero share short-circuits before any
        // check, so even a count that would overrun a latency export is simply
        // no prefix rather than a rejection.
        pi_consumer_slot_t slot{};
        slot.phase_done = 1000;
        pi_producer_result_t result{};
        result.latency_count = 2000;
        const SamplePrefix prefix = latency_sample_prefix(slot, result, l, UINT32_MAX);
        EXPECT_TRUE(prefix.ok()) << prefix.describe();
        EXPECT_EQ(prefix.count, 0u);
    }
}

TEST(PacketInjectionAdmission, HostLayoutIsTheAbiLayoutAndNothingElse) {
    Config cfg{};
    cfg.mode = PI_MODE_LATENCY;
    cfg.include_creation = 0;
    cfg.family = WorkerFamily::Paired;
    cfg.tile_count = 140;  // P = 280
    cfg.S = 16;
    cfg.B = 8;
    cfg.C = 1;
    cfg.K = 1;
    cfg.W = 1000;
    cfg.N = 48;
    cfg.ordered_tiles.reserve(140);
    for (uint32_t i = 0; i < 140; ++i) {
        cfg.ordered_tiles.push_back(LogicalCoord{i % 10, i / 10});
    }

    const Admission adm = validate_config(cfg, FullCapacity(), 0x1000);
    ASSERT_TRUE(adm.ok()) << adm.describe();

    pi_layout_t direct{};
    ASSERT_EQ(pi_compute_layout(280, 140, 16, 8, 48, PI_MODE_LATENCY, 0x1000, &direct).code, (uint32_t)PI_LAYOUT_OK);

    const Layout host = compute_layout(cfg, 0x1000);
    ASSERT_TRUE(host.ok()) << host.describe();

    // The host derives no offset of its own: both its layouts are byte-for-byte
    // the ABI's, so a producer or X280 deriving from the same parameters agrees.
    EXPECT_EQ(std::memcmp(&host.abi, &direct, sizeof(direct)), 0);
    EXPECT_EQ(std::memcmp(&adm.layout.abi, &direct, sizeof(direct)), 0);
    EXPECT_EQ(host.abi.arena_base_bytes, (uint64_t)PI_ARENA_BASE_GOLDEN_P280_T140_S16);
    EXPECT_EQ(host.abi.latency_export_bytes, 107520u);
    EXPECT_EQ(host.abi.arena_bytes, 727232u);
    EXPECT_EQ(adm.required_bytes, host.abi.arena_bytes);
    EXPECT_EQ(adm.available_bytes, (uint64_t)PI_ARENA_BYTES);
    EXPECT_EQ(worker_count(cfg), 280u);

    // Admission proves the ms->ticks conversion for both clocks and hands back
    // the values the launch transmits, so no caller repeats it.
    uint64_t x280_ticks = 0;
    uint64_t tensix_ticks = 0;
    ASSERT_TRUE(
        x280::harness::measurement::phase_timeout_ticks(cfg.phase_timeout_ms, FullCapacity().mtime_hz, &x280_ticks));
    ASSERT_TRUE(x280::harness::measurement::phase_timeout_ticks(
        cfg.phase_timeout_ms, FullCapacity().producer_timer_hz, &tensix_ticks));
    EXPECT_EQ(adm.x280_timeout_ticks, x280_ticks);
    EXPECT_EQ(adm.producer_timeout_ticks, tensix_ticks);
    EXPECT_EQ(x280_ticks, 3000000000ULL);     // 60000 ms at 50 MHz
    EXPECT_EQ(tensix_ticks, 81000000000ULL);  // 60000 ms at 1.35 GHz
}

TEST(PacketInjectionAdmission, PhaseTimeoutConversionRoundsUpAndRefusesBadInput) {
    uint64_t ticks = 0xA5A5A5A5A5A5A5A5ULL;

    EXPECT_TRUE(
        x280::harness::measurement::phase_timeout_ticks(60000u, x280::harness::measurement::kX280MtimeHz, &ticks));
    EXPECT_EQ(ticks, 3000000000ULL);

    // A sub-tick budget rounds up rather than down to an instant timeout.
    EXPECT_TRUE(x280::harness::measurement::phase_timeout_ticks(1u, 1u, &ticks));
    EXPECT_EQ(ticks, 1u);
    EXPECT_TRUE(x280::harness::measurement::phase_timeout_ticks(1u, 1000u, &ticks));
    EXPECT_EQ(ticks, 1u);
    EXPECT_TRUE(x280::harness::measurement::phase_timeout_ticks(1u, 1001u, &ticks));
    EXPECT_EQ(ticks, 2u);

    // Rejections leave *out untouched.
    const uint64_t poison = 0x5A5A5A5A5A5A5A5AULL;
    for (const auto& bad : std::vector<std::pair<uint64_t, uint64_t>>{
             {0u, x280::harness::measurement::kX280MtimeHz},  // zero milliseconds
             {60000u, 0u},                                    // zero clock
             {UINT64_MAX, 2u},                                // ms * hz overflows u64
             {2u, UINT64_MAX},                                // and the other way round
             {UINT64_MAX, UINT64_MAX}}) {
        ticks = poison;
        EXPECT_FALSE(x280::harness::measurement::phase_timeout_ticks(bad.first, bad.second, &ticks))
            << bad.first << " ms at " << bad.second << " Hz";
        EXPECT_EQ(ticks, poison) << bad.first << " ms at " << bad.second << " Hz";
    }
    EXPECT_FALSE(
        x280::harness::measurement::phase_timeout_ticks(60000u, x280::harness::measurement::kX280MtimeHz, nullptr));
}

// The host deadline must cover nine phase budgets plus 5000 ms.
TEST(PacketInjectionAdmission, HostTimeoutBelowNinePhasesPlusSlackIsRefused) {
    Config cfg{};
    cfg.mode = PI_MODE_LATENCY;
    cfg.family = WorkerFamily::Paired;
    cfg.tile_count = 140;
    cfg.S = 16;
    cfg.B = 8;
    cfg.C = 1;
    cfg.K = 1;
    cfg.W = 1000;
    cfg.N = 58;
    for (uint32_t i = 0; i < 140; ++i) {
        cfg.ordered_tiles.push_back(LogicalCoord{i % 10, i / 10});
    }
    cfg.phase_timeout_ms = 1000;
    cfg.host_timeout_ms = 14000;
    const Admission admitted = validate_config(cfg, FullCapacity());
    EXPECT_TRUE(admitted.ok()) << admitted.describe();

    cfg.host_timeout_ms = 13999;
    const Admission refused = validate_config(cfg, FullCapacity());
    EXPECT_FALSE(refused.ok());
    EXPECT_EQ(refused.error, PI_ERROR_BAD_ARGS);
}

TEST(PacketInjectionAdmission, RejectsALayoutTheAbiWillNotAccept) {
    Config cfg{};
    cfg.mode = PI_MODE_LATENCY;
    cfg.family = WorkerFamily::Paired;
    cfg.tile_count = 140;  // P = 280
    cfg.S = 16;
    cfg.B = 8;
    cfg.C = 1;
    cfg.K = 1;
    cfg.W = 1000;
    cfg.N = 59;  // align64(280 * 8 * 59) exceeds PI_CACHE_LIMIT_BYTES
    cfg.ordered_tiles.reserve(140);
    for (uint32_t i = 0; i < 140; ++i) {
        cfg.ordered_tiles.push_back(LogicalCoord{i % 10, i / 10});
    }

    const Admission adm = validate_config(cfg, FullCapacity());
    EXPECT_FALSE(adm.ok());
    // The layout's rejection class maps to the admission error, and the two
    // compared values travel with it.
    EXPECT_EQ(adm.error, PI_ERROR_CAPACITY);
    EXPECT_EQ(adm.check, AdmissionCheck::Layout);
    EXPECT_EQ(adm.value, 132160u);  // align64(280 * 8 * 59)
    EXPECT_EQ(adm.bound, (uint64_t)PI_CACHE_LIMIT_BYTES);

    pi_layout_t direct{};
    EXPECT_EQ(
        pi_compute_layout(280, 140, 16, 8, 59, PI_MODE_LATENCY, 0, &direct).code, (uint32_t)PI_LAYOUT_EXPORT_CAPACITY);

    // N = 58 is the largest that fits, and both agree on it.
    cfg.N = 58;
    const Admission admitted = validate_config(cfg, FullCapacity());
    EXPECT_TRUE(admitted.ok()) << admitted.describe();
    EXPECT_EQ(admitted.layout.abi.latency_export_bytes, 129920u);
}

TEST(PacketInjectionAdmission, ARejectedAdmissionCarriesTheLayoutItDerived) {
    // The layout is assigned before the first rejection that can follow it, so
    // a refused run still hands back the check that explains it. That makes
    // adm.layout.ok() a statement about the LAYOUT, not about the run: it is
    // false only when the layout itself was refused, and asserting it false on
    // every rejection would be wrong.
    Config cfg{};
    cfg.mode = PI_MODE_BACKPRESSURE;
    cfg.family = WorkerFamily::Paired;
    cfg.tile_count = 1;
    cfg.S = 16;
    cfg.B = 8;
    cfg.C = 1;
    cfg.K = 1;
    cfg.W = 0;
    cfg.N = 40000;
    cfg.ordered_tiles.push_back(LogicalCoord{0, 0});

    // The layout is fine and so is the L1 budget.
    const Admission admitted = validate_config(cfg, FullCapacity());
    ASSERT_TRUE(admitted.ok()) << admitted.describe();
    EXPECT_TRUE(admitted.layout.ok());
    EXPECT_EQ(admitted.layout.abi.l1_tile_bytes, 1283392u);

    // The layout is still fine; the tile's L1 requirement is what fails. Two
    // backpressure sample arrays per worker at N=50000 need 1603136 bytes of a
    // tile against the 1400832 the allocator can hand out.
    cfg.N = 50000;
    const Admission no_l1 = validate_config(cfg, FullCapacity());
    EXPECT_FALSE(no_l1.ok());
    EXPECT_EQ(no_l1.error, PI_ERROR_CAPACITY);
    EXPECT_TRUE(no_l1.layout.ok()) << "the layout was derived; only the L1 budget failed";
    EXPECT_EQ(no_l1.layout.check.code, (uint32_t)PI_LAYOUT_OK);
    EXPECT_EQ(no_l1.layout.abi.l1_tile_bytes, 1603392u);
    EXPECT_EQ(no_l1.check, AdmissionCheck::L1Tile);
    EXPECT_EQ(no_l1.value, 1603392u);
    EXPECT_EQ(no_l1.bound, 1400832u);

    // A layout rejection leaves both false, and the check names the class.
    cfg.mode = PI_MODE_LATENCY;
    cfg.tile_count = PI_MAX_TILES;
    cfg.N = 59;
    cfg.ordered_tiles.clear();
    for (uint32_t i = 0; i < PI_MAX_TILES; ++i) {
        cfg.ordered_tiles.push_back(LogicalCoord{i % 10, i / 10});
    }
    const Admission refused = validate_config(cfg, FullCapacity());
    EXPECT_FALSE(refused.ok());
    EXPECT_FALSE(refused.layout.ok());
    EXPECT_EQ(refused.layout.check.code, (uint32_t)PI_LAYOUT_EXPORT_CAPACITY);
    EXPECT_EQ(refused.layout.check.value, 132160u);
    EXPECT_EQ(refused.layout.check.bound, (uint64_t)PI_CACHE_LIMIT_BYTES);
    EXPECT_EQ(refused.check, AdmissionCheck::Layout) << refused.describe();

    // And the case the hoist cannot fix: a rejection BEFORE the layout is
    // computed leaves the member default, which is indistinguishable from a
    // successful layout because PI_LAYOUT_OK is zero. So layout.ok() answers
    // "was the layout refused", never "is this Admission good" -- adm.ok() is
    // the only thing that answers the latter.
    Config bad_policy{};
    bad_policy.mode = PI_MODE_THROUGHPUT;
    bad_policy.include_creation = 1;  // throughput requires 0
    bad_policy.ordered_tiles.push_back(LogicalCoord{0, 0});
    const Admission pre_layout = validate_config(bad_policy, FullCapacity());
    EXPECT_FALSE(pre_layout.ok());
    EXPECT_EQ(pre_layout.error, PI_ERROR_MODE_MISMATCH);
    EXPECT_TRUE(pre_layout.layout.ok());
    EXPECT_EQ(pre_layout.layout.abi.arena_bytes, 0u) << "no layout was derived";
}

TEST(PacketInjectionAdmission, LayoutRejectionClassChoosesTheAdmissionError) {
    Config cfg{};
    cfg.mode = PI_MODE_THROUGHPUT;
    cfg.family = WorkerFamily::BriscOnly;
    cfg.tile_count = 1;
    cfg.S = 16;
    cfg.B = 8;
    cfg.C = 1;
    cfg.K = 1;
    cfg.W = 0;
    cfg.N = 1024;
    cfg.ordered_tiles.push_back(LogicalCoord{0, 0});

    // A misaligned L1 base is a placement fault, not a capacity one: the two
    // classes reach admission through pi_layout_code_to_error rather than every
    // layout rejection collapsing to PI_ERROR_CAPACITY.
    const Admission misaligned = validate_config(cfg, FullCapacity(), 98305u);
    EXPECT_FALSE(misaligned.ok());
    EXPECT_EQ(misaligned.error, PI_ERROR_BAD_PLACEMENT);
    EXPECT_EQ(misaligned.check, AdmissionCheck::Layout);
    EXPECT_EQ(misaligned.value, 98305u);
    EXPECT_EQ(misaligned.bound, 64u);

    // The same base, aligned, is admitted.
    const Admission aligned = validate_config(cfg, FullCapacity(), 98304u);
    EXPECT_TRUE(aligned.ok()) << aligned.describe();
    EXPECT_EQ(aligned.layout.abi.l1_tile_base, 98304u);

    // A tile count outside the domain is reported, with its value and limit,
    // rather than returning a not-ok layout with no check.
    cfg.family = WorkerFamily::Paired;
    cfg.tile_count = PI_MAX_TILES + 1u;
    const Admission too_many = validate_config(cfg, FullCapacity());
    EXPECT_FALSE(too_many.ok());
    EXPECT_EQ(too_many.check, AdmissionCheck::TileCount);
    EXPECT_EQ(too_many.value, PI_MAX_TILES + 1u);
    EXPECT_EQ(too_many.bound, PI_MAX_TILES);
}

TEST(PacketInjectionAdmission, AdmitsWPlusNCrossing2Pow32AndRejectsTheirOverflow) {
    Config cfg{};
    cfg.mode = PI_MODE_THROUGHPUT;
    cfg.include_creation = 0;
    cfg.family = WorkerFamily::BriscOnly;
    cfg.tile_count = 1;
    cfg.S = 16;
    cfg.B = 1;
    cfg.C = 1;
    cfg.K = 1;
    cfg.W = 3000000000ULL;
    cfg.N = 2000000000ULL;  // W + N = 5*10^9 > 2^32
    cfg.ordered_tiles.push_back(LogicalCoord{0, 0});

    const Admission adm = validate_config(cfg, FullCapacity());
    EXPECT_TRUE(adm.ok()) << adm.describe();

    // The per-worker total is counted in 64 bits, so only a real overflow fails.
    cfg.W = UINT64_MAX;
    cfg.N = 2;
    EXPECT_FALSE(validate_config(cfg, FullCapacity()).ok());
}

TEST(PacketInjectionAdmission, RejectsInvalidPolicyParameters) {
    Config cfg{};
    cfg.ordered_tiles.push_back(LogicalCoord{0, 0});

    // C > K
    cfg.C = 2;
    cfg.K = 1;
    EXPECT_EQ(validate_config(cfg, FullCapacity()).error, PI_ERROR_BAD_ARGS);
    cfg.C = 1;

    // include_creation must be 0 or 1
    cfg.include_creation = 2;
    EXPECT_EQ(validate_config(cfg, FullCapacity()).error, PI_ERROR_BAD_ARGS);
    cfg.include_creation = 0;

    // Throughput with include_creation=1
    cfg.mode = PI_MODE_THROUGHPUT;
    cfg.include_creation = 1;
    EXPECT_EQ(validate_config(cfg, FullCapacity()).error, PI_ERROR_MODE_MISMATCH);
    cfg.include_creation = 0;
    cfg.mode = PI_MODE_LATENCY;

    // Duplicate tiles
    cfg.tile_count = 2;
    cfg.ordered_tiles = {LogicalCoord{0, 0}, LogicalCoord{0, 0}};
    EXPECT_EQ(validate_config(cfg, FullCapacity()).error, PI_ERROR_BAD_PLACEMENT);
}

TEST(PacketInjectionMemoryProfile, ArenaExcludesImageStacksTrapsSocket) {
    const uint64_t backed_bytes = kRequiredLimBackedBytes;
    const auto arena = x280::available_lim_for_layout(x280::LimLayout::PacketInjection, backed_bytes);

    EXPECT_EQ(arena.begin, 0x08020000ULL);
    EXPECT_EQ(arena.end, 0x080FFF80ULL);
    EXPECT_EQ(arena.size(), 917376ULL);

    // Excludes image text/data [0x08001000, 0x08020000).
    EXPECT_EQ(
        X280_ACTIVE_FW_LOAD_ADDR + X280_LIM_PACKET_INJECTION_IMAGE_MAX_BYTES, X280_LIM_PACKET_INJECTION_IMAGE_END);
    EXPECT_LE(X280_LIM_PACKET_INJECTION_IMAGE_END, X280_LIM_PACKET_INJECTION_ARENA_BASE);
    EXPECT_GE(arena.begin, 0x08020000ULL);

    // Excludes active stacks [0x080FFF80, 0x0811FF80)
    EXPECT_LE(arena.end, 0x080FFF80ULL);

    // Excludes active traps [0x0811FF80, 0x08120000)
    EXPECT_LE(arena.end, 0x0811FF80ULL);

    // Excludes everything from 0x08120000 up: the hart protocol block and the socket layout
    EXPECT_LE(arena.end, 0x08120000ULL);
}

TEST(PacketInjectionMemoryProfile, RespectsBackedSize) {
    // If backed bytes does not reach the end of the hart protocol block, arena is empty
    constexpr uint64_t kReservedBytes = X280_HART_PROTOCOL_END - X280_LIM_BASE;
    const auto empty_arena = x280::available_lim_for_layout(x280::LimLayout::PacketInjection, kReservedBytes - 1u);
    EXPECT_EQ(empty_arena.size(), 0u);

    // Reaching it enables the full lower arena
    const auto valid_arena = x280::available_lim_for_layout(x280::LimLayout::PacketInjection, kReservedBytes);
    EXPECT_EQ(valid_arena.size(), 917376ULL);
}

TEST(PacketInjectionMemoryProfile, ValidateActiveImagePacketInjectionBound) {
    // Under PacketInjection, max image size is 0x1F000 (126976 bytes)
    EXPECT_TRUE(
        x280::validate_active_image(0x08001000ULL, 0x1F000ULL, X280_LIM_SIZE, x280::LimLayout::PacketInjection).ok);
    EXPECT_FALSE(
        x280::validate_active_image(0x08001000ULL, 0x1F004ULL, X280_LIM_SIZE, x280::LimLayout::PacketInjection).ok);

    // Both layouts accept valid sub-0x1F000 images inside the primed window.
    EXPECT_TRUE(
        x280::validate_active_image(0x08001000ULL, 0x1000ULL, x280::kLimWayBytes, x280::LimLayout::PacketInjection).ok);
    EXPECT_TRUE(x280::validate_active_image(0x08001000ULL, 0x1000ULL, x280::kLimWayBytes).ok);
    EXPECT_FALSE(x280::validate_active_image(
                     0x08001000ULL, 0x1000ULL, x280::kLimCacheLineBytes, x280::LimLayout::PacketInjection)
                     .ok);
}

TEST(PacketInjectionMemoryProfile, ArgumentBlockValidationInArena) {
    const uint64_t backed_bytes = kRequiredLimBackedBytes;
    const auto arena = x280::available_lim_for_layout(x280::LimLayout::PacketInjection, backed_bytes);

    // Args inside arena
    EXPECT_TRUE(x280::validate_argument_block(arena, 0x08020000ULL, sizeof(pi_x280_args_t)).ok);

    // Args outside arena
    EXPECT_FALSE(x280::validate_argument_block(arena, 0x0801F000ULL, sizeof(pi_x280_args_t)).ok);
    EXPECT_FALSE(x280::validate_argument_block(arena, 0x080FFF80ULL, sizeof(pi_x280_args_t)).ok);
}

TEST(PacketInjectionHeader, UnicastHeaderSeedAll96Bytes) {
    fs_hybrid_mesh_packet_header_t header{};
    fs_seed_unicast_header(&header, 42u, 17u);

    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&header);
    // Bytes 0..39: command union zero
    for (size_t i = 0; i < 40; ++i) {
        EXPECT_EQ(bytes[i], 0u) << "byte " << i;
    }
    // Payload bytes 40..41: zero
    EXPECT_EQ(bytes[40], 0u);
    EXPECT_EQ(bytes[41], 0u);
    // Byte 42: NOC_UNICAST_WRITE = 0
    EXPECT_EQ(bytes[42], 0u);
    // Byte 43: src_ch_id = 0
    EXPECT_EQ(bytes[43], 0u);
    // Bytes 44..47: routing_fields = 0
    EXPECT_EQ(header.routing_fields.value, 0u);
    // Bytes 48..82: 35 route bytes FS_HOP_NOOP
    for (size_t i = 0; i < 35; ++i) {
        EXPECT_EQ(header.route_buffer[i], FS_HOP_NOOP) << "route byte " << i;
    }
    // Destination chip at offset 83
    EXPECT_EQ(bytes[83], 42u);
    EXPECT_EQ(bytes[84], 0u);
    // Destination mesh at offset 85
    EXPECT_EQ(bytes[85], 17u);
    EXPECT_EQ(bytes[86], 0u);
    // Multicast parameters bytes 87..94: zero
    EXPECT_EQ(header.mcast_params_64, 0u);
    // is_mcast_active at 95: zero
    EXPECT_EQ(header.is_mcast_active, 0u);
}

TEST(PacketInjectionTimer, LiveTimerSuccessAndRetryRollover) {
    constexpr uintptr_t low_addr = 0x12340000u;
    static std::vector<uintptr_t> addresses;
    addresses.clear();
    // 1. Consistent read on first attempt
    static uint32_t seq1[] = {10u, 1000u, 10u};  // high1=10, low=1000, high2=10
    static size_t idx1 = 0;
    idx1 = 0;
    g_tensix_live_timer_mock_read = [](uintptr_t addr) -> uint32_t {
        addresses.push_back(addr);
        return seq1[idx1++];
    };

    uint64_t ticks = 0;
    int ok = tensix_live_timer_read(low_addr, 16, &ticks);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(ticks, (10ULL << 32) | 1000ULL);
    EXPECT_EQ(addresses, (std::vector<uintptr_t>{low_addr + 4, low_addr, low_addr + 4}));

    // 2. Rollover on attempt 0 (high1=10, low=0xFFFFFFF0, high2=11), then consistent on attempt 1 (high1=11, low=50,
    // high2=11)
    static uint32_t seq2[] = {
        10u,
        0xFFFFFFF0u,
        11u,  // attempt 0: high1 != high2
        11u,
        50u,
        11u  // attempt 1: high1 == high2
    };
    static size_t idx2 = 0;
    idx2 = 0;
    addresses.clear();
    g_tensix_live_timer_mock_read = [](uintptr_t addr) -> uint32_t {
        addresses.push_back(addr);
        return seq2[idx2++];
    };

    ticks = 0;
    ok = tensix_live_timer_read(low_addr, 16, &ticks);
    EXPECT_EQ(ok, 1);
    EXPECT_EQ(ticks, (11ULL << 32) | 50ULL);
    EXPECT_EQ(
        addresses,
        (std::vector<uintptr_t>{low_addr + 4, low_addr, low_addr + 4, low_addr + 4, low_addr, low_addr + 4}));

    // 3. Retry exhaustion
    static uint32_t seq3_val = 0;
    g_tensix_live_timer_mock_read = [](uintptr_t) -> uint32_t {
        return ++seq3_val;  // Always different!
    };
    ticks = 0xDEADBEEFULL;
    ok = tensix_live_timer_read(0, 4, &ticks);
    EXPECT_EQ(ok, 0);
    EXPECT_EQ(ticks, 0xDEADBEEFULL);  // Unmodified on failure

    g_tensix_live_timer_mock_read = nullptr;
}

TEST(PacketInjectionWindowMap, ReservationsAreDisjointAndBounded) {
    EXPECT_EQ(X280_TLB_PACKET_INJECTION_CREDIT_BASE, 16u);
    EXPECT_EQ(X280_TLB_PACKET_INJECTION_CREDIT_COUNT, 140u);
    EXPECT_EQ(X280_TLB_PACKET_INJECTION_TIMER, 156u);

    // [16, 156) is exactly 140 entries
    EXPECT_EQ(
        X280_TLB_PACKET_INJECTION_CREDIT_BASE + X280_TLB_PACKET_INJECTION_CREDIT_COUNT,
        X280_TLB_PACKET_INJECTION_TIMER);

    // Timer, core-sync block, and host block are adjacent or ordered without overlap.
    EXPECT_LT(X280_TLB_PACKET_INJECTION_TIMER, (uint32_t)X280_TLB_CORE_SYNC_BASE);
    EXPECT_LE(X280_TLB_CORE_SYNC_BASE + X280_TLB_CORE_SYNC_COUNT, (uint32_t)X280_TLB_HOST_BASE);
}

// --------------------------------------------------------------------------
// Producer state model
// --------------------------------------------------------------------------

#include <functional>
#include <memory>
#include <string>

// Stands in for the Tensix NoC and live timer. Every LIM destination the
// producer derives is declared up front, so a write to any other address is a
// derivation failure rather than a silently dropped packet.
struct MockTensixTransport {
    // Target identity and header seeding are the Tensix transport's own.
    using Device = packet_injection::DeviceTensixTransport;
    static constexpr uint32_t kNoc = Device::kNoc;
    static constexpr uint32_t kRisc = Device::kRisc;
    static constexpr uint32_t kL1SpanLimit = Device::kL1SpanLimit;
    static constexpr pi_actor_t kActor = Device::kActor;

    static void prepare_header(volatile fs_hybrid_mesh_packet_header_t* header, uint16_t dst_chip, uint16_t dst_mesh) {
        Device::prepare_header(header, dst_chip, dst_mesh);
    }

    static inline uint32_t produced_dst = 0;
    static inline uint32_t ready_dst = 0;
    static inline uint32_t ring_base = 0;
    static inline uint32_t last_published_produced = 0;
    static inline uint32_t last_published_ready = 0;
    static inline std::vector<uint32_t> published_produced_history;
    static inline std::vector<uint32_t> published_ready_history;
    static inline std::vector<uint32_t> header_dsts;
    static inline std::vector<uint64_t> header_times;
    static inline std::vector<uint32_t> timer_consumed_history;
    static inline uint32_t u32_write_count = 0;
    static inline uint32_t barrier_count = 0;
    static inline uint64_t mock_time = 1000;
    static inline uintptr_t host_l1_base = 0;
    // Device-space base of the slice host_l1_base stands in for, so a source
    // address can be mapped back into the host buffer.
    static inline uint32_t l1_worker_base = 0;
    static inline std::function<void()> on_poll_cache = nullptr;
    static inline std::vector<std::pair<uintptr_t, std::vector<uint8_t>>> pending_sources;
    static inline uint32_t fail_timer_after_publication = UINT32_MAX;
    // The barrier call, counted from 1, that expires instead of completing.
    static inline uint32_t expire_barrier = UINT32_MAX;
    static inline uint64_t last_barrier_budget = 0;

    static void reset() {
        produced_dst = 0;
        ready_dst = 0;
        ring_base = 0;
        last_published_produced = 0;
        last_published_ready = 0;
        published_produced_history.clear();
        published_ready_history.clear();
        header_dsts.clear();
        header_times.clear();
        timer_consumed_history.clear();
        u32_write_count = 0;
        barrier_count = 0;
        mock_time = 1000;
        host_l1_base = 0;
        l1_worker_base = 0;
        on_poll_cache = nullptr;
        pending_sources.clear();
        fail_timer_after_publication = UINT32_MAX;
        expire_barrier = UINT32_MAX;
        last_barrier_budget = 0;
    }

    static void invalidate_cache() {
        if (on_poll_cache) {
            on_poll_cache();
        }
    }

    static bool barrier(uint32_t, uint64_t budget) {
        barrier_count++;
        last_barrier_budget = budget;
        if (barrier_count == expire_barrier) {
            return false;
        }
        for (const auto& source : pending_sources) {
            const auto* bytes = reinterpret_cast<const uint8_t*>(source.first);
            EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + source.second.size()), source.second)
                << "NoC source changed before its completion barrier at " << source.first;
        }
        pending_sources.clear();
        return true;
    }

    // The producer names its own L1 in device space; a host run holds that one
    // slice in a heap buffer, so undo the slice base the override replaced.
    static uintptr_t host_addr(uint32_t src_l1_addr) {
        if (host_l1_base == 0) {
            return (uintptr_t)src_l1_addr;
        }
        EXPECT_GE(src_l1_addr, l1_worker_base) << "source below this worker's slice";
        return host_l1_base + (src_l1_addr - l1_worker_base);
    }

    // Preserve the issued source bytes until the mock completion barrier.
    static void record_source(uint32_t src_l1_addr, size_t bytes) {
        const uintptr_t addr = host_addr(src_l1_addr);
        const auto* source = reinterpret_cast<const uint8_t*>(addr);
        pending_sources.emplace_back(addr, std::vector<uint8_t>(source, source + bytes));
    }

    static void issue_header_write(uint32_t src_l1_addr, uint32_t, uint32_t, uint32_t dst_lim_addr, uint32_t) {
        record_source(src_l1_addr, PI_SLOT_BYTES);
        header_dsts.push_back(dst_lim_addr);
        header_times.push_back(mock_time);
    }

    static void issue_u32_write(uint32_t src_l1_addr, uint32_t, uint32_t, uint32_t dst_lim_addr, uint32_t) {
        record_source(src_l1_addr, sizeof(uint32_t));
        u32_write_count++;
        const uint32_t val = *(volatile uint32_t*)host_addr(src_l1_addr);
        if (dst_lim_addr == produced_dst) {
            last_published_produced = val;
            published_produced_history.push_back(val);
        } else if (dst_lim_addr == ready_dst) {
            last_published_ready = val;
            published_ready_history.push_back(val);
        } else {
            ADD_FAILURE() << "u32 write to an undeclared LIM destination 0x" << std::hex << dst_lim_addr;
        }
    }

    static int read_cpu(uint64_t* ticks) {
        *ticks = mock_time;
        return 1;
    }

    static int read_timer(uint32_t, uint64_t* ticks) {
        if (last_published_produced >= fail_timer_after_publication) {
            return 0;
        }
        if (host_l1_base != 0u) {
            const uint32_t consumed = *(volatile uint32_t*)(host_l1_base + PI_L1_CONSUMED_OFFSET);
            timer_consumed_history.push_back(consumed);
        }
        *ticks = mock_time;
        mock_time += 10;
        return 1;
    }
};

struct MockPacketSyncTransport {
    using Roster = core_sync::X280Roster;
    using ArrivalTarget = core_sync::MetalWordArrivalTarget;
    using WaitContext = core_sync::X280WaitContext;

    static constexpr bool kHasTransport = true;
    static constexpr bool kMayDirect = true;

    static inline bool notify_arrival_result = true;
    static inline bool wait_result = true;
    static inline bool wait_for_arrivals_result = true;
    static inline bool notify_group_result = true;
    static inline uint32_t notify_arrival_count = 0u;
    static inline uint32_t wait_count = 0u;
    static inline uint32_t wait_for_arrivals_count = 0u;
    static inline uint32_t notify_group_count = 0u;
    static inline uint32_t last_wait_addr = 0u;
    static inline ArrivalTarget last_arrival{};
    static inline Roster last_roster{};
    static inline std::vector<core_sync::WordArrival> arrivals;
    static inline std::vector<core_sync::X280Group> groups;
    static inline std::vector<core_sync::X280Endpoint> endpoints;
    static inline std::function<void()> on_wait_for_arrivals = nullptr;
    static inline std::function<void()> on_notify_group = nullptr;
    static inline std::function<void()> on_wait = nullptr;

    static void reset() {
        notify_arrival_result = true;
        wait_result = true;
        wait_for_arrivals_result = true;
        notify_group_result = true;
        notify_arrival_count = 0u;
        wait_count = 0u;
        wait_for_arrivals_count = 0u;
        notify_group_count = 0u;
        last_wait_addr = 0u;
        last_arrival = ArrivalTarget{};
        last_roster = Roster{};
        arrivals.clear();
        groups.clear();
        endpoints.clear();
        on_wait_for_arrivals = nullptr;
        on_notify_group = nullptr;
        on_wait = nullptr;
    }

    static uint64_t arrival_count(const Roster& roster) { return roster.word_arrival_count; }

    static bool has_release_targets(const Roster& roster) {
        return roster.group_count != 0u || roster.endpoint_count != 0u;
    }

    static bool validate_roster(const Roster& roster) { return core_sync::X280Transport<>::validate_roster(roster); }

    static bool notify_arrival(const ArrivalTarget& target, uint32_t) {
        ++notify_arrival_count;
        last_arrival = target;
        return notify_arrival_result;
    }

    static bool wait(uint32_t local_addr, uint32_t, const WaitContext&) {
        ++wait_count;
        last_wait_addr = local_addr;
        if (on_wait) {
            on_wait();
        }
        return wait_result;
    }

    static bool wait_for_arrivals(const Roster& roster, uint32_t, const WaitContext&) {
        ++wait_for_arrivals_count;
        record_roster(roster);
        if (on_wait_for_arrivals) {
            on_wait_for_arrivals();
        }
        return wait_for_arrivals_result;
    }

    static bool notify_group(const Roster& roster, uint32_t) {
        ++notify_group_count;
        record_roster(roster);
        if (on_notify_group) {
            on_notify_group();
        }
        return notify_group_result;
    }

private:
    static void record_roster(const Roster& roster) {
        arrivals.assign(roster.word_arrivals, roster.word_arrivals + roster.word_arrival_count);
        groups.assign(roster.groups, roster.groups + roster.group_count);
        endpoints.assign(roster.endpoints, roster.endpoints + roster.endpoint_count);
        last_roster = roster;
        last_roster.word_arrivals = arrivals.data();
        last_roster.groups = groups.data();
        last_roster.endpoints = endpoints.data();
    }
};

// One way a launch can be malformed, and the code the producer owes for it.
struct LaunchRejection {
    const char* name;
    void (*mutate)(pi_producer_args_t&);
    pi_error_t expected;
};

// One worker on tile 0 with its L1 slice held in a host buffer. Derives the
// same layout the producer will, so the mock knows every address it should see.
struct ProducerTestFixture {
    std::vector<uint8_t> storage;
    uintptr_t l1_base = 0;
    pi_producer_args_t args{};
    pi_layout_t layout{};

    // `P`, `T` and `l1_tile_base` default to the single-worker slice at offset
    // zero that most tests want; the L1-bound cases set all three.
    ProducerTestFixture(
        uint32_t mode,
        uint32_t include_creation,
        uint32_t S,
        uint32_t B,
        uint64_t W,
        uint64_t N,
        uint32_t P = 1u,
        uint32_t T = 1u,
        uint32_t l1_tile_base = 0u) :
        storage(32768 + 64, 0) {
        MockTensixTransport::reset();
        MockPacketSyncTransport::reset();
        EXPECT_EQ(pi_compute_layout(P, T, S, B, N, mode, l1_tile_base, &layout).code, (uint32_t)PI_LAYOUT_OK);
        EXPECT_LE(layout.l1_worker_stride, 32768u);

        l1_base = (uintptr_t)pi_align64((uint64_t)(uintptr_t)storage.data());

        args.abi_tag = PI_ABI_TAG(mode, include_creation);
        args.run_nonce = 0x1234ABCDu;
        args.worker_index = 0u;
        args.worker_count = (uint16_t)P;
        args.tile_count = (uint16_t)T;
        args.S = S;
        args.B = B;
        args.poll_cadence = PI_DEFAULT_POLL_CADENCE;
        args.dst_chip = 7u;
        args.dst_mesh = 11u;
        args.peer = {8u, 3u};
        args.l1_tile_base = l1_tile_base;
        args.W = W;
        args.N = N;
        args.timeout_ticks = 1000000ULL;

        MockTensixTransport::host_l1_base = l1_base;
        MockTensixTransport::l1_worker_base = pi_worker_l1_base(&layout, 0u);
        MockTensixTransport::produced_dst = (uint32_t)(X280_LIM_PACKET_INJECTION_ARENA_BASE + layout.produced_offset);
        MockTensixTransport::ready_dst = (uint32_t)(X280_LIM_PACKET_INJECTION_ARENA_BASE + layout.ready_offset);
        MockTensixTransport::ring_base = (uint32_t)(X280_LIM_PACKET_INJECTION_ARENA_BASE + layout.rings_offset);
    }

    volatile uint32_t* consumed_mirror() const { return (volatile uint32_t*)(l1_base + PI_L1_CONSUMED_OFFSET); }
    volatile uint32_t* control_line() const { return (volatile uint32_t*)(l1_base + PI_L1_CONTROL_OFFSET); }
    pi_producer_result_t* result() const { return (pi_producer_result_t*)(l1_base + PI_L1_RESULT_OFFSET); }
    pi_local_stamp_t* starts() const { return (pi_local_stamp_t*)(l1_base + layout.l1_starts_offset); }
    pi_local_sample_t* stalls() const { return (pi_local_sample_t*)(l1_base + layout.l1_stalls_offset); }
    pi_local_sample_t* rtts() const { return (pi_local_sample_t*)(l1_base + layout.l1_rtts_offset); }
};

TEST(PacketInjectionProducerStateModel, FlushesOnFullRingWhenBExceedsS) {
    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, /*S=*/2, /*B=*/4, /*W=*/0, /*N=*/4);

    MockTensixTransport::on_poll_cache = [&]() {
        if (MockTensixTransport::last_published_produced == 2) {
            *f.consumed_mirror() = 2;
        } else if (MockTensixTransport::last_published_produced == 4) {
            *f.consumed_mirror() = 4;
        }
    };

    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_NONE);

    // With B=4 > S=2 the producer flushed at 2 (full ring) and 4 (phase end).
    EXPECT_EQ(MockTensixTransport::published_produced_history, (std::vector<uint32_t>{2, 4}));
    EXPECT_TRUE(MockTensixTransport::pending_sources.empty());

    const pi_producer_result_t* r = f.result();
    EXPECT_EQ(r->abi_tag, PI_ABI_TAG(PI_MODE_THROUGHPUT, 0));
    EXPECT_EQ(r->run_nonce, f.args.run_nonce);
    EXPECT_EQ(r->worker_index, 0u);
    EXPECT_EQ(r->quiescent, 1u);
    EXPECT_EQ(r->measured_completed, 4u);
    EXPECT_EQ(r->published_progress, 4u);
    EXPECT_EQ(r->consumed_progress, 4u);
    EXPECT_EQ(r->reserved_progress, 4u);
    EXPECT_EQ(r->phase, (uint32_t)PI_PHASE_DONE);
    EXPECT_EQ(r->failure.code, (uint32_t)PI_ERROR_NONE);
    EXPECT_EQ(MockPacketSyncTransport::notify_arrival_count, 1u);
    EXPECT_EQ(MockPacketSyncTransport::wait_count, 1u);
    EXPECT_EQ(MockPacketSyncTransport::last_arrival.noc.x, f.args.peer.x);
    EXPECT_EQ(MockPacketSyncTransport::last_arrival.noc.y, f.args.peer.y);
    EXPECT_EQ(
        MockPacketSyncTransport::last_arrival.ready_base,
        X280_LIM_PACKET_INJECTION_ARENA_BASE + f.layout.sync_arrivals_offset);
    EXPECT_EQ(MockPacketSyncTransport::last_arrival.worker_index, 0u);
    EXPECT_EQ(MockPacketSyncTransport::last_wait_addr, f.layout.l1_tile_base + f.layout.l1_release_offset);
    EXPECT_EQ(MockTensixTransport::published_ready_history, (std::vector<uint32_t>{PI_READY_DONE}));
}

TEST(PacketInjectionProducerStateModel, HeadersLandInThisWorkersRingSlots) {
    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, /*S=*/2, /*B=*/1, /*W=*/0, /*N=*/4);

    MockTensixTransport::on_poll_cache = [&]() { *f.consumed_mirror() = MockTensixTransport::last_published_produced; };

    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_NONE);

    // The ring is addressed as rings_offset + w*ring_stride, indexed modulo S.
    ASSERT_EQ(MockTensixTransport::header_dsts.size(), 4u);
    for (size_t i = 0; i < MockTensixTransport::header_dsts.size(); ++i) {
        EXPECT_EQ(
            MockTensixTransport::header_dsts[i],
            MockTensixTransport::ring_base + (uint32_t)(i % f.args.S) * PI_SLOT_BYTES)
            << "header " << i;
    }
}

TEST(PacketInjectionProducerStateModel, ThroughputEndpointsBracketTheMeasuredHeaders) {
    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, /*S=*/2, /*B=*/1, /*W=*/2, /*N=*/4);
    size_t headers_at_release = 0u;
    MockPacketSyncTransport::on_wait = [&]() { headers_at_release = MockTensixTransport::header_dsts.size(); };

    MockTensixTransport::on_poll_cache = [&]() { *f.consumed_mirror() = MockTensixTransport::last_published_produced; };

    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_NONE);

    const pi_producer_result_t* r = f.result();
    EXPECT_GT(r->throughput_end, r->throughput_start);

    // The start endpoint falls after the last warmup header and before the
    // first measured one, and the end endpoint after them all, so the interval
    // spans exactly the measured injection and nothing else.
    const auto& times = MockTensixTransport::header_times;
    ASSERT_EQ(times.size(), f.args.W + f.args.N);
    EXPECT_EQ(headers_at_release, f.args.W);
    for (size_t i = 0; i < times.size(); ++i) {
        if (i < f.args.W) {
            EXPECT_LE(times[i], r->throughput_start);
        } else {
            EXPECT_GE(times[i], r->throughput_start);
            EXPECT_LE(times[i], r->throughput_end);
        }
    }
    ASSERT_FALSE(MockTensixTransport::timer_consumed_history.empty());
    EXPECT_EQ(MockTensixTransport::timer_consumed_history.back(), f.args.W + f.args.N);
    EXPECT_EQ(MockTensixTransport::timer_consumed_history.back(), MockTensixTransport::last_published_produced);
}

TEST(PacketInjectionProducerStateModel, S1ProducerDeadlockFree) {
    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, /*S=*/1, /*B=*/1, /*W=*/0, /*N=*/3);

    MockTensixTransport::on_poll_cache = [&]() { *f.consumed_mirror() = MockTensixTransport::last_published_produced; };

    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_NONE);

    EXPECT_EQ(MockTensixTransport::published_produced_history, (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_EQ(f.result()->measured_completed, 3u);
}

TEST(PacketInjectionProducerStateModel, WrappingCounterDistanceAndRejection) {
    uint32_t c_seen = 0xFFFFFFFEu;
    uint32_t r = 0x00000002u;  // 4 slots forward
    uint32_t occupied = pi_counter_distance(r, c_seen);
    EXPECT_EQ(occupied, 4u);

    // Valid forward credit jump
    uint32_t next = 0x00000001u;
    uint32_t jump = pi_counter_distance(next, c_seen);
    EXPECT_EQ(jump, 3u);
    EXPECT_LE(jump, occupied);

    // A credit jump past what the ring holds is rejected the first time the
    // ring fills, before the request that would have overrun it is issued.
    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, /*S=*/2, /*B=*/1, /*W=*/0, /*N=*/4);
    *f.consumed_mirror() = 10u;  // jump=10 > occupied=2

    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_COUNTER_DISTANCE);

    EXPECT_EQ(MockTensixTransport::header_dsts.size(), 2u);
    EXPECT_EQ(f.result()->failure.code, (uint32_t)PI_ERROR_COUNTER_DISTANCE);
    EXPECT_EQ(f.result()->failure.phase, (uint32_t)PI_PHASE_MEASURE);
    EXPECT_EQ(f.result()->quiescent, 1u);
    EXPECT_EQ(MockTensixTransport::last_published_ready, (uint32_t)PI_READY_FAILED);
}

TEST(PacketInjectionProducerStateModel, LatencyCountCountsRetirementsNotIssues) {
    // latency_count is the producer's own retirement evidence, kept separate
    // from the consumer's phase_done because the two can legitimately differ:
    // the consumer counts a request when it services it, the producer only when
    // it has observed the credit for it. That makes latency_count <= phase_done
    // an invariant of the pair, and this is the producer half of it.
    {
        ProducerTestFixture f(PI_MODE_LATENCY, 0, /*S=*/2, /*B=*/1, /*W=*/3, /*N=*/4);
        MockTensixTransport::on_poll_cache = [&]() {
            *f.consumed_mirror() = MockTensixTransport::last_published_produced;
        };

        packet_injection::Producer<PI_MODE_LATENCY, 0, MockTensixTransport, MockPacketSyncTransport> producer(
            f.args, f.l1_base);
        EXPECT_EQ(producer.run(), PI_ERROR_NONE);

        const pi_producer_result_t* r = f.result();
        EXPECT_EQ(r->latency_count, 4u);
        // Both are the measured retirement count, so they cannot diverge here;
        // the warmup requests are excluded from each.
        EXPECT_EQ(r->latency_count, r->measured_completed);
        EXPECT_EQ(r->warmup_completed, 3u);
        EXPECT_EQ(r->consumed_progress, 7u);
    }
    {
        // A run that fails mid-measurement publishes the prefix it retired, not
        // the count it issued, and never more than it retired.
        ProducerTestFixture f(PI_MODE_LATENCY, 0, /*S=*/2, /*B=*/1, /*W=*/0, /*N=*/4);
        MockTensixTransport::fail_timer_after_publication = 2;
        MockTensixTransport::on_poll_cache = [&]() {
            *f.consumed_mirror() = MockTensixTransport::last_published_produced;
        };

        packet_injection::Producer<PI_MODE_LATENCY, 0, MockTensixTransport, MockPacketSyncTransport> producer(
            f.args, f.l1_base);
        EXPECT_EQ(producer.run(), PI_ERROR_TIMER_RETRY_EXHAUSTED);

        const pi_producer_result_t* r = f.result();
        EXPECT_EQ(r->phase, (uint32_t)PI_PHASE_FAILED);
        EXPECT_EQ(r->latency_count, r->measured_completed);
        EXPECT_LE(r->latency_count, 4u);
        EXPECT_LE(r->latency_count, r->consumed_progress);
    }
}

TEST(PacketInjectionProducerStateModel, BackpressureStallAndCoalescedStarts) {
    ProducerTestFixture f(PI_MODE_BACKPRESSURE, 1, /*S=*/2, /*B=*/1, /*W=*/0, /*N=*/4);

    // Simulated timeline:
    // Req 0 starts at t=1000
    // Req 1 starts at t=1020
    // Ring full (2 requests in flight). Producer enters stall loop at t=2000.
    // Consumer releases coalesced credit of 2 at t=2300!
    // Producer detects progress: end=2300.
    // Stall duration = 2300 - 2000 = 300.
    // RTTs recorded using original starts:
    //   rtts[0] = 2300 - 1000 = 1300
    //   rtts[1] = 2300 - 1020 = 1280
    // Then req 2 & 3 proceed and final drain completes.
    MockTensixTransport::mock_time = 1000;
    int poll_count = 0;
    uint64_t req0_start = 0;
    uint64_t req1_start = 0;

    MockTensixTransport::on_poll_cache = [&]() {
        // When the producer has issued 2 requests, the ring is full (r=2, c_seen=0)
        if (MockTensixTransport::last_published_produced == 2) {
            poll_count++;
            if (poll_count == 1) {
                // Capture the original starts before slots are reused
                req0_start = f.starts()[0].wall;
                req1_start = f.starts()[1].wall;
                // Stall start
                MockTensixTransport::mock_time = 2000;
            } else if (poll_count >= 3) {
                // Progress observed! Coalesced jump of 2
                MockTensixTransport::mock_time = 2300;
                *f.consumed_mirror() = 2;
            }
        } else if (MockTensixTransport::last_published_produced == 4) {
            // Final drain
            *f.consumed_mirror() = 4;
        }
    };

    packet_injection::Producer<PI_MODE_BACKPRESSURE, 1, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_NONE);

    const pi_producer_result_t* r = f.result();
    EXPECT_EQ(r->measured_completed, 4u);
    EXPECT_EQ(r->stall_count, 1u);
    EXPECT_EQ(r->rtt_count, 2u);

    // Verify stall and RTT samples match the original request starts
    EXPECT_EQ(f.stalls()[0].wall, 300u);
    EXPECT_EQ(f.rtts()[0].wall, 2300ULL - req0_start);
    EXPECT_EQ(f.rtts()[1].wall, 2300ULL - req1_start);
}

TEST(PacketInjectionProducerStateModel, WarmupAndFinalDrainExcludeSamples) {
    // W > S forces warmup acquisition, not just drain.
    ProducerTestFixture f(PI_MODE_BACKPRESSURE, 0, /*S=*/2, /*B=*/1, /*W=*/4, /*N=*/2);

    constexpr uint64_t sentinel = 0xBADCAFEu;
    for (uint32_t i = 0; i < 2; ++i) {
        f.starts()[i].wall = sentinel;
        f.stalls()[i].wall = sentinel;
        f.rtts()[i].wall = sentinel;
    }
    int warmup_polls = 0;
    bool reached_measure = false;

    MockPacketSyncTransport::on_wait = [&]() {
        reached_measure = true;
        EXPECT_GE(warmup_polls, 3);
        for (uint32_t i = 0; i < 2; ++i) {
            EXPECT_EQ(f.starts()[i].wall, sentinel);
            EXPECT_EQ(f.stalls()[i].wall, sentinel);
            EXPECT_EQ(f.rtts()[i].wall, sentinel);
        }
    };

    MockTensixTransport::on_poll_cache = [&]() {
        if (MockTensixTransport::last_published_produced == 2 && *f.consumed_mirror() == 0) {
            warmup_polls++;
            if (warmup_polls >= 3) {
                *f.consumed_mirror() = 2;  // Release after stall in warmup
            }
        } else if (MockTensixTransport::last_published_produced == 4) {
            *f.consumed_mirror() = 4;
        } else if (MockTensixTransport::last_published_produced == 6) {
            *f.consumed_mirror() = 6;
        }
    };

    packet_injection::Producer<PI_MODE_BACKPRESSURE, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_NONE);

    const pi_producer_result_t* r = f.result();
    EXPECT_TRUE(reached_measure);
    EXPECT_EQ(MockPacketSyncTransport::notify_arrival_count, 1u);
    EXPECT_EQ(MockPacketSyncTransport::wait_count, 1u);
    EXPECT_EQ(r->warmup_completed, 4u);
    EXPECT_EQ(r->measured_completed, 2u);
    // Warmup stall was excluded from measured counts
    EXPECT_EQ(r->stall_count, 0u);
    EXPECT_EQ(r->rtt_count, 0u);
    for (uint32_t i = 0; i < 2; ++i) {
        EXPECT_EQ(f.stalls()[i].wall, sentinel);
        EXPECT_EQ(f.rtts()[i].wall, sentinel);
    }
}

TEST(PacketInjectionProducerStateModel, ThroughputCountsOnlyMeasuredStallEpisodes) {
    // At S=2 every third request finds the ring full. The warmup and the first
    // measured full ring wait for credit; the second measured one finds its
    // credit already returned, which is not an episode.
    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, /*S=*/2, /*B=*/1, /*W=*/4, /*N=*/6);

    int stalled_polls = 0;
    MockTensixTransport::on_poll_cache = [&]() {
        const uint32_t published = MockTensixTransport::last_published_produced;
        volatile uint32_t* consumed = f.consumed_mirror();
        if ((published == 2u && *consumed == 0u) || (published == 6u && *consumed == 4u)) {
            if (++stalled_polls == 3) {
                *consumed = published;
                stalled_polls = 0;
            }
        } else if (published == 4u || published == 8u || published == 10u) {
            *consumed = published;
        }
    };

    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    ASSERT_EQ(producer.run(), PI_ERROR_NONE);

    const pi_producer_result_t* r = f.result();
    EXPECT_EQ(r->warmup_completed, 4u);
    EXPECT_EQ(r->measured_completed, 6u);
    EXPECT_EQ(r->stall_count, 1u);
}

TEST(PacketInjectionProducerStateModel, CountsOnlyMeasuredProducedCounterWrites) {
    // At S=4, B=3 each phase publishes on a full batch, a full ring and the
    // phase end: warmup at 3, 4, 5 and the measured phase at 8, 9, 12.
    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, /*S=*/4, /*B=*/3, /*W=*/5, /*N=*/7);

    MockTensixTransport::on_poll_cache = [&]() { *f.consumed_mirror() = MockTensixTransport::last_published_produced; };

    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    ASSERT_EQ(producer.run(), PI_ERROR_NONE);

    EXPECT_EQ(MockTensixTransport::published_produced_history, (std::vector<uint32_t>{3, 4, 5, 8, 9, 12}));
    EXPECT_EQ(f.result()->measured_counter_writes, 3u);
}

TEST(PacketInjectionCoreSync, ProducerChecksStopBeforeMeasuredTraffic) {
    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, /*S=*/2, /*B=*/1, /*W=*/2, /*N=*/4);
    MockTensixTransport::on_poll_cache = [&]() { *f.consumed_mirror() = MockTensixTransport::last_published_produced; };
    MockPacketSyncTransport::on_wait = [&]() { *f.control_line() = PI_CONTROL_STOP; };

    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_STOPPED);

    EXPECT_EQ(MockTensixTransport::header_dsts.size(), f.args.W);
    EXPECT_EQ(f.result()->warmup_completed, f.args.W);
    EXPECT_EQ(f.result()->measured_completed, 0u);
    EXPECT_EQ(f.result()->throughput_start, 0u);
    EXPECT_EQ(MockTensixTransport::published_ready_history, (std::vector<uint32_t>{PI_READY_FAILED}));
}

TEST(PacketInjectionCoreSync, ProducerMapsBarrierFailuresToPacketErrors) {
    for (const bool arrival_failure : {false, true}) {
        SCOPED_TRACE(arrival_failure);
        ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, 1, 1, 0, 1);
        if (arrival_failure) {
            MockPacketSyncTransport::notify_arrival_result = false;
        } else {
            MockPacketSyncTransport::wait_result = false;
        }

        packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
            f.args, f.l1_base);
        const pi_error_t expected = arrival_failure ? PI_ERROR_LAYOUT_INVARIANT : PI_ERROR_STOPPED;
        EXPECT_EQ(producer.run(), expected);
        EXPECT_EQ(f.result()->failure.code, (uint32_t)expected);
        EXPECT_EQ(MockTensixTransport::published_ready_history, (std::vector<uint32_t>{PI_READY_FAILED}));
    }

    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, 1, 1, 0, 1);
    MockPacketSyncTransport::on_wait = [&]() { *f.control_line() = UINT32_MAX; };
    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_BAD_ARGS);
    EXPECT_EQ(f.result()->measured_completed, 0u);
    EXPECT_EQ(MockTensixTransport::published_ready_history, (std::vector<uint32_t>{PI_READY_FAILED}));
}

TEST(PacketInjectionProducerStateModel, FailedRunPreservesOnlyMeasuredSamplePrefixes) {
    for (bool measured_failure : {false, true}) {
        SCOPED_TRACE(measured_failure);
        ProducerTestFixture f(
            PI_MODE_BACKPRESSURE,
            1,
            /*S=*/2,
            /*B=*/1,
            /*W=*/measured_failure ? 0u : 4u,
            /*N=*/measured_failure ? 4u : 1u);
        MockTensixTransport::fail_timer_after_publication = 3;

        uint32_t polls = 0;
        MockTensixTransport::on_poll_cache = [&]() {
            if (MockTensixTransport::last_published_produced == 2 && ++polls >= 3) {
                *f.consumed_mirror() = 2;
            }
        };

        packet_injection::Producer<PI_MODE_BACKPRESSURE, 1, MockTensixTransport, MockPacketSyncTransport> producer(
            f.args, f.l1_base);
        EXPECT_EQ(producer.run(), PI_ERROR_TIMER_RETRY_EXHAUSTED);

        const pi_producer_result_t* r = f.result();
        EXPECT_EQ(r->phase, (uint32_t)PI_PHASE_FAILED);
        EXPECT_EQ(r->quiescent, 1u);
        EXPECT_EQ(r->warmup_completed, measured_failure ? 0u : 2u);
        EXPECT_EQ(r->measured_completed, measured_failure ? 2u : 0u);
        EXPECT_EQ(r->stall_count, measured_failure ? 1u : 0u);
        EXPECT_EQ(r->rtt_count, measured_failure ? 2u : 0u);
        EXPECT_EQ(r->measured_counter_writes, measured_failure ? 3u : 0u);
        EXPECT_EQ(r->failure.code, (uint32_t)PI_ERROR_TIMER_RETRY_EXHAUSTED);
        EXPECT_EQ(r->failure.worker_index, 0u);
        EXPECT_EQ(r->failure.actor, (uint32_t)PI_ACTOR_BRISC);
        EXPECT_EQ(MockTensixTransport::last_published_ready, (uint32_t)PI_READY_FAILED);
        EXPECT_TRUE(MockTensixTransport::pending_sources.empty());
    }
}

TEST(PacketInjectionProducerStateModel, RejectsABadLaunchBeforeTheFirstHeader) {
    // A launch the L1 derivation itself rejects leaves no result record: the
    // slice address it would be written through is the thing in doubt.
    const LaunchRejection undeliverable[] = {
        {"mode in the tag",
         [](pi_producer_args_t& a) { a.abi_tag = PI_ABI_TAG(PI_MODE_LATENCY, 0); },
         PI_ERROR_MODE_MISMATCH},
        {"zero tag", [](pi_producer_args_t& a) { a.abi_tag = 0u; }, PI_ERROR_MODE_MISMATCH},
        {"layout the ABI refuses", [](pi_producer_args_t& a) { a.S = 15u; }, PI_ERROR_BAD_ARGS},
    };
    for (const auto& c : undeliverable) {
        SCOPED_TRACE(c.name);
        ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, 2, 1, 0, 4);
        c.mutate(f.args);
        packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
            f.args, f.l1_base);
        EXPECT_EQ(producer.run(), c.expected);
        EXPECT_EQ(f.result()->abi_tag, 0u) << "identity written through an unproven slice";
        EXPECT_EQ(MockTensixTransport::u32_write_count, 0u);
        EXPECT_TRUE(MockTensixTransport::header_dsts.empty());
    }
}

TEST(PacketInjectionProducerStateModel, TileSpanMayFillL1ExactlyButNotExceedIt) {
    // The producer's L1 bound mirrors MEM_L1_SIZE rather than tightening it, so
    // both sides of it matter: a span one cache line past the end must be
    // refused, and a span that ends exactly at the end must still run.
    constexpr uint32_t kTensixL1Bytes = 1536u * 1024u;
    // P=2, T=1, S=2, B=4 gives two 640-byte worker slices after one cache line.
    constexpr uint32_t kTileBytes = 1344u;

    {
        ProducerTestFixture f(
            PI_MODE_THROUGHPUT,
            0,
            /*S=*/2,
            /*B=*/4,
            /*W=*/0,
            /*N=*/2,
            /*P=*/2,
            /*T=*/1,
            /*l1_tile_base=*/kTensixL1Bytes - 128u);
        ASSERT_EQ(f.layout.l1_worker_stride, 640u);
        ASSERT_EQ(f.layout.l1_tile_bytes, kTileBytes);
        ASSERT_GT(f.args.l1_tile_base + kTileBytes, kTensixL1Bytes);

        packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
            f.args, f.l1_base);
        EXPECT_EQ(producer.run(), PI_ERROR_CAPACITY);
        // Refused inside the L1 derivation, so no result record and no traffic.
        EXPECT_EQ(f.result()->abi_tag, 0u);
        EXPECT_EQ(MockTensixTransport::u32_write_count, 0u);
        EXPECT_TRUE(MockTensixTransport::header_dsts.empty());
    }
    {
        ProducerTestFixture f(
            PI_MODE_THROUGHPUT,
            0,
            /*S=*/2,
            /*B=*/4,
            /*W=*/0,
            /*N=*/2,
            /*P=*/2,
            /*T=*/1,
            /*l1_tile_base=*/kTensixL1Bytes - kTileBytes);
        ASSERT_EQ(f.args.l1_tile_base + kTileBytes, kTensixL1Bytes);

        MockTensixTransport::on_poll_cache = [&]() {
            *f.consumed_mirror() = MockTensixTransport::last_published_produced;
        };

        packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
            f.args, f.l1_base);
        EXPECT_EQ(producer.run(), PI_ERROR_NONE);
        EXPECT_EQ(f.result()->measured_completed, 2u);
        EXPECT_EQ(f.result()->phase, (uint32_t)PI_PHASE_DONE);
    }
}

TEST(PacketInjectionProducerStateModel, RecordsAnUnannouncableFailureWithoutPublishingIt) {
    // Once the slice is proven the identity is in L1, so these failures are
    // legible to the host. None of them can be announced: the ready slot they
    // would publish on is either another worker's or out of the arena.
    const LaunchRejection unannouncable[] = {
        {"worker index past worker_count", [](pi_producer_args_t& a) { a.worker_index = 1u; }, PI_ERROR_BAD_ARGS},
        {"peer x past the NoC node-id field", [](pi_producer_args_t& a) { a.peer.x = 64u; }, PI_ERROR_BAD_PLACEMENT},
        {"peer y past the NoC node-id field", [](pi_producer_args_t& a) { a.peer.y = 64u; }, PI_ERROR_BAD_PLACEMENT},
    };
    for (const auto& c : unannouncable) {
        SCOPED_TRACE(c.name);
        ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, 2, 1, 0, 4);
        c.mutate(f.args);
        packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
            f.args, f.l1_base);
        EXPECT_EQ(producer.run(), c.expected);

        const pi_producer_result_t* r = f.result();
        EXPECT_EQ(r->abi_tag, PI_ABI_TAG(PI_MODE_THROUGHPUT, 0));
        EXPECT_EQ(r->run_nonce, f.args.run_nonce);
        EXPECT_EQ(r->failure.code, (uint32_t)c.expected);
        EXPECT_EQ(r->failure.actor, (uint32_t)PI_ACTOR_BRISC);
        EXPECT_EQ(MockTensixTransport::u32_write_count, 0u) << "announced an unreachable failure";
        EXPECT_NE(MockTensixTransport::last_published_ready, (uint32_t)PI_READY_FAILED);
        EXPECT_TRUE(MockTensixTransport::header_dsts.empty());
    }
}

// A write wait, bounded by the phase timeout, that expires fails the run as a
// recorded TIMEOUT; the failure is announced once a later wait drains.
TEST(PacketInjectionProducerStateModel, AnExpiredWriteWaitIsARecordedTimeout) {
    ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, /*S=*/2, /*B=*/1, /*W=*/0, /*N=*/4);
    MockTensixTransport::on_poll_cache = [&]() { *f.consumed_mirror() = MockTensixTransport::last_published_produced; };
    MockTensixTransport::expire_barrier = 2;
    packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
        f.args, f.l1_base);
    EXPECT_EQ(producer.run(), PI_ERROR_TIMEOUT);

    const pi_producer_result_t* r = f.result();
    EXPECT_EQ(r->failure.code, (uint32_t)PI_ERROR_TIMEOUT);
    EXPECT_EQ(r->failure.phase, (uint32_t)PI_PHASE_MEASURE);
    EXPECT_EQ(r->phase, (uint32_t)PI_PHASE_FAILED);
    EXPECT_EQ(MockTensixTransport::last_barrier_budget, f.args.timeout_ticks);
    EXPECT_EQ(MockTensixTransport::last_published_ready, (uint32_t)PI_READY_FAILED);
}

TEST(PacketInjectionProducerStateModel, AnnouncesAControlFailureOnTheReadySlot) {
    // A control failure reaches a proven ready slot, so it is published as well
    // as recorded.
    const LaunchRejection announced[] = {
        {"poll cadence not a power of two", [](pi_producer_args_t& a) { a.poll_cadence = 3u; }, PI_ERROR_BAD_ARGS},
        {"zero poll cadence", [](pi_producer_args_t& a) { a.poll_cadence = 0u; }, PI_ERROR_BAD_ARGS},
        {"zero timeout", [](pi_producer_args_t& a) { a.timeout_ticks = 0u; }, PI_ERROR_BAD_ARGS},
        {"timeout outside the timer half-range",
         [](pi_producer_args_t& a) { a.timeout_ticks = 1ULL << 63; },
         PI_ERROR_BAD_ARGS},
        {"W + N overflows",
         [](pi_producer_args_t& a) {
             a.W = UINT64_MAX;
             a.N = 4u;
         },
         PI_ERROR_BAD_ARGS},
    };
    for (const auto& c : announced) {
        SCOPED_TRACE(c.name);
        ProducerTestFixture f(PI_MODE_THROUGHPUT, 0, 2, 1, 0, 4);
        c.mutate(f.args);
        packet_injection::Producer<PI_MODE_THROUGHPUT, 0, MockTensixTransport, MockPacketSyncTransport> producer(
            f.args, f.l1_base);
        EXPECT_EQ(producer.run(), c.expected);

        const pi_producer_result_t* r = f.result();
        EXPECT_EQ(r->failure.code, (uint32_t)c.expected);
        EXPECT_EQ(r->failure.phase, (uint32_t)PI_PHASE_INIT);
        EXPECT_EQ(r->phase, (uint32_t)PI_PHASE_FAILED);
        EXPECT_EQ(r->quiescent, 1u);
        EXPECT_EQ(MockTensixTransport::last_published_ready, (uint32_t)PI_READY_FAILED);
        EXPECT_GT(MockTensixTransport::u32_write_count, 0u);
        EXPECT_TRUE(MockTensixTransport::header_dsts.empty());
    }
}

// --------------------------------------------------------------------------
// Consumer state model
// --------------------------------------------------------------------------

// Stands in for the X280's fences, mtime, TLB windows, and pinned way. One
// credit window per tile, each a fixed span of a host buffer, so a store is
// decoded back to the worker and L1 record it was aimed at.
struct MockX280Transport {
    static constexpr uint64_t kWindowSpan = 0x4000u;

    static inline uint64_t mock_mtime = 50000;
    static inline uint64_t mock_rdcycle = 7000;
    static inline std::vector<uint32_t> credit_history;
    static inline std::vector<uint32_t> control_history;
    static inline std::vector<uint32_t> timer_window_tiles;
    static inline std::vector<uint32_t> credit_window_tiles;
    static inline uintptr_t window_base_addr = 0;
    static inline uintptr_t timer_window_addr = 0;
    static inline uintptr_t pinned_addr = 0;
    static inline uintptr_t arena_addr = 0;
    // Admission code the cache reports for the baseline way masks; a non-
    // latency run requires the baseline, so a stale prepared mask fails it.
    static inline uint32_t baseline_policy = X280_PINNED_INPUT_ADMITTED;
    // Non-zero once the host has asked the run to stop.
    static inline int stop_requested = 0;
    // Tile whose credit-window configuration fails, or UINT32_MAX for none;
    // and the timer window's own result. Both readbacks succeed by default.
    static inline uint32_t credit_window_fail_tile = UINT32_MAX;
    static inline int timer_window_result = 0;
    static inline pi_layout_t layout{};
    static inline std::vector<std::string> lifecycle;

    static inline std::function<void(uint32_t worker, uint64_t record, uint32_t val)> on_store = nullptr;
    static inline std::function<void()> on_mtime = nullptr;
    static inline int clear_result = 0;
    static inline uint64_t teardown_result = X280_PINNED_INPUT_RESULT_TORN_DOWN;
    static inline uint64_t prepare_result = X280_PINNED_INPUT_RESULT_PREPARED;

    static void reset() {
        mock_mtime = 50000;
        mock_rdcycle = 7000;
        credit_history.clear();
        control_history.clear();
        timer_window_tiles.clear();
        credit_window_tiles.clear();
        window_base_addr = 0;
        timer_window_addr = 0;
        pinned_addr = 0;
        arena_addr = 0;
        baseline_policy = X280_PINNED_INPUT_ADMITTED;
        stop_requested = 0;
        credit_window_fail_tile = UINT32_MAX;
        timer_window_result = 0;
        layout = pi_layout_t{};
        lifecycle.clear();
        on_store = nullptr;
        on_mtime = nullptr;
        clear_result = 0;
        teardown_result = X280_PINNED_INPUT_RESULT_TORN_DOWN;
        prepare_result = X280_PINNED_INPUT_RESULT_PREPARED;
    }

    static void fence_full() {}
    static void fence_load() {}
    static uint64_t mtime() {
        if (on_mtime) {
            on_mtime();
        }
        mock_mtime += 100;
        return mock_mtime;
    }
    static uint64_t rdcycle() {
        mock_rdcycle += 3;
        return mock_rdcycle;
    }
    static int check_stop(uint64_t, uint64_t) { return stop_requested; }
    static uint64_t pinned_base() { return (uint64_t)pinned_addr; }
    static uint32_t check_pinned_baseline() { return baseline_policy; }
    // The base every LIM address is derived from: the fixed arena on device,
    // a heap buffer under a host test.
    static uint64_t arena_base() { return (uint64_t)arena_addr; }
    static int configure_credit_window(uint32_t index, uint32_t noc_x, uint32_t noc_y) {
        credit_window_tiles.push_back((noc_x << 16) | noc_y);
        const uint32_t tile = index - X280_TLB_PACKET_INJECTION_CREDIT_BASE;
        return tile == credit_window_fail_tile ? -1 : 0;
    }
    static int configure_timer_window(uint32_t, uint32_t noc_x, uint32_t noc_y) {
        timer_window_tiles.push_back((noc_x << 16) | noc_y);
        return timer_window_result;
    }
    static int clear_window(uint32_t) {
        lifecycle.push_back("clear");
        return clear_result;
    }
    static volatile void* window_base(uint32_t index) {
        if (index == X280_TLB_PACKET_INJECTION_TIMER) {
            return (volatile void*)timer_window_addr;
        }
        const uint32_t tile = index - X280_TLB_PACKET_INJECTION_CREDIT_BASE;
        return (volatile void*)(window_base_addr + (uintptr_t)tile * kWindowSpan);
    }
    static uint64_t prepare_pinned(uint64_t, uint64_t, const x280_pinned_input_region_t*) {
        lifecycle.push_back("pin");
        return prepare_result;
    }
    static uint64_t teardown_pinned(uint64_t, uint64_t, const x280_pinned_input_region_t*) {
        lifecycle.push_back("unpin");
        return teardown_result;
    }
    static void counter_store(volatile uint32_t* addr, uint32_t val) {
        if (addr) {
            *addr = val;
        }

        const uintptr_t offset = (uintptr_t)addr - window_base_addr;
        const uint32_t tile = (uint32_t)(offset / kWindowSpan);
        const uint64_t within = offset % kWindowSpan;
        EXPECT_GE(within, (uint64_t)layout.l1_tile_base + layout.l1_worker_base_offset);
        const uint64_t worker_region = within - layout.l1_tile_base - layout.l1_worker_base_offset;
        const uint32_t risc = (uint32_t)(worker_region / layout.l1_worker_stride);
        const uint64_t record = worker_region % layout.l1_worker_stride;
        const uint32_t worker = tile * layout.risc_per_tile + risc;

        // The decode is the inverse of the mapping the consumer wrote through,
        // so hold it against the helpers rather than trusting the arithmetic.
        EXPECT_EQ(pi_worker_tile_index(&layout, worker), tile);
        EXPECT_EQ(pi_worker_risc_index(&layout, worker), risc);
        EXPECT_EQ(pi_worker_l1_base(&layout, worker), within - record);

        if (record == PI_L1_CONTROL_OFFSET) {
            control_history.push_back(val);
            if (val == PI_CONTROL_STOP) {
                lifecycle.push_back("stop");
            }
        } else if (record == PI_L1_CONSUMED_OFFSET) {
            credit_history.push_back(val);
        } else {
            ADD_FAILURE() << "store to an unmapped L1 record at +" << record;
        }
        if (on_store) {
            on_store(worker, record, val);
        }
    }
};

pi_layout_t ComputeLayoutOrDie(uint32_t P, uint32_t T, uint32_t S, uint64_t N, uint32_t mode) {
    pi_layout_t layout{};
    EXPECT_EQ(pi_compute_layout(P, T, S, 1u, N, mode, 0x1000u, &layout).code, (uint32_t)PI_LAYOUT_OK);
    return layout;
}

// Holds the whole arena in a host buffer and relocates the consumer onto it.
// Every address is arena base plus a pi_layout_t offset, exactly as the X280
// derives it on silicon, the args record at the arena base included.
struct ConsumerTestFixture {
    pi_layout_t layout;
    std::vector<uint8_t> arena_storage;
    // Where the arena actually lives, which every accessor below reads through.
    uintptr_t storage_base;
    // The base the consumer derives its addresses from. Normally the same, but
    // a test can move it to hand the consumer a base that disagrees with where
    // the args and slots really are.
    uintptr_t consumer_arena_base;
    pi_x280_args_t& args;
    std::vector<uint8_t> tile_l1_storage;
    std::vector<uint64_t> pinned;
    uint64_t run_value = 0x50494E4A1234ULL;
    uint32_t P_ = 0;
    uint32_t T_ = 0;

    ConsumerTestFixture(
        uint32_t P, uint32_t T, uint32_t S, uint32_t C, uint32_t K, uint64_t N, uint32_t mode = PI_MODE_THROUGHPUT) :
        layout(ComputeLayoutOrDie(P, T, S, N, mode)),
        arena_storage((size_t)layout.arena_bytes + 64u, 0),
        storage_base((uintptr_t)pi_align64((uint64_t)(uintptr_t)arena_storage.data())),
        consumer_arena_base(storage_base),
        args(*(pi_x280_args_t*)(storage_base + layout.args_offset)),
        tile_l1_storage((size_t)T * MockX280Transport::kWindowSpan, 0),
        pinned((size_t)P * N, 0),
        P_(P),
        T_(T) {
        MockX280Transport::reset();
        MockPacketSyncTransport::reset();
        relocate_consumer_arena(storage_base);
        MockX280Transport::window_base_addr = (uintptr_t)tile_l1_storage.data();
        MockX280Transport::timer_window_addr = 0x40000000u;
        MockX280Transport::pinned_addr = (uintptr_t)pinned.data();
        MockX280Transport::layout = layout;

        args.abi_tag = PI_ABI_TAG(mode, 0u);
        args.worker_count = (uint16_t)P;
        args.tile_count = (uint16_t)T;
        args.S = S;
        args.B = 1u;
        args.C = C;
        args.K = K;
        args.poll_cadence = PI_DEFAULT_POLL_CADENCE;
        args.l1_tile_base = layout.l1_tile_base;
        args.W = 0u;
        args.N = N;
        args.x280_timeout_ticks = 1000000ULL;
        args.run_nonce = 0x1234ABCDu;

        for (uint32_t t = 0; t < T; ++t) {
            *coord(t) = {8u + t % 10u, 3u + t / 10u};
        }
    }

    x280_noc_coord_t* coord(uint32_t t) const {
        return (x280_noc_coord_t*)(storage_base + layout.tile_coords_offset + (uint64_t)t * sizeof(x280_noc_coord_t));
    }

    // Gives the consumer the group the host selector would for a table that
    // fills [start, end].
    void set_release_group(x280_noc_coord_t start, x280_noc_coord_t end) {
        args.release_num_dests = T_;
        args.release_rect = {start, end};
    }
    pi_consumer_slot_t* slot(uint32_t w) const {
        return (pi_consumer_slot_t*)(storage_base + layout.consumer_slots_offset +
                                     (uint64_t)w * sizeof(pi_consumer_slot_t));
    }
    pi_counter_slot_t* produced(uint32_t w) const {
        return (pi_counter_slot_t*)(storage_base + layout.produced_offset + (uint64_t)w * sizeof(pi_counter_slot_t));
    }
    pi_ready_slot_t* ready(uint32_t w) const {
        return (pi_ready_slot_t*)(storage_base + layout.ready_offset + (uint64_t)w * sizeof(pi_ready_slot_t));
    }
    pi_run_result_t* run_res() const { return (pi_run_result_t*)(storage_base + layout.run_result_offset); }
    pi_host_control_t* hctrl() const { return (pi_host_control_t*)(storage_base + layout.host_control_offset); }
    const uint64_t* exported() const { return (const uint64_t*)(storage_base + layout.latency_export_offset); }

    void acknowledge_collection() const { hctrl()->collect_ack_run_value = run_value; }

    // Points the consumer at `base` without moving the storage the accessors
    // read. Sets both the constructor argument and the Transport property, so
    // it holds whichever of the two the consumer takes its base from.
    void relocate_consumer_arena(uintptr_t base) {
        consumer_arena_base = base;
        MockX280Transport::arena_addr = base;
    }

    // Emulates every producer: warmup begins immediately, barrier release
    // publishes measured traffic, and credits advance the bounded rings.
    void wire_auto_producer() {
        std::shared_ptr<std::vector<uint32_t>> last_credited = std::make_shared<std::vector<uint32_t>>(P_, 0u);
        std::shared_ptr<bool> released = std::make_shared<bool>(false);

        for (uint32_t i = 0; i < P_; ++i) {
            produced(i)->value = (uint32_t)std::min<uint64_t>(args.S, args.W);
        }
        MockPacketSyncTransport::on_notify_group = [this, released]() {
            if (std::find(
                    MockX280Transport::control_history.begin(),
                    MockX280Transport::control_history.end(),
                    (uint32_t)PI_CONTROL_STOP) != MockX280Transport::control_history.end()) {
                return;
            }
            *released = true;
            for (uint32_t i = 0; i < P_; ++i) {
                produced(i)->value = (uint32_t)(args.W + std::min<uint64_t>(args.S, args.N));
            }
        };

        MockX280Transport::on_store = [this, last_credited, released](uint32_t w, uint64_t record, uint32_t val) {
            if (record == PI_L1_CONTROL_OFFSET) {
                return;
            }

            (*last_credited)[w] = val;
            uint32_t next_p = val + args.S;
            const uint64_t total = *released ? args.W + args.N : args.W;
            if (next_p > total) {
                next_p = (uint32_t)total;
            }
            // A volatile slot, stored only when it advances.
            if (next_p > produced(w)->value) {  // NOLINT(readability-use-std-min-max)
                produced(w)->value = next_p;
            }

            bool all_done = true;
            for (uint32_t i = 0; i < P_; ++i) {
                if ((*last_credited)[i] != args.W + args.N) {
                    all_done = false;
                    break;
                }
            }
            if (all_done) {
                for (uint32_t i = 0; i < P_; ++i) {
                    ready(i)->value = PI_READY_DONE;
                }
                acknowledge_collection();
            }
        };
    }
};

// The host's export accounting. Eight bytes are exported per measured
// completion, but only once the measured phase has begun: begin_phase() zeroes
// phase_done at each phase boundary, so a run that fails before the measured
// phase starts leaves the WARMUP counts in phase_done while exporting nothing.
// The failure phase is what separates the two cases, since PI_PHASE_MEASURE is
// the first phase recorded after the counters are rearmed.
::testing::AssertionResult ExportAccountingHolds(const ConsumerTestFixture& tf) {
    const pi_run_result_t* r = tf.run_res();
    uint64_t phase_done_total = 0;
    for (uint32_t w = 0; w < tf.P_; ++w) {
        phase_done_total += tf.slot(w)->phase_done;
    }
    const bool measured_started =
        r->failure.code == (uint32_t)PI_ERROR_NONE || r->failure.phase >= (uint32_t)PI_PHASE_MEASURE;
    const uint64_t expected = measured_started ? 8u * phase_done_total : 0u;

    if (r->export_valid_bytes != expected) {
        return ::testing::AssertionFailure()
               << "export_valid_bytes " << r->export_valid_bytes << " != expected " << expected << " (phase_done total "
               << phase_done_total << ", failure phase " << r->failure.phase << ", measured_started "
               << measured_started << ")";
    }
    if (r->export_valid_bytes > tf.layout.latency_export_bytes) {
        return ::testing::AssertionFailure() << "export_valid_bytes " << r->export_valid_bytes
                                             << " exceeds the export region " << tf.layout.latency_export_bytes;
    }
    return ::testing::AssertionSuccess();
}

TEST(PacketInjectionConsumerStateModel, RoundRobinServiceOrderAndVisitCapK) {
    // 2 workers, K=2, C=1, N=4
    ConsumerTestFixture tf(2, 1, 16, 1, 2, 4);
    tf.wire_auto_producer();

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_NONE);

    // Both workers completed exactly 4 requests
    EXPECT_EQ(tf.slot(0)->consumed, 4u);
    EXPECT_EQ(tf.slot(1)->consumed, 4u);
    EXPECT_EQ(tf.slot(0)->phase_done, 4u);
    EXPECT_EQ(tf.slot(1)->phase_done, 4u);
    EXPECT_EQ(tf.run_res()->measured_completed, 8u);
    EXPECT_EQ(tf.run_res()->worker_count, 2u);
    EXPECT_EQ(tf.run_res()->state, (uint32_t)PI_PHASE_DONE);

    // With K=2, each visit consumed 2 slots, round-robining between worker 0 and worker 1:
    // W0 receives credits 1, 2 -> W1 receives 1, 2 -> W0 receives 3, 4 -> W1 receives 3, 4
    EXPECT_EQ(MockX280Transport::credit_history, (std::vector<uint32_t>{1, 2, 1, 2, 3, 4, 3, 4}));
}

// A produced counter further ahead of the consumer than the ring is deep,
// directly or by wrapping 2^32, is a COUNTER_DISTANCE on that worker before any
// slot is read.
TEST(PacketInjectionConsumerStateModel, ProducedCounterBeyondTheRingIsRefused) {
    for (const uint32_t produced : {17u, 0xFFFFFFFFu}) {
        ConsumerTestFixture tf(1, 1, 16, 1, 1, 4);
        tf.wire_auto_producer();
        const auto release = MockPacketSyncTransport::on_notify_group;
        MockPacketSyncTransport::on_notify_group = [&tf, release, produced]() {
            release();
            tf.produced(0)->value = produced;
        };

        packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
            0, tf.run_value, tf.args);
        EXPECT_EQ(consumer.run(), PI_ERROR_COUNTER_DISTANCE) << "produced " << produced;
        EXPECT_EQ(tf.run_res()->failure.code, (uint32_t)PI_ERROR_COUNTER_DISTANCE) << "produced " << produced;
        EXPECT_EQ(tf.run_res()->failure.worker_index, 0u) << "produced " << produced;
        EXPECT_EQ(tf.slot(0)->consumed, 0u) << "produced " << produced;
    }
}

TEST(PacketInjectionConsumerStateModel, CreditThresholdC3WithK5Gives3Plus2) {
    // 1 worker, K=5, C=3, N=5
    ConsumerTestFixture tf(1, 1, 16, 3, 5, 5);
    tf.wire_auto_producer();

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_NONE);

    // Group of 3 at threshold C, then remaining 2 at visit end (3 + 2)
    EXPECT_EQ(MockX280Transport::credit_history, (std::vector<uint32_t>{3, 5}));
    EXPECT_EQ(tf.slot(0)->consumed, 5u);
    EXPECT_EQ(tf.slot(0)->credited, 5u);
}

TEST(PacketInjectionConsumerStateModel, ShortSnapshotEarlyFlush) {
    // Available = 2 < C (C=3, K=5)
    ConsumerTestFixture tf(1, 1, 16, 3, 5, 2);
    tf.wire_auto_producer();

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_NONE);

    // Credit returned immediately at snapshot exhaustion
    EXPECT_EQ(MockX280Transport::credit_history, (std::vector<uint32_t>{2}));
    EXPECT_EQ(tf.slot(0)->consumed, 2u);
}

TEST(PacketInjectionConsumerStateModel, S1ConsumerSingleSlot) {
    // S=1, K=1, C=1, N=3
    ConsumerTestFixture tf(1, 1, 1, 1, 1, 3);
    tf.wire_auto_producer();

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_NONE);

    EXPECT_EQ(MockX280Transport::credit_history, (std::vector<uint32_t>{1, 2, 3}));
    EXPECT_EQ(tf.slot(0)->consumed, 3u);
}

TEST(PacketInjectionConsumerStateModel, PairedTilesUseOwnVirtualCreditAndTimerTargets) {
    ConsumerTestFixture tf(4, 2, 4, 1, 1, 1, PI_MODE_LATENCY);
    *tf.coord(0) = {1u, 2u};
    *tf.coord(1) = {10u, 4u};
    tf.wire_auto_producer();
    g_tensix_live_timer_mock_read = [](uintptr_t) -> uint32_t { return 1; };

    packet_injection::X280Consumer<PI_MODE_LATENCY, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    const pi_error_t err = consumer.run();
    g_tensix_live_timer_mock_read = nullptr;

    EXPECT_EQ(err, PI_ERROR_NONE);
    const std::vector<uint32_t> targets{(1u << 16) | 2u, (10u << 16) | 4u};
    EXPECT_EQ(MockX280Transport::credit_window_tiles, targets);
    EXPECT_EQ(MockX280Transport::timer_window_tiles, targets);
    for (uint32_t w = 0; w < 4; ++w) {
        EXPECT_EQ(tf.slot(w)->phase_done, 1u) << "worker " << w;
        EXPECT_EQ(tf.slot(w)->consumed, 1u) << "worker " << w;
    }
    EXPECT_EQ(tf.run_res()->measured_completed, 4u);
    EXPECT_EQ(tf.run_res()->export_valid_bytes, 32u);
    EXPECT_TRUE(ExportAccountingHolds(tf));
}

TEST(PacketInjectionCoreSync, MaximumLayoutsBuildOneDenseReleaseGroup) {
    struct Shape {
        uint32_t P;
        uint32_t T;
    };
    for (const Shape shape : {Shape{PI_MAX_TILES, PI_MAX_TILES}, Shape{PI_MAX_WORKERS, PI_MAX_TILES}}) {
        SCOPED_TRACE(shape.P);
        ConsumerTestFixture tf(shape.P, shape.T, 1, 1, 1, 1);
        tf.set_release_group({8u, 3u}, {17u, 3u + (shape.T - 1u) / 10u});
        tf.wire_auto_producer();

        packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
            0, tf.run_value, tf.args);
        ASSERT_EQ(consumer.run(), PI_ERROR_NONE);

        EXPECT_EQ(MockPacketSyncTransport::wait_for_arrivals_count, 1u);
        EXPECT_EQ(MockPacketSyncTransport::notify_group_count, 1u);
        ASSERT_EQ(MockPacketSyncTransport::arrivals.size(), shape.P);
        ASSERT_EQ(MockPacketSyncTransport::groups.size(), 1u);
        EXPECT_EQ(MockPacketSyncTransport::last_roster.endpoint_count, 0u);

        const auto& group = MockPacketSyncTransport::groups[0];
        EXPECT_EQ(group.num_dests, shape.T);
        EXPECT_EQ(group.rect.start.x, tf.args.release_rect.start.x);
        EXPECT_EQ(group.rect.end.y, tf.args.release_rect.end.y);
        EXPECT_EQ(tf.run_res()->released_group_dests, shape.T);
        EXPECT_EQ(tf.run_res()->released_endpoints, 0u);
        EXPECT_EQ(group.addr, tf.layout.l1_tile_base + tf.layout.l1_release_offset);
        for (uint32_t w = 0; w < shape.P; ++w) {
            const auto& arrival = MockPacketSyncTransport::arrivals[w];
            EXPECT_EQ(arrival.worker_index, w);
            EXPECT_EQ(
                arrival.addr,
                core_sync_slot_addr((uint32_t)(tf.consumer_arena_base + tf.layout.sync_arrivals_offset), w));

            const uint32_t tile = pi_worker_tile_index(&tf.layout, w);
            const uint32_t x = tf.coord(tile)->x;
            const uint32_t y = tf.coord(tile)->y;
            EXPECT_GE(x, group.rect.start.x);
            EXPECT_LE(x, group.rect.end.x);
            EXPECT_GE(y, group.rect.start.y);
            EXPECT_LE(y, group.rect.end.y);
        }
        if (shape.P == 2u * shape.T) {
            for (uint32_t tile = 0; tile < shape.T; ++tile) {
                EXPECT_EQ(pi_worker_tile_index(&tf.layout, 2u * tile), tile);
                EXPECT_EQ(pi_worker_tile_index(&tf.layout, 2u * tile + 1u), tile);
            }
        }
        EXPECT_TRUE(MockX280Transport::control_history.empty());
    }
}

TEST(PacketInjectionCoreSync, SparseTilesReleaseOrderedEndpoints) {
    ConsumerTestFixture tf(8, 4, 1, 1, 1, 1);
    const x280_noc_coord_t coords[] = {{8u, 3u}, {11u, 3u}, {8u, 7u}, {12u, 9u}};
    for (uint32_t t = 0; t < 4u; ++t) {
        *tf.coord(t) = coords[t];
    }
    tf.wire_auto_producer();

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    ASSERT_EQ(consumer.run(), PI_ERROR_NONE);

    EXPECT_TRUE(MockPacketSyncTransport::groups.empty());
    ASSERT_EQ(MockPacketSyncTransport::endpoints.size(), 4u);
    for (uint32_t t = 0; t < 4u; ++t) {
        const auto& endpoint = MockPacketSyncTransport::endpoints[t];
        EXPECT_EQ(endpoint.noc.x, coords[t].x);
        EXPECT_EQ(endpoint.noc.y, coords[t].y);
        EXPECT_EQ(endpoint.addr, tf.layout.l1_tile_base + tf.layout.l1_release_offset);
        EXPECT_FALSE(endpoint.same_tile);
    }
    EXPECT_EQ(tf.run_res()->released_group_dests, 0u);
    EXPECT_EQ(tf.run_res()->released_endpoints, 4u);

    for (uint32_t w = 0; w < tf.P_; ++w) {
        const uint32_t tile = pi_worker_tile_index(&tf.layout, w);
        uint32_t matching_endpoints = 0u;
        for (const auto& endpoint : MockPacketSyncTransport::endpoints) {
            matching_endpoints += endpoint.noc.x == tf.coord(tile)->x && endpoint.noc.y == tf.coord(tile)->y;
        }
        EXPECT_EQ(matching_endpoints, 1u) << "worker " << w;
    }
}

TEST(PacketInjectionCoreSync, DenseTilesWithoutAHostGroupReleaseEndpoints) {
    // The tiles fill their box, but the host chose no group: the consumer
    // executes that choice rather than selecting a multicast itself. A one-tile
    // group would still be a broadcast, which every tile but Tensix drops, so
    // each tile takes an endpoint write.
    ConsumerTestFixture tf(4, 4, 1, 1, 1, 1);
    const x280_noc_coord_t coords[] = {{1u, 2u}, {2u, 2u}, {1u, 3u}, {2u, 3u}};
    for (uint32_t t = 0; t < 4u; ++t) {
        *tf.coord(t) = coords[t];
    }
    ASSERT_EQ(tf.args.release_num_dests, 0u);
    tf.wire_auto_producer();
    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    ASSERT_EQ(consumer.run(), PI_ERROR_NONE);
    EXPECT_EQ(MockPacketSyncTransport::notify_group_count, 1u);
    EXPECT_TRUE(MockPacketSyncTransport::groups.empty());
    ASSERT_EQ(MockPacketSyncTransport::endpoints.size(), 4u);
    for (uint32_t t = 0; t < 4u; ++t) {
        const auto& endpoint = MockPacketSyncTransport::endpoints[t];
        EXPECT_EQ(endpoint.noc.x, coords[t].x);
        EXPECT_EQ(endpoint.noc.y, coords[t].y);
        EXPECT_EQ(endpoint.addr, tf.layout.l1_tile_base + tf.layout.l1_release_offset);
        EXPECT_FALSE(endpoint.same_tile);
    }
}

TEST(PacketInjectionCoreSync, GroupInconsistentWithTheTileTableIsRefused) {
    using Mutation = std::pair<const char*, void (*)(ConsumerTestFixture&)>;
    const Mutation mutations[] = {
        {"count short of the table", [](ConsumerTestFixture& tf) { tf.args.release_num_dests -= 1u; }},
        {"tile outside the rectangle", [](ConsumerTestFixture& tf) { tf.args.release_rect.end.x -= 1u; }},
        {"reversed rectangle",
         [](ConsumerTestFixture& tf) { std::swap(tf.args.release_rect.start, tf.args.release_rect.end); }},
        {"corner outside six bits", [](ConsumerTestFixture& tf) { tf.args.release_rect.end.y = 64u; }},
        {"rectangle without a group", [](ConsumerTestFixture& tf) { tf.args.release_num_dests = 0u; }},
    };
    for (const auto& [name, mutate] : mutations) {
        SCOPED_TRACE(name);
        ConsumerTestFixture tf(4, 4, 1, 1, 1, 1);
        tf.set_release_group({8u, 3u}, {11u, 3u});
        mutate(tf);
        tf.wire_auto_producer();
        packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
            0, tf.run_value, tf.args);
        EXPECT_EQ(consumer.run(), PI_ERROR_BAD_ARGS);
        EXPECT_EQ(MockPacketSyncTransport::notify_group_count, 0u);
    }
}

// A failure before the measured release still releases the producers, through
// the same validated host group.
TEST(PacketInjectionCoreSync, RecoveryReleasesTheHostGroup) {
    ConsumerTestFixture tf(4, 4, 1, 1, 1, 1);
    tf.set_release_group({8u, 3u}, {11u, 3u});
    MockPacketSyncTransport::wait_for_arrivals_result = false;
    for (uint32_t w = 0; w < tf.P_; ++w) {
        tf.ready(w)->value = (uint32_t)PI_READY_FAILED;
    }
    MockX280Transport::on_mtime = [&]() {
        if (tf.run_res()->data_quiescent) {
            tf.acknowledge_collection();
        }
    };
    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_NE(consumer.run(), PI_ERROR_NONE);
    EXPECT_EQ(MockPacketSyncTransport::notify_group_count, 1u);
    ASSERT_EQ(MockPacketSyncTransport::groups.size(), 1u);
    EXPECT_EQ(MockPacketSyncTransport::groups[0].num_dests, 4u);
    EXPECT_EQ(MockPacketSyncTransport::groups[0].rect.end.x, 11u);
    EXPECT_TRUE(MockPacketSyncTransport::endpoints.empty());
}

TEST(PacketInjectionCoreSync, WarmupCreditsDrainBeforeTheOnlyDirectorBarrier) {
    ConsumerTestFixture tf(2, 1, 2, 1, 1, 2);
    tf.args.W = 3u;
    tf.wire_auto_producer();
    bool observed_drained_warmup = false;
    MockPacketSyncTransport::on_wait_for_arrivals = [&]() {
        observed_drained_warmup = true;
        for (uint32_t w = 0; w < tf.P_; ++w) {
            EXPECT_EQ(tf.slot(w)->phase_done, tf.args.W);
            EXPECT_EQ(tf.slot(w)->consumed, tf.args.W);
            EXPECT_EQ(tf.slot(w)->credited, tf.args.W);
            EXPECT_EQ(tf.produced(w)->value, tf.args.W);
        }
    };

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    ASSERT_EQ(consumer.run(), PI_ERROR_NONE);

    EXPECT_TRUE(observed_drained_warmup);
    EXPECT_EQ(MockPacketSyncTransport::wait_for_arrivals_count, 1u);
    EXPECT_EQ(MockPacketSyncTransport::notify_group_count, 1u);
    EXPECT_TRUE(MockX280Transport::control_history.empty());
    for (uint32_t w = 0; w < tf.P_; ++w) {
        EXPECT_EQ(tf.ready(w)->value, (uint32_t)PI_READY_DONE);
    }
}

TEST(PacketInjectionConsumerStateModel, AggregateIntervalAndEmptyVisitsCoverOnlyTheMeasuredPhase) {
    // Worker 1 publishes each phase only once worker 0's is credited, so the
    // consumer finds worker 1 empty on exactly one visit per phase, and visits
    // worker 0 after it has finished each phase.
    ConsumerTestFixture tf(2, 2, 16, 1, 1, 2);
    tf.args.W = 2u;
    tf.produced(0)->value = 2u;
    uint64_t release_mtime = 0;
    uint64_t first_measured_credit_mtime = 0;
    uint64_t last_credit_mtime = 0;
    MockPacketSyncTransport::on_notify_group = [&]() {
        release_mtime = MockX280Transport::mock_mtime;
        tf.produced(0)->value = 4u;
    };
    MockX280Transport::on_store = [&](uint32_t w, uint64_t record, uint32_t val) {
        if (record != PI_L1_CONSUMED_OFFSET) {
            return;
        }
        if (val > tf.args.W && first_measured_credit_mtime == 0) {
            first_measured_credit_mtime = MockX280Transport::mock_mtime;
        }
        last_credit_mtime = MockX280Transport::mock_mtime;
        if (w == 0u && (val == 2u || val == 4u)) {
            tf.produced(1)->value = val;
        }
        if (w == 1u && val == 4u) {
            for (uint32_t i = 0; i < tf.P_; ++i) {
                tf.ready(i)->value = PI_READY_DONE;
            }
            tf.acknowledge_collection();
        }
    };

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    ASSERT_EQ(consumer.run(), PI_ERROR_NONE);

    const pi_run_result_t* r = tf.run_res();
    EXPECT_EQ(r->empty_visits, 1u);
    EXPECT_GT(r->aggregate_start_mtime, release_mtime);
    EXPECT_LT(r->aggregate_start_mtime, first_measured_credit_mtime);
    EXPECT_GT(r->aggregate_end_mtime, last_credit_mtime);
    EXPECT_NE(r->aggregate_start_rdcycle, 0u);
    EXPECT_GT(r->aggregate_end_rdcycle, r->aggregate_start_rdcycle);
}

TEST(PacketInjectionConsumerStateModel, TimerFailureDoesNotAdvance) {
    ConsumerTestFixture tf(1, 1, 16, 1, 1, 2, PI_MODE_LATENCY);
    tf.wire_auto_producer();

    // Force timer retry failure
    g_tensix_live_timer_mock_read = [](uintptr_t) -> uint32_t {
        static uint32_t s = 0;
        return ++s;  // Never matches high1 == high2
    };
    MockX280Transport::on_mtime = [&]() {
        if (tf.run_res()->data_quiescent) {
            tf.acknowledge_collection();
        }
    };

    packet_injection::X280Consumer<PI_MODE_LATENCY, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    const pi_error_t err = consumer.run();
    g_tensix_live_timer_mock_read = nullptr;

    EXPECT_EQ(err, PI_ERROR_TIMER_RETRY_EXHAUSTED);

    // Invariant: Failed timer endpoint does NOT advance consumed or phase_done!
    EXPECT_EQ(tf.slot(0)->consumed, 0u);
    EXPECT_EQ(tf.slot(0)->phase_done, 0u);
    EXPECT_EQ(tf.run_res()->export_valid_bytes, 0u);
    EXPECT_TRUE(ExportAccountingHolds(tf));
}

TEST(PacketInjectionConsumerStateModel, CooperativeFailureRetainsWhenCountersDoNotQuiesce) {
    ConsumerTestFixture tf(1, 1, 16, 1, 1, 2, PI_MODE_LATENCY);
    tf.wire_auto_producer();
    const auto auto_producer = MockX280Transport::on_store;
    MockX280Transport::on_store = [&](uint32_t w, uint64_t record, uint32_t val) {
        if (record == PI_L1_CONTROL_OFFSET && val == PI_CONTROL_STOP) {
            tf.ready(0)->value = PI_READY_FAILED;
        } else {
            auto_producer(w, record, val);
        }
    };
    bool collected = false;
    uint32_t export_polls = 0;
    MockX280Transport::on_mtime = [&]() {
        if (tf.run_res()->data_quiescent && !collected) {
            MockX280Transport::mock_mtime += 2 * tf.args.x280_timeout_ticks;
            if (++export_polls < 3) {
                return;
            }
            EXPECT_EQ(tf.ready(0)->value, (uint32_t)PI_READY_FAILED);
            EXPECT_EQ(tf.run_res()->data_quiescent, 1u);
            EXPECT_EQ(tf.run_res()->windows_cleared, 0u);
            EXPECT_EQ(tf.run_res()->cache_restored, 0u);
            EXPECT_EQ(tf.run_res()->export_valid_bytes, 0u);
            MockX280Transport::lifecycle.push_back("ack");
            tf.acknowledge_collection();
            collected = true;
        }
    };
    g_tensix_live_timer_mock_read = [](uintptr_t) -> uint32_t {
        static uint32_t value = 0;
        return ++value;
    };

    packet_injection::X280Consumer<PI_MODE_LATENCY, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    const pi_error_t err = consumer.run();
    g_tensix_live_timer_mock_read = nullptr;

    EXPECT_EQ(err, PI_ERROR_TIMER_RETRY_EXHAUSTED);
    EXPECT_FALSE(collected);
    EXPECT_EQ(tf.run_res()->failure.code, (uint32_t)PI_ERROR_TIMER_RETRY_EXHAUSTED);
    EXPECT_EQ(tf.run_res()->failure.actor, (uint32_t)PI_ACTOR_X280);
    EXPECT_EQ(tf.run_res()->data_quiescent, 0u);
    EXPECT_EQ(tf.run_res()->windows_cleared, 0u);
    EXPECT_EQ(tf.run_res()->cache_restored, 0u);
    EXPECT_EQ(MockX280Transport::lifecycle, (std::vector<std::string>{"pin", "stop"}));
}

TEST(PacketInjectionConsumerStateModel, ExportAckTimeoutRetainsUntilLateAck) {
    ConsumerTestFixture tf(1, 1, 16, 1, 1, 2);
    tf.wire_auto_producer();
    const auto auto_producer = MockX280Transport::on_store;
    MockX280Transport::on_store = [&](uint32_t w, uint64_t record, uint32_t val) {
        auto_producer(w, record, val);
        tf.hctrl()->collect_ack_run_value = 0u;
    };
    bool late_ack = false;
    MockX280Transport::on_mtime = [&]() {
        if (tf.run_res()->failure.code == PI_ERROR_TIMEOUT && !late_ack) {
            EXPECT_EQ(tf.run_res()->data_quiescent, 1u);
            EXPECT_EQ(tf.run_res()->windows_cleared, 0u);
            EXPECT_EQ(tf.run_res()->cache_restored, 0u);
            tf.acknowledge_collection();
            late_ack = true;
        }
    };

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_TIMEOUT);
    EXPECT_TRUE(late_ack);
    // The collect wait is blocked on the host, not on a worker, so the
    // attribution is the sentinel rather than worker 0.
    EXPECT_EQ(tf.run_res()->failure.phase, (uint32_t)PI_PHASE_WAIT_COLLECT);
    EXPECT_EQ(tf.run_res()->failure.worker_index, UINT32_MAX);
    EXPECT_EQ(tf.slot(0)->consumed, 2u);
    EXPECT_EQ(tf.slot(0)->credited, 2u);
    EXPECT_EQ(tf.ready(0)->value, (uint32_t)PI_READY_DONE);
    EXPECT_EQ(tf.run_res()->data_quiescent, 1u);
    EXPECT_EQ(tf.run_res()->windows_cleared, 1u);
    EXPECT_EQ(tf.run_res()->cache_restored, 1u);
}

TEST(PacketInjectionConsumerStateModel, PartialMeasuredFailureRetainsUnquiescedResources) {
    ConsumerTestFixture tf(2, 1, 16, 1, 1, 2, PI_MODE_LATENCY);
    tf.wire_auto_producer();
    const auto auto_producer = MockX280Transport::on_store;
    MockX280Transport::on_store = [&](uint32_t w, uint64_t record, uint32_t val) {
        if (record == PI_L1_CONTROL_OFFSET && val == PI_CONTROL_STOP) {
            tf.ready(0)->value = PI_READY_FAILED;
            tf.ready(1)->value = PI_READY_FAILED;
        } else {
            auto_producer(w, record, val);
        }
    };
    MockX280Transport::on_mtime = [&]() {
        if (tf.run_res()->data_quiescent) {
            tf.acknowledge_collection();
        }
    };
    static uint32_t reads = 0;
    reads = 0;
    g_tensix_live_timer_mock_read = [](uintptr_t) -> uint32_t {
        ++reads;
        return reads <= 3 ? 1 : reads;
    };

    packet_injection::X280Consumer<PI_MODE_LATENCY, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    const auto err = consumer.run();
    g_tensix_live_timer_mock_read = nullptr;

    EXPECT_EQ(err, PI_ERROR_TIMER_RETRY_EXHAUSTED);
    EXPECT_EQ(tf.slot(0)->phase_done, 1u);
    EXPECT_EQ(tf.slot(1)->phase_done, 0u);
    EXPECT_EQ(tf.run_res()->export_valid_bytes, 0u);
    EXPECT_EQ(tf.run_res()->failure.worker_index, 1u);
    EXPECT_EQ(tf.exported()[0], 0u);
    EXPECT_EQ(tf.exported()[1], 0u);
    EXPECT_EQ(tf.exported()[2], 0u);
    EXPECT_EQ(tf.run_res()->data_quiescent, 0u);
    EXPECT_EQ(tf.run_res()->windows_cleared, 0u);
    EXPECT_EQ(tf.run_res()->cache_restored, 0u);
}

TEST(PacketInjectionConsumerStateModel, WarmupFailureExportsNoMeasuredSamples) {
    ConsumerTestFixture tf(1, 1, 16, 1, 1, 2, PI_MODE_LATENCY);
    tf.args.W = 3;
    tf.produced(0)->value = 3u;
    std::fill(tf.pinned.begin(), tf.pinned.end(), 123456);
    MockX280Transport::on_store = [&](uint32_t, uint64_t record, uint32_t val) {
        if ((record == PI_L1_CONSUMED_OFFSET && val == 3) ||
            (record == PI_L1_CONTROL_OFFSET && val == PI_CONTROL_STOP)) {
            tf.ready(0)->value = PI_READY_FAILED;
        }
    };
    MockPacketSyncTransport::wait_for_arrivals_result = false;
    MockX280Transport::on_mtime = [&]() {
        if (tf.run_res()->data_quiescent) {
            tf.acknowledge_collection();
        }
    };
    g_tensix_live_timer_mock_read = [](uintptr_t) -> uint32_t { return 1; };

    packet_injection::X280Consumer<PI_MODE_LATENCY, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    const auto err = consumer.run();
    g_tensix_live_timer_mock_read = nullptr;

    EXPECT_EQ(err, PI_ERROR_STOPPED);
    EXPECT_EQ(tf.slot(0)->phase_done, 3u);
    EXPECT_EQ(tf.slot(0)->warmup_done, 3u);
    EXPECT_EQ(tf.run_res()->measured_completed, 0u);
    EXPECT_EQ(tf.run_res()->export_valid_bytes, 0u);
    EXPECT_EQ(tf.exported()[0], 0u);
    EXPECT_TRUE(ExportAccountingHolds(tf));

    // phase_done still holds the three warmup completions here, so an
    // unconditional "export_valid_bytes == 8 * sum(phase_done)" cross-check
    // would report a false mismatch on every pre-measured failure.
    EXPECT_EQ(tf.run_res()->failure.phase, (uint32_t)PI_PHASE_WAIT_MEASURE);
    EXPECT_LT(tf.run_res()->failure.phase, (uint32_t)PI_PHASE_MEASURE);
    EXPECT_NE(tf.run_res()->export_valid_bytes, 8u * tf.slot(0)->phase_done);
}

TEST(PacketInjectionCoreSync, PersistentHostStopAllowsRecoveryAfterRelease) {
    ConsumerTestFixture tf(1, 1, 1, 1, 1, 2);
    MockPacketSyncTransport::wait_for_arrivals_result = false;
    MockPacketSyncTransport::on_notify_group = [&]() { MockX280Transport::lifecycle.push_back("release"); };
    MockX280Transport::stop_requested = 1;
    uint32_t recovery_polls = 0u;
    MockX280Transport::on_store = [&](uint32_t, uint64_t record, uint32_t value) {
        if (record == PI_L1_CONTROL_OFFSET && value == PI_CONTROL_STOP) {
            EXPECT_EQ(MockPacketSyncTransport::notify_group_count, 0u);
        }
    };
    MockX280Transport::on_mtime = [&]() {
        if (++recovery_polls == 3u) {
            tf.ready(0)->value = PI_READY_FAILED;
        }
        if (tf.run_res()->data_quiescent) {
            tf.acknowledge_collection();
        }
    };

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_STOPPED);

    const auto stop = std::find(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "stop");
    const auto release = std::find(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "release");
    ASSERT_NE(stop, MockX280Transport::lifecycle.end());
    ASSERT_NE(release, MockX280Transport::lifecycle.end());
    EXPECT_LT(stop, release);
    EXPECT_GE(recovery_polls, 3u);
    EXPECT_EQ(MockPacketSyncTransport::wait_for_arrivals_count, 1u);
    EXPECT_EQ(MockPacketSyncTransport::notify_group_count, 1u);
    EXPECT_EQ(tf.run_res()->data_quiescent, 1u);
    EXPECT_EQ(tf.run_res()->windows_cleared, 1u);
    EXPECT_EQ(tf.run_res()->cache_restored, 1u);
}

TEST(PacketInjectionConsumerStateModel, MidWarmupTimeoutExportsNothingWithWarmupProgressHeld) {
    ConsumerTestFixture tf(1, 1, 16, 1, 1, 2, PI_MODE_LATENCY);
    tf.args.W = 4;
    tf.args.x280_timeout_ticks = 1000;  // reached in a few mtime polls

    // The producer delivers half of the warmup and then stalls, so the timeout
    // lands inside the warmup phase with partial progress in phase_done.
    tf.produced(0)->value = 2u;
    MockX280Transport::on_store = [&](uint32_t, uint64_t record, uint32_t val) {
        if (record == PI_L1_CONTROL_OFFSET && val == (uint32_t)PI_CONTROL_STOP) {
            tf.ready(0)->value = PI_READY_FAILED;
        }
    };
    MockX280Transport::on_mtime = [&]() {
        if (tf.run_res()->data_quiescent) {
            tf.acknowledge_collection();
        }
    };
    g_tensix_live_timer_mock_read = [](uintptr_t) -> uint32_t { return 1; };

    packet_injection::X280Consumer<PI_MODE_LATENCY, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    const auto err = consumer.run();
    g_tensix_live_timer_mock_read = nullptr;

    EXPECT_EQ(err, PI_ERROR_TIMEOUT);
    EXPECT_EQ(tf.run_res()->failure.phase, (uint32_t)PI_PHASE_WARMUP);
    EXPECT_EQ(tf.slot(0)->phase_done, 2u);
    EXPECT_EQ(tf.slot(0)->warmup_done, 0u);
    EXPECT_EQ(tf.run_res()->measured_completed, 0u);
    EXPECT_EQ(tf.run_res()->export_valid_bytes, 0u);
    EXPECT_TRUE(ExportAccountingHolds(tf));

    // A second pre-measure failure phase, so the accounting condition is not
    // fitted to the WAIT_MEASURE case alone.
    EXPECT_LT(tf.run_res()->failure.phase, (uint32_t)PI_PHASE_MEASURE);
    EXPECT_NE(tf.run_res()->export_valid_bytes, 8u * tf.slot(0)->phase_done);
}

TEST(PacketInjectionFailure, EveryLayoutRejectionClassMapsToOneError) {
    EXPECT_EQ(pi_layout_code_to_error(PI_LAYOUT_OK), PI_ERROR_NONE);
    EXPECT_EQ(pi_layout_code_to_error(PI_LAYOUT_ALIGNMENT), PI_ERROR_BAD_PLACEMENT);
    EXPECT_EQ(pi_layout_code_to_error(PI_LAYOUT_LIM_CAPACITY), PI_ERROR_CAPACITY);
    EXPECT_EQ(pi_layout_code_to_error(PI_LAYOUT_EXPORT_CAPACITY), PI_ERROR_CAPACITY);
    EXPECT_EQ(pi_layout_code_to_error(PI_LAYOUT_L1_CAPACITY), PI_ERROR_CAPACITY);
    EXPECT_EQ(pi_layout_code_to_error(PI_LAYOUT_BAD_PARAM), PI_ERROR_BAD_ARGS);
    EXPECT_EQ(pi_layout_code_to_error(PI_LAYOUT_OVERFLOW), PI_ERROR_BAD_ARGS);

    // INVARIANT is the layout function catching its own postcondition, so it
    // gets its own error rather than BAD_ARGS, which would blame the caller for
    // a defect in this code. No input reaches it -- the three checks it guards
    // are unreachable while the arithmetic above them stands -- so the mapping
    // is asserted here rather than through a pi_compute_layout() call.
    EXPECT_EQ(pi_layout_code_to_error(PI_LAYOUT_INVARIANT), PI_ERROR_LAYOUT_INVARIANT);
    EXPECT_NE(pi_layout_code_to_error(PI_LAYOUT_INVARIANT), PI_ERROR_BAD_ARGS);

    // An unknown code still fails the run rather than reading as success.
    EXPECT_EQ(pi_layout_code_to_error(UINT32_MAX), PI_ERROR_BAD_ARGS);
}

TEST(PacketInjectionConsumerStateModel, PrepareRefusalRetainsTheReservedWay) {
    // The other half of the pinned split: a Mode 1 prepare refusal is the one
    // PI_ERROR_PIN_POLICY at PI_PHASE_INIT, and unlike a baseline failure it
    // reserved something it cannot prove it released, so nothing is claimed
    // restored. Code plus phase is what tells the two apart.
    ConsumerTestFixture tf(1, 1, 16, 1, 1, 2, PI_MODE_LATENCY);
    tf.args.x280_timeout_ticks = 1000;
    tf.wire_auto_producer();
    MockX280Transport::prepare_result = UINT64_MAX;

    packet_injection::X280Consumer<PI_MODE_LATENCY, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_PIN_POLICY);

    EXPECT_EQ(tf.run_res()->failure.code, (uint32_t)PI_ERROR_PIN_POLICY);
    EXPECT_EQ(tf.run_res()->failure.phase, (uint32_t)PI_PHASE_INIT);
    EXPECT_EQ(tf.run_res()->cache_restored, 0u);
    EXPECT_EQ(tf.run_res()->windows_cleared, 0u);
    // It tried to reserve and never tore down.
    EXPECT_EQ(std::count(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "pin"), 1);
    EXPECT_EQ(std::count(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "unpin"), 0);
}

TEST(PacketInjectionConsumerStateModel, NonLatencyRunNeitherReservesNorTearsDownTheSampleRegion) {
    // A throughput image reserves no sample region, so the only pinned-cache
    // requirement is that the baseline way masks are still in place.
    ConsumerTestFixture tf(2, 1, 16, 1, 1, 4);
    tf.wire_auto_producer();
    ASSERT_EQ(MockX280Transport::baseline_policy, (uint32_t)X280_PINNED_INPUT_ADMITTED);

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_NONE);

    EXPECT_EQ(tf.run_res()->state, (uint32_t)PI_PHASE_DONE);
    EXPECT_EQ(tf.run_res()->measured_completed, 8u);
    // Never reserved, so never torn down.
    EXPECT_EQ(std::count(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "pin"), 0);
    EXPECT_EQ(std::count(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "unpin"), 0);
    EXPECT_EQ(tf.run_res()->cache_restored, 1u);
    EXPECT_EQ(tf.run_res()->windows_cleared, 1u);
}

TEST(PacketInjectionConsumerStateModel, BaselineMaskFailureStillReleasesItsWindows) {
    // The asymmetry worth pinning: a Mode 1 prepare failure sets the pin-policy
    // uncertainty flag and keeps the reserved way, but a baseline failure
    // reserved nothing, so it must not -- otherwise the run would retain TLB
    // windows it never had a reason to hold.
    // Each admission code carries its own cause; anything outside the enum
    // names none, which is what PI_ERROR_PIN_POLICY means on this path.
    struct Cause {
        uint32_t admission;
        pi_error_t expected;
    };
    const Cause causes[] = {
        {(uint32_t)X280_PINNED_INPUT_WRONG_WAYENABLE, PI_ERROR_PIN_WAYENABLE},
        {(uint32_t)X280_PINNED_INPUT_WRONG_WAYMASK, PI_ERROR_PIN_WAYMASK},
        {(uint32_t)X280_PINNED_INPUT_WRONG_SHAPE, PI_ERROR_PIN_SHAPE},
        {UINT32_MAX, PI_ERROR_PIN_POLICY},
    };
    for (const auto& cause : causes) {
        const uint32_t code = cause.admission;
        SCOPED_TRACE(code);
        ConsumerTestFixture tf(2, 1, 16, 1, 1, 4);
        tf.wire_auto_producer();
        const auto auto_producer = MockX280Transport::on_store;
        // Confirm the drain so cleanup is reached at all; the unproven-drain
        // path is covered separately below.
        MockX280Transport::on_store = [&](uint32_t w, uint64_t record, uint32_t val) {
            if (record == PI_L1_CONTROL_OFFSET && val == PI_CONTROL_STOP) {
                tf.ready(0)->value = PI_READY_FAILED;
                tf.ready(1)->value = PI_READY_FAILED;
            } else {
                auto_producer(w, record, val);
            }
        };
        MockX280Transport::on_mtime = [&]() {
            if (tf.run_res()->data_quiescent) {
                tf.acknowledge_collection();
            }
        };
        MockX280Transport::baseline_policy = code;

        packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
            0, tf.run_value, tf.args);
        EXPECT_EQ(consumer.run(), cause.expected);

        EXPECT_EQ(tf.run_res()->failure.code, (uint32_t)cause.expected);
        EXPECT_EQ(tf.run_res()->failure.phase, (uint32_t)PI_PHASE_INIT);
        EXPECT_EQ(tf.run_res()->failure.worker_index, UINT32_MAX);
        EXPECT_EQ(tf.run_res()->failure.actor, (uint32_t)PI_ACTOR_X280);
        EXPECT_EQ(tf.run_res()->state, (uint32_t)PI_PHASE_FAILED);

        // Nothing was reserved, and the windows are handed back.
        EXPECT_EQ(std::count(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "pin"), 0);
        EXPECT_EQ(std::count(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "unpin"), 0);
        EXPECT_EQ(tf.run_res()->cache_restored, 1u);
        EXPECT_EQ(tf.run_res()->windows_cleared, 1u);

        // STOP is the only control write; phase release belongs to core-sync.
        EXPECT_TRUE(std::all_of(
            MockX280Transport::control_history.begin(), MockX280Transport::control_history.end(), [](uint32_t value) {
                return value == (uint32_t)PI_CONTROL_STOP;
            }));
        EXPECT_TRUE(MockX280Transport::credit_history.empty());
    }
}

TEST(PacketInjectionConsumerStateModel, StalePinnedMaskIsRefusedBeforeAnyTraffic) {
    // A run that reserves no sample region needs the cache to still carry the
    // baseline way masks; a mask a previous run left prepared would starve the
    // measured phase into a timeout instead of failing here.
    ConsumerTestFixture tf(1, 1, 16, 1, 1, 2);
    tf.args.x280_timeout_ticks = 1000;  // the drain wait below is bounded by this
    tf.wire_auto_producer();
    MockX280Transport::baseline_policy = X280_PINNED_INPUT_WRONG_WAYMASK;

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    // The admission code names the cause, so the run reports which mask is
    // wrong rather than the generic policy failure.
    EXPECT_EQ(consumer.run(), PI_ERROR_PIN_WAYMASK);

    EXPECT_EQ(tf.run_res()->failure.code, (uint32_t)PI_ERROR_PIN_WAYMASK);
    EXPECT_EQ(tf.run_res()->failure.phase, (uint32_t)PI_PHASE_INIT);
    EXPECT_EQ(tf.run_res()->state, (uint32_t)PI_PHASE_FAILED);
    EXPECT_EQ(tf.slot(0)->consumed, 0u);
    EXPECT_TRUE(MockX280Transport::credit_history.empty());

    // The producers never confirm they are drained, so nothing may be claimed
    // as restored.
    EXPECT_EQ(tf.run_res()->data_quiescent, 0u);
    EXPECT_EQ(tf.run_res()->windows_cleared, 0u);
    EXPECT_EQ(tf.run_res()->cache_restored, 0u);
    EXPECT_EQ(std::count(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "clear"), 0);
    // STOP still reaches them through the windows that were configured.
    EXPECT_EQ(
        std::count(
            MockX280Transport::control_history.begin(),
            MockX280Transport::control_history.end(),
            (uint32_t)PI_CONTROL_STOP),
        1);
}

TEST(PacketInjectionConsumerStateModel, UnverifiedWindowMappingIsRetainedNotReleased) {
    // A TLB window whose readback failed was never proved to be ours, so the
    // run must keep it rather than hand it back: releasing a mapping it could
    // not verify is the failure the uncertainty flag guards. Tile 0 maps, tile
    // 1's readback fails, so one window is verified and one is not.
    ConsumerTestFixture tf(4, 2, 16, 1, 1, 2);
    tf.args.x280_timeout_ticks = 1000;
    MockX280Transport::credit_window_fail_tile = 1u;

    // Pre-failed producers, so the drain completes and the run actually reaches
    // cleanup. Without that, cleanup is never entered and the retention this
    // test is about would go unexercised either way.
    for (uint32_t w = 0; w < 4; ++w) {
        tf.ready(w)->value = PI_READY_FAILED;
    }
    MockX280Transport::on_mtime = [&]() {
        if (tf.run_res()->data_quiescent) {
            tf.acknowledge_collection();
        }
    };

    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        0, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_WINDOW_CONFIG);

    EXPECT_EQ(tf.run_res()->failure.code, (uint32_t)PI_ERROR_WINDOW_CONFIG);
    EXPECT_EQ(tf.run_res()->failure.phase, (uint32_t)PI_PHASE_INIT);
    EXPECT_EQ(tf.run_res()->failure.worker_index, UINT32_MAX);

    // It got as far as cleanup and restored the cache, so the windows staying
    // put is a decision rather than an early return.
    EXPECT_EQ(tf.run_res()->data_quiescent, 1u);
    EXPECT_EQ(tf.run_res()->cache_restored, 1u);
    EXPECT_EQ(tf.run_res()->windows_cleared, 0u);
    EXPECT_EQ(std::count(MockX280Transport::lifecycle.begin(), MockX280Transport::lifecycle.end(), "clear"), 0)
        << "released a window whose mapping was never verified";

    // Two configuration attempts: tile 0 succeeded and tile 1 did not.
    EXPECT_EQ(MockX280Transport::credit_window_tiles.size(), 2u);
    EXPECT_TRUE(MockX280Transport::credit_history.empty());

    // STOP goes only through the window that was verified, so tile 0's two
    // workers are told and tile 1's are not -- the same rule as the cleanup,
    // applied to the write rather than the release.
    EXPECT_EQ(
        MockX280Transport::control_history,
        (std::vector<uint32_t>{(uint32_t)PI_CONTROL_STOP, (uint32_t)PI_CONTROL_STOP}));
}

TEST(PacketInjectionConsumerStateModel, CleanupFailuresDoNotClaimRestoredResources) {
    for (bool fail_pin : {false, true}) {
        SCOPED_TRACE(fail_pin);
        ConsumerTestFixture tf(1, 1, 16, 1, 1, 2, PI_MODE_LATENCY);
        tf.wire_auto_producer();
        if (fail_pin) {
            MockX280Transport::teardown_result = UINT64_MAX;
        } else {
            MockX280Transport::clear_result = -1;
        }
        g_tensix_live_timer_mock_read = [](uintptr_t) -> uint32_t { return 1; };

        packet_injection::X280Consumer<PI_MODE_LATENCY, 0, MockX280Transport, MockPacketSyncTransport> consumer(
            0, tf.run_value, tf.args);
        const auto err = consumer.run();
        g_tensix_live_timer_mock_read = nullptr;

        EXPECT_EQ(err, fail_pin ? PI_ERROR_PIN_POLICY : PI_ERROR_WINDOW_CLEAR);
        EXPECT_EQ(tf.run_res()->state, (uint32_t)PI_PHASE_FAILED);
        EXPECT_EQ(tf.run_res()->cache_restored, fail_pin ? 0u : 1u);
        EXPECT_EQ(tf.run_res()->windows_cleared, 0u);
    }
}

TEST(PacketInjectionConsumerStateModel, InvalidArgumentsAreRejectedBeforeTraffic) {
    struct InvalidCase {
        const char* name;
        std::function<void(ConsumerTestFixture&)> mutate;
        pi_error_t expected;
    };
    const std::vector<InvalidCase> cases{
        {"wrong mode in tag",
         [](ConsumerTestFixture& t) { t.args.abi_tag = PI_ABI_TAG(PI_MODE_LATENCY, 0); },
         PI_ERROR_MODE_MISMATCH},
        {"wrong include_creation in tag",
         [](ConsumerTestFixture& t) { t.args.abi_tag = PI_ABI_TAG(PI_MODE_THROUGHPUT, 1); },
         PI_ERROR_MODE_MISMATCH},
        {"zero tag", [](ConsumerTestFixture& t) { t.args.abi_tag = 0u; }, PI_ERROR_MODE_MISMATCH},
        {"schema four physical coordinate table",
         [](ConsumerTestFixture& t) { t.args.abi_tag = 0x50490420u; },
         PI_ERROR_MODE_MISMATCH},
        {"zero run value", [](ConsumerTestFixture& t) { t.run_value = 0u; }, PI_ERROR_BAD_ARGS},
        // The args record heads the arena, so a base the args do not sit at
        // means every derived address is measured from the wrong origin. Both
        // of these are refused by that placement check, the misaligned one
        // included: args_offset is 0 and pi_x280_args_t is 64-aligned, so any
        // base the placement check accepts is already aligned and the separate
        // alignment guard has no input that reaches it first.
        {"arena base off by one",
         [](ConsumerTestFixture& t) { t.relocate_consumer_arena(t.consumer_arena_base + 1u); },
         PI_ERROR_BAD_ARGS},
        // What reset() leaves behind, so a fixture that forgets to publish a
        // base fails here instead of dereferencing address zero.
        {"arena base never published",
         [](ConsumerTestFixture& t) { t.relocate_consumer_arena(0u); },
         PI_ERROR_BAD_ARGS},
        {"args published away from the arena base",
         [](ConsumerTestFixture& t) { t.relocate_consumer_arena(t.consumer_arena_base + 64u); },
         PI_ERROR_BAD_ARGS},
        {"zero worker count", [](ConsumerTestFixture& t) { t.args.worker_count = 0u; }, PI_ERROR_BAD_PLACEMENT},
        {"worker count is neither T nor 2T",
         [](ConsumerTestFixture& t) { t.args.worker_count = 3u; },
         PI_ERROR_BAD_PLACEMENT},
        {"non-power-of-two S", [](ConsumerTestFixture& t) { t.args.S = 15u; }, PI_ERROR_BAD_ARGS},
        {"C greater than K",
         [](ConsumerTestFixture& t) {
             t.args.C = 2u;
             t.args.K = 1u;
         },
         PI_ERROR_BAD_ARGS},
        {"zero N", [](ConsumerTestFixture& t) { t.args.N = 0u; }, PI_ERROR_BAD_ARGS},
        {"non-power-of-two poll cadence", [](ConsumerTestFixture& t) { t.args.poll_cadence = 3u; }, PI_ERROR_BAD_ARGS},
        {"zero timeout", [](ConsumerTestFixture& t) { t.args.x280_timeout_ticks = 0u; }, PI_ERROR_BAD_ARGS},
        {"misaligned l1_tile_base", [](ConsumerTestFixture& t) { t.args.l1_tile_base = 32u; }, PI_ERROR_BAD_ARGS},
        {"ring depth the arena cannot hold", [](ConsumerTestFixture& t) { t.args.S = 1u << 20; }, PI_ERROR_CAPACITY},
        {"duplicate tile coordinates",
         [](ConsumerTestFixture& t) { *t.coord(1) = *t.coord(0); },
         PI_ERROR_BAD_PLACEMENT},
        {"tile coordinate past the NoC node-id field",
         [](ConsumerTestFixture& t) { t.coord(0)->x = 64u; },
         PI_ERROR_BAD_PLACEMENT},
    };

    for (const auto& invalid : cases) {
        SCOPED_TRACE(invalid.name);
        ConsumerTestFixture tf(4, 2, 16, 1, 1, 2);
        tf.wire_auto_producer();
        invalid.mutate(tf);

        packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
            0, tf.run_value, tf.args);
        EXPECT_EQ(consumer.run(), invalid.expected);
        EXPECT_TRUE(MockX280Transport::credit_window_tiles.empty());
        EXPECT_TRUE(MockX280Transport::control_history.empty());
        EXPECT_TRUE(MockX280Transport::credit_history.empty());
    }
}

TEST(PacketInjectionConsumerStateModel, ANonZeroHartIsRejected) {
    ConsumerTestFixture tf(1, 1, 16, 1, 1, 2);
    tf.wire_auto_producer();
    packet_injection::X280Consumer<PI_MODE_THROUGHPUT, 0, MockX280Transport, MockPacketSyncTransport> consumer(
        1, tf.run_value, tf.args);
    EXPECT_EQ(consumer.run(), PI_ERROR_BAD_ARGS);
    EXPECT_TRUE(MockX280Transport::credit_window_tiles.empty());
}

// --------------------------------------------------------------------------
// Sample population output
// --------------------------------------------------------------------------

TEST(PacketInjectionOutput, SerializesRawPopulations) {
    using x280::harness::output::sample_population_json;
    EXPECT_EQ(
        sample_population_json({}), "{\"count\":0,\"units\":\"ticks\",\"raw_ticks\":[],\"reason\":\"no_samples\"}");
    EXPECT_EQ(sample_population_json({30, 10, 20}), "{\"count\":3,\"units\":\"ticks\",\"raw_ticks\":[30,10,20]}");
    EXPECT_THROW(sample_population_json({1ULL << 63}), std::runtime_error);
}

TEST(PacketInjectionOutput, ThroughputRetainsRawEndpointsAndErrors) {
    using x280::harness::output::throughput_json;
    EXPECT_EQ(
        throughput_json(10, 10, 100, true, kUsefulHeaderBytes, PI_SLOT_BYTES),
        "{\"start_ticks\":10,\"end_ticks\":10,\"requests\":100,"
        "\"useful_bytes_per_request\":96,\"slot_bytes_per_request\":128,"
        "\"interval_error\":\"invalid_interval\"}");
    EXPECT_EQ(
        throughput_json(10, 20, std::nullopt, true, kUsefulHeaderBytes, PI_SLOT_BYTES),
        "{\"start_ticks\":10,\"end_ticks\":20,\"useful_bytes_per_request\":96,"
        "\"slot_bytes_per_request\":128,\"interval_error\":null}");
    const auto failed = throughput_json(10, 20, 100, false, kUsefulHeaderBytes, PI_SLOT_BYTES);
    EXPECT_NE(failed.find("\"interval_error\":\"run_failed\""), std::string::npos);
    const auto success = throughput_json(UINT64_MAX - 5, 4, 100, true, kUsefulHeaderBytes, PI_SLOT_BYTES);
    EXPECT_NE(success.find("\"start_ticks\":18446744073709551610,\"end_ticks\":4"), std::string::npos) << success;
    EXPECT_NE(success.find("\"interval_error\":null"), std::string::npos);
    EXPECT_EQ(success.find("requests_per_second"), std::string::npos);
}
