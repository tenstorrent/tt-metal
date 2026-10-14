// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "p2p_latency_x280_noc_args.h"
#include "metal_peer_args.h"
#include "ubench_section_load.h"
#include "tests/device/suites/p2p_latency/metal_peer_arguments.hpp"
#include "harness/regions/l1_window.hpp"
#include "harness/regions/regions.hpp"
#include "tests/device/suites/p2p_latency/metal_peer.hpp"
#include "tests/host/translated_coord_factory.hpp"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <sys/mman.h>
#include <unistd.h>

namespace {

// Drives the frontend's timed loop: warm-up polls succeed, timed polls succeed
// timed_polls_left times before reporting a hang or STOP, and deltas succeed
// deltas_left times before reporting a refused interval.
struct ScriptedBackend {
    struct state_t {
        uint32_t timed_polls_left = UINT32_MAX;
        uint32_t deltas_left = UINT32_MAX;
        uint64_t clock = 0;
    };
    struct timestamp_t {
        uint64_t primary = 0;
        uint64_t secondary = 0;
    };
    using counter_delta_t = uint64_t;

    static void bind_direction(state_t&, p2p_latency_direction_t) {}
    static timestamp_t measurement_now(state_t& s) {
        s.clock += 10u;
        return {s.clock, s.clock};
    }
    static bool delta(state_t& s, timestamp_t t0, timestamp_t t1, counter_delta_t* primary) {
        if (s.deltas_left == 0u) {
            return false;
        }
        --s.deltas_left;
        *primary = t1.primary - t0.primary;
        return true;
    }
    template <p2p_latency_transfer_shape_t, p2p_latency_transmit_mode_t>
    static void transmit(state_t&, uint32_t) {}
    template <p2p_latency_poll_mode_t>
    static bool poll(state_t& s, uint32_t) {
        if (s.timed_polls_left == 0u) {
            return false;
        }
        --s.timed_polls_left;
        return true;
    }
    template <p2p_latency_poll_mode_t>
    static bool poll_for(state_t&, uint32_t, uint64_t) {
        return true;
    }
    static uint32_t observed(state_t&) { return 0u; }
};

struct Mapping {
    void* address = MAP_FAILED;
    size_t size = 0;

    ~Mapping() {
        if (address != MAP_FAILED) {
            munmap(address, size);
        }
    }
};

void expect_semantic_fields_equal(
    const p2p_latency_x280_noc_section_t& expected, const p2p_latency_x280_noc_section_t& actual) {
    EXPECT_EQ(actual.cookie_common, expected.cookie_common);
    EXPECT_EQ(actual.common.self, expected.common.self);
    EXPECT_EQ(actual.common.first_direction, expected.common.first_direction);
    EXPECT_EQ(actual.common.payload_size, expected.common.payload_size);
    EXPECT_EQ(actual.common.warmup_count, expected.common.warmup_count);
    EXPECT_EQ(actual.common.sample_count, expected.common.sample_count);
    EXPECT_EQ(actual.common.run_nonce, expected.common.run_nonce);
    EXPECT_EQ(actual.common.warmup_max_poll_cycles, expected.common.warmup_max_poll_cycles);
    EXPECT_EQ(actual.cookie_backend, expected.cookie_backend);
    EXPECT_EQ(actual.backend.own_inbound_addr, expected.backend.own_inbound_addr);
    EXPECT_EQ(actual.backend.own_result_addr, expected.backend.own_result_addr);
    EXPECT_EQ(actual.backend.peer_inbound_addr, expected.backend.peer_inbound_addr);
    EXPECT_EQ(actual.backend.peer_noc.x, expected.backend.peer_noc.x);
    EXPECT_EQ(actual.backend.peer_noc.y, expected.backend.peer_noc.y);
    EXPECT_EQ(actual.backend.peer_tlb_index, expected.backend.peer_tlb_index);
}

void expect_rejected_destination_unchanged(
    const p2p_latency_x280_noc_section_t& expected, const p2p_latency_x280_noc_section_t& actual) {
    expect_semantic_fields_equal(expected, actual);
    EXPECT_EQ(actual.pad0, expected.pad0);
    EXPECT_EQ(actual.pad1, expected.pad1);
    EXPECT_EQ(actual.backend.reserved, expected.backend.reserved);
}

p2p_latency_x280_noc_section_t make_section() {
    p2p_latency_x280_noc_section_t section{};
    section.cookie_common = p2p_latency_cookie_for(UBENCH_SECTION_COMMON);
    section.common.self = P2P_LATENCY_PARTICIPANT_B;
    section.common.first_direction = P2P_LATENCY_DIRECTION_B_TO_A;
    section.common.payload_size = 4u;
    section.common.warmup_count = 17u;
    section.common.sample_count = 19u;
    section.common.run_nonce = 0x23456789u;
    section.common.warmup_max_poll_cycles = UINT64_C(0x1020304050607080);
    section.cookie_backend = p2p_latency_cookie_for(UBENCH_SECTION_BACKEND);
    section.backend.own_inbound_addr = UINT64_C(0x1111222233334444);
    section.backend.own_result_addr = UINT64_C(0x5555666677778888);
    section.backend.peer_inbound_addr = UINT64_C(0x9999AAAABBBBCCCC);
    section.backend.peer_noc = {13u, 7u};
    section.backend.peer_tlb_index = 21u;
    return section;
}

// Distinct high address halves prevent a 32-bit copy from satisfying the test.
TEST(P2PLatencyX280Args, CopyRoundTripsSemanticFields) {
    const p2p_latency_x280_noc_section_t source = make_section();
    UBENCH_SECTION_STORAGE(p2p_latency_x280_noc_section_t, section) destination{};

    ASSERT_TRUE(load_section<p2p_latency_x280_noc_section_t>(&source, sizeof(source), destination.as_uint32));
    expect_semantic_fields_equal(source, destination.section);
}

// Every rule the participants share, each refused on its own, and the final
// token's 32-bit bound at its edge.
TEST(P2PLatencyCommonArgs, EachSharedRuleRefusesAndTheFinalTokenFitsAtItsEdge) {
    constexpr uint32_t warmup = 8u;
    constexpr uint32_t samples = 64u;
    const p2p_latency_common_args_t valid{
        .self = P2P_LATENCY_PARTICIPANT_B,
        .first_direction = P2P_LATENCY_DIRECTION_B_TO_A,
        .payload_size = 4u,
        .warmup_count = warmup,
        .sample_count = samples,
        .run_nonce = UINT32_MAX - 2u * (warmup + samples) + 1u,
        .warmup_max_poll_cycles = 1u,
    };
    ASSERT_TRUE(p2p_latency_common_valid(&valid)) << "the final token is exactly UINT32_MAX";

    const std::array<std::pair<const char*, void (*)(p2p_latency_common_args_t&)>, 9> refusals{{
        {"warmup 0", [](p2p_latency_common_args_t& c) { c.warmup_count = 0u; }},
        {"samples 0", [](p2p_latency_common_args_t& c) { c.sample_count = 0u; }},
        {"samples above the bound",
         [](p2p_latency_common_args_t& c) {
             c.sample_count = P2P_LATENCY_MAX_SAMPLES + 1u;
             c.run_nonce = 1u;
         }},
        {"run_nonce 0", [](p2p_latency_common_args_t& c) { c.run_nonce = 0u; }},
        {"poll bound 0", [](p2p_latency_common_args_t& c) { c.warmup_max_poll_cycles = 0u; }},
        {"payload 8", [](p2p_latency_common_args_t& c) { c.payload_size = 8u; }},
        {"direction 2", [](p2p_latency_common_args_t& c) { c.first_direction = 2u; }},
        {"final token one past UINT32_MAX", [](p2p_latency_common_args_t& c) { c.run_nonce += 1u; }},
        {"final token wraps the doubled count",
         [](p2p_latency_common_args_t& c) {
             c.warmup_count = 0x80000000u;
             c.run_nonce = 1u;
         }},
    }};
    for (const auto& [name, mutate] : refusals) {
        p2p_latency_common_args_t c = valid;
        mutate(c);
        EXPECT_FALSE(p2p_latency_common_valid(&c)) << name;
    }
}

// Section h of a block is the h-th sizeof(Section) slice, and its extent must
// cover it; a nonzero exact extent must be the block's extent itself.
TEST(P2PLatencyX280Args, IndexedSectionsAndExactExtents) {
    using Section = p2p_latency_x280_noc_section_t;
    std::array<Section, 2> block{make_section(), make_section()};
    block[1].common.run_nonce ^= 0xFFu;
    UBENCH_SECTION_STORAGE(Section, section) destination{};

    ASSERT_TRUE(load_section<Section>(block.data(), sizeof(block), destination.as_uint32, 1u));
    expect_semantic_fields_equal(block[1], destination.section);
    EXPECT_FALSE(load_section<Section>(block.data(), sizeof(Section), destination.as_uint32, 1u))
        << "an extent covering section 0 only admits section 1";
    EXPECT_FALSE(load_section<Section>(block.data(), sizeof(block), destination.as_uint32, 2u));

    EXPECT_TRUE(load_section<Section>(block.data(), sizeof(block), destination.as_uint32, 0u, sizeof(block)));
    EXPECT_FALSE(load_section<Section>(block.data(), sizeof(block), destination.as_uint32, 0u, sizeof(Section)))
        << "an extent other than the exact one is admitted";
    EXPECT_FALSE(load_section<Section>(block.data(), sizeof(Section), destination.as_uint32, 0u, sizeof(block)));
}

// PROT_NONE makes any read before extent rejection observable.
TEST(P2PLatencyX280Args, NullAndBoundaryExtentsAreRejectedBeforeWordZero) {
    const long page_size_result = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size_result, 0);
    const size_t page_size = static_cast<size_t>(page_size_result);
    Mapping unreadable{mmap(nullptr, page_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0), page_size};
    ASSERT_NE(unreadable.address, MAP_FAILED);

    UBENCH_SECTION_STORAGE(p2p_latency_x280_noc_section_t, section) destination{};
    destination.section = make_section();
    destination.section.pad1 = 0x12345678u;
    destination.section.backend.reserved = 0x87654321u;
    const p2p_latency_x280_noc_section_t sentinel = destination.section;

    EXPECT_FALSE(load_section<p2p_latency_x280_noc_section_t>(nullptr, 0u, destination.as_uint32));
    expect_rejected_destination_unchanged(sentinel, destination.section);
    EXPECT_FALSE(load_section<p2p_latency_x280_noc_section_t>(
        nullptr, sizeof(p2p_latency_x280_noc_section_t), destination.as_uint32));
    expect_rejected_destination_unchanged(sentinel, destination.section);

    const auto* const unreadable_words = static_cast<const volatile uint32_t*>(unreadable.address);
    for (const uint64_t source_bytes : std::array<uint64_t, 2>{0u, sizeof(p2p_latency_x280_noc_section_t) - 1u}) {
        EXPECT_FALSE(
            load_section<p2p_latency_x280_noc_section_t>(unreadable_words, source_bytes, destination.as_uint32))
            << "undersized source_bytes " << source_bytes;
        expect_rejected_destination_unchanged(sentinel, destination.section);
    }
}

// The guard immediately after byte 63 makes a changed-size overread observable.
TEST(P2PLatencyX280Args, CurrentCookieInA64ByteSourceCannotTriggerAnOverread) {
    const long page_size_result = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size_result, 64);
    const size_t page_size = static_cast<size_t>(page_size_result);
    Mapping mapping{
        mmap(nullptr, page_size * 2u, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0), page_size * 2u};
    ASSERT_NE(mapping.address, MAP_FAILED);
    ASSERT_EQ(mprotect(static_cast<char*>(mapping.address) + page_size, page_size, PROT_NONE), 0);

    volatile uint32_t* const source =
        reinterpret_cast<volatile uint32_t*>(static_cast<char*>(mapping.address) + page_size - 64u);
    source[0] = p2p_latency_cookie_for(UBENCH_SECTION_COMMON);

    UBENCH_SECTION_STORAGE(p2p_latency_x280_noc_section_t, section) destination{};
    destination.section = make_section();
    destination.section.pad1 = 0x12345678u;
    destination.section.backend.reserved = 0x87654321u;
    const p2p_latency_x280_noc_section_t sentinel = destination.section;

    EXPECT_FALSE(load_section<p2p_latency_x280_noc_section_t>(source, 64u, destination.as_uint32));
    expect_rejected_destination_unchanged(sentinel, destination.section);
}

// The copy gate owns the outer layout; inner cookies remain visible to the
// section validator.
TEST(P2PLatencyX280Args, CopyPreservesMalformedSectionCookiesForValidation) {
    p2p_latency_x280_noc_section_t bad_common = make_section();
    bad_common.cookie_common ^= 0x1u;
    UBENCH_SECTION_STORAGE(p2p_latency_x280_noc_section_t, section) copied_common{};
    ASSERT_TRUE(load_section<p2p_latency_x280_noc_section_t>(&bad_common, sizeof(bad_common), copied_common.as_uint32));
    EXPECT_EQ(copied_common.section.cookie_common, bad_common.cookie_common);
    EXPECT_NE(copied_common.section.cookie_common, p2p_latency_cookie_for(UBENCH_SECTION_COMMON));

    p2p_latency_x280_noc_section_t bad_backend = make_section();
    bad_backend.cookie_backend ^= 0x1u;
    UBENCH_SECTION_STORAGE(p2p_latency_x280_noc_section_t, section) copied_backend{};
    ASSERT_TRUE(
        load_section<p2p_latency_x280_noc_section_t>(&bad_backend, sizeof(bad_backend), copied_backend.as_uint32));
    EXPECT_EQ(copied_backend.section.cookie_backend, bad_backend.cookie_backend);
    EXPECT_NE(copied_backend.section.cookie_backend, p2p_latency_cookie_for(UBENCH_SECTION_BACKEND));
}

// Participant values cannot alias the versioned layout cookie.
TEST(P2PLatencyMetalPeerArgs, LayoutGateAcceptsOnlyTheCurrentVersionedCookie) {
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE, 0x50324D03u);
    EXPECT_TRUE(ubench_layout_is_current(P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE, P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE));

    for (const uint32_t layout_cookie : {
             static_cast<uint32_t>(P2P_LATENCY_PARTICIPANT_A),
             static_cast<uint32_t>(P2P_LATENCY_PARTICIPANT_B),
             0x50324D00u,
             0x50325402u,
             0x50324501u,
             0x50324401u,
         }) {
        EXPECT_FALSE(ubench_layout_is_current(layout_cookie, P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE))
            << "old or unknown layout cookie " << layout_cookie;
    }
}

TEST(P2PLatencyMetalPeerArgs, SharedSlotOrderRemainsTheSixteenWordContract) {
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_LAYOUT_COOKIE, 0);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_SELF, 1);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_FIRST_DIRECTION, 2);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_PAYLOAD_SIZE, 3);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_WARMUP_COUNT, 4);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_SAMPLE_COUNT, 5);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_RUN_NONCE, 6);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_LO, 7);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_HI, 8);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_OWN_INBOUND_ADDR, 9);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_OWN_STAGING_ADDR, 10);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_OWN_RESULT_ADDR, 11);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_LO, 12);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_HI, 13);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_X, 14);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_Y, 15);
    EXPECT_EQ(P2P_LATENCY_METAL_PEER_ARG_COUNT, 16);
}

TEST(P2PLatencyMetalPeerArgs, BackendIdsProveTheSelectedImplementation) {
    const uint32_t ids[] = {
        P2P_LATENCY_BACKEND_ID_X280_METAL_PEER_TILE,
        P2P_LATENCY_BACKEND_ID_TENSIX_TENSIX_PEER_TILE,
        P2P_LATENCY_BACKEND_ID_ERISC_ERISC_PEER_TILE,
        P2P_LATENCY_BACKEND_ID_DRISC_DRISC_PEER_TILE,
    };
    for (unsigned i = 0; i < 4; ++i) {
        for (unsigned j = i + 1; j < 4; ++j) {
            EXPECT_NE(ids[i], ids[j]);
        }
    }
}

TEST(P2PLatencyMetalPeerArgs, SharedPlacementPackerAndReductionPreserveTheWireContract) {
    namespace peer = x280::suites::p2p_latency::metal_peer;
    const auto regions = peer::MetalRegions::compute({.local = {.base = 0x10000, .size = 0x20000}});
    EXPECT_EQ(regions.mailbox.addr, 0x10000u);
    EXPECT_EQ(regions.staging.addr, regions.mailbox.end());
    EXPECT_EQ(regions.result.addr, regions.staging.end());
    EXPECT_THROW(peer::MetalRegions::compute({.local = {.base = 0x10000, .size = 1}}), std::invalid_argument);
    const auto block = peer::make_x280_args(
        {.self = P2P_LATENCY_PARTICIPANT_A, .sample_count = 3},
        {.own_inbound_addr = 0x8001000,
         .own_result_addr = 0x8002000,
         .peer_inbound_addr = regions.mailbox.addr,
         .peer_noc = {7u, 8u}});
    EXPECT_EQ(block.cookie_common, p2p_latency_cookie_for(UBENCH_SECTION_COMMON));
    EXPECT_EQ(block.cookie_backend, p2p_latency_cookie_for(UBENCH_SECTION_BACKEND));
    EXPECT_EQ(block.common.sample_count, 3u);
    EXPECT_EQ(block.backend.peer_noc.x, 7u);
    EXPECT_EQ(block.backend.peer_noc.y, 8u);
    EXPECT_EQ(block.pad0, 0u);
    EXPECT_EQ(block.pad1, 0u);
    EXPECT_EQ(block.backend.reserved, 0u);
    p2p_latency_phase_record_t record{};
    EXPECT_THROW(peer::validate_phase_record(record), std::runtime_error);
    record.samples_completed = P2P_LATENCY_MAX_SAMPLES + 1;
    EXPECT_THROW(peer::validate_phase_record(record), std::runtime_error);
    record.samples_completed = P2P_LATENCY_MAX_SAMPLES;
    EXPECT_NO_THROW(peer::validate_phase_record(record));
}

// Spec (phase_samples_nonzero): every stored sample in both counters nonzero;
// the failure names the counter and the sample. Only samples_completed are read.
TEST(P2PLatencySampleBounds, AZeroSampleInEitherCounterFailsAndIsNamed) {
    namespace peer = x280::suites::p2p_latency::metal_peer;
    p2p_latency_phase_record_t ph{};
    ph.samples_completed = 3;
    for (uint32_t i = 0; i < 3; ++i) {
        ph.raw_deltas[i] = 10 + i;
        ph.secondary_deltas[i] = 20 + i;
    }
    EXPECT_TRUE(peer::phase_samples_nonzero(ph));
    ph.secondary_deltas[1] = 0;
    const auto result = peer::phase_samples_nonzero(ph);
    EXPECT_FALSE(result);
    EXPECT_NE(std::string(result.message()).find("secondary sample 1 of 3 is 0"), std::string::npos)
        << result.message();
    ph.samples_completed = 1;
    EXPECT_TRUE(peer::phase_samples_nonzero(ph));
}

// Spec (phase_record): a Metal clock is given exactly for a Metal participant.
TEST(P2PLatencyRecords, MetalFieldsAreRequiredExactlyForAMetalParticipant) {
    namespace peer = x280::suites::p2p_latency::metal_peer;
    p2p_latency_result_t x280{};
    x280.backend_id = P2P_LATENCY_BACKEND_ID_X280_METAL_PEER_TILE;
    x280.phases[0].samples_completed = 1;
    p2p_latency_result_t tensix = x280;
    tensix.backend_id = P2P_LATENCY_BACKEND_ID_TENSIX_TENSIX_PEER_TILE;
    const p2p_latency_common_args_t common{};
    const x280::harness::measurement::AiclkRunEvidence clock{};
    const auto metal = std::optional{peer::MetalClock{x280::harness::measurement::Counter::TensixLiveTimer, clock}};
    const x280::harness::measurement::WarmupSelection explicit_warmup{.warmup = 8};
    EXPECT_NO_THROW(peer::phase_record("m", {}, x280, 0, common, 0, NOC_ORDER_STRICT, std::nullopt, explicit_warmup));
    EXPECT_NO_THROW(peer::phase_record("m", {}, tensix, 0, common, 0, NOC_ORDER_STRICT, metal, explicit_warmup));
    EXPECT_THROW(
        peer::phase_record("m", {}, x280, 0, common, 0, NOC_ORDER_STRICT, metal, explicit_warmup),
        std::invalid_argument);
    EXPECT_THROW(
        peer::phase_record("m", {}, tensix, 0, common, 0, NOC_ORDER_STRICT, std::nullopt, explicit_warmup),
        std::invalid_argument);
}

TEST(P2PLatencyDriscArgs, WorkingRegionPlacementKeepsLocalAndTaggedViewsDistinct) {
    namespace hreg = x280::harness::regions;
    constexpr uint64_t unreserved_base = 0x10000u;
    constexpr uint64_t gcb_zone_size = 0x400u;
    const hreg::L1Window unreserved{
        .local = {.base = unreserved_base, .size = 0x20000u},
        .host_noc_base = UINT64_C(0x2000010000),
    };
    const auto working =
        hreg::subwindow(unreserved, unreserved_base + gcb_zone_size, unreserved.local.size - gcb_zone_size);
    EXPECT_EQ(working.local.base, 0x10400u);
    EXPECT_EQ(working.host_noc_base, UINT64_C(0x2000010400));
    EXPECT_THROW(hreg::subwindow(working, unreserved_base, hreg::kCacheLineBytes), std::invalid_argument);
    EXPECT_THROW(
        hreg::subwindow(
            hreg::L1Window{
                .local = {.base = 0u, .size = 2u},
                .host_noc_base = std::numeric_limits<uint64_t>::max(),
            },
            0u,
            2u),
        std::invalid_argument);

    const uint64_t rounded_result = hreg::round_up(sizeof(p2p_latency_result_t), hreg::kCacheLineBytes);
    const uint64_t required = P2P_LATENCY_MAILBOX_BYTES + P2P_LATENCY_STAGING_BYTES + rounded_result;
    const auto exact = hreg::subwindow(working, working.local.base, required);
    hreg::Arena arena(hreg::Region{exact.local.base, exact.local.size});
    const auto mailbox = arena.allocate(P2P_LATENCY_MAILBOX_BYTES);
    const auto staging = arena.allocate(P2P_LATENCY_STAGING_BYTES);
    const auto result = arena.allocate(sizeof(p2p_latency_result_t));
    EXPECT_EQ(arena.remaining(), 0u);
    EXPECT_EQ(hreg::host_noc_address(exact, mailbox.addr, mailbox.bytes), exact.host_noc_base);
    EXPECT_NE(mailbox.addr, exact.host_noc_base);
    EXPECT_EQ(staging.addr % hreg::kCacheLineBytes, 0u);
    EXPECT_EQ(result.addr % hreg::kCacheLineBytes, 0u);

    hreg::Arena undersized(hreg::Region{exact.local.base, exact.local.size - 1u});
    (void)undersized.allocate(P2P_LATENCY_MAILBOX_BYTES);
    (void)undersized.allocate(P2P_LATENCY_STAGING_BYTES);
    EXPECT_THROW((void)undersized.allocate(sizeof(p2p_latency_result_t)), std::invalid_argument);
}

TEST(P2PLatencyDriscArgs, PackerPreservesNoc0CoordinatesAndRawOnlyTiming) {
    namespace hargs = x280::harness::arguments;
    constexpr uint32_t warmup = 8u;
    constexpr uint32_t samples = 64u;
    constexpr uint32_t run_nonce = UINT32_MAX - 2u * (warmup + samples) + 1u;
    const uint64_t final_token =
        static_cast<uint64_t>(run_nonce) + 2ull * (static_cast<uint64_t>(warmup) + samples) - 1ull;
    ASSERT_EQ(final_token, UINT32_MAX);

    const auto args = hargs::pack_metal_peer_runtime_args(hargs::MetalPeerRuntimeArgs{
        .layout_cookie = P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE,
        .common =
            {
                .self = P2P_LATENCY_PARTICIPANT_B,
                .first_direction = P2P_LATENCY_DIRECTION_A_TO_B,
                .payload_size = 4u,
                .warmup_count = warmup,
                .sample_count = samples,
                .run_nonce = run_nonce,
                .warmup_max_poll_cycles = UINT64_C(3000000000),
            },
        .own_inbound_addr = 0x10400u,
        .own_staging_addr = 0x10440u,
        .own_result_addr = 0x10480u,
        .peer_inbound_addr = UINT64_C(0x80001000),
        .peer_noc = x280::testing::TranslatedCoordFactory::make(11u, 7u),
    });
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_LAYOUT_COOKIE], P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_X], 11u);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_Y], 7u);
}

TEST(P2PLatencyEriscArgs, RoundedRegionsFitExactlyAndRejectOneByteLess) {
    namespace hreg = x280::harness::regions;
    const uint64_t rounded_result = hreg::round_up(sizeof(p2p_latency_result_t), hreg::kCacheLineBytes);
    const uint64_t required = P2P_LATENCY_MAILBOX_BYTES + P2P_LATENCY_STAGING_BYTES + rounded_result;

    const hreg::L1Window reported_exact{
        .local = {.base = 0x10003u, .size = required + 61u},
        .host_noc_base = 0x90003u,
    };
    const auto aligned_exact = hreg::align_window_start(reported_exact, hreg::kCacheLineBytes);
    EXPECT_EQ(aligned_exact.local.base, 0x10040u);
    EXPECT_EQ(aligned_exact.local.size, required);
    EXPECT_EQ(aligned_exact.host_noc_base, 0x90040u);

    hreg::Arena exact(hreg::Region{aligned_exact.local.base, aligned_exact.local.size});
    const auto mailbox = exact.allocate(P2P_LATENCY_MAILBOX_BYTES);
    const auto staging = exact.allocate(P2P_LATENCY_STAGING_BYTES);
    const auto result = exact.allocate(sizeof(p2p_latency_result_t));
    EXPECT_EQ(exact.remaining(), 0u);
    EXPECT_EQ(mailbox.addr % hreg::kCacheLineBytes, 0u);
    EXPECT_EQ(staging.addr % hreg::kCacheLineBytes, 0u);
    EXPECT_EQ(result.addr % hreg::kCacheLineBytes, 0u);
    EXPECT_GE(result.bytes, sizeof(p2p_latency_result_t));

    hreg::L1Window reported_undersized = reported_exact;
    --reported_undersized.local.size;
    const auto aligned_undersized = hreg::align_window_start(reported_undersized, hreg::kCacheLineBytes);
    hreg::Arena undersized(hreg::Region{aligned_undersized.local.base, aligned_undersized.local.size});
    (void)undersized.allocate(P2P_LATENCY_MAILBOX_BYTES);
    (void)undersized.allocate(P2P_LATENCY_STAGING_BYTES);
    EXPECT_THROW((void)undersized.allocate(sizeof(p2p_latency_result_t)), std::invalid_argument);
}

TEST(P2PLatencyEriscArgs, WindowStartAlignmentRejectsInvalidInputs) {
    namespace hreg = x280::harness::regions;
    const hreg::L1Window valid{
        .local = {.base = 0x10003u, .size = 0x1000u},
        .host_noc_base = 0x90003u,
    };
    EXPECT_THROW(hreg::align_window_start(valid, 0u), std::invalid_argument);
    EXPECT_THROW(hreg::align_window_start(valid, 48u), std::invalid_argument);
    EXPECT_THROW(
        hreg::align_window_start(
            hreg::L1Window{
                .local = {.base = std::numeric_limits<uint64_t>::max(), .size = 1u},
                .host_noc_base = 0u,
            },
            hreg::kCacheLineBytes),
        std::invalid_argument);
    EXPECT_THROW(
        hreg::align_window_start(
            hreg::L1Window{
                .local = {.base = 0x10003u, .size = 1u},
                .host_noc_base = 0x90003u,
            },
            hreg::kCacheLineBytes),
        std::invalid_argument);
    EXPECT_THROW(
        hreg::align_window_start(
            hreg::L1Window{
                .local = {.base = 0x10003u, .size = 0x1000u},
                .host_noc_base = std::numeric_limits<uint64_t>::max() - 10u,
            },
            hreg::kCacheLineBytes),
        std::invalid_argument);
}

TEST(P2PLatencyMetalPeerArgs, SharedPackerPreservesEveryRuntimeField) {
    namespace hargs = x280::harness::arguments;
    constexpr uint64_t peer_address = UINT64_C(0x1122334455667788);
    constexpr uint32_t noc_x = 47u;
    constexpr uint32_t noc_y = 19u;
    constexpr uint64_t poll_budget = UINT64_C(0x1020304050607080);
    const auto args = hargs::pack_metal_peer_runtime_args(hargs::MetalPeerRuntimeArgs{
        .layout_cookie = P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE,
        .common =
            {
                .self = P2P_LATENCY_PARTICIPANT_B,
                .first_direction = P2P_LATENCY_DIRECTION_B_TO_A,
                .payload_size = 4u,
                .warmup_count = 17u,
                .sample_count = 19u,
                .run_nonce = 0x23456789u,
                .warmup_max_poll_cycles = poll_budget,
            },
        .own_inbound_addr = 0x10000u,
        .own_staging_addr = 0x10100u,
        .own_result_addr = 0x10200u,
        .peer_inbound_addr = peer_address,
        .peer_noc = x280::testing::TranslatedCoordFactory::make(noc_x, noc_y),
    });

    ASSERT_EQ(args.size(), P2P_LATENCY_METAL_PEER_ARG_COUNT);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_LAYOUT_COOKIE], P2P_LATENCY_METAL_PEER_LAYOUT_COOKIE);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_SELF], P2P_LATENCY_PARTICIPANT_B);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_FIRST_DIRECTION], P2P_LATENCY_DIRECTION_B_TO_A);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_PAYLOAD_SIZE], 4u);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_WARMUP_COUNT], 17u);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_SAMPLE_COUNT], 19u);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_RUN_NONCE], 0x23456789u);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_LO], static_cast<uint32_t>(poll_budget));
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_WARMUP_MAX_POLL_HI], static_cast<uint32_t>(poll_budget >> 32));
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_OWN_INBOUND_ADDR], 0x10000u);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_OWN_STAGING_ADDR], 0x10100u);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_OWN_RESULT_ADDR], 0x10200u);

    const uint64_t unpacked = (static_cast<uint64_t>(args[P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_HI]) << 32) |
                              args[P2P_LATENCY_METAL_PEER_ARG_PEER_INBOUND_ADDR_LO];
    EXPECT_EQ(unpacked, peer_address);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_X], noc_x);
    EXPECT_EQ(args[P2P_LATENCY_METAL_PEER_ARG_PEER_NOC_Y], noc_y);
}

// Runs both participants of the A_TO_B first phase (A initiates, B responds)
// through the frontend on `script`, expecting phase 0 to end `state` after
// `completed` samples.
void ExpectPhaseZeroEnds(const ScriptedBackend::state_t& script, p2p_latency_state_t state, uint32_t completed) {
    for (const auto self : {P2P_LATENCY_PARTICIPANT_A, P2P_LATENCY_PARTICIPANT_B}) {
        p2p_latency_common_args_t args{};
        args.self = self;
        args.first_direction = P2P_LATENCY_DIRECTION_A_TO_B;
        args.payload_size = 4u;
        args.warmup_count = 2u;
        args.sample_count = 5u;
        args.run_nonce = 1u;
        args.warmup_max_poll_cycles = 100u;
        ScriptedBackend::state_t s = script;
        auto result = std::make_unique<p2p_latency_result_t>();

        EXPECT_FALSE((p2p_latency::run_participant<
                      ScriptedBackend,
                      P2P_LATENCY_TRANSFER_TOKEN_ONLY,
                      P2P_LATENCY_POLL_TIGHT,
                      P2P_LATENCY_TRANSMIT_WAIT_FOR_COMPLETION>(s, args, *result)));
        EXPECT_EQ(result->phases[0].terminal_state, static_cast<uint32_t>(state)) << "participant " << self;
        EXPECT_EQ(result->phases[0].samples_completed, completed) << "participant " << self;
        EXPECT_EQ(result->terminal_state, static_cast<uint32_t>(state)) << "participant " << self;
    }
}

TEST(P2pLatencyFrontend, TimedPollFailureEndsThePhaseTimeout) {
    ExpectPhaseZeroEnds({.timed_polls_left = 3u}, P2P_LATENCY_STATE_TIMEOUT, 3u);
}

TEST(P2pLatencyFrontend, RefusedDeltaEndsThePhaseTrap) {
    ExpectPhaseZeroEnds({.deltas_left = 3u}, P2P_LATENCY_STATE_TRAP, 3u);
}

}  // namespace
