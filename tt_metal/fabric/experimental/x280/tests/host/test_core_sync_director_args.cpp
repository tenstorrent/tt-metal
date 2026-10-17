// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core_sync_barrier.hpp"
#include "core_sync_director_args.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

/* Returns a complete two-participant director section for mutation tests. */
core_sync_director_section_t valid_director_section() {
    core_sync_director_section_t section{};
    section.cookie_common = core_sync_director_cookie_for(UBENCH_SECTION_COMMON);
    section.cookie_backend = core_sync_director_cookie_for(UBENCH_SECTION_BACKEND);
    section.common.run_nonce = 1u;
    section.common.role = CORE_SYNC_DIRECTOR_ROLE_DIRECTOR;
    section.backend.result_addr = 0x2000u;
    section.backend.ready_base = 0x1000u;
    section.backend.ready_bytes = 2u * CORE_SYNC_SLOT_BYTES;
    section.backend.arrival_count = 2u;
    section.backend.grid = {{{1u, 1u}, {1u, 1u}}, 0x3000u, 1u};
    section.backend.endpoint_count = 1u;
    section.backend.endpoints[0] = {{2u, 2u}, 0x4000u, 0u};
    section.backend.arrivals[0] = {0x1000u, 0u};
    section.backend.arrivals[1] = {0x1008u, 1u};
    return section;
}

/* Returns a complete participant section for mutation tests. */
core_sync_director_section_t valid_participant_section() {
    core_sync_director_section_t section{};
    section.cookie_common = core_sync_director_cookie_for(UBENCH_SECTION_COMMON);
    section.cookie_backend = core_sync_director_cookie_for(UBENCH_SECTION_BACKEND);
    section.common.run_nonce = 1u;
    section.common.role = CORE_SYNC_DIRECTOR_ROLE_PARTICIPANT;
    section.backend.result_addr = 0x2000u;
    section.backend.local_release_addr = 0x3000u;
    section.backend.arrival = {{{1u, 2u}, 0x4000u, 1u, 0u}, 2u * CORE_SYNC_SLOT_BYTES};
    return section;
}

TEST(CoreSyncWireValidation, AcceptsCompleteRoleSpecificSections) {
    const auto director = valid_director_section();
    const auto participant = valid_participant_section();

    EXPECT_TRUE(core_sync_director_validate(&director));
    EXPECT_TRUE(core_sync_director_validate(&participant));
}

// One tile's release word can wake several participants, so arrivals need not
// equal the release destinations.
TEST(CoreSyncWireValidation, CountsArrivalsIndependentlyOfReleaseTargets) {
    auto director = valid_director_section();
    director.backend.endpoint_count = 0u;
    EXPECT_TRUE(core_sync_director_validate(&director));
}

TEST(CoreSyncWireValidation, RejectsMalformedDirectorSections) {
    using Mutation = std::pair<const char*, void (*)(core_sync_director_section_t&)>;
    const std::array<Mutation, 28> mutations{{
        {"common cookie", [](auto& s) { s.cookie_common ^= 1u; }},
        {"backend cookie", [](auto& s) { s.cookie_backend ^= 1u; }},
        {"zero nonce", [](auto& s) { s.common.run_nonce = 0u; }},
        {"unknown role", [](auto& s) { s.common.role = 99u; }},
        {"zero result address", [](auto& s) { s.backend.result_addr = 0u; }},
        {"misaligned result address", [](auto& s) { s.backend.result_addr = 0x2002u; }},
        {"endpoint count overflow", [](auto& s) { s.backend.endpoint_count = CORE_SYNC_DIRECTOR_MAX_ENDPOINTS + 1u; }},
        {"arrival count overflow", [](auto& s) { s.backend.arrival_count = CORE_SYNC_DIRECTOR_MAX_ARRIVALS + 1u; }},
        {"empty director", [](auto& s) { s.backend.arrival_count = 0u; }},
        {"no release target",
         [](auto& s) {
             s.backend.endpoint_count = 0u;
             s.backend.grid = {};
         }},
        {"zero ready base", [](auto& s) { s.backend.ready_base = 0u; }},
        {"misaligned ready base", [](auto& s) { s.backend.ready_base += 1u; }},
        {"misaligned ready size", [](auto& s) { s.backend.ready_bytes -= 1u; }},
        {"zero ready size", [](auto& s) { s.backend.ready_bytes = 0u; }},
        {"undersized ready region", [](auto& s) { s.backend.ready_bytes = CORE_SYNC_SLOT_BYTES; }},
        {"overflowing ready region",
         [](auto& s) {
             s.backend.ready_base = UINT32_MAX - 7u;
             s.backend.ready_bytes = 2u * CORE_SYNC_SLOT_BYTES;
         }},
        {"participant release field", [](auto& s) { s.backend.local_release_addr = 8u; }},
        {"participant arrival field", [](auto& s) { s.backend.arrival.target.ready_base = 8u; }},
        {"empty grid address", [](auto& s) { s.backend.grid.addr = 0u; }},
        {"reversed grid", [](auto& s) { s.backend.grid.rect.start.x = 2u; }},
        {"grid corner outside six bits", [](auto& s) { s.backend.grid.rect.end.x = 64u; }},
        {"fields on an empty grid",
         [](auto& s) {
             s.backend.grid = {{{1u, 0u}, {0u, 0u}}, 0u, 0u};
             s.backend.endpoint_count = 2u;
             s.backend.endpoints[1] = {{3u, 3u}, 0x5000u, 0u};
         }},
        {"empty endpoint address", [](auto& s) { s.backend.endpoints[0].addr = 0u; }},
        {"endpoint outside six bits", [](auto& s) { s.backend.endpoints[0].noc.x = 64u; }},
        {"invalid endpoint kind", [](auto& s) { s.backend.endpoints[0].same_tile = 2u; }},
        {"misaligned arrival address", [](auto& s) { s.backend.arrivals[1].addr += 1u; }},
        {"arrival index outside population", [](auto& s) { s.backend.arrivals[1].worker_index = 2u; }},
        {"duplicate arrival", [](auto& s) { s.backend.arrivals[1] = s.backend.arrivals[0]; }},
    }};

    for (const auto& [name, mutate] : mutations) {
        auto section = valid_director_section();
        mutate(section);
        EXPECT_FALSE(core_sync_director_validate(&section)) << name;
    }
}

TEST(CoreSyncWireValidation, RejectsMalformedParticipantSections) {
    using Mutation = std::pair<const char*, void (*)(core_sync_director_section_t&)>;
    const std::array<Mutation, 16> mutations{{
        {"zero release address", [](auto& s) { s.backend.local_release_addr = 0u; }},
        {"misaligned release address", [](auto& s) { s.backend.local_release_addr += 1u; }},
        {"director ready field", [](auto& s) { s.backend.ready_base = 8u; }},
        {"director arrival count", [](auto& s) { s.backend.arrival_count = 1u; }},
        {"director grid field", [](auto& s) { s.backend.grid.addr = 8u; }},
        {"director endpoint count", [](auto& s) { s.backend.endpoint_count = 1u; }},
        {"arrival outside six bits", [](auto& s) { s.backend.arrival.target.noc.y = 64u; }},
        {"zero arrival base", [](auto& s) { s.backend.arrival.target.ready_base = 0u; }},
        {"misaligned arrival base", [](auto& s) { s.backend.arrival.target.ready_base += 1u; }},
        {"misaligned arrival size", [](auto& s) { s.backend.arrival.ready_bytes -= 1u; }},
        {"zero arrival size", [](auto& s) { s.backend.arrival.ready_bytes = 0u; }},
        {"arrival index outside capacity", [](auto& s) { s.backend.arrival.target.worker_index = 2u; }},
        {"overflowing arrival region",
         [](auto& s) {
             s.backend.arrival.target.ready_base = UINT32_MAX - 7u;
             s.backend.arrival.ready_bytes = 2u * CORE_SYNC_SLOT_BYTES;
         }},
        {"same-tile arrival", [](auto& s) { s.backend.arrival.target.same_tile = 1u; }},
        {"populated trailing endpoint", [](auto& s) { s.backend.endpoints[0].addr = 8u; }},
        {"populated trailing record", [](auto& s) { s.backend.arrivals[0].addr = 8u; }},
    }};

    for (const auto& [name, mutate] : mutations) {
        auto section = valid_participant_section();
        mutate(section);
        EXPECT_FALSE(core_sync_director_validate(&section)) << name;
    }
}

TEST(CoreSyncWireValidation, AcceptsAReadyRegionEndingAtTheAddressSpaceBoundary) {
    auto director = valid_director_section();
    director.backend.ready_base = UINT32_MAX - 15u;
    director.backend.arrivals[0].addr = UINT32_MAX - 15u;
    director.backend.arrivals[1].addr = UINT32_MAX - 7u;

    EXPECT_TRUE(core_sync_director_validate(&director));
}

}  // namespace

// Only a completed director admits a measurement, and a refused or failed one
// never reads as still in progress, whatever barrier outcome it carries.
TEST(CoreSyncOutcome, OnlyDoneIsAMeasurementAndEveryFailureIsTerminal) {
    using core_sync::Outcome;
    for (const Outcome o : {Outcome::Success, Outcome::StopRequested, Outcome::ConfigError}) {
        const ubench_outcome_t barrier = core_sync::ubench_outcome_of(o);
        EXPECT_EQ(ubench_outcome_is_measurement(barrier), o == Outcome::Success)
            << "barrier outcome " << static_cast<uint32_t>(o);
        for (const core_sync_director_state_t state :
             {CORE_SYNC_DIRECTOR_STATE_INIT,
              CORE_SYNC_DIRECTOR_STATE_DONE,
              CORE_SYNC_DIRECTOR_STATE_ARG_LAYOUT_MISMATCH,
              CORE_SYNC_DIRECTOR_STATE_SYNC_FAILURE}) {
            const ubench_outcome_t outcome = core_sync_director_outcome(state, barrier);
            EXPECT_EQ(ubench_outcome_is_measurement(outcome), state == CORE_SYNC_DIRECTOR_STATE_DONE)
                << "state " << state << ", barrier outcome " << static_cast<uint32_t>(o);
            EXPECT_EQ(outcome == UBENCH_OUTCOME_IN_PROGRESS, state == CORE_SYNC_DIRECTOR_STATE_INIT)
                << "state " << state << ", barrier outcome " << static_cast<uint32_t>(o);
        }
    }
}
