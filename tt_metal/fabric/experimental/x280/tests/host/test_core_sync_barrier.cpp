// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core_sync_barrier.hpp"
#include "core_sync_host_transport.hpp"
#include "core_sync_layout.h"
#include "core_sync_transport_metal.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

using WordTransport = core_sync::HostTransport<false, core_sync::ArrivalKind::Word>;
using CounterTransport = core_sync::HostTransport<false, core_sync::ArrivalKind::Counter>;
using MulticastTransport = core_sync::HostTransport<true, core_sync::ArrivalKind::Word>;
using DriscWordTransport = core_sync::MetalTransport<ProgrammableCoreType::DRAM, core_sync::ArrivalKind::Word>;

constexpr uint32_t kReadyBase = 0x1000u;
constexpr uint32_t kReleaseAddr = 0x2000u;
constexpr uint32_t kCounterAddr = 0x9000u;
constexpr uint32_t kWorkerIndex = 2u;

struct WordWorkerConfig {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = false;
    static constexpr uint32_t kLocalReleaseAddr = kReleaseAddr;
    static constexpr core_sync::HostArrivalTarget kArrivalTarget{kReadyBase, kWorkerIndex};
};

struct CounterWorkerConfig : WordWorkerConfig {
    static constexpr core_sync::HostArrivalTarget kArrivalTarget{kCounterAddr, 0u};
};

constexpr core_sync::X280Endpoint kEndpoints[] = {
    {{1u, 1u}, 0x3000u, false},
    {{1u, 2u}, 0x3008u, false},
};

constexpr core_sync::WordArrival kWordArrivals[] = {
    {core_sync_slot_addr(kReadyBase, kWorkerIndex), kWorkerIndex},
};

constexpr core_sync::WordArrival kMixedWordArrivals[] = {
    {core_sync_slot_addr(kReadyBase, 2u), 2u},
    {core_sync_slot_addr(kReadyBase, 3u), 3u},
};

constexpr core_sync::HostGroup kReleaseGroup{{{10u, 10u}, {11u, 10u}}, 0x4000u, 2u};
constexpr core_sync::HostGroup kSingletonGroup{{{10u, 10u}, {10u, 10u}}, 0x4000u, 1u};
constexpr core_sync::HostGroup kThreeGroups[] = {
    {{{10u, 10u}, {10u, 10u}}, 0x4000u, 1u},
    {{{12u, 10u}, {13u, 10u}}, 0x4100u, 2u},
    {{{15u, 10u}, {17u, 10u}}, 0x4200u, 3u},
};

constexpr core_sync::HostGroup kMalformedGroups[] = {
    {{{10u, 10u}, {10u, 10u}}, 0x4000u, 1u},
    {{{12u, 10u}, {11u, 10u}}, 0x4100u, 1u},
};

/* Compares every field of the production group value used by the host fake. */
void expect_group_eq(const core_sync::HostGroup& actual, const core_sync::HostGroup& expected) {
    EXPECT_EQ(actual.rect.start.x, expected.rect.start.x);
    EXPECT_EQ(actual.rect.start.y, expected.rect.start.y);
    EXPECT_EQ(actual.rect.end.x, expected.rect.end.x);
    EXPECT_EQ(actual.rect.end.y, expected.rect.end.y);
    EXPECT_EQ(actual.addr, expected.addr);
    EXPECT_EQ(actual.num_dests, expected.num_dests);
}

struct DirectorConfig {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = true;
    static constexpr core_sync::HostRoster kRoster{
        .endpoints = kEndpoints,
        .endpoint_count = 2u,
        .counter_addr = kCounterAddr,
        .counter_arrivals = 1u,
        .word_arrivals = kWordArrivals,
        .word_arrival_count = 1u};
};

struct MixedDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{
        .endpoints = kEndpoints,
        .endpoint_count = 2u,
        .counter_addr = kCounterAddr,
        .counter_arrivals = 2u,
        .word_arrivals = kMixedWordArrivals,
        .word_arrival_count = 2u,
        .groups = &kReleaseGroup,
        .group_count = 1u};
};

struct EmptyDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{};
};

/* Arrivals present, no release target: rejected because release would write
 * nothing, not because arrivals and targets disagree in count. */
struct EmptyReleaseDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{.counter_addr = kCounterAddr, .counter_arrivals = 1u};
};

/* Two arrivals, one release group. The barrier does not equate the counts. */
struct SharedTargetDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{
        .counter_addr = kCounterAddr, .counter_arrivals = 2u, .groups = &kReleaseGroup, .group_count = 1u};
};

struct NullGroupsDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{
        .counter_addr = kCounterAddr, .counter_arrivals = 1u, .groups = nullptr, .group_count = 1u};
};

struct NullArrivalsDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{
        .word_arrivals = nullptr, .word_arrival_count = 1u, .groups = &kSingletonGroup, .group_count = 1u};
};

struct NullEndpointsDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{
        .endpoints = nullptr, .endpoint_count = 1u, .counter_addr = kCounterAddr, .counter_arrivals = 1u};
};

struct MalformedGroupsDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{
        .counter_addr = kCounterAddr, .counter_arrivals = 1u, .groups = kMalformedGroups, .group_count = 2u};
};

struct SingletonGroupDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{
        .counter_addr = kCounterAddr, .counter_arrivals = 1u, .groups = &kSingletonGroup, .group_count = 1u};
};

struct MultiGroupDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{
        .counter_addr = kCounterAddr, .counter_arrivals = 3u, .groups = kThreeGroups, .group_count = 3u};
};

struct GroupsThenEndpointsDirectorConfig : DirectorConfig {
    static constexpr core_sync::HostRoster kRoster{
        .endpoints = kEndpoints,
        .endpoint_count = 2u,
        .counter_addr = kCounterAddr,
        .counter_arrivals = 3u,
        .groups = kThreeGroups,
        .group_count = 3u};
};

using DisabledBarrier = core_sync::core_sync_barrier<core_sync::DisabledSyncConfig, WordTransport>;
using WordWorkerBarrier = core_sync::core_sync_barrier<WordWorkerConfig, WordTransport>;
using CounterWorkerBarrier = core_sync::core_sync_barrier<CounterWorkerConfig, CounterTransport>;
using DirectorBarrier = core_sync::core_sync_barrier<DirectorConfig, WordTransport>;
using MixedDirectorBarrier = core_sync::core_sync_barrier<MixedDirectorConfig, MulticastTransport>;
using EmptyDirectorBarrier = core_sync::core_sync_barrier<EmptyDirectorConfig, WordTransport>;
using EmptyReleaseDirectorBarrier = core_sync::core_sync_barrier<EmptyReleaseDirectorConfig, WordTransport>;
using SharedTargetDirectorBarrier = core_sync::core_sync_barrier<SharedTargetDirectorConfig, MulticastTransport>;
using NullGroupsDirectorBarrier = core_sync::core_sync_barrier<NullGroupsDirectorConfig, MulticastTransport>;
using NullArrivalsDirectorBarrier = core_sync::core_sync_barrier<NullArrivalsDirectorConfig, MulticastTransport>;
using NullEndpointsDirectorBarrier = core_sync::core_sync_barrier<NullEndpointsDirectorConfig, WordTransport>;
using MalformedGroupsDirectorBarrier = core_sync::core_sync_barrier<MalformedGroupsDirectorConfig, MulticastTransport>;
using SingletonGroupDirectorBarrier = core_sync::core_sync_barrier<SingletonGroupDirectorConfig, MulticastTransport>;
using MultiGroupDirectorBarrier = core_sync::core_sync_barrier<MultiGroupDirectorConfig, MulticastTransport>;
using GroupsThenEndpointsDirectorBarrier =
    core_sync::core_sync_barrier<GroupsThenEndpointsDirectorConfig, MulticastTransport>;

TEST(CoreSyncLayout, TokenAndSlotArithmetic) {
    constexpr uint32_t nonce = 0x12345678u;
    constexpr uint32_t other_nonce = 0x87654321u;

    EXPECT_EQ(core_sync_slot_addr(kReadyBase, 3u), kReadyBase + 3u * CORE_SYNC_SLOT_BYTES);
    EXPECT_NE(core_sync_ready_token(nonce, 0u), core_sync_ready_token(nonce, 1u));
    EXPECT_NE(core_sync_ready_token(nonce, 0u), core_sync_release_token(nonce));
    EXPECT_NE(core_sync_ready_token(nonce, 0u), core_sync_ready_token(other_nonce, 0u));
    EXPECT_NE(core_sync_release_token(nonce), core_sync_release_token(other_nonce));
}

TEST(CoreSyncTransportPolicy, DriscUsesRawReleaseStorageAndWordArrival) {
    EXPECT_FALSE(DriscWordTransport::kUsesSemaphoreRelease);
    EXPECT_EQ(DriscWordTransport::kArrivalKind, core_sync::ArrivalKind::Word);

    constexpr auto endpoint = core_sync::MetalEndpoint::raw_l1({3u, 4u}, 0x5000u, ProgrammableCoreType::DRAM);
    EXPECT_EQ(endpoint.core_type, ProgrammableCoreType::DRAM);
    EXPECT_FALSE(endpoint.is_semaphore);
    EXPECT_EQ(endpoint.semaphore_id, 0x5000u);
}

TEST(CoreSyncBarrier, DisabledConfigDoesNotUseTransport) {
    WordTransport::reset();

    EXPECT_EQ(DisabledBarrier::sync(3u, 4u, 5u), core_sync::Outcome::Success);
    EXPECT_TRUE(WordTransport::calls.empty());
}

TEST(CoreSyncBarrier, WordWorkerAnnouncesBeforeWaiting) {
    WordTransport::reset();
    constexpr uint32_t nonce = 0x55AA55AAu;

    EXPECT_EQ(WordWorkerBarrier::sync(7u, 8u, nonce), core_sync::Outcome::Success);

    ASSERT_EQ(WordTransport::calls.size(), 2u);
    const auto& arrival = WordTransport::calls[0];
    EXPECT_EQ(arrival.kind, core_sync::HostCallKind::NotifyArrival);
    EXPECT_EQ(arrival.arrival_target.addr, kReadyBase);
    EXPECT_EQ(arrival.arrival_target.worker_index, kWorkerIndex);
    EXPECT_EQ(arrival.run_nonce, nonce);

    const auto& wait = WordTransport::calls[1];
    EXPECT_EQ(wait.kind, core_sync::HostCallKind::Wait);
    EXPECT_EQ(wait.addr, kReleaseAddr);
    EXPECT_EQ(wait.value, core_sync_release_token(nonce));
    EXPECT_EQ(wait.hartid, 7u);
    EXPECT_EQ(wait.run_value, 8u);
}

TEST(CoreSyncBarrier, CounterWorkerUsesCounterTarget) {
    CounterTransport::reset();
    constexpr uint32_t nonce = 0x0F0F1234u;

    EXPECT_EQ(CounterWorkerBarrier::sync(0u, 0u, nonce), core_sync::Outcome::Success);

    ASSERT_EQ(CounterTransport::calls.size(), 2u);
    EXPECT_EQ(CounterTransport::kArrivalKind, core_sync::ArrivalKind::Counter);
    EXPECT_EQ(CounterTransport::calls[0].kind, core_sync::HostCallKind::NotifyArrival);
    EXPECT_EQ(CounterTransport::calls[0].arrival_target.addr, kCounterAddr);
    EXPECT_EQ(CounterTransport::calls[0].arrival_target.worker_index, 0u);
}

TEST(CoreSyncBarrier, ArrivalRefusalIsConfigError) {
    WordTransport::reset();
    WordTransport::notify_arrival_result = false;

    EXPECT_EQ(WordWorkerBarrier::sync(0u, 0u, 1u), core_sync::Outcome::ConfigError);
    ASSERT_EQ(WordTransport::calls.size(), 1u);
    EXPECT_EQ(WordTransport::calls[0].kind, core_sync::HostCallKind::NotifyArrival);
}

TEST(CoreSyncBarrier, ReleaseWaitInterruptionIsStopRequested) {
    WordTransport::reset();
    WordTransport::wait_result = false;

    EXPECT_EQ(WordWorkerBarrier::sync(0u, 9u, 1u), core_sync::Outcome::StopRequested);
    ASSERT_EQ(WordTransport::calls.size(), 2u);
    EXPECT_EQ(WordTransport::calls[1].kind, core_sync::HostCallKind::Wait);
    EXPECT_EQ(WordTransport::calls[1].run_value, 9u);
}

TEST(CoreSyncBarrier, DirectorWaitsBeforeReleasing) {
    WordTransport::reset();
    constexpr uint32_t nonce = 0x12344321u;

    EXPECT_EQ(DirectorBarrier::sync(4u, 5u, nonce), core_sync::Outcome::Success);

    ASSERT_EQ(WordTransport::calls.size(), 2u);
    const auto& wait = WordTransport::calls[0];
    EXPECT_EQ(wait.kind, core_sync::HostCallKind::WaitForArrivals);
    ASSERT_EQ(wait.roster.endpoints.size(), 2u);
    EXPECT_EQ(wait.roster.endpoints[0], kEndpoints[0]);
    EXPECT_EQ(wait.roster.endpoints[1], kEndpoints[1]);
    EXPECT_EQ(wait.roster.endpoint_count, 2u);
    EXPECT_EQ(wait.roster.counter_addr, kCounterAddr);
    EXPECT_EQ(wait.roster.counter_arrivals, 1u);
    ASSERT_EQ(wait.roster.word_arrivals.size(), 1u);
    EXPECT_EQ(wait.roster.word_arrivals[0].addr, kWordArrivals[0].addr);
    EXPECT_EQ(wait.roster.word_arrivals[0].worker_index, kWordArrivals[0].worker_index);
    EXPECT_EQ(wait.roster.word_arrival_count, 1u);
    EXPECT_EQ(wait.run_nonce, nonce);
    EXPECT_EQ(wait.hartid, 4u);
    EXPECT_EQ(wait.run_value, 5u);

    const auto& release = WordTransport::calls[1];
    EXPECT_EQ(release.kind, core_sync::HostCallKind::NotifyGroup);
    ASSERT_EQ(release.roster.endpoints.size(), 2u);
    EXPECT_EQ(release.roster.endpoints[0], kEndpoints[0]);
    EXPECT_EQ(release.roster.endpoints[1], kEndpoints[1]);
    EXPECT_EQ(release.value, core_sync_release_token(nonce));
}

TEST(CoreSyncBarrier, FailedDirectorWaitReleasesNobody) {
    WordTransport::reset();
    WordTransport::wait_for_arrivals_result = false;

    EXPECT_EQ(DirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::StopRequested);
    ASSERT_EQ(WordTransport::calls.size(), 1u);
    EXPECT_EQ(WordTransport::calls[0].kind, core_sync::HostCallKind::WaitForArrivals);
}

TEST(CoreSyncBarrier, ReleaseRefusalIsConfigError) {
    WordTransport::reset();
    WordTransport::notify_group_result = false;

    EXPECT_EQ(DirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::ConfigError);
    ASSERT_EQ(WordTransport::calls.size(), 2u);
    EXPECT_EQ(WordTransport::calls[1].kind, core_sync::HostCallKind::NotifyGroup);
}

TEST(CoreSyncBarrier, EmptyDirectorRosterIsConfigError) {
    WordTransport::reset();

    EXPECT_EQ(EmptyDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::ConfigError);
    EXPECT_TRUE(WordTransport::calls.empty());
}

TEST(CoreSyncBarrier, EmptyReleaseRosterIsConfigError) {
    WordTransport::reset();

    EXPECT_EQ(EmptyReleaseDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::ConfigError);
    EXPECT_TRUE(WordTransport::calls.empty());
}

TEST(CoreSyncBarrier, ArrivalsMayShareReleaseTarget) {
    MulticastTransport::reset();
    constexpr uint32_t nonce = 0x0BADF00Du;

    EXPECT_EQ(SharedTargetDirectorBarrier::sync(0u, 0u, nonce), core_sync::Outcome::Success);

    ASSERT_EQ(MulticastTransport::calls.size(), 2u);
    EXPECT_EQ(MulticastTransport::calls[0].kind, core_sync::HostCallKind::WaitForArrivals);
    const auto& release = MulticastTransport::calls[1];
    EXPECT_EQ(release.kind, core_sync::HostCallKind::NotifyGroup);
    ASSERT_EQ(release.releases.size(), 1u);
    EXPECT_EQ(release.releases[0].kind, core_sync::HostReleaseKind::Group);
    expect_group_eq(release.releases[0].group, kReleaseGroup);
}

TEST(CoreSyncX280Roster, InvalidBorrowedArraysFailBeforeWaiting) {
    MulticastTransport::reset();
    EXPECT_EQ(NullGroupsDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::ConfigError);
    EXPECT_TRUE(MulticastTransport::calls.empty());

    MulticastTransport::reset();
    EXPECT_EQ(NullArrivalsDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::ConfigError);
    EXPECT_TRUE(MulticastTransport::calls.empty());

    WordTransport::reset();
    EXPECT_EQ(NullEndpointsDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::ConfigError);
    EXPECT_TRUE(WordTransport::calls.empty());
}

TEST(CoreSyncX280Roster, MalformedLaterGroupFailsBeforeWaitingOrReleasing) {
    MulticastTransport::reset();
    MulticastTransport::wait_for_arrivals_result = false;

    EXPECT_EQ(MalformedGroupsDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::ConfigError);
    EXPECT_TRUE(MulticastTransport::calls.empty());
}

TEST(CoreSyncX280Roster, ProductionPreflightRejectsMalformedTargets) {
    constexpr core_sync::WordArrival arrivals[] = {{0x1000u, 0u}};
    constexpr core_sync::X280Endpoint endpoints[] = {
        {{1u, 1u}, 0x2000u, false},
    };
    constexpr core_sync::X280Group groups[] = {
        {{{10u, 10u}, {10u, 10u}}, 0x3000u, 1u},
    };
    core_sync::X280Roster roster{
        .endpoints = endpoints,
        .endpoint_count = 1u,
        .word_arrivals = arrivals,
        .word_arrival_count = 1u,
        .groups = groups,
        .group_count = 1u};
    EXPECT_TRUE(core_sync::X280Transport<>::validate_roster(roster));

    roster.word_arrivals = nullptr;
    EXPECT_FALSE(core_sync::X280Transport<>::validate_roster(roster));
    roster.word_arrivals = arrivals;

    roster.endpoints = nullptr;
    EXPECT_FALSE(core_sync::X280Transport<>::validate_roster(roster));
    roster.endpoints = endpoints;

    roster.groups = nullptr;
    EXPECT_FALSE(core_sync::X280Transport<>::validate_roster(roster));
    roster.groups = groups;

    auto malformed_group = groups[0];
    malformed_group.rect.start.x = malformed_group.rect.end.x + 1u;
    roster.groups = &malformed_group;
    EXPECT_FALSE(core_sync::X280Transport<>::validate_roster(roster));
    roster.groups = groups;

    auto overcounted_group = groups[0];
    overcounted_group.num_dests = 2u;
    roster.groups = &overcounted_group;
    EXPECT_FALSE(core_sync::X280Transport<>::validate_roster(roster))
        << "a one-tile rectangle cannot release two destinations";
    roster.groups = groups;

    auto malformed_endpoint = endpoints[0];
    malformed_endpoint.noc.x = 64u;
    roster.endpoints = &malformed_endpoint;
    EXPECT_FALSE(core_sync::X280Transport<>::validate_roster(roster));

    auto covered_endpoint = endpoints[0];
    covered_endpoint.noc = groups[0].rect.start;
    roster.endpoints = &covered_endpoint;
    EXPECT_FALSE(core_sync::X280Transport<>::validate_roster(roster))
        << "a tile the group reaches would be released twice";
}

TEST(CoreSyncMetalRoster, PreflightRejectsOutOfRangeTargets) {
    using Transport = core_sync::MetalTransport<ProgrammableCoreType::TENSIX>;
    core_sync::MetalEndpoint endpoint({1u, 1u}, 2u, ProgrammableCoreType::TENSIX);
    core_sync::MetalRoster roster{.endpoints = &endpoint, .count = 1u};
    EXPECT_TRUE(Transport::validate_roster(roster));

    endpoint.noc.y = 64u;
    EXPECT_FALSE(Transport::validate_roster(roster));
    endpoint.noc.y = 1u;

    roster.grid = core_sync::MetalGroup{{{3u, 3u}, {2u, 3u}}, 2u, 1u};
    EXPECT_FALSE(Transport::validate_roster(roster));
}

TEST(CoreSyncHostTransport, RecordedRosterOwnsBorrowedArrays) {
    MulticastTransport::reset();
    core_sync::WordArrival arrivals[] = {{0x1000u, 0u}};
    core_sync::X280Endpoint endpoints[] = {{{1u, 1u}, 0x2000u, false}};
    core_sync::HostGroup groups[] = {{{{10u, 10u}, {10u, 10u}}, 0x3000u, 1u}};
    const core_sync::HostRoster roster{
        .endpoints = endpoints,
        .endpoint_count = 1u,
        .word_arrivals = arrivals,
        .word_arrival_count = 1u,
        .groups = groups,
        .group_count = 1u};
    const core_sync::HostWaitContext context{4u, 5u};

    ASSERT_TRUE(MulticastTransport::wait_for_arrivals(roster, 6u, context));
    ASSERT_TRUE(MulticastTransport::notify_group(roster, 7u));
    arrivals[0] = {0x4000u, 3u};
    endpoints[0] = {{2u, 2u}, 0x5000u, false};
    groups[0] = {{{20u, 20u}, {20u, 20u}}, 0x6000u, 1u};

    ASSERT_EQ(MulticastTransport::calls.size(), 2u);
    for (const auto& call : MulticastTransport::calls) {
        ASSERT_EQ(call.roster.word_arrivals.size(), 1u);
        EXPECT_EQ(call.roster.word_arrivals[0].addr, 0x1000u);
        EXPECT_EQ(call.roster.word_arrivals[0].worker_index, 0u);
        ASSERT_EQ(call.roster.endpoints.size(), 1u);
        EXPECT_EQ(call.roster.endpoints[0].addr, 0x2000u);
        ASSERT_EQ(call.roster.groups.size(), 1u);
        EXPECT_EQ(call.roster.groups[0].addr, 0x3000u);
    }
}

TEST(CoreSyncX280Roster, SingletonGroupIsAccepted) {
    MulticastTransport::reset();

    EXPECT_EQ(SingletonGroupDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::Success);

    ASSERT_EQ(MulticastTransport::calls.size(), 2u);
    const auto& release = MulticastTransport::calls[1];
    ASSERT_EQ(release.releases.size(), 1u);
    EXPECT_EQ(release.releases[0].kind, core_sync::HostReleaseKind::Group);
    expect_group_eq(release.releases[0].group, kSingletonGroup);
}

TEST(CoreSyncX280Roster, MultipleGroupsReleaseInOrder) {
    MulticastTransport::reset();

    EXPECT_EQ(MultiGroupDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::Success);

    ASSERT_EQ(MulticastTransport::calls.size(), 2u);
    const auto& release = MulticastTransport::calls[1];
    ASSERT_EQ(release.releases.size(), 3u);
    for (uint32_t i = 0u; i < 3u; ++i) {
        EXPECT_EQ(release.releases[i].kind, core_sync::HostReleaseKind::Group);
        expect_group_eq(release.releases[i].group, kThreeGroups[i]);
    }
}

TEST(CoreSyncX280Roster, GroupFailureStopsLaterGroupsAndEndpoints) {
    MulticastTransport::reset();
    MulticastTransport::notify_group_fail_at = 1u;

    EXPECT_EQ(GroupsThenEndpointsDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::ConfigError);

    ASSERT_EQ(MulticastTransport::calls.size(), 2u);
    const auto& release = MulticastTransport::calls[1];
    ASSERT_EQ(release.releases.size(), 1u);
    EXPECT_EQ(release.releases[0].kind, core_sync::HostReleaseKind::Group);
    expect_group_eq(release.releases[0].group, kThreeGroups[0]);
}

TEST(CoreSyncX280Roster, EndpointsFollowGroups) {
    MulticastTransport::reset();

    EXPECT_EQ(GroupsThenEndpointsDirectorBarrier::sync(0u, 0u, 1u), core_sync::Outcome::Success);

    ASSERT_EQ(MulticastTransport::calls.size(), 2u);
    const auto& release = MulticastTransport::calls[1];
    ASSERT_EQ(release.releases.size(), 5u);
    for (uint32_t i = 0u; i < 3u; ++i) {
        EXPECT_EQ(release.releases[i].kind, core_sync::HostReleaseKind::Group);
        expect_group_eq(release.releases[i].group, kThreeGroups[i]);
    }
    for (uint32_t i = 0u; i < 2u; ++i) {
        EXPECT_EQ(release.releases[3u + i].kind, core_sync::HostReleaseKind::Endpoint);
        EXPECT_EQ(release.releases[3u + i].endpoint, kEndpoints[i]);
    }
}

TEST(CoreSyncBarrier, MixedRosterReachesTransportUnchanged) {
    MulticastTransport::reset();
    constexpr uint32_t nonce = 0xABCD1234u;

    EXPECT_EQ(MixedDirectorBarrier::sync(0u, 0u, nonce), core_sync::Outcome::Success);

    ASSERT_EQ(MulticastTransport::calls.size(), 2u);
    for (const auto& call : MulticastTransport::calls) {
        ASSERT_EQ(call.roster.group_count, 1u);
        expect_group_eq(call.roster.groups[0], kReleaseGroup);
        ASSERT_EQ(call.roster.endpoints.size(), 2u);
        EXPECT_EQ(call.roster.endpoints[0], kEndpoints[0]);
        EXPECT_EQ(call.roster.endpoints[1], kEndpoints[1]);
        EXPECT_EQ(call.roster.endpoint_count, 2u);
        EXPECT_EQ(call.roster.counter_addr, kCounterAddr);
        EXPECT_EQ(call.roster.counter_arrivals, 2u);
        ASSERT_EQ(call.roster.word_arrivals.size(), 2u);
        EXPECT_EQ(call.roster.word_arrivals[0].addr, kMixedWordArrivals[0].addr);
        EXPECT_EQ(call.roster.word_arrivals[1].addr, kMixedWordArrivals[1].addr);
        EXPECT_EQ(call.roster.word_arrival_count, 2u);
    }
}

}  // namespace
