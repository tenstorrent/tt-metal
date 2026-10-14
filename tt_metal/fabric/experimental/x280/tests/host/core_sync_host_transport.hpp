// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_TESTS_HOST_CORE_SYNC_HOST_TRANSPORT_HPP
#define X280_TESTS_HOST_CORE_SYNC_HOST_TRANSPORT_HPP

#include <stdint.h>

#include <vector>

#include "core_sync_arrival.hpp"
#include "core_sync_transport_x280.hpp"

namespace core_sync {

using HostGroup = X280Group;

/// Barrier roster whose arrays remain owned by the test. Mirrors X280Roster:
/// endpoints and groups are borrowed, and a null pointer is allowed only when
/// its count is zero.
struct HostRoster {
    const X280Endpoint* endpoints = nullptr;
    uint32_t endpoint_count = 0u;
    uint32_t counter_addr = 0u;
    uint32_t counter_arrivals = 0u;
    const WordArrival* word_arrivals = nullptr;
    uint32_t word_arrival_count = 0u;
    const HostGroup* groups = nullptr;
    uint32_t group_count = 0u;
};

/// Owned snapshot of a HostRoster, including declared counts for invalid spans.
struct RecordedHostRoster {
    std::vector<X280Endpoint> endpoints{};
    uint32_t endpoint_count = 0u;
    uint32_t counter_addr = 0u;
    uint32_t counter_arrivals = 0u;
    std::vector<WordArrival> word_arrivals{};
    uint32_t word_arrival_count = 0u;
    std::vector<HostGroup> groups{};
    uint32_t group_count = 0u;
};

struct HostArrivalTarget {
    uint32_t addr = 0u;
    uint32_t worker_index = 0u;
};

struct HostWaitContext {
    uint64_t hartid = 0u;
    uint64_t run_value = 0u;

    constexpr HostWaitContext(uint64_t hart, uint64_t run) : hartid(hart), run_value(run) {}
};

enum class HostCallKind {
    NotifyArrival,
    Wait,
    WaitForArrivals,
    NotifyGroup,
};

/// One release operation notify_group issued, in the order it issued them: a
/// group broadcast or an exceptional endpoint. The payloads are owned copies, so
/// they survive the borrowed roster arrays the adapter may hold on its stack.
enum class HostReleaseKind { Group, Endpoint };

struct HostRelease {
    HostReleaseKind kind;
    HostGroup group{};
    X280Endpoint endpoint{};
};

/// Arguments recorded for one transport operation.
struct HostCall {
    HostCallKind kind;
    HostArrivalTarget arrival_target{};
    RecordedHostRoster roster{};
    std::vector<HostRelease> releases{};
    uint32_t addr = 0u;
    uint32_t value = 0u;
    uint32_t run_nonce = 0u;
    uint64_t hartid = 0u;
    uint64_t run_value = 0u;
};

/// Test-selected transport results and the ordered calls they produced.
struct HostRecorder {
    static constexpr uint32_t kNoFail = 0xFFFFFFFFu;

    static inline std::vector<HostCall> calls{};
    static inline bool notify_arrival_result = true;
    static inline bool wait_result = true;
    static inline bool wait_for_arrivals_result = true;
    static inline bool notify_group_result = true;
    /// Group index whose release fails, modelling a refused multicast window.
    static inline uint32_t notify_group_fail_at = kNoFail;

    static void reset() {
        calls.clear();
        notify_arrival_result = true;
        wait_result = true;
        wait_for_arrivals_result = true;
        notify_group_result = true;
        notify_group_fail_at = kNoFail;
    }
};

/// Recording transport with no memory or delivery model.
template <bool HasMulticast = false, ArrivalKind Arrival = ArrivalKind::Word>
struct HostTransport {
    using Endpoint = X280Endpoint;
    using Group = HostGroup;
    using Roster = HostRoster;
    using WaitContext = HostWaitContext;
    using ArrivalTarget = HostArrivalTarget;

    static constexpr bool kHasTransport = true;
    static constexpr bool kHasMulticast = HasMulticast;
    static constexpr bool kMayDirect = true;
    static constexpr ArrivalKind kArrivalKind = Arrival;
    static constexpr bool kMayHostCounter = true;

    static inline std::vector<HostCall>& calls = HostRecorder::calls;
    static inline bool& notify_arrival_result = HostRecorder::notify_arrival_result;
    static inline bool& wait_result = HostRecorder::wait_result;
    static inline bool& wait_for_arrivals_result = HostRecorder::wait_for_arrivals_result;
    static inline bool& notify_group_result = HostRecorder::notify_group_result;
    static inline uint32_t& notify_group_fail_at = HostRecorder::notify_group_fail_at;

    static void reset() { HostRecorder::reset(); }

    /// Configured arrival primitives: the shared counter plus every word.
    static uint64_t arrival_count(const Roster& roster) {
        return static_cast<uint64_t>(roster.counter_arrivals) + roster.word_arrival_count;
    }

    /// Whether any release operation exists; never a participant count.
    static bool has_release_targets(const Roster& roster) {
        return roster.group_count != 0u || roster.endpoint_count != 0u;
    }

    /// Validates borrowed pointer/count pairs and every release target without
    /// recording a call.
    static bool validate_roster(const Roster& roster) {
        if (roster.counter_arrivals > 0u && (roster.counter_addr == 0u || (roster.counter_addr & 3u) != 0u)) {
            return false;
        }
        const X280Roster x280_roster{
            .endpoints = roster.endpoints,
            .endpoint_count = roster.endpoint_count,
            .word_arrivals = roster.word_arrivals,
            .word_arrival_count = roster.word_arrival_count,
            .groups = roster.groups,
            .group_count = roster.group_count};
        return X280Transport<>::validate_roster(x280_roster);
    }

    /// Copies a roster for post-call assertions without retaining borrowed arrays.
    static RecordedHostRoster record_roster(const Roster& roster) {
        RecordedHostRoster recorded{};
        recorded.endpoint_count = roster.endpoint_count;
        recorded.counter_addr = roster.counter_addr;
        recorded.counter_arrivals = roster.counter_arrivals;
        recorded.word_arrival_count = roster.word_arrival_count;
        recorded.group_count = roster.group_count;
        if (roster.endpoints != nullptr) {
            recorded.endpoints.assign(roster.endpoints, roster.endpoints + roster.endpoint_count);
        }
        if (roster.word_arrivals != nullptr) {
            recorded.word_arrivals.assign(roster.word_arrivals, roster.word_arrivals + roster.word_arrival_count);
        }
        if (roster.groups != nullptr) {
            recorded.groups.assign(roster.groups, roster.groups + roster.group_count);
        }
        return recorded;
    }

    /// Records the target and nonce and returns `notify_arrival_result`.
    static bool notify_arrival(const ArrivalTarget& target, uint32_t run_nonce) {
        calls.push_back(HostCall{
            .kind = HostCallKind::NotifyArrival,
            .arrival_target = target,
            .run_nonce = run_nonce,
        });
        return notify_arrival_result;
    }

    /// Records the local address, expected value, and context and returns `wait_result`.
    static bool wait(uint32_t local_addr, uint32_t expected, const WaitContext& ctx) {
        calls.push_back(HostCall{
            .kind = HostCallKind::Wait,
            .addr = local_addr,
            .value = expected,
            .hartid = ctx.hartid,
            .run_value = ctx.run_value,
        });
        return wait_result;
    }

    /// Records the roster, nonce, and context and returns `wait_for_arrivals_result`.
    static bool wait_for_arrivals(const Roster& roster, uint32_t run_nonce, const WaitContext& ctx) {
        calls.push_back(HostCall{
            .kind = HostCallKind::WaitForArrivals,
            .roster = record_roster(roster),
            .run_nonce = run_nonce,
            .hartid = ctx.hartid,
            .run_value = ctx.run_value,
        });
        return wait_for_arrivals_result;
    }

    /// Models X280Transport::notify_group: rejects a nonzero count with a null
    /// array, releases every group in order then every endpoint, and stops at
    /// the first refusal. Records owned copies of each release so assertions
    /// outlive the borrowed roster arrays.
    static bool notify_group(const Roster& roster, uint32_t value) {
        HostCall call{};
        call.kind = HostCallKind::NotifyGroup;
        call.roster = record_roster(roster);
        call.value = value;

        bool ok = notify_group_result;
        if (ok && !validate_roster(roster)) {
            ok = false;
        }
        if (ok) {
            for (uint32_t i = 0u; i < roster.group_count; ++i) {
                if (i == notify_group_fail_at) {
                    ok = false;
                    break;
                }
                call.releases.push_back(HostRelease{HostReleaseKind::Group, roster.groups[i], {}});
            }
        }
        if (ok) {
            for (uint32_t i = 0u; i < roster.endpoint_count; ++i) {
                call.releases.push_back(HostRelease{HostReleaseKind::Endpoint, {}, roster.endpoints[i]});
            }
        }
        calls.push_back(std::move(call));
        return ok;
    }
};

}  // namespace core_sync

#endif  // X280_TESTS_HOST_CORE_SYNC_HOST_TRANSPORT_HPP
