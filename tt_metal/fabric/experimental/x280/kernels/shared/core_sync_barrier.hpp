// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// kernels/shared/core_sync_barrier.hpp -- Compile-time star-topology barrier
// across cores for microbenchmarks.
//
// Protocol:
//   1. Each worker core runs local setup, then announces its arrival
//      to the Director through the transport's arrival primitive: an atomic
//      increment of a shared counter, or a core_sync_ready_token written to its
//      own slot in the Director's memory.
//   2. Each worker then polls its own local release slot.
//   3. The Director core waits on its arrival storage until every worker has
//      arrived (or host STOP is signaled).
//   4. The Director writes core_sync_release_token to each worker's local release
//      slot (via NoC write or LIM store) and issues a release fence.
//   5. All cores continue to timed execution.
//
// Supported lifecycle:
//   1. The host describes the complete participant topology.
//   2. The host queries its storage requirements, allocates caller-owned raw
//      regions, zeroes counters and release storage, and binds the regions.
//   3. The host launches every participant while retaining the allocation
//      owners, and each participant calls sync exactly once.
//   4. The host waits for workload completion before releasing those owners.
// A later host launch is a new barrier invocation and must reinitialize its
// synchronization storage.
#ifndef X280_CORE_SYNC_BARRIER_HPP
#define X280_CORE_SYNC_BARRIER_HPP

#include <stdint.h>

#include "core_sync_layout.h"
#include "core_sync_transport_x280.hpp"
#include "ubench_outcome.h"

namespace core_sync {

/** What sync() returns. StopRequested means a wait ended without its token:
 * the host published STOP for run_value on the X280 transport, or the poll's
 * iteration cap expired on the Metal transport, which takes the cap from
 * run_value (MetalWaitContext). Either way the caller must unwind. */
enum class Outcome : uint32_t { Success = 0u, StopRequested = 1u, ConfigError = 2u };

/** The shared outcome of a sync() result: Success is OK, StopRequested
 * TIMEOUT (a wait ended without its token), ConfigError ARG_LAYOUT_MISMATCH
 * (a roster or coordinate was refused). A value outside Outcome is TRAP. */
constexpr ubench_outcome_t ubench_outcome_of(Outcome o) {
    switch (o) {
        case Outcome::Success: return UBENCH_OUTCOME_OK;
        case Outcome::StopRequested: return UBENCH_OUTCOME_TIMEOUT;
        case Outcome::ConfigError: return UBENCH_OUTCOME_ARG_LAYOUT_MISMATCH;
    }
    return UBENCH_OUTCOME_TRAP;
}

/* Disables synchronization without instantiating a transport operation. */
struct DisabledSyncConfig {
    static constexpr bool kEnabled = false;
};

template <typename Config, typename Transport = X280Transport<>>
class core_sync_barrier {
public:
    /* TODO(core-sync): Support multiple sync calls in one kernel launch. Word
     * arrivals are nonce-qualified; counter arrivals need an explicit
     * cumulative target or generation. Resetting the active counter between
     * calls can erase a fast worker's next arrival. Cover counter-only,
     * word-only, mixed, and counter-overflow behavior on host and silicon. */
    static inline Outcome sync(uint64_t hartid, uint64_t run_value, uint32_t run_nonce) {
        /* The else is what keeps a disabled config from naming a transport
         * operation at all: the branch is discarded before instantiation, so a
         * build whose transport has no hardware behind it still compiles. */
        if constexpr (!Config::kEnabled) {
            (void)hartid;
            (void)run_value;
            (void)run_nonce;
            return Outcome::Success;
        } else {
            static_assert(
                Transport::kHasTransport,
                "core_sync_barrier has no transport for this core: the protocol "
                "is written against a hardware transport. An enabled config here "
                "would neither post nor observe a token.");
            if constexpr (Config::kIsDirector) {
                static_assert(Transport::kMayDirect, "this core family may not coordinate as director");
                return run_director(hartid, run_value, run_nonce);
            } else {
                return run_worker(hartid, run_value, run_nonce);
            }
        }
    }

private:
    static inline Outcome run_director(uint64_t hartid, uint64_t run_value, uint32_t run_nonce) {
        const typename Transport::Roster& roster = Config::kRoster;

        /* At least one arrival to wait on and at least one target to release.
         * The barrier does not equate their counts: one release target can wake
         * several local participants, so target cardinality is not participant
         * cardinality. Proving that every participant is covered by some target
         * belongs to the topology owner that knows the mapping. */
        if (!Transport::validate_roster(roster) || Transport::arrival_count(roster) == 0u ||
            !Transport::has_release_targets(roster)) {
            return Outcome::ConfigError;
        }

        /* 1. Wait for every worker to arrive, on whichever tier each uses */
        const typename Transport::WaitContext ctx{hartid, run_value};
        if (!Transport::wait_for_arrivals(roster, run_nonce, ctx)) {
            return Outcome::StopRequested;
        }

        /* 2. Release each worker */
        const uint32_t release = core_sync_release_token(run_nonce);
        if (!Transport::notify_group(roster, release)) {
            return Outcome::ConfigError;
        }

        return Outcome::Success;
    }

    static inline Outcome run_worker(uint64_t hartid, uint64_t run_value, uint32_t run_nonce) {
        /* 1. Announce arrival to the Director */
        if (!Transport::notify_arrival(Config::kArrivalTarget, run_nonce)) {
            return Outcome::ConfigError;
        }

        /* 2. Poll own local release slot */
        const uint32_t release = core_sync_release_token(run_nonce);
        const typename Transport::WaitContext ctx{hartid, run_value};

        if (!Transport::wait(Config::kLocalReleaseAddr, release, ctx)) {
            return Outcome::StopRequested;
        }

        return Outcome::Success;
    }
};

}  // namespace core_sync

#endif /* X280_CORE_SYNC_BARRIER_HPP */
