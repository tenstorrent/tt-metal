// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// suites/core_sync/probe_args.hpp -- the runtime-argument packers for
// tests/device/kernels/core_sync_probe.cpp.
//
// Two host cases launch that kernel -- the Tensix-director measurement and the
// X280-director one -- and the kernel's unpacker is positional, so the packing
// lives here rather than once per case.

#ifndef X280_SUITES_CORE_SYNC_PROBE_ARGS_HPP
#define X280_SUITES_CORE_SYNC_PROBE_ARGS_HPP

#include "harness/regions/core_sync_roster.hpp"

#include "core_sync_probe_args.h"

#include <cstdint>
#include <vector>

namespace x280::suites::core_sync {

/* The values every role's runtime args carry after the layout cookie. */
struct ProbeCommon {
    uint32_t run_nonce = 0;
    /* The director's tier-2 ready table. */
    uint32_t director_ready_base = 0;
    /* This core's result record. */
    uint32_t result_l1_addr = 0;
    /* On the director, the counter every worker increments. */
    uint32_t arrival_semaphore_id = 0;
};

/* One role's full runtime args: the layout cookie and `common` in their named
 * slots (core_sync_probe_args.h), then `role` from CORE_SYNC_PROBE_ARG_ROLE_BASE. */
inline std::vector<uint32_t> probe_common_args(const ProbeCommon& common, const std::vector<uint32_t>& role) {
    std::vector<uint32_t> args(CORE_SYNC_PROBE_ARG_ROLE_BASE, 0u);
    args.reserve(CORE_SYNC_PROBE_ARG_ROLE_BASE + role.size());
    args[CORE_SYNC_PROBE_ARG_LAYOUT_COOKIE] = CORE_SYNC_PROBE_LAYOUT_COOKIE;
    args[CORE_SYNC_PROBE_ARG_RUN_NONCE] = common.run_nonce;
    args[CORE_SYNC_PROBE_ARG_DIRECTOR_READY_BASE] = common.director_ready_base;
    args[CORE_SYNC_PROBE_ARG_RESULT_L1_ADDR] = common.result_l1_addr;
    args[CORE_SYNC_PROBE_ARG_ARRIVAL_SEMAPHORE_ID] = common.arrival_semaphore_id;
    args.insert(args.end(), role.begin(), role.end());
    return args;
}

/* A director's runtime args. Its role args are the selector's group and its
 * destination count, how many arrivals of each tier it waits for, the
 * endpoints outside the rectangle, and the tier-2 ready words.
 *
 * counter_arrivals overrides the roster's own count so a case can name one more
 * arrival than any worker will write, which is how the abandonment path is
 * reached; passing the roster's count is the ordinary run. */
inline std::vector<uint32_t> director_args(
    const ProbeCommon& common, const x280::harness::regions::CoreSyncRoster& roster, uint32_t counter_arrivals) {
    std::vector<uint32_t> role = {
        roster.grid.rect.start.x,
        roster.grid.rect.start.y,
        roster.grid.rect.end.x,
        roster.grid.rect.end.y,
        roster.grid.semaphore_id,
        roster.grid.num_dests,
        counter_arrivals,
        static_cast<uint32_t>(roster.worker_endpoints.size()),
        static_cast<uint32_t>(roster.word_arrivals.size()),
    };
    for (const auto& ep : roster.worker_endpoints) {
        role.push_back(ep.noc.x);
        role.push_back(ep.noc.y);
        role.push_back(ep.semaphore_id);
        role.push_back(ep.is_semaphore ? 1u : 0u);
        /* The endpoint's own core family, which is what resolves a semaphore id
         * to an address on the destination rather than at the Tensix base. */
        role.push_back(static_cast<uint32_t>(ep.core_type));
    }
    for (const auto& w : roster.word_arrivals) {
        role.push_back(w.addr);
        role.push_back(w.worker_index);
    }
    return probe_common_args(common, role);
}

/* A counter-tier worker's runtime args. The last two role args name the tier-2
 * slot and the raw release address a word worker uses, and are zero here: this
 * worker arrives on the counter and is released on a semaphore. */
inline std::vector<uint32_t> counter_worker_args(
    const ProbeCommon& common, const x280::harness::regions::CoreSyncRoster& roster) {
    return probe_common_args(
        common, {roster.director_noc.x(), roster.director_noc.y(), roster.release_semaphore_id, 0u, 0u});
}

/* A word-tier worker's runtime args. Its role args are the director it writes
 * its tier-2 slot on, the release semaphore slot (which the word path does not
 * read), its slot index, and the raw L1 address its release lands on (an X280
 * director has no semaphore to name). */
inline std::vector<uint32_t> word_worker_args(
    const ProbeCommon& common,
    const x280::TranslatedCoord& director,
    uint32_t release_semaphore_id,
    uint32_t word_index,
    uint32_t release_l1_addr) {
    return probe_common_args(common, {director.x(), director.y(), release_semaphore_id, word_index, release_l1_addr});
}

}  // namespace x280::suites::core_sync

#endif  // X280_SUITES_CORE_SYNC_PROBE_ARGS_HPP
