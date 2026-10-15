// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * core_sync_director.cpp -- X280 image that joins a core_sync_barrier.
 *
 * The mechanism test's X280 half. `role` in the args block selects whether this
 * hart directs the barrier -- waiting on a ready table in its own LIM and then
 * releasing a Tensix rectangle in one broadcast through the system-port TLB
 * window -- or joins one as a participant, announcing arrival into a director's
 * ready table and polling its own LIM release word.
 *
 * Only hart 0 participates; harts 1..3 return immediately, the convention every
 * X280-side NOC image in this tree uses. That is also what keeps the barrier's
 * TLB slot uncontended: the four harts of an L2CPU tile share one TLB
 * configuration block, so a second participating hart would need a slot of its
 * own (core_sync_transport_x280.hpp, X280Transport::kTlbSlot).
 *
 * The image validates the complete section before preparing either role. A
 * barrier outcome other than Success is published as SYNC_FAILURE rather than
 * left for the host to infer from DONE-not-set.
 *
 * The director executes the section's endpoints and group in place: the
 * section lives in main's frame for the whole barrier. The arrival array and
 * the mutable config members are file-scope objects rather than locals,
 * because the transport holds the roster by reference for the whole barrier.
 * They are zero-initialised and a per-hart image carries no .bss, so this
 * object is built with -fno-zero-initialized-in-bss, as every image is (X280_CFLAGS
 * in x280_image.cmake).
 */

extern "C" {
#include "x280_hart_protocol.h"
#include "rt/x280_idle.h"
}

#include "core_sync_barrier.hpp"
#include "core_sync_director_args.h"
#include "core_sync_layout.h"
#include "core_sync_transport_x280.hpp"
#include "x280_section_load.hpp"

namespace {

constexpr uint64_t kParticipants = 1u;

using Transport = core_sync::X280Transport<>;

core_sync::WordArrival g_arrivals[CORE_SYNC_DIRECTOR_MAX_ARRIVALS]{};

struct DirectorSyncConfig {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = true;
    static inline core_sync::X280Roster kRoster{};
};

struct ParticipantSyncConfig {
    static constexpr bool kEnabled = true;
    static constexpr bool kIsDirector = false;
    static inline uint32_t kLocalReleaseAddr = 0u;
    static inline core_sync::X280ArrivalTarget kArrivalTarget{};
};

/* Publishes the terminal state after every preceding result write. */
inline void publish(core_sync_director_result_t& result, uint32_t state) {
    result.terminal_state = state;
    x280_fence_ow();
}

/* Initializes every result field before the barrier can publish a failure. */
void initialize_result(core_sync_director_result_t& result, const core_sync_director_section_t& section) {
    result.cookie = CORE_SYNC_DIRECTOR_RESULT_COOKIE;
    result.terminal_state = CORE_SYNC_DIRECTOR_STATE_INIT;
    result.barrier_outcome = static_cast<uint32_t>(core_sync::Outcome::ConfigError);
    result.role = section.common.role;
    result.run_nonce = section.common.run_nonce;
    result.awaited_arrivals = 0u;
    result.released_endpoints = 0u;
    result.released_dests = 0u;
}

/* Binds the validated roster, borrowing the section's endpoints and group,
 * which outlive sync(), and reports it as the roster executed. */
void prepare_director(const core_sync_director_backend_args_t& args, core_sync_director_result_t& result) {
    for (uint32_t i = 0u; i < args.arrival_count; ++i) {
        g_arrivals[i] = core_sync::WordArrival{args.arrivals[i].addr, args.arrivals[i].worker_index};
    }
    /* The wire ABI carries one embedded grid; one with no destinations is no
     * group at all. */
    const uint32_t group_count = args.grid.num_dests > 0u ? 1u : 0u;
    DirectorSyncConfig::kRoster = core_sync::X280Roster{
        args.endpoints,
        args.endpoint_count,
        0u,
        0u,
        g_arrivals,
        args.arrival_count,
        group_count > 0u ? &args.grid : nullptr,
        group_count};
    result.awaited_arrivals = args.arrival_count;
    result.released_endpoints = args.endpoint_count;
    result.released_dests = args.grid.num_dests;
}

/* Binds the arrival target and clears the local release word before arrival. */
void prepare_participant(const core_sync_director_backend_args_t& args) {
    ParticipantSyncConfig::kArrivalTarget = args.arrival.target;
    ParticipantSyncConfig::kLocalReleaseAddr = args.local_release_addr;

    volatile uint32_t* release = reinterpret_cast<volatile uint32_t*>(args.local_release_addr);
    *release = 0u;
    x280_fence_ow();
}

/* Executes the validated section's selected role and returns its barrier outcome. */
core_sync::Outcome synchronize(
    const core_sync_director_section_t& section,
    core_sync_director_result_t& result,
    uint64_t hartid,
    uint64_t run_value) {
    if (section.common.role == CORE_SYNC_DIRECTOR_ROLE_DIRECTOR) {
        prepare_director(section.backend, result);
        return core_sync::core_sync_barrier<DirectorSyncConfig, Transport>::sync(
            hartid, run_value, section.common.run_nonce);
    }
    prepare_participant(section.backend);
    return core_sync::core_sync_barrier<ParticipantSyncConfig, Transport>::sync(
        hartid, run_value, section.common.run_nonce);
}

}  // namespace

extern "C" int main(uint64_t hartid) {
    UBENCH_SECTION_STORAGE(core_sync_director_section_t, section);
    if (!x280_load_section<core_sync_director_section_t>(hartid, kParticipants, as_uint32)) {
        return 0;
    }
    const uint64_t run_value = x280_hart_control(hartid)->run_value;
    if (section.backend.result_addr == 0u || (section.backend.result_addr & 3u) != 0u) {
        return 0;
    }

    auto& result = *reinterpret_cast<core_sync_director_result_t*>(section.backend.result_addr);
    initialize_result(result, section);

    if (!core_sync_director_validate(&section)) {
        publish(result, CORE_SYNC_DIRECTOR_STATE_ARG_LAYOUT_MISMATCH);
        return 0;
    }

    const core_sync::Outcome outcome = synchronize(section, result, hartid, run_value);
    result.barrier_outcome = static_cast<uint32_t>(outcome);
    if (outcome != core_sync::Outcome::Success) {
        publish(result, CORE_SYNC_DIRECTOR_STATE_SYNC_FAILURE);
        return 0;
    }

    publish(result, CORE_SYNC_DIRECTOR_STATE_DONE);
    return 0;
}
