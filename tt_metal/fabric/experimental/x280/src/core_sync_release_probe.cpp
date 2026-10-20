// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * core_sync_release_probe.cpp -- Test-only X280 singleton and multi-group
 * release image.
 */

extern "C" {
#include "rt/x280_idle.h"
}

#include "core_sync_release_probe_args.h"
#include "core_sync_transport_x280.hpp"
#include "x280_section_load.hpp"

namespace {

constexpr uint64_t kParticipants = 1u;

using Transport = core_sync::X280Transport<>;

core_sync::X280Roster g_roster{};

/* Publishes a terminal or intermediate result state after preceding fields. */
inline void publish(core_sync_release_probe_result_t& result, uint32_t state) {
    result.state = state;
    __asm__ volatile("fence ow, ow" ::: "memory");
}

}  // namespace

extern "C" int main(uint64_t hartid) {
    UBENCH_SECTION_STORAGE(core_sync_release_probe_args_t, args);
    if (!x280_load_section<core_sync_release_probe_args_t>(hartid, kParticipants, as_uint32)) {
        return 0;
    }
    if (args.result_addr == 0u || (args.result_addr & 3u) != 0u) {
        return 0;
    }

    auto& result = *reinterpret_cast<core_sync_release_probe_result_t*>(args.result_addr);
    result.cookie = CORE_SYNC_RELEASE_PROBE_RESULT_COOKIE;
    result.group_count = args.group_count;
    result.release_value = args.release_value;
    publish(result, CORE_SYNC_RELEASE_PROBE_INIT);

    if (!core_sync_release_probe_validate(&args)) {
        publish(result, CORE_SYNC_RELEASE_PROBE_BAD_ARGS);
        return 0;
    }

    g_roster.groups = args.groups;
    g_roster.group_count = args.group_count;
    if (!Transport::notify_group(g_roster, args.release_value)) {
        publish(result, CORE_SYNC_RELEASE_PROBE_RELEASE_FAILED);
        return 0;
    }

    publish(result, CORE_SYNC_RELEASE_PROBE_DONE);
    return 0;
}
