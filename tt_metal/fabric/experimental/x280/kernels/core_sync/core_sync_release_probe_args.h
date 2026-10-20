// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * core_sync_release_probe_args.h -- Test-only X280 multi-group release ABI.
 *
 * Transport-only coverage: the host names explicit groups, including one-tile
 * ones, and the X280 transport executes them. It is not a barrier path and
 * selects no release; normal releases come from the host release selector.
 */
#ifndef X280_CORE_SYNC_RELEASE_PROBE_ARGS_H
#define X280_CORE_SYNC_RELEASE_PROBE_ARGS_H

#include <stdint.h>

#include "core_sync_x280_roster.h"

#define CORE_SYNC_RELEASE_PROBE_RESULT_COOKIE 0x43535252u
#define CORE_SYNC_RELEASE_PROBE_MAX_GROUPS 4u

/* Host-written probe request. group_count is in [1, MAX_GROUPS], and the
 * fixed array is owned by this record for the complete firmware call. The
 * record carries no layout version, so a host built against another layout is
 * not detected. */
typedef struct {
    uint32_t pad0;
    uint32_t result_addr;
    uint32_t release_value;
    uint32_t group_count;
    core_sync_x280_group_t groups[CORE_SYNC_RELEASE_PROBE_MAX_GROUPS];
} core_sync_release_probe_args_t;

enum {
    CORE_SYNC_RELEASE_PROBE_INIT = 0u,
    CORE_SYNC_RELEASE_PROBE_DONE = 1u,
    CORE_SYNC_RELEASE_PROBE_BAD_ARGS = 2u,
    CORE_SYNC_RELEASE_PROBE_RELEASE_FAILED = 3u,
};

/* Firmware result published in caller-owned LIM at args.result_addr. */
typedef struct {
    uint32_t cookie;
    uint32_t state;
    uint32_t group_count;
    uint32_t release_value;
} core_sync_release_probe_result_t;

/* Validates the fixed test ABI before firmware uses a result or group address. */
static inline int core_sync_release_probe_validate(const core_sync_release_probe_args_t* args) {
    if (args == 0) {
        return 0;
    }
    if (args->result_addr == 0u || (args->result_addr & 3u) != 0u) {
        return 0;
    }
    if (args->release_value == 0u) {
        return 0;
    }
    if (args->group_count == 0u || args->group_count > CORE_SYNC_RELEASE_PROBE_MAX_GROUPS) {
        return 0;
    }
    return 1;
}

#if defined(__cplusplus)
static_assert(sizeof(core_sync_x280_group_t) == 24u, "release-probe group wire size changed");
static_assert(sizeof(core_sync_release_probe_args_t) == 112u, "release-probe args wire size changed");
static_assert(sizeof(core_sync_release_probe_result_t) == 16u, "release-probe result wire size changed");
#else
_Static_assert(sizeof(core_sync_x280_group_t) == 24u, "release-probe group wire size changed");
_Static_assert(sizeof(core_sync_release_probe_args_t) == 112u, "release-probe args wire size changed");
_Static_assert(sizeof(core_sync_release_probe_result_t) == 16u, "release-probe result wire size changed");
#endif

#endif /* X280_CORE_SYNC_RELEASE_PROBE_ARGS_H */
