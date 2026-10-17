// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * kernels/core_sync/core_sync_director_args.h -- argument block and result
 * layout for the X280 core_sync participant image.
 *
 * One image serves both roles of the mechanism test. `role` selects which:
 * a director waits on the ready table at `ready_base` and then releases the
 * Tensix rectangle and the endpoint list; a participant announces arrival at
 * `arrival` and polls `local_release_addr`. The fields the other role uses are
 * zero and unread, and validate() refuses a block whose role and populated
 * fields disagree.
 *
 * The block is LIM-resident and read once through load_section, so every field
 * is a 32-bit word or a pair of them. Addresses in it are byte addresses in the
 * space their owner reads: an `addr` on a Tensix destination is an L1 address,
 * an `addr` on this core is a LIM address.
 *
 * Tensix coordinates are virtual, the space the L2CPU outbound TLB window
 * takes: a director's release rectangle and endpoints, and a participant's
 * arrival target. A Tensix kernel addresses this tile at the same virtual (UMD
 * TRANSLATED) coordinate, which the host resolves per NoC through
 * X280Runtime::virtual_core_from_tile. An Ethernet endpoint is
 * virtual too: a window at an Ethernet tile's virtual coordinate reaches its
 * ERISC L1 (X280CoreSyncDirector.ActiveEriscWordWorkersOnBothNocs), as
 * suites/p2p_latency/metal_peer_runner.hpp aims it.
 */
#ifndef X280_CORE_SYNC_DIRECTOR_ARGS_H
#define X280_CORE_SYNC_DIRECTOR_ARGS_H

#include <stdint.h>

#include "core_sync_layout.h"
#include "core_sync_x280_roster.h"
#include "ubench_cookies.h"
#include "ubench_outcome.h"

#define CORE_SYNC_DIRECTOR_SUITE_MAGIC 0x43535930u   /* 'CSY', 0 */
#define CORE_SYNC_DIRECTOR_RESULT_COOKIE 0x43535952u /* 'CSY', result */

/* Bounds on the roster one args block carries. */
#define CORE_SYNC_DIRECTOR_MAX_ENDPOINTS 16u
#define CORE_SYNC_DIRECTOR_MAX_ARRIVALS 64u

typedef enum {
    CORE_SYNC_DIRECTOR_ROLE_DIRECTOR = 0u,
    CORE_SYNC_DIRECTOR_ROLE_PARTICIPANT = 1u
} core_sync_director_role_t;

/* Terminal states distinguish rejected arguments from a failed barrier. */
typedef enum {
    CORE_SYNC_DIRECTOR_STATE_INIT = 0u,
    CORE_SYNC_DIRECTOR_STATE_DONE = 1u,
    CORE_SYNC_DIRECTOR_STATE_ARG_LAYOUT_MISMATCH = 2u,
    CORE_SYNC_DIRECTOR_STATE_SYNC_FAILURE = 4u
} core_sync_director_state_t;

/* The shared outcome of a director result: DONE is OK, ARG_LAYOUT_MISMATCH
 * itself, SYNC_FAILURE the barrier's own outcome `barrier` (its sync() result
 * through core_sync::ubench_outcome_of; TRAP if that reads OK), INIT
 * IN_PROGRESS. Every state is mapped so -Wswitch catches additions. */
static inline ubench_outcome_t core_sync_director_outcome(core_sync_director_state_t state, ubench_outcome_t barrier) {
    switch (state) {
        case CORE_SYNC_DIRECTOR_STATE_DONE: return UBENCH_OUTCOME_OK;
        case CORE_SYNC_DIRECTOR_STATE_ARG_LAYOUT_MISMATCH: return UBENCH_OUTCOME_ARG_LAYOUT_MISMATCH;
        case CORE_SYNC_DIRECTOR_STATE_SYNC_FAILURE: return barrier == UBENCH_OUTCOME_OK ? UBENCH_OUTCOME_TRAP : barrier;
        case CORE_SYNC_DIRECTOR_STATE_INIT: return UBENCH_OUTCOME_IN_PROGRESS;
    }
    return UBENCH_OUTCOME_TRAP;
}

/* One tier-2 arrival this director waits on: the ready-table slot and the
 * worker index whose token that slot must hold. */
typedef struct {
    uint32_t addr;
    uint32_t worker_index;
} core_sync_director_arrival_args_t;

/* Where this participant announces arrival, and the size of the ready table
 * target.ready_base starts, which bounds target.worker_index. */
typedef struct {
    core_sync_x280_arrival_target_t target;
    uint32_t ready_bytes;
} core_sync_director_arrival_target_args_t;

typedef struct {
    uint32_t run_nonce;
    uint32_t role;
} core_sync_director_common_args_t;

typedef struct {
    /* LIM address of this participant's core_sync_director_result_t. */
    uint64_t result_addr;

    /* Director: base of the ready table in this tile's LIM, and how many of
     * the arrivals below are populated. */
    uint32_t ready_base;
    uint32_t ready_bytes;
    uint32_t arrival_count;

    /* Director: the host-selected Tensix group, executed in place.
     * grid.num_dests zero means no group is released. */
    core_sync_x280_group_t grid;

    /* Director: how many of the endpoints below are populated. */
    uint32_t endpoint_count;

    /* Participant: the LIM word this core polls for its release token. */
    uint32_t local_release_addr;

    core_sync_director_arrival_target_args_t arrival;

    core_sync_x280_endpoint_t endpoints[CORE_SYNC_DIRECTOR_MAX_ENDPOINTS];
    core_sync_director_arrival_args_t arrivals[CORE_SYNC_DIRECTOR_MAX_ARRIVALS];
} core_sync_director_backend_args_t;

/* The section carries no layout version: the cookies name its sections, and a
 * peer built against another layout is not detected. */
typedef struct {
    uint32_t pad0;
    uint32_t cookie_common;
    core_sync_director_common_args_t common;
    uint32_t cookie_backend;
    uint32_t pad;
    core_sync_director_backend_args_t backend;
} core_sync_director_section_t;

/* What the host reads back. awaited_arrivals, released_endpoints and
 * released_dests are the roster the director handed its transport -- zero
 * until a validated section is bound -- so a run states the shape it executed
 * rather than leaving the host to infer it from what it asked for. */
typedef struct {
    uint32_t cookie;
    uint32_t terminal_state;
    uint32_t barrier_outcome;
    uint32_t role;
    uint32_t run_nonce;
    uint32_t awaited_arrivals;
    uint32_t released_endpoints;
    uint32_t released_dests;
} core_sync_director_result_t;

#if defined(__cplusplus)
constexpr uint32_t core_sync_director_cookie_for(ubench_section_t s) {
    return ubench_cookie_for(CORE_SYNC_DIRECTOR_SUITE_MAGIC, s);
}
#else
static inline uint32_t core_sync_director_cookie_for(ubench_section_t s) {
    return ubench_cookie_for(CORE_SYNC_DIRECTOR_SUITE_MAGIC, s);
}
#endif
/* Validates a complete host-written section before firmware uses an address or
 * performs a transport operation. */
static inline int core_sync_director_validate(const core_sync_director_section_t* s) {
    if (s == 0) {
        return 0;
    }
    if (s->cookie_common != core_sync_director_cookie_for(UBENCH_SECTION_COMMON)) {
        return 0;
    }
    if (s->cookie_backend != core_sync_director_cookie_for(UBENCH_SECTION_BACKEND)) {
        return 0;
    }
    if (s->common.run_nonce == 0u) {
        return 0;
    }
    if (s->common.role != CORE_SYNC_DIRECTOR_ROLE_DIRECTOR && s->common.role != CORE_SYNC_DIRECTOR_ROLE_PARTICIPANT) {
        return 0;
    }

    const core_sync_director_backend_args_t* b = &s->backend;
    if (b->result_addr == 0u || (b->result_addr & 3u) != 0u) {
        return 0;
    }
    if (b->endpoint_count > CORE_SYNC_DIRECTOR_MAX_ENDPOINTS) {
        return 0;
    }
    if (b->arrival_count > CORE_SYNC_DIRECTOR_MAX_ARRIVALS) {
        return 0;
    }

    if (s->common.role == CORE_SYNC_DIRECTOR_ROLE_DIRECTOR) {
        /* Arrivals and release targets are counted independently: one tile's
         * release word can wake several participants, and a participant on
         * this tile arrives without being released remotely. */
        if (b->arrival_count == 0u) {
            return 0;
        }
        if (b->endpoint_count == 0u && b->grid.num_dests == 0u) {
            return 0;
        }
        if (b->ready_base == 0u || (b->ready_base % CORE_SYNC_SLOT_BYTES) != 0u) {
            return 0;
        }
        if (b->ready_bytes == 0u || (b->ready_bytes % CORE_SYNC_SLOT_BYTES) != 0u) {
            return 0;
        }
        if (b->arrival_count > b->ready_bytes / CORE_SYNC_SLOT_BYTES) {
            return 0;
        }
        if ((uint64_t)b->ready_base + b->ready_bytes > (uint64_t)UINT32_MAX + 1u) {
            return 0;
        }
        if (b->local_release_addr != 0u) {
            return 0;
        }
        if (b->arrival.target.noc.x != 0u || b->arrival.target.noc.y != 0u || b->arrival.target.ready_base != 0u ||
            b->arrival.target.worker_index != 0u || b->arrival.target.same_tile != 0u || b->arrival.ready_bytes != 0u) {
            return 0;
        }
        if (b->grid.num_dests == 0u) {
            if (b->grid.rect.start.x != 0u || b->grid.rect.start.y != 0u || b->grid.rect.end.x != 0u ||
                b->grid.rect.end.y != 0u || b->grid.addr != 0u) {
                return 0;
            }
        } else if (b->grid.addr == 0u || !x280_noc_rect_valid(b->grid.rect)) {
            return 0;
        }
        for (uint32_t i = 0u; i < b->endpoint_count; ++i) {
            const core_sync_x280_endpoint_t* e = &b->endpoints[i];
            if (e->addr == 0u || e->same_tile > 1u) {
                return 0;
            }
            if (e->same_tile == 0u && !x280_noc_coord_valid(e->noc)) {
                return 0;
            }
        }
        for (uint32_t i = 0u; i < b->arrival_count; ++i) {
            const core_sync_director_arrival_args_t* a = &b->arrivals[i];
            if (a->worker_index >= b->arrival_count) {
                return 0;
            }
            if (a->addr != core_sync_slot_addr(b->ready_base, a->worker_index)) {
                return 0;
            }
            for (uint32_t j = 0u; j < i; ++j) {
                if (a->worker_index == b->arrivals[j].worker_index) {
                    return 0;
                }
            }
        }
        return 1;
    }

    if (b->local_release_addr == 0u || (b->local_release_addr & 3u) != 0u) {
        return 0;
    }
    if (b->ready_base != 0u || b->ready_bytes != 0u || b->arrival_count != 0u) {
        return 0;
    }
    if (b->grid.rect.start.x != 0u || b->grid.rect.start.y != 0u || b->grid.rect.end.x != 0u ||
        b->grid.rect.end.y != 0u || b->grid.addr != 0u || b->grid.num_dests != 0u || b->endpoint_count != 0u) {
        return 0;
    }
    const core_sync_x280_arrival_target_t* t = &b->arrival.target;
    if (!x280_noc_coord_valid(t->noc)) {
        return 0;
    }
    if (t->ready_base == 0u || (t->ready_base % CORE_SYNC_SLOT_BYTES) != 0u) {
        return 0;
    }
    if (b->arrival.ready_bytes == 0u || (b->arrival.ready_bytes % CORE_SYNC_SLOT_BYTES) != 0u) {
        return 0;
    }
    if (t->worker_index >= b->arrival.ready_bytes / CORE_SYNC_SLOT_BYTES) {
        return 0;
    }
    if ((uint64_t)t->ready_base + b->arrival.ready_bytes > (uint64_t)UINT32_MAX + 1u) {
        return 0;
    }
    if (t->same_tile != 0u) {
        return 0;
    }
    for (uint32_t i = 0u; i < CORE_SYNC_DIRECTOR_MAX_ENDPOINTS; ++i) {
        if (b->endpoints[i].noc.x != 0u || b->endpoints[i].noc.y != 0u || b->endpoints[i].addr != 0u ||
            b->endpoints[i].same_tile != 0u) {
            return 0;
        }
    }
    for (uint32_t i = 0u; i < CORE_SYNC_DIRECTOR_MAX_ARRIVALS; ++i) {
        if (b->arrivals[i].addr != 0u || b->arrivals[i].worker_index != 0u) {
            return 0;
        }
    }
    return 1;
}

#if defined(__cplusplus)
#define CORE_SYNC_DIRECTOR_ASSERT static_assert
#else
#define CORE_SYNC_DIRECTOR_ASSERT _Static_assert
#endif

CORE_SYNC_DIRECTOR_ASSERT(
    sizeof(core_sync_director_section_t) % 4u == 0u, "the section is copied out of LIM as whole words");

#undef CORE_SYNC_DIRECTOR_ASSERT

#endif /* X280_CORE_SYNC_DIRECTOR_ARGS_H */
