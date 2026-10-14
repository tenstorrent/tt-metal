// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* Shared wire and result contracts; member argument sections remain separate. */
#ifndef X280_PIPELINE_STAGE_COMMON_H
#define X280_PIPELINE_STAGE_COMMON_H

#include <stddef.h>
#include <stdint.h>

#include "../shared/fabric_shadow.h"
#include "../shared/ubench_backend_ids.h"
#include "../shared/ubench_cookies.h"
#include "../shared/ubench_outcome.h"
#include "../shared/ubench_region.h"
#include "../shared/ubench_section_load.h"
#include "../shared/ubench_units.h"
#include "../../include/rt/measurement.h"
#include "../../include/rt/x280_pinned_input.h"
#include "../../include/socket_layout.h"
#include "../../include/x280_address_map.h"

#define PIPELINE_STAGE_COOKIE_BASE 0x50535447u

#define PIPELINE_STAGE_LOOKUP_HEADER_COUNT 16u
#define PIPELINE_STAGE_LOOKUP_FIXTURE_V1 1u

/* With WAYENABLE=1, fourteen of the fifteen 128 KiB LIM ways remain.
 * Available LIM starts after the fixed active-image reservation. */
#define PIPELINE_STAGE_PREPARED_LIM_BEGIN (X280_LIM_BASE + X280_SOCKET_HIGHEST_END)
#define PIPELINE_STAGE_PREPARED_LIM_END (X280_LIM_END - X280_PINNED_INPUT_WAYENABLE * (X280_LIM_SIZE / 15u))

#if defined(__cplusplus)
constexpr uint32_t pipeline_stage_cookie_for(ubench_section_t section) {
    return ubench_cookie_for(PIPELINE_STAGE_COOKIE_BASE, section);
}
#else
static inline uint32_t pipeline_stage_cookie_for(ubench_section_t section) {
    return ubench_cookie_for(PIPELINE_STAGE_COOKIE_BASE, section);
}
#endif

typedef enum {
    PIPELINE_STAGE_STATE_INIT = 0u,
    PIPELINE_STAGE_STATE_VERIFY_STAGE = 1u,
    PIPELINE_STAGE_STATE_WARMUP = 2u,
    PIPELINE_STAGE_STATE_TIMED_PLAIN = 3u,
    PIPELINE_STAGE_STATE_TIMED_INSTRUMENTED = 4u,
    PIPELINE_STAGE_STATE_PUBLISH = 5u,
    PIPELINE_STAGE_STATE_DONE = 6u,
    PIPELINE_STAGE_STATE_VERIFY_MISMATCH = 7u,
    /* 8 and 9 are unused: no member times out on the device or traps. */
    PIPELINE_STAGE_STATE_ARG_LAYOUT_MISMATCH = 10u,
    PIPELINE_STAGE_STATE_VERIFY_AFTER = 11u,
    PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED = 12u,
    PIPELINE_STAGE_STATE_COUNTER_ERROR = 13u
} pipeline_stage_state_t;

static inline ubench_outcome_t pipeline_stage_outcome(pipeline_stage_state_t state) {
    switch (state) {
        case PIPELINE_STAGE_STATE_DONE: return UBENCH_OUTCOME_OK;
        case PIPELINE_STAGE_STATE_VERIFY_MISMATCH: return UBENCH_OUTCOME_PATH_MISMATCH;
        case PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED:
        case PIPELINE_STAGE_STATE_COUNTER_ERROR: return UBENCH_OUTCOME_TRAP;
        case PIPELINE_STAGE_STATE_ARG_LAYOUT_MISMATCH: return UBENCH_OUTCOME_ARG_LAYOUT_MISMATCH;
        case PIPELINE_STAGE_STATE_INIT:
        case PIPELINE_STAGE_STATE_VERIFY_STAGE:
        case PIPELINE_STAGE_STATE_WARMUP:
        case PIPELINE_STAGE_STATE_TIMED_PLAIN:
        case PIPELINE_STAGE_STATE_TIMED_INSTRUMENTED:
        case PIPELINE_STAGE_STATE_PUBLISH:
        case PIPELINE_STAGE_STATE_VERIFY_AFTER: return UBENCH_OUTCOME_IN_PROGRESS;
    }
    return UBENCH_OUTCOME_IN_PROGRESS;
}

typedef enum {
    PIPELINE_STAGE_INSTRUMENTATION_OFF = 0u,
    PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE = 1u
} pipeline_stage_instrumentation_t;

typedef struct {
    uint32_t warmup_count;
    uint32_t timed_count;
    uint32_t instrumentation;
    uint32_t run_nonce;
} pipeline_stage_common_args_t;

typedef struct {
    ubench_region_t window;
    uint64_t source_base;
} host_device_layout_t;

#if defined(__cplusplus)
inline bool is_span_contained(uint64_t base, uint64_t bytes, const host_device_layout_t& layout) {
    if (bytes == 0u) {
        return false;
    }
    const uint64_t window_end = layout.window.addr + layout.window.bytes;
    if (base < layout.window.addr || base >= window_end) {
        return false;
    }
    return bytes <= window_end - base;
}

inline bool spans_disjoint(uint64_t a_base, uint64_t a_bytes, uint64_t b_base, uint64_t b_bytes) {
    return (a_base + a_bytes <= b_base) || (b_base + b_bytes <= a_base);
}

inline bool layout_is_sane(const host_device_layout_t& layout) {
    if (layout.window.bytes == 0u) {
        return false;
    }
    if (layout.window.addr + layout.window.bytes < layout.window.addr) {
        return false;
    }
    if (layout.window.addr + layout.window.bytes > static_cast<uint64_t>(UINT32_MAX) + 1ull) {
        return false;
    }
    if (layout.source_base == 0u || layout.source_base % UBENCH_LIM_LINE_BYTES != 0u) {
        return false;
    }
    if (layout.source_base > UINT32_MAX) {
        return false;
    }
    return true;
}

inline void fill_layout(host_device_layout_t& layout, ubench_region_t window, uint64_t source_base) {
    layout.window = window;
    layout.source_base = source_base;
}
#endif

/* Cache-line result; workload_size and stage_witness are member-specific. */
typedef struct {
    uint32_t backend_id;
    uint32_t terminal_state;
    uint32_t packet_header_bytes;
    uint32_t instrumentation;
    uint32_t warmup_completed;
    uint32_t timed_count;
    uint64_t raw_batch_delta;
    uint64_t secondary_batch_delta;
    uint64_t items_completed;
    uint32_t verify_sentinel;
    uint32_t reserved; /* zero */
    uint32_t workload_size;
    uint32_t stage_witness;
} pipeline_stage_result_t;

#define PIPELINE_STAGE_VERIFY_MATCH 0xFFFFFFFFu

/**
 * Writes every field of *result: backend_id, terminal_state, verify_sentinel
 * and workload_size as given, every other field zero. A member starts its
 * result with (backend_id, PIPELINE_STAGE_STATE_INIT, 0, workload_size) and
 * publishes a failure before its frontend runs with (backend_id, the failing
 * state, reason, 0), so a host reading a refused result can tell why. Stores
 * field by field: the X280 images link no memset.
 */
static inline void pipeline_stage_write_result(
    pipeline_stage_result_t* result,
    uint32_t backend_id,
    uint32_t terminal_state,
    uint32_t verify_sentinel,
    uint32_t workload_size) {
    result->backend_id = backend_id;
    result->terminal_state = terminal_state;
    result->packet_header_bytes = 0u;
    result->instrumentation = 0u;
    result->warmup_completed = 0u;
    result->timed_count = 0u;
    result->raw_batch_delta = 0u;
    result->secondary_batch_delta = 0u;
    result->items_completed = 0u;
    result->verify_sentinel = verify_sentinel;
    result->reserved = 0u;
    result->workload_size = workload_size;
    result->stage_witness = 0u;
}

#if defined(__cplusplus)
constexpr uint32_t pipeline_stage_verify_mismatch(uint32_t index, uint8_t produced) {
    return (index & 0xFFFFu) | ((uint32_t)produced << 16u);
}
#else
static inline uint32_t pipeline_stage_verify_mismatch(uint32_t index, uint8_t produced) {
    return (index & 0xFFFFu) | ((uint32_t)produced << 16u);
}
#endif

#define PIPELINE_STAGE_MAX_SAMPLES 4096u
#define PIPELINE_STAGE_SAMPLES_BYTES (8u + 8u * PIPELINE_STAGE_MAX_SAMPLES)
#define PIPELINE_STAGE_SAMPLES_REGION_BYTES \
    ((PIPELINE_STAGE_SAMPLES_BYTES + UBENCH_LIM_LINE_BYTES - 1u) & ~(UBENCH_LIM_LINE_BYTES - 1u))

typedef struct {
    uint32_t samples_completed;
    uint32_t reserved;
    uint64_t raw_deltas[PIPELINE_STAGE_MAX_SAMPLES];
} pipeline_stage_samples_t;

#if defined(__cplusplus)
/* Non-zero counts and nonce, a known instrumentation mode, and under
 * per-sample instrumentation a sample region that can hold timed_count;
 * otherwise no sample region. */
inline bool common_args_are_valid(const pipeline_stage_common_args_t& common, uint64_t sample_region_base) {
    if (common.warmup_count == 0u || common.timed_count == 0u || common.run_nonce == 0u) {
        return false;
    }
    switch (common.instrumentation) {
        case PIPELINE_STAGE_INSTRUMENTATION_OFF: return sample_region_base == 0u;
        case PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE:
            return sample_region_base != 0u && common.timed_count <= PIPELINE_STAGE_MAX_SAMPLES;
        default: return false;
    }
}
#endif

/**
 * Publishes a timed batch's wall and CPU deltas, each through
 * measurement_delta. A refused interval -- a backward reading, or one at or
 * past half range -- latches PIPELINE_STAGE_STATE_COUNTER_ERROR in
 * result->terminal_state and stores zero in that delta.
 */
static inline void pipeline_stage_publish_batch_delta(
    pipeline_stage_result_t* result, uint64_t wall_start, uint64_t wall_end, uint64_t cpu_start, uint64_t cpu_end) {
    uint64_t wall = 0;
    uint64_t cpu = 0;
    const int wall_ok = measurement_delta(wall_start, wall_end, &wall);
    const int cpu_ok = measurement_delta(cpu_start, cpu_end, &cpu);
    if (!wall_ok || !cpu_ok) {
        result->terminal_state = (uint32_t)PIPELINE_STAGE_STATE_COUNTER_ERROR;
    }
    result->raw_batch_delta = wall;
    result->secondary_batch_delta = cpu;
}

UBENCH_LAYOUT_ASSERT(sizeof(pipeline_stage_result_t) == 64u, "pipeline result is exactly one cache line");
UBENCH_LAYOUT_ASSERT(
    PIPELINE_STAGE_PREPARED_LIM_BEGIN == 0x08154000u,
    "available LIM begins after every runtime and socket reservation");
UBENCH_LAYOUT_ASSERT(PIPELINE_STAGE_PREPARED_LIM_END == 0x081C0000u, "WAYENABLE 1 leaves fourteen 128 KiB LIM ways");
UBENCH_LAYOUT_ASSERT(
    offsetof(pipeline_stage_result_t, raw_batch_delta) == 24u, "pipeline counters retain their published offset");
UBENCH_LAYOUT_ASSERT(
    offsetof(pipeline_stage_result_t, verify_sentinel) == 48u, "pipeline verification retains its published offset");
UBENCH_LAYOUT_ASSERT(
    sizeof(pipeline_stage_samples_t) == PIPELINE_STAGE_SAMPLES_BYTES,
    "sample region contains a two-word header and fixed delta array");
UBENCH_LAYOUT_ASSERT(sizeof(host_device_layout_t) == 24u, "host_device_layout size");
UBENCH_LAYOUT_ASSERT(offsetof(host_device_layout_t, window.addr) == 0u, "window.addr offset");
UBENCH_LAYOUT_ASSERT(offsetof(host_device_layout_t, window.bytes) == 8u, "window.bytes offset");
UBENCH_LAYOUT_ASSERT(offsetof(host_device_layout_t, source_base) == 16u, "source_base offset");

#endif
