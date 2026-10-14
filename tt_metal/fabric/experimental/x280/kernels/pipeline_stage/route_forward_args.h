// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Fixed-layout forwarding section. The section is 64 bytes, with 44 defined
 * bytes and a zero tail. The composed 88-byte section-plus-layout block is
 * published as a 128-byte cache-line-rounded span.
 */
#ifndef X280_PIPELINE_STAGE_ROUTE_FORWARD_ARGS_H
#define X280_PIPELINE_STAGE_ROUTE_FORWARD_ARGS_H

#include <stddef.h>
#include <stdint.h>

#include "../shared/ubench_section_load.h"
#include "pipeline_stage_common.h"

#define PIPELINE_STAGE_DIR_EAST 0u
#define PIPELINE_STAGE_DIR_WEST 1u
#define PIPELINE_STAGE_DIR_NORTH 2u
#define PIPELINE_STAGE_DIR_SOUTH 3u
#define PIPELINE_STAGE_DIR_COUNT 4u

/* Defined bytes of the section; the remainder of the section is its zero tail. */
#define PIPELINE_STAGE_ROUTE_FORWARD_ARGS_USED_BYTES 44u

/* Extent the host publishes, which is the block rounded to a whole cache line
 * because Arena allocates in whole lines. Firmware compares
 * x280_hart_args_bytes() against this, not against sizeof(args_block_t). */
#define PIPELINE_STAGE_ROUTE_FORWARD_ARGS_BYTES 128u

typedef struct {
    uint64_t result_base;
    uint64_t sample_region_base;
} pipeline_stage_route_forward_args_t;

typedef struct {
    uint32_t cookie_common;
    pipeline_stage_common_args_t common;
    uint32_t cookie_backend;
    pipeline_stage_route_forward_args_t backend;
    uint32_t hop_count;
    uint32_t pad[5];
} pipeline_stage_route_forward_section_t;

typedef struct {
    pipeline_stage_route_forward_section_t section;
    host_device_layout_t layout;
} pipeline_stage_route_forward_args_block_t;

#if defined(__cplusplus)
inline bool validate_section(const pipeline_stage_route_forward_section_t& s) {
    return s.cookie_common == pipeline_stage_cookie_for(UBENCH_SECTION_COMMON) &&
           s.cookie_backend == pipeline_stage_cookie_for(UBENCH_SECTION_BACKEND) && s.pad[0] == 0u && s.pad[1] == 0u &&
           s.pad[2] == 0u && s.pad[3] == 0u && s.pad[4] == 0u;
}

/**
 * The route_forward arguments: the common counts (common_args_are_valid)
 * with instrumentation off, and so no sample region, PIPELINE_STAGE_DIR_COUNT
 * hops, and a nonzero line-aligned result inside the layout window. The
 * backend adds its own placement rules.
 */
inline bool route_forward_args_valid(const pipeline_stage_route_forward_args_block_t& a) {
    const pipeline_stage_route_forward_args_t& b = a.section.backend;
    if (!common_args_are_valid(a.section.common, b.sample_region_base) ||
        a.section.common.instrumentation != static_cast<uint32_t>(PIPELINE_STAGE_INSTRUMENTATION_OFF)) {
        return false;
    }
    if (a.section.hop_count != PIPELINE_STAGE_DIR_COUNT) {
        return false;
    }
    return b.result_base != 0u && b.result_base % UBENCH_LIM_LINE_BYTES == 0u &&
           is_span_contained(b.result_base, sizeof(pipeline_stage_result_t), a.layout);
}

inline void fill_section(
    pipeline_stage_route_forward_section_t& s,
    uint32_t warmup_count,
    uint32_t timed_count,
    uint32_t instrumentation,
    uint32_t run_nonce,
    uint32_t hop_count,
    uint64_t result_base,
    uint64_t sample_region_base) {
    s.cookie_common = pipeline_stage_cookie_for(UBENCH_SECTION_COMMON);
    s.common.warmup_count = warmup_count;
    s.common.timed_count = timed_count;
    s.common.instrumentation = instrumentation;
    s.common.run_nonce = run_nonce;
    s.cookie_backend = pipeline_stage_cookie_for(UBENCH_SECTION_BACKEND);
    s.hop_count = hop_count;
    s.backend.result_base = result_base;
    s.backend.sample_region_base = sample_region_base;
    s.pad[0] = 0u;
    s.pad[1] = 0u;
    s.pad[2] = 0u;
    s.pad[3] = 0u;
    s.pad[4] = 0u;
}
#endif

UBENCH_LAYOUT_ASSERT(sizeof(pipeline_stage_route_forward_args_t) == 16u, "forward args size");
UBENCH_LAYOUT_ASSERT(offsetof(pipeline_stage_route_forward_args_t, result_base) == 0u, "result_base offset");
UBENCH_LAYOUT_ASSERT(
    offsetof(pipeline_stage_route_forward_args_t, sample_region_base) == 8u, "sample_region_base offset");

UBENCH_LAYOUT_ASSERT(sizeof(pipeline_stage_route_forward_section_t) == 64u, "forward section size");
UBENCH_LAYOUT_ASSERT(offsetof(pipeline_stage_route_forward_section_t, cookie_common) == 0u, "cookie_common offset");
UBENCH_LAYOUT_ASSERT(offsetof(pipeline_stage_route_forward_section_t, common) == 4u, "common offset");
UBENCH_LAYOUT_ASSERT(offsetof(pipeline_stage_route_forward_section_t, cookie_backend) == 20u, "cookie_backend offset");
UBENCH_LAYOUT_ASSERT(offsetof(pipeline_stage_route_forward_section_t, backend) == 24u, "backend offset");
UBENCH_LAYOUT_ASSERT(offsetof(pipeline_stage_route_forward_section_t, hop_count) == 40u, "hop_count offset");
UBENCH_LAYOUT_ASSERT(
    offsetof(pipeline_stage_route_forward_section_t, pad) == PIPELINE_STAGE_ROUTE_FORWARD_ARGS_USED_BYTES,
    "pad offset");

UBENCH_LAYOUT_ASSERT(sizeof(pipeline_stage_route_forward_args_block_t) == 88u, "forward args block size");
UBENCH_LAYOUT_ASSERT(
    offsetof(pipeline_stage_route_forward_args_block_t, section) == 0u, "section offset in forward block");
UBENCH_LAYOUT_ASSERT(
    offsetof(pipeline_stage_route_forward_args_block_t, layout) == 64u, "layout offset in forward block");
UBENCH_LAYOUT_ASSERT(
    PIPELINE_STAGE_ROUTE_FORWARD_ARGS_BYTES ==
        ((sizeof(pipeline_stage_route_forward_args_block_t) + UBENCH_LIM_LINE_BYTES - 1u) &
         ~(uint64_t)(UBENCH_LIM_LINE_BYTES - 1u)),
    "published extent is the block rounded to a whole cache line, which is what "
    "Arena allocates and therefore what x280_hart_args_bytes() reports");

#endif
