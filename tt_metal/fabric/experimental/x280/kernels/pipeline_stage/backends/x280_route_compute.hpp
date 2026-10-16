// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef X280_PIPELINE_STAGE_BACKEND_ROUTE_COMPUTE_HPP
#define X280_PIPELINE_STAGE_BACKEND_ROUTE_COMPUTE_HPP

#include <stddef.h>
#include <stdint.h>

extern "C" {
#include "rt/x280_pinned_input.h"
}

#include "fabric_shadow.h"
#include "fabric_shadow_header.hpp"
#include "pipeline_stage_common.h"
#include "pipeline_stage_frontend_cxx.hpp"
#include "route_compute_args.h"
#include "route_compute_subject.hpp"
#include "x280_member_backend.hpp"

namespace pipeline_stage::backend_route_compute {

static_assert(
    X280_PINNED_INPUT_HEADER_COUNT == PIPELINE_STAGE_LOOKUP_HEADER_COUNT,
    "pipeline fixture count matches the prepared-region contract");
static_assert(
    X280_PINNED_INPUT_HEADER_BYTES == FS_PACKET_HEADER_BYTES, "packet layout matches the prepared-region contract");
static_assert(
    X280_PINNED_INPUT_ROUTING_BYTES == sizeof(fs_routing_l1_info_t),
    "routing image matches the prepared-region contract");

struct state_t {
    UBENCH_SECTION_STORAGE(pipeline_stage_route_compute_args_block_t, args);
    pipeline_stage_result_t* result;
    pipeline_stage_samples_t* samples;
    volatile uint8_t* input;
    const volatile uint8_t* headers;
};

inline bool sanity_check(const state_t& state) {
    return route_compute_args_valid(state.args) &&
           x280_pinned_input_contains(&X280_PINNED_INPUT_ROUTER_REGION, X280_PINNED_INPUT_BASE, X280_PINNED_INPUT_SPAN);
}

/* Both spans satisfy the required 64-bit alignment. */
inline void copy_staged_input(state_t& state) {
    const volatile uint64_t* source =
        reinterpret_cast<const volatile uint64_t*>(state.args.section.backend.staged_input_base);
    volatile uint64_t* destination = reinterpret_cast<volatile uint64_t*>(state.input);
    for (uint32_t i = 0u; i < X280_PINNED_INPUT_SPAN / sizeof(uint64_t); ++i) {
        destination[i] = source[i];
    }
    __asm__ volatile("fence rw, rw" ::: "memory");
}

inline bool compare_staged_input(state_t& state, uint32_t& mismatch) {
    const volatile uint8_t* expected =
        reinterpret_cast<const volatile uint8_t*>(state.args.section.backend.staged_input_base);
    const volatile uint8_t* actual = state.input;
    for (uint32_t offset = 0u; offset < X280_PINNED_INPUT_SPAN; ++offset) {
        const uint8_t produced = actual[offset];
        if (produced != expected[offset]) {
            mismatch = pipeline_stage_verify_mismatch(offset, produced);
            return false;
        }
    }
    mismatch = PIPELINE_STAGE_VERIFY_MATCH;
    return true;
}

/* Compares the prepared copy with the staged input, then checks the fixture,
 * and stores the first mismatch (or PIPELINE_STAGE_VERIFY_MATCH) in the
 * result's verify_sentinel. */
inline bool verify_input(state_t& state) {
    uint32_t mismatch = 0u;
    const bool verified = compare_staged_input(state, mismatch) &&
                          pipeline_stage::verify_fixture(
                              pipeline_stage::prepared_input_t{state.input, state.headers}, mismatch);
    state.result->verify_sentinel = mismatch;
    return verified;
}

/* The routing image is read from the prepared copy, the headers from the
 * staged input in LIM. */
inline bool initialize_input(state_t& state, volatile void* prepared_base) {
    state.input = reinterpret_cast<volatile uint8_t*>(prepared_base);
    state.headers = reinterpret_cast<const volatile uint8_t*>(
        state.args.section.backend.staged_input_base + X280_PINNED_INPUT_HEADERS_OFFSET);
    copy_staged_input(state);
    return verify_input(state);
}

struct Backend : member_backend_t<state_t> {
    static inline __attribute__((always_inline)) void stage_once(state_t& state, uint32_t slot, uint32_t& witness) {
        const prepared_input_t access{state.input, state.headers};
        pipeline_stage::stage_once(access, slot, witness);
    }

    static uint32_t gen_seed(state_t& state) { return state.args.section.common.run_nonce; }

    static inline __attribute__((always_inline)) uint32_t next_index(state_t&, uint32_t& generator, uint32_t) {
        return next_slot(generator);
    }

    static bool verify_stage(state_t& state) { return verify_input(state); }

    static bool verify_after(state_t& state) { return verify_input(state); }

    static bool publish_footer(state_t&) { return true; }
};

}  // namespace pipeline_stage::backend_route_compute

#endif
