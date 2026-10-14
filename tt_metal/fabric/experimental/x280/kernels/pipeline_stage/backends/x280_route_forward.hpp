// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* DECIDE: the hop decode and routing_fields update, with no packet issued. */
#ifndef X280_PIPELINE_STAGE_BACKEND_ROUTE_FORWARD_HPP
#define X280_PIPELINE_STAGE_BACKEND_ROUTE_FORWARD_HPP

#include <stddef.h>
#include <stdint.h>

extern "C" {
#include "rt/x280_pinned_input.h"
#include "x280_address_map.h"
}

#include "fabric_shadow.h"
#include "fabric_shadow_header.hpp"
#include "pipeline_stage_common.h"
#include "pipeline_stage_frontend_cxx.hpp"
#include "route_forward_args.h"
#include "route_forward_subject.hpp"
#include "x280_member_backend.hpp"

namespace pipeline_stage::backend_route_forward {

struct state_t {
    UBENCH_SECTION_STORAGE(pipeline_stage_route_forward_args_block_t, args);
    pipeline_stage_result_t* result;
    pipeline_stage_samples_t* samples;
    volatile fs_hybrid_mesh_packet_header_t* header;
};

inline bool sanity_check(const state_t& state) {
    return route_forward_args_valid(state.args) &&
           x280_pinned_input_contains(
               &X280_PINNED_INPUT_ROUTER_REGION, X280_PINNED_INPUT_FIRST_HEADER_BASE, FS_PACKET_HEADER_BYTES);
}

/* Binds state.header to the pinned first header, seeds it
 * (fs_seed_forward_header) and fences the seed out. */
inline void seed_route(state_t& state) {
    state.header = reinterpret_cast<volatile fs_hybrid_mesh_packet_header_t*>(X280_PINNED_INPUT_FIRST_HEADER_BASE);
    fs_seed_forward_header(state.header);
    x280_fence_ow();
}

struct Backend : member_backend_t<state_t> {
    static inline __attribute__((always_inline)) void stage_once(state_t& state, uint32_t hop, uint32_t& witness) {
        pipeline_stage::stage_once(state.header, hop, witness);
    }

    static uint32_t gen_seed(state_t&) { return 0u; }

    static inline __attribute__((always_inline)) uint32_t next_index(state_t& state, uint32_t& generator, uint32_t) {
        return next_hop(generator, state.args.section.hop_count);
    }

    static bool verify_stage(state_t& state) {
        return verify_route(state.header, state.args.section.hop_count, state.result->verify_sentinel);
    }

    static bool verify_after(state_t& state) { return verify_stage(state); }

    static bool publish_footer(state_t&) { return true; }
};

}  // namespace pipeline_stage::backend_route_forward

#endif
