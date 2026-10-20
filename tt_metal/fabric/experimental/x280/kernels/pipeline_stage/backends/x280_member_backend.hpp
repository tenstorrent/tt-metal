// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/* The Backend operations both X280 pipeline_stage members share. */
#ifndef X280_PIPELINE_STAGE_BACKEND_MEMBER_BACKEND_HPP
#define X280_PIPELINE_STAGE_BACKEND_MEMBER_BACKEND_HPP

#include <stdint.h>

extern "C" {
#include "rt/x280_measurement.h"
#include "rt/x280_pinned_input.h"
}

#include "pipeline_stage_common.h"

namespace pipeline_stage {

struct timestamp_t {
    uint64_t mtime_ticks;
    uint64_t cycle_ticks;
};

/* A Backend derives from this and adds stage_once, gen_seed, next_index,
 * verify_stage, verify_after and publish_footer. State carries the args,
 * result and samples members. */
template <typename State>
struct member_backend_t {
    using state_t = State;

    static const pipeline_stage_common_args_t& common(state_t& state) { return state.args.section.common; }
    static pipeline_stage_result_t& result(state_t& state) { return *state.result; }
    static pipeline_stage_samples_t* samples(state_t& state) { return state.samples; }

    static inline __attribute__((always_inline)) timestamp_t measurement_now(state_t&) {
        const x280_paired_t now = x280_paired_now();
        return timestamp_t{now.wall, now.cpu};
    }

    static inline __attribute__((always_inline)) uint64_t sample_now(state_t&) { return x280_rdcycle_fenced(); }

    static void publish_stage_witness(state_t& state, uint32_t witness) { state.result->stage_witness = witness; }

    static void publish_delta(state_t& state, const timestamp_t& start, const timestamp_t& end) {
        pipeline_stage_publish_batch_delta(
            state.result, start.mtime_ticks, end.mtime_ticks, start.cycle_ticks, end.cycle_ticks);
    }

    static void publish_barrier(state_t&) { x280_fence_ow(); }
};

/**
 * Writes a failure record (pipeline_stage_write_result) to state.result and
 * fences it out. state.result must point at a result region
 * the section admitted.
 */
template <typename State>
inline void publish_failure(State& state, uint32_t backend_id, pipeline_stage_state_t terminal_state, uint32_t reason) {
    pipeline_stage_write_result(state.result, backend_id, static_cast<uint32_t>(terminal_state), reason, 0u);
    x280_fence_ow();
}

/**
 * The refusal of a section that failed validation: when it loaded and its
 * result_base is a 64-byte-aligned span inside its layout window, binds
 * state.result there and publishes an ARG_LAYOUT_MISMATCH record with reason
 * 0xA0. Otherwise writes nothing, and the host sees no completion.
 */
template <typename State>
inline void refuse_section(State& state, bool loaded, uint32_t backend_id) {
    const uint64_t result_base = state.args.section.backend.result_base;
    if (!loaded || result_base % UBENCH_LIM_LINE_BYTES != 0u ||
        !is_span_contained(result_base, sizeof(pipeline_stage_result_t), state.args.layout)) {
        return;
    }
    state.result = reinterpret_cast<pipeline_stage_result_t*>(result_base);
    publish_failure(state, backend_id, PIPELINE_STAGE_STATE_ARG_LAYOUT_MISMATCH, 0xA0u);
}

/**
 * Binds state.result, and state.samples under per-sample instrumentation, to
 * the regions an admitted section names, zeroes the sample header, and starts
 * the result in PIPELINE_STAGE_STATE_INIT under backend_id and workload_size.
 */
template <typename State>
inline void bind_result(State& state, uint32_t backend_id, uint32_t workload_size) {
    state.result = reinterpret_cast<pipeline_stage_result_t*>(state.args.section.backend.result_base);
    if (state.args.section.common.instrumentation == static_cast<uint32_t>(PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE)) {
        state.samples = reinterpret_cast<pipeline_stage_samples_t*>(state.args.section.backend.sample_region_base);
        state.samples->samples_completed = 0u;
        state.samples->reserved = 0u;
    }
    pipeline_stage_write_result(
        state.result, backend_id, static_cast<uint32_t>(PIPELINE_STAGE_STATE_INIT), 0u, workload_size);
}

#if defined(__riscv) && __riscv_xlen == 64
/**
 * True when the router region's pinned-input policy admits a prepared run.
 * Otherwise publishes an ARG_LAYOUT_MISMATCH record whose reason is
 * 0xB000 | code << 8 | master_id (low bytes), with the policy's master_id in
 * workload_size and its observed word in secondary_batch_delta. Call after
 * bind_result.
 */
template <typename State>
inline bool admit_pinned_input(State& state, uint32_t backend_id) {
    const x280_pinned_input_admission_t admission =
        x280_pinned_input_check_policy(&X280_PINNED_INPUT_ROUTER_REGION, X280_PINNED_INPUT_PREPARED);
    if (admission.code == X280_PINNED_INPUT_ADMITTED) {
        return true;
    }
    const uint32_t reason = 0xB000u | ((admission.code & 0xFFu) << 8u) | (admission.master_id & 0xFFu);
    pipeline_stage_write_result(
        state.result,
        backend_id,
        static_cast<uint32_t>(PIPELINE_STAGE_STATE_ARG_LAYOUT_MISMATCH),
        reason,
        admission.master_id);
    state.result->secondary_batch_delta = admission.observed;
    x280_fence_ow();
    return false;
}
#endif

}  // namespace pipeline_stage

#endif
