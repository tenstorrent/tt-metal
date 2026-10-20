// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * Shared measurement sequencing for the fixed lookup and forwarding members,
 * with the state prefix, argument parsing and validation their backends share.
 * Backend::stage_once is the timed subject.
 */
#ifndef X280_PIPELINE_STAGE_FRONTEND_CXX_HPP
#define X280_PIPELINE_STAGE_FRONTEND_CXX_HPP

#include <stdint.h>

#include "pipeline_stage_common.h"
#include "../shared/ubench_finish.hpp"

namespace pipeline_stage {

static inline __attribute__((always_inline)) void pipeline_stage_sink(uint32_t value) {
    __asm__ volatile("" ::"r"(value));
}

/* True while no phase has recorded a terminal state in the result. */
inline bool in_progress(const pipeline_stage_result_t& result) {
    return pipeline_stage_outcome(static_cast<pipeline_stage_state_t>(result.terminal_state)) ==
           UBENCH_OUTCOME_IN_PROGRESS;
}

template <typename Backend>
inline void run_timed_plain(typename Backend::state_t& state, uint32_t count) {
    pipeline_stage_result_t& result = Backend::result(state);
    result.terminal_state = static_cast<uint32_t>(PIPELINE_STAGE_STATE_TIMED_PLAIN);
    result.packet_header_bytes = FS_PACKET_HEADER_BYTES;
    result.instrumentation = static_cast<uint32_t>(PIPELINE_STAGE_INSTRUMENTATION_OFF);

    uint32_t generator = Backend::gen_seed(state);
    uint32_t witness = 0u;
    const auto start = Backend::measurement_now(state);
    for (uint32_t i = 0u; i < count; ++i) {
        const uint32_t index = Backend::next_index(state, generator, i);
        Backend::stage_once(state, index, witness);
    }
    const auto end = Backend::measurement_now(state);
    pipeline_stage_sink(witness);
    Backend::publish_stage_witness(state, witness);
    Backend::publish_delta(state, start, end);
    result.items_completed = static_cast<uint64_t>(count);
}

template <typename Backend>
inline void run_timed_instrumented(typename Backend::state_t& state, uint32_t count) {
    pipeline_stage_result_t& result = Backend::result(state);
    result.terminal_state = static_cast<uint32_t>(PIPELINE_STAGE_STATE_TIMED_INSTRUMENTED);
    result.packet_header_bytes = FS_PACKET_HEADER_BYTES;
    result.instrumentation = static_cast<uint32_t>(PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE);

    pipeline_stage_samples_t* samples = Backend::samples(state);
    uint32_t generator = Backend::gen_seed(state);
    uint32_t witness = 0u;
    uint32_t completed = 0u;
    const auto start = Backend::measurement_now(state);
    for (; completed < count; ++completed) {
        const uint32_t index = Backend::next_index(state, generator, completed);
        const uint64_t sample_start = Backend::sample_now(state);
        Backend::stage_once(state, index, witness);
        const uint64_t sample_end = Backend::sample_now(state);
        if (!measurement_delta(sample_start, sample_end, &samples->raw_deltas[completed])) {
            result.terminal_state = static_cast<uint32_t>(PIPELINE_STAGE_STATE_COUNTER_ERROR);
            break;
        }
    }
    const auto end = Backend::measurement_now(state);
    pipeline_stage_sink(witness);
    Backend::publish_stage_witness(state, witness);
    if (in_progress(result)) {
        Backend::publish_delta(state, start, end);
    }
    result.items_completed = static_cast<uint64_t>(completed);
    samples->samples_completed = completed;
}

template <typename Backend>
inline void warmup(typename Backend::state_t& state, uint32_t count) {
    uint32_t generator = Backend::gen_seed(state);
    uint32_t witness = 0u;
    for (uint32_t i = 0u; i < count; ++i) {
        const uint32_t index = Backend::next_index(state, generator, i);
        Backend::stage_once(state, index, witness);
    }
    pipeline_stage_sink(witness);
}

template <typename Backend>
inline void run_member(typename Backend::state_t& state) {
    pipeline_stage_result_t& result = Backend::result(state);
    const pipeline_stage_common_args_t& args = Backend::common(state);

    result.terminal_state = static_cast<uint32_t>(PIPELINE_STAGE_STATE_VERIFY_STAGE);
    if (!Backend::verify_stage(state)) {
        ubench_finish<Backend>(state, PIPELINE_STAGE_STATE_VERIFY_MISMATCH, PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED);
        return;
    }

    result.terminal_state = static_cast<uint32_t>(PIPELINE_STAGE_STATE_WARMUP);
    warmup<Backend>(state, args.warmup_count);
    result.warmup_completed = args.warmup_count;

    if (args.instrumentation == static_cast<uint32_t>(PIPELINE_STAGE_INSTRUMENTATION_PER_SAMPLE)) {
        run_timed_instrumented<Backend>(state, args.timed_count);
    } else {
        run_timed_plain<Backend>(state, args.timed_count);
    }

    /* A backend or the sample loop may have latched a terminal state (such
     * as COUNTER_ERROR) during the timed phase; it stands, and is already
     * visible while the footer runs. */
    if (!in_progress(result)) {
        ubench_finish<Backend>(
            state,
            static_cast<pipeline_stage_state_t>(result.terminal_state),
            PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED);
        return;
    }

    result.terminal_state = static_cast<uint32_t>(PIPELINE_STAGE_STATE_VERIFY_AFTER);
    if (!Backend::verify_after(state)) {
        ubench_finish<Backend>(state, PIPELINE_STAGE_STATE_VERIFY_MISMATCH, PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED);
        return;
    }

    result.terminal_state = static_cast<uint32_t>(PIPELINE_STAGE_STATE_PUBLISH);
    result.timed_count = args.timed_count;
    ubench_finish<Backend>(state, PIPELINE_STAGE_STATE_DONE, PIPELINE_STAGE_STATE_WINDOW_CLEANUP_FAILED);
}

}  // namespace pipeline_stage

#endif
