// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// No transport at all, for the measurements: the compute kernel's input
// buffers are pushed as soon as they are free, nothing is read from DRAM or
// from another core, and whatever the buffers hold is computed on. The
// column state is loaded once per residency interval only in the sense that
// K and V are pushed then; its gradients are consumed by the null writer.
// With the compute kernel's arithmetic on, this is the kernel's compute time
// per timestep alone -- what the data movement must hide under.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_dataflow_utils.hpp"

#ifndef SEED_COLUMN_GRADIENTS
#define SEED_COLUMN_GRADIENTS 0
#endif
#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif
#if DENSE_MODE
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Dense;
#else
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Causal;
#endif

void kernel_main() {
    const uint32_t my_core = get_arg_val<uint32_t>(0);
    const uint32_t slice_count = get_arg_val<uint32_t>(1);

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t Bt = get_compile_time_arg_val(3);
    constexpr uint32_t row_tiles = Bt * qWt;
    constexpr uint32_t val_tiles = Bt * vWt;

    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_15;
    constexpr uint32_t cb_grad_query_out = tt::CBIndex::c_17;
    constexpr uint32_t cb_grad_key_seed = tt::CBIndex::c_18;
    constexpr uint32_t cb_grad_value_seed = tt::CBIndex::c_21;

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();
    const auto owned = sched.owned_columns(my_core);
    bool visited[2] = {false, false};

    for (uint32_t s = 0; s < slice_count; ++s) {
        visited[0] = false;
        visited[1] = false;
        for (uint32_t t = 0; t < kTimesteps; ++t) {
            const auto pair = sched.pair(my_core, t);
            const bool column_changed = (t == 0u) || (sched.pair(my_core, t - 1u).j != pair.j);
            if (column_changed) {
                cb_reserve_back(cb_key, row_tiles);
                cb_push_back(cb_key, row_tiles);
                cb_reserve_back(cb_value, val_tiles);
                cb_push_back(cb_value, val_tiles);
                const uint32_t owned_slot = (pair.j == owned.first) ? 0u : 1u;
                if (visited[owned_slot] || SEED_COLUMN_GRADIENTS) {
                    cb_reserve_back(cb_grad_key_seed, row_tiles);
                    cb_push_back(cb_grad_key_seed, row_tiles);
                    cb_reserve_back(cb_grad_value_seed, val_tiles);
                    cb_push_back(cb_grad_value_seed, val_tiles);
                }
                visited[owned_slot] = true;
            }
            cb_reserve_back(cb_query, row_tiles);
            cb_push_back(cb_query, row_tiles);
            cb_reserve_back(cb_grad_output, val_tiles);
            cb_push_back(cb_grad_output, val_tiles);
            cb_reserve_back(cyclic_dataflow::kStatsReadyCb, 1);
            cb_push_back(cyclic_dataflow::kStatsReadyCb, 1);
            cb_reserve_back(cb_grad_query_seed, row_tiles);
            cb_push_back(cb_grad_query_seed, row_tiles);
            cb_wait_front(cb_grad_query_out, row_tiles);
            cb_pop_front(cb_grad_query_out, row_tiles);
        }
    }
}
