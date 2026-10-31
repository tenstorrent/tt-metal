// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The write side of the null transport (see the null reader): the statistic
// tiles made from whatever the scratch holds (the compute kernel's inputs
// have to be finite numbers, which zeroed tiles are), the mask and ones
// tiles once, and the column gradients consumed at every interval's end.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_dataflow_utils.hpp"

#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif
#ifndef NO_STAT_WORK
#define NO_STAT_WORK 0
#endif
#ifndef L_SEED_SHIFT_BITS
#define L_SEED_SHIFT_BITS 0u
#endif
#ifndef L_SEED_SCALE_BITS
#define L_SEED_SCALE_BITS 0x3F800000u
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

    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_20;
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_23;
    constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
    constexpr uint32_t cb_lse_row = tt::CBIndex::c_13;
    constexpr uint32_t cb_u_row = tt::CBIndex::c_14;
    constexpr uint32_t cb_lse_rem = tt::CBIndex::c_9;
    constexpr uint32_t cb_u_rem = tt::CBIndex::c_29;

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();

    const uint32_t interm_bytes = get_tile_size(cb_u_scalar);
    const uint32_t base_lse = get_write_ptr(cb_u_scalar);
    const uint32_t base_u_scalar = base_lse + 2u * Bt * interm_bytes;
    {
        const uint32_t zero_l1 = get_write_ptr(tt::CBIndex::c_8);
        const uint32_t zero_bytes = get_tile_size(tt::CBIndex::c_8);
        cyclic_dataflow::zero_tile(zero_l1, zero_bytes);
        cyclic_dataflow::zero_region_via_noc(get_write_ptr(cb_lse_row), 2u * Bt * interm_bytes, zero_l1, zero_bytes);
        cyclic_dataflow::zero_region_via_noc(get_write_ptr(cb_u_row), 2u * Bt * interm_bytes, zero_l1, zero_bytes);
        cyclic_dataflow::zero_region_via_noc(
            get_write_ptr(cb_lse_rem), 2u * Bt * get_tile_size(cb_lse_rem), zero_l1, zero_bytes);
        cyclic_dataflow::zero_region_via_noc(
            get_write_ptr(cb_u_rem), 2u * Bt * get_tile_size(cb_u_rem), zero_l1, zero_bytes);
        // The L / D scratch too, so the statistics are finite.
        cyclic_dataflow::zero_region_via_noc(base_lse, 4u * Bt * interm_bytes, zero_l1, zero_bytes);
        cyclic_dataflow::generate_causal_mask_tiles(cb_attn_mask);
        cyclic_dataflow::generate_ones_column_tile(tt::CBIndex::c_28);
        noc_async_write_barrier();
    }

    for (uint32_t s = 0; s < slice_count; ++s) {
        for (uint32_t t = 0; t < kTimesteps; ++t) {
            const auto pair = sched.pair(my_core, t);
            cb_wait_front(cyclic_dataflow::kStatsReadyCb, 1);
#if NO_STAT_WORK
            cb_reserve_back(cb_lse_row, Bt);
            cb_reserve_back(cb_u_row, Bt);
            cb_reserve_back(cb_lse_rem, Bt);
            cb_reserve_back(cb_u_rem, Bt);
            cb_push_back(cb_lse_row, Bt);
            cb_push_back(cb_u_row, Bt);
            cb_push_back(cb_lse_rem, Bt);
            cb_push_back(cb_u_rem, Bt);
#else
            cyclic_dataflow::produce_statistic_tiles(
                base_lse, base_u_scalar, Bt, interm_bytes, cb_lse_row, cb_u_row, cb_lse_rem, cb_u_rem,
                L_SEED_SHIFT_BITS, L_SEED_SCALE_BITS);
#endif
            cb_pop_front(cyclic_dataflow::kStatsReadyCb, 1);
            const bool column_ends = (t + 1u == kTimesteps) || (sched.pair(my_core, t + 1u).j != pair.j);
            if (column_ends) {
                cb_wait_front(cb_grad_value, val_tiles);
                cb_pop_front(cb_grad_value, val_tiles);
                cb_wait_front(cb_grad_key, row_tiles);
                cb_pop_front(cb_grad_key, row_tiles);
            }
        }
    }
}
