// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The write kernel of Algorithm 1 (see the DRAM reader): the statistic tiles
// from the L and D the reader loads every timestep, the column gradients at
// the end of each residency interval, and the chip-wide barrier, at which
// this core arrives once its reader has written the timestep's dQ back.
//
// NO_STAT_WORK (measurement knob): the soft-float work on the statistics is
// skipped, every handshake kept.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/waypoint.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
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

#if defined(PROFILE_NOC_EVENTS)
#undef DeviceZoneScopedN
#define DeviceZoneScopedN(name)
#endif

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t my_core = get_arg_val<uint32_t>(arg++);
    const uint32_t first_slice = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t coord_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t coord_noc_y = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_x_start = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_y_start = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_x_end = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_y_end = get_arg_val<uint32_t>(arg++);
    const uint32_t is_coordinator = get_arg_val<uint32_t>(arg++);
    const uint32_t slice_count = get_arg_val<uint32_t>(arg++);
    const uint32_t slice_stride = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t Bt = get_compile_time_arg_val(5);
    constexpr uint32_t row_tiles = Bt * qWt;
    constexpr uint32_t val_tiles = Bt * vWt;
    constexpr auto grad_key_args = TensorAccessorArgs<6>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_20;
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_23;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_25;
    constexpr uint32_t cb_column_progress = tt::CBIndex::c_26;
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
    const uint32_t stride_interm = Bt * interm_bytes;
    {
        DeviceZoneScopedN("WRITER-INIT");
        const uint32_t zero_l1 = get_write_ptr(tt::CBIndex::c_8);
        const uint32_t zero_bytes = get_tile_size(tt::CBIndex::c_8);
        cyclic_dataflow::zero_tile(zero_l1, zero_bytes);
        cyclic_dataflow::zero_region_via_noc(get_write_ptr(cb_lse_row), 2u * Bt * interm_bytes, zero_l1, zero_bytes);
        cyclic_dataflow::zero_region_via_noc(get_write_ptr(cb_u_row), 2u * Bt * interm_bytes, zero_l1, zero_bytes);
        cyclic_dataflow::zero_region_via_noc(
            get_write_ptr(cb_lse_rem), 2u * Bt * get_tile_size(cb_lse_rem), zero_l1, zero_bytes);
        cyclic_dataflow::zero_region_via_noc(
            get_write_ptr(cb_u_rem), 2u * Bt * get_tile_size(cb_u_rem), zero_l1, zero_bytes);
        cyclic_dataflow::generate_causal_mask_tiles(cb_attn_mask);
        cyclic_dataflow::generate_ones_column_tile(tt::CBIndex::c_28);
        noc_async_write_barrier();
    }

    const uint32_t grad_bytes = get_tile_size(cb_grad_key);
    const auto grad_key = TensorAccessor(grad_key_args, grad_key_addr, grad_bytes);
    const auto grad_value = TensorAccessor(grad_value_args, grad_value_addr, grad_bytes);

    volatile tt_l1_ptr uint32_t* column_progress =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_column_progress));
    *column_progress = 0u;
    volatile tt_l1_ptr uint32_t* arrive_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrive_sem_id));
    const uint32_t scratch_l1 = get_write_ptr(cb_scratch);
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
    const uint64_t arrive_noc_addr = get_noc_addr(coord_noc_x, coord_noc_y, get_semaphore(arrive_sem_id));
    const uint64_t release_mcast_addr = get_noc_multicast_addr(
        mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, get_semaphore(release_sem_id));

    for (uint32_t s = 0; s < slice_count; ++s) {
        const uint32_t sl = first_slice + s * slice_stride;
        const uint32_t row_base = sl * 2u * kCores * row_tiles;
        const uint32_t val_base = sl * 2u * kCores * val_tiles;
        for (uint32_t t = 0; t < kTimesteps; ++t) {
            const auto pair = sched.pair(my_core, t);
            const uint32_t g = s * kTimesteps + t;
            const uint32_t slot = g % 2u;

            {
                DeviceZoneScopedN("STAT-ROWS");
                cb_wait_front(cyclic_dataflow::kStatsReadyCb, 1);
                invalidate_l1_cache();
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
                    base_lse + slot * stride_interm, base_u_scalar + slot * stride_interm, Bt, interm_bytes,
                    cb_lse_row, cb_u_row, cb_lse_rem, cb_u_rem, L_SEED_SHIFT_BITS, L_SEED_SCALE_BITS);
#endif
                cb_pop_front(cyclic_dataflow::kStatsReadyCb, 1);
            }

            const bool column_ends = (t + 1u == kTimesteps) || (sched.pair(my_core, t + 1u).j != pair.j);
            if (column_ends) {
                DeviceZoneScopedN("STORE-COL");
                write_tiles_by_row(cb_grad_value, grad_value, val_base + (pair.j - 1u) * val_tiles, val_tiles, grad_bytes, val_tiles);
                write_tiles_by_row(cb_grad_key, grad_key, row_base + (pair.j - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
            }
            *column_progress = g + 1u;

            // This core's dQ write-back is complete: arrive.
            {
                DeviceZoneScopedN("ARRIVE");
                cb_wait_front(cyclic_dataflow::kStatsDoneCb, 1);
                cb_pop_front(cyclic_dataflow::kStatsDoneCb, 1);
                noc_semaphore_inc(arrive_noc_addr, 1u);
            }
            if (is_coordinator != 0u) {
                DeviceZoneScopedN("RELEASE");
                WAYPOINT("ARVW");
                do {
                    invalidate_l1_cache();
                } while ((*arrive_sem) < kCores * (g + 1u));
                WAYPOINT("ARVD");
                if constexpr (kCores == 1u) {
                    volatile tt_l1_ptr uint32_t* release_local =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_sem_id));
                    noc_semaphore_set(release_local, g + 1u);
                } else {
                    *scratch = g + 1u;
                    noc_semaphore_set_multicast_loopback_src(scratch_l1, release_mcast_addr, kCores);
                    noc_async_write_barrier();
                }
            }
        }
    }
    noc_async_atomic_barrier();
}
