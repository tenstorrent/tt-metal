// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Algorithm 1 of the paper, for the measurements: the cyclic schedule with
// the column state resident in L1 and every row packet going through DRAM
// at every timestep. The compute kernel is the relay's (COLUMN_RESIDENT, no
// release token); the buffers are laid out as the relay lays them out; only
// where the row packet comes from and goes to differs.
//
// Per timestep: the column state where the column changes (K, V; the
// gradients' running sums where the column is revisited), then Q, dO, L and
// D of the row, then -- after the chip-wide barrier of the previous timestep,
// which certifies that every core's dQ write-back of that timestep has
// completed -- dQ, and once the compute kernel has added to it, dQ back to
// DRAM, after which the write kernel arrives at the barrier.
//
// NO_STAT_READS (measurement knob): L and D are not read at all, which is
// what a DRAM implementation that stored the statistics compactly (as the
// relay's packet carries them) would save; they are stored as full tiles,
// so their pages are 4 KB a row tile against 128 bytes of values.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_dataflow_utils.hpp"

#ifndef SEED_COLUMN_GRADIENTS
#define SEED_COLUMN_GRADIENTS 0
#endif
#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif
#ifndef NO_STAT_READS
#define NO_STAT_READS 0
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
    const uint32_t query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t value_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lse_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t u_scalar_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(arg++);
    // Slices this core's group runs in turn, and the stride between them
    // (0 repeats one slice, for a measurement).
    const uint32_t slice_count = get_arg_val<uint32_t>(arg++);
    const uint32_t slice_stride = get_arg_val<uint32_t>(arg++);

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t Bt = get_compile_time_arg_val(4);
    constexpr uint32_t row_tiles = Bt * qWt;
    constexpr uint32_t val_tiles = Bt * vWt;
    constexpr auto query_args = TensorAccessorArgs<5>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto grad_output_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto lse_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto u_args = TensorAccessorArgs<lse_args.next_compile_time_args_offset()>();
    constexpr auto grad_query_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto grad_key_args = TensorAccessorArgs<grad_query_args.next_compile_time_args_offset()>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;  // L / D scratch, two slots of each
    constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_15;
    constexpr uint32_t cb_grad_query_out = tt::CBIndex::c_17;
    constexpr uint32_t cb_grad_key_seed = tt::CBIndex::c_18;
    constexpr uint32_t cb_grad_value_seed = tt::CBIndex::c_21;
    constexpr uint32_t cb_column_progress = tt::CBIndex::c_26;

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();

    const uint32_t tile_bytes = get_tile_size(cb_query);
    const uint32_t interm_bytes = get_tile_size(cb_u_scalar);
    const uint32_t grad_bytes = get_tile_size(cb_grad_query_seed);

    const auto query = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto grad_output = TensorAccessor(grad_output_args, grad_output_addr, tile_bytes);
    const auto lse = TensorAccessor(lse_args, lse_addr, interm_bytes);
    const auto u_scalar = TensorAccessor(u_args, u_scalar_addr, interm_bytes);
    const auto grad_query = TensorAccessor(grad_query_args, grad_query_addr, grad_bytes);
    const auto grad_key = TensorAccessor(grad_key_args, grad_key_addr, grad_bytes);
    const auto grad_value = TensorAccessor(grad_value_args, grad_value_addr, grad_bytes);

    // The L / D scratch: the relay's layout (L slot 0, L slot 1, D slot 0,
    // D slot 1), which the write kernel reads the statistics from.
    const uint32_t base_lse = get_write_ptr(cb_u_scalar);
    const uint32_t base_u_scalar = base_lse + 2u * Bt * interm_bytes;
    const uint32_t stride_interm = Bt * interm_bytes;

    volatile tt_l1_ptr uint32_t* release_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_sem_id));
    volatile tt_l1_ptr uint32_t* column_progress =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_column_progress));

    const auto owned = sched.owned_columns(my_core);
    const uint32_t owned_column[2] = {owned.first, owned.second};
    bool visited[2] = {false, false};

    for (uint32_t s = 0; s < slice_count; ++s) {
        const uint32_t sl = first_slice + s * slice_stride;
        const uint32_t row_base = sl * 2u * kCores * row_tiles;
        const uint32_t val_base = sl * 2u * kCores * val_tiles;
        const uint32_t stat_base = sl * 2u * kCores * Bt;
        visited[0] = false;
        visited[1] = false;
        for (uint32_t t = 0; t < kTimesteps; ++t) {
            DeviceZoneScopedN("DRAM-READER-STEP");
            const auto pair = sched.pair(my_core, t);
            const uint32_t i = pair.i;
            const uint32_t j = pair.j;
            const uint32_t g = s * kTimesteps + t;
            const uint32_t slot = g % 2u;

            const bool column_changed = (t == 0u) || (sched.pair(my_core, t - 1u).j != j);
            if (column_changed) {
                DeviceZoneScopedN("LOAD-COL");
                read_tiles_by_row(cb_key, key, row_base + (j - 1u) * row_tiles, row_tiles, tile_bytes, row_tiles);
                read_tiles_by_row(cb_value, value, val_base + (j - 1u) * val_tiles, val_tiles, tile_bytes, val_tiles);
                const uint32_t owned_slot = (j == owned_column[0]) ? 0u : 1u;
                if (visited[owned_slot] || SEED_COLUMN_GRADIENTS) {
                    // The running sums this core's own write kernel stored at
                    // the end of the column's previous interval.
                    WAYPOINT("COLW");
                    do {
                        invalidate_l1_cache();
                    } while ((*column_progress) < g);
                    WAYPOINT("COLD");
                    read_tiles_by_row(
                        cb_grad_key_seed, grad_key, row_base + (j - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
                    read_tiles_by_row(
                        cb_grad_value_seed, grad_value, val_base + (j - 1u) * val_tiles, val_tiles, grad_bytes,
                        val_tiles);
                }
                visited[owned_slot] = true;
            }

            // The row's constant fields, before the barrier: they depend on
            // nothing that is in flight.
            {
                DeviceZoneScopedN("LOAD-IMM");
                cb_reserve_back(cb_query, row_tiles);
                cb_reserve_back(cb_grad_output, val_tiles);
                const uint32_t qs = get_write_ptr(cb_query);
                const uint32_t os = get_write_ptr(cb_grad_output);
                for (uint32_t k = 0; k < row_tiles; ++k) {
                    noc_async_read_page(row_base + (i - 1u) * row_tiles + k, query, qs + k * tile_bytes);
                }
                for (uint32_t k = 0; k < val_tiles; ++k) {
                    noc_async_read_page(val_base + (i - 1u) * val_tiles + k, grad_output, os + k * tile_bytes);
                }
#if !NO_STAT_READS
                const uint32_t ls = base_lse + slot * stride_interm;
                const uint32_t ds = base_u_scalar + slot * stride_interm;
                for (uint32_t k = 0; k < Bt; ++k) {
                    noc_async_read_page(stat_base + (i - 1u) * Bt + k, lse, ls + k * interm_bytes);
                    noc_async_read_page(stat_base + (i - 1u) * Bt + k, u_scalar, ds + k * interm_bytes);
                }
#endif
                noc_async_read_barrier();
                // L and D are in L1: the write kernel makes the statistic tiles.
                cb_reserve_back(cyclic_dataflow::kStatsReadyCb, 1);
                cb_push_back(cyclic_dataflow::kStatsReadyCb, 1);
                cb_push_back(cb_query, row_tiles);
                cb_push_back(cb_grad_output, val_tiles);
            }

            // The barrier of the previous timestep: every core's dQ
            // write-back of g - 1 has completed, so this row's dQ carries
            // every earlier update.
            if (g > 0u) {
                DeviceZoneScopedN("BARRIER");
                WAYPOINT("BARW");
                do {
                    invalidate_l1_cache();
                } while ((*release_sem) < g);
                WAYPOINT("BARD");
            }
            {
                DeviceZoneScopedN("LOAD-DQ");
                read_tiles_by_row(
                    cb_grad_query_seed, grad_query, row_base + (i - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
            }
            // The updated dQ back to DRAM, complete, then the write kernel is
            // told so it can arrive at this timestep's barrier.
            {
                DeviceZoneScopedN("STORE-DQ");
                write_tiles_by_row(
                    cb_grad_query_out, grad_query, row_base + (i - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
                cb_reserve_back(cyclic_dataflow::kStatsDoneCb, 1);
                cb_push_back(cyclic_dataflow::kStatsDoneCb, 1);
            }
        }
    }
}
