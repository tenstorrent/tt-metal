// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Column gradients and the chip-wide barrier, for the relay variant.
//
// The row packet is the relay reader's business: it receives it, hands it to
// the compute kernel and forwards it. What is left here is the column side,
// which still goes through DRAM every timestep, and the barrier that orders a
// streak-start reload after the preceding streak's spill.
//
// The mask tile is generated here, once, as sdpa_bw's writer does.
//
// ENDPOINT_SYNC selects Algorithm 4: no chip-wide barrier at all. What the
// barrier did here was tell this core's reader that the column gradients it
// is about to load have been written -- an ordering between two RISCs of the
// same core, which needs no chip-wide anything. A local progress word does
// it. The other thing the barrier did, ordering a streak-start reload after
// the preceding streak's spill, is the reader's business and becomes the
// endpoint counters.
//
// The column gradients only pass through DRAM at all because this step has
// not yet restored the paper's column residency; with them resident, this
// handoff disappears too.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/waypoint.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/kernels/dataflow/cyclic_dataflow_utils.hpp"

#ifndef ENDPOINT_SYNC
#define ENDPOINT_SYNC 0
#endif

// DENSE_MODE selects the unmasked schedule: every block pair is live, which
// is what a ring-attention step needs when the visiting key/value chunk is
// earlier in the sequence than the local query chunk. It changes the schedule
// (2T timesteps in two passes rather than T + 1) and, in the compute kernel,
// removes the intra-block mask; nothing else about the relay changes.
#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif

#if DENSE_MODE
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Dense;
#else
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Causal;
#endif

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t my_core = get_arg_val<uint32_t>(arg++);
    const uint32_t first_slice = get_arg_val<uint32_t>(arg++);  // this group's first slice
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t coord_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t coord_noc_y = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_x_start = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_y_start = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_x_end = get_arg_val<uint32_t>(arg++);
    const uint32_t mcast_y_end = get_arg_val<uint32_t>(arg++);
    const uint32_t is_coordinator = get_arg_val<uint32_t>(arg++);
    // Slices this group runs in sequence, and the stride between them; see
    // the relay reader.
    const uint32_t slice_count = get_arg_val<uint32_t>(arg++);
    const uint32_t slice_stride = get_arg_val<uint32_t>(arg++);
    // Chunk pairs, as in the relay reader. Only the column chunk matters
    // here: dK and dV belong to the key side.
    const uint32_t chunks = get_arg_val<uint32_t>(arg++);
    const uint32_t pairs = get_arg_val<uint32_t>(arg++);
    const uint32_t pair_table_arg = arg;

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(4);
    // Row-tiles per block: B = Bt * 32, so a block is Bt * qWt tiles wide in
    // memory and the statistics are Bt tiles instead of one.
    constexpr uint32_t Bt = get_compile_time_arg_val(5);
    constexpr uint32_t row_tiles = Bt * qWt;
    constexpr uint32_t val_tiles = Bt * vWt;
    constexpr auto grad_key_args = TensorAccessorArgs<6>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();
    // The packet-readiness semaphores, one per slot, after the accessor args.
    constexpr uint32_t ready_imm_sem_id[2] = {
        get_compile_time_arg_val(grad_value_args.next_compile_time_args_offset()),
        get_compile_time_arg_val(grad_value_args.next_compile_time_args_offset() + 1u)};

    constexpr uint32_t cb_attn_mask = tt::CBIndex::c_6;
    constexpr uint32_t cb_grad_key = tt::CBIndex::c_20;
    constexpr uint32_t cb_grad_value = tt::CBIndex::c_23;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_25;
    constexpr uint32_t cb_column_progress = tt::CBIndex::c_26;
    // The packet's statistics, and the tiles this kernel makes of them for the
    // compute kernel (see cyclic_dataflow_utils.hpp).
    constexpr uint32_t cb_lse = tt::CBIndex::c_4;
    constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
    constexpr uint32_t cb_lse_row = tt::CBIndex::c_13;
    constexpr uint32_t cb_u_row = tt::CBIndex::c_14;
    constexpr uint32_t cb_lse_rem = tt::CBIndex::c_30;
    constexpr uint32_t cb_u_rem = tt::CBIndex::c_29;

    using ttml::metal::ops::cyclic_sdpa_bw::CyclicSchedule;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();

    // The compute kernel forms S^T, so its diagonal tile takes the transposed
    // causal mask: live where the key index is at most the query index.
    cyclic_dataflow::generate_transposed_causal_mask_tile(cb_attn_mask);
    cyclic_dataflow::generate_ones_column_tile(tt::CBIndex::c_28);  // for the D remainder

    // Slot-0 addresses of the packet's statistic buffers (this kernel never
    // reserves them, so its write pointers stay at the base), and the reader's
    // "statistics in L1" word.
    const uint32_t interm_bytes = get_tile_size(cb_lse);
    const uint32_t base_lse = get_write_ptr(cb_lse);
    const uint32_t base_u_scalar = get_write_ptr(cb_u_scalar);
    const uint32_t stride_interm = Bt * interm_bytes;
    volatile tt_l1_ptr uint32_t* ready_imm_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_imm_sem_id[0])),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_imm_sem_id[1]))};
    // The row-layout tiles are written in row 0 only, the remainder columns
    // in column 0 only; zero them once so the rest holds zeros.
    cyclic_dataflow::zero_tile(get_write_ptr(cb_lse_row), 2u * Bt * interm_bytes);
    cyclic_dataflow::zero_tile(get_write_ptr(cb_u_row), 2u * Bt * interm_bytes);
    cyclic_dataflow::zero_tile(get_write_ptr(cb_lse_rem), 2u * Bt * interm_bytes);
    cyclic_dataflow::zero_tile(get_write_ptr(cb_u_rem), 2u * Bt * get_tile_size(cb_u_rem));

    const uint32_t grad_bytes = get_tile_size(cb_grad_key);
    const auto grad_key = TensorAccessor(grad_key_args, grad_key_addr, grad_bytes);
    const auto grad_value = TensorAccessor(grad_value_args, grad_value_addr, grad_bytes);
    uint32_t row_base = 0;  // per slice
    uint32_t val_base = 0;

#if ENDPOINT_SYNC
    volatile tt_l1_ptr uint32_t* column_progress =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_column_progress));
    *column_progress = 0u;
#endif

    volatile tt_l1_ptr uint32_t* arrive_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrive_sem_id));
    const uint32_t scratch_l1 = get_write_ptr(cb_scratch);
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
    const uint64_t arrive_noc_addr = get_noc_addr(coord_noc_x, coord_noc_y, get_semaphore(arrive_sem_id));
    const uint64_t release_mcast_addr = get_noc_multicast_addr(
        mcast_x_start, mcast_y_start, mcast_x_end, mcast_y_end, get_semaphore(release_sem_id));

    for (uint32_t s = 0; s < slice_count; ++s) {
    const uint32_t sl = first_slice + s * slice_stride;
    const uint32_t bh = sl / pairs;
    const uint32_t col_chunk = get_arg_val<uint32_t>(pair_table_arg + 2u * (sl % pairs) + 1u);
    row_base = (bh * chunks + col_chunk) * 2u * kCores * row_tiles;
    val_base = (bh * chunks + col_chunk) * 2u * kCores * val_tiles;
    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);
        // Global timestep across slices; the progress word and the barrier
        // counter must keep rising across a slice boundary.
        const uint32_t g = s * kTimesteps + t;

        // First the statistic tiles for this timestep, as soon as the reader
        // has L and D in slot g mod 2: the compute kernel waits on nothing
        // else to start its score pass. Before the column-gradient writes
        // below, which wait on the compute kernel and so must not hold this up.
        {
            DeviceZoneScopedN("STAT-ROWS");
            const uint32_t slot = g % 2u;
            const bool forwarded = sched.producer(my_core, t).internal;
            if (forwarded) {
                // The producer's readiness write, the same one the reader
                // waits on -- but the reader gets to it only after the previous
                // timestep's dQ relay, and this is a timestep earlier.
                WAYPOINT("STRW");
                do {
                    invalidate_l1_cache();
                } while ((*ready_imm_sem[slot]) < g + 1u);
                WAYPOINT("STRD");
            } else {
                cb_wait_front(cyclic_dataflow::kStatsReadyCb, 1);
                invalidate_l1_cache();
            }
            cyclic_dataflow::produce_statistic_tiles(
                base_lse + slot * stride_interm, base_u_scalar + slot * stride_interm, Bt, interm_bytes,
                cb_lse_row, cb_u_row, cb_lse_rem, cb_u_rem);
            if (!forwarded) {
                cb_pop_front(cyclic_dataflow::kStatsReadyCb, 1);
            }
        }

        // The column gradients are handed over once per residency interval,
        // at its end -- which the schedule says is a column change or the
        // last timestep. Writing them back before reusing their storage is
        // the paper's rule; here the storage is released by the handover
        // itself.
        const bool column_ends =
            (t + 1u == kTimesteps) || (sched.pair(my_core, t + 1u).j != pair.j);
        if (column_ends) {
            write_tiles_by_row(cb_grad_value, grad_value, val_base + (pair.j - 1u) * val_tiles, val_tiles, grad_bytes, val_tiles);
            write_tiles_by_row(cb_grad_key, grad_key, row_base + (pair.j - 1u) * row_tiles, row_tiles, grad_bytes, row_tiles);
        }

#if ENDPOINT_SYNC
        // This core's own reader is the only thing waiting on these writes.
        *column_progress = g + 1u;
#else
        noc_semaphore_inc(arrive_noc_addr, 1u);

        if (is_coordinator != 0u) {
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
#endif
    }
    }  // slices
}
