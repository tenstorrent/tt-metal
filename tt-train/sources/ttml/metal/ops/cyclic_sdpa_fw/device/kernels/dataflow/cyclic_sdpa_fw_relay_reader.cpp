// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The forward's row-packet relay (tt-flash-attn, Algorithm 8): receive a
// packet, hand it to the compute kernel, forward it to the next consumer.
// The transport is the backward relay's (cyclic_sdpa_bw_relay_reader.cpp
// has the full account: slots as circular-buffer pages, per-pair monotone
// credits, payload before readiness, release after the compute kernel's
// pop, endpoint words ordering a reload after the preceding streak's spill,
// read remotely rather than published). What differs is the packet:
//
//   immutable   Q_i                       forwarded before any arithmetic,
//   the state   O_i^T (Float32, tiles transposed within themselves), and
//               m_i, l_i as row-layout Float32 tiles (one value per query
//               in row 0, one tile per row tile of the block),
//
// updated in place by the compute kernel and forwarded after it, with its
// own readiness tag. At a row's first streak the state is nothing -- the
// compute kernel's first update writes -- so nothing is read; at a later
// streak start it is reloaded from the spill scratch; at the row's last
// visit the compute kernel finishes the row and the write kernel stores it,
// so nothing is spilled.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_fw/device/kernels/dataflow/cyclic_dataflow_utils.hpp"

#ifndef DENSE_MODE
#define DENSE_MODE 0
#endif

#if DENSE_MODE
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Dense;
#else
constexpr auto kMaskMode = ttml::metal::ops::cyclic_sdpa_bw::MaskMode::Causal;
#endif

// Under NoC event tracing the zones below are noise, and at a whole launch
// they overflow the per-core marker buffer, which breaks the trace's zone
// pairing; compiled out there, unchanged in an ordinary profiling build.
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
    const uint32_t state_acc_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t state_stats_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t prev_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t prev_noc_y = get_arg_val<uint32_t>(arg++);
    const uint32_t next_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t next_noc_y = get_arg_val<uint32_t>(arg++);
    const uint32_t core_coords_arg = arg;

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    const uint32_t slice_count = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores);
    const uint32_t slice_stride = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 1u);
    const uint32_t chunks = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 2u);
    const uint32_t pairs = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 3u);
    const uint32_t heads = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 4u);
    const uint32_t kv_slices = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 5u);
    const uint32_t q_heads = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 6u);
    const uint32_t kv_heads = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 7u);
    const uint32_t heads_per_group = get_arg_val<uint32_t>(core_coords_arg + 2u * kCores + 8u);
    const uint32_t pair_table_arg = core_coords_arg + 2u * kCores + 9u;
    (void)pairs;
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t ready_imm_sem_id[2] = {get_compile_time_arg_val(3), get_compile_time_arg_val(4)};
    constexpr uint32_t ready_st_sem_id[2] = {get_compile_time_arg_val(5), get_compile_time_arg_val(6)};
    constexpr uint32_t credit_prev_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t credit_next_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t credit_self_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t endpoint1_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t endpoint2_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t Bt = get_compile_time_arg_val(12);
    constexpr uint32_t row_tiles = Bt * qWt;
    constexpr uint32_t val_tiles = Bt * vWt;
    constexpr auto query_args = TensorAccessorArgs<13>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto acc_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto stats_args = TensorAccessorArgs<acc_args.next_compile_time_args_offset()>();

    // The packet: the compute kernel's inputs, two slots deep. The state is
    // read by the compute kernel through the seed views and packed, in place,
    // through the out views.
    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_max_seed = tt::CBIndex::c_13;
    constexpr uint32_t cb_max_plain = tt::CBIndex::c_25;  // the same memory, the compute kernel's FPU view
    constexpr uint32_t cb_sum_seed = tt::CBIndex::c_14;
    constexpr uint32_t cb_sum_plain = tt::CBIndex::c_26;  // the same memory, the compute kernel's matmul view
    constexpr uint32_t cb_out_seed = tt::CBIndex::c_15;
    constexpr uint32_t cb_max_out = tt::CBIndex::c_18;
    constexpr uint32_t cb_sum_out = tt::CBIndex::c_19;
    constexpr uint32_t cb_out_out = tt::CBIndex::c_17;
    // The resident column.
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
    constexpr uint32_t cb_slot_release = tt::CBIndex::c_7;
    constexpr uint32_t cb_scratch = tt::CBIndex::c_24;

    using namespace ttml::metal::ops::cyclic_sdpa_bw;
    constexpr CyclicSchedule sched(kCores, kMaskMode);
    constexpr uint32_t kTimesteps = sched.num_timesteps();
    const auto neighbors = snake_neighbors(kCores, my_core);

    const uint32_t tile_bytes = get_tile_size(cb_query);
    const uint32_t fp32_bytes = get_tile_size(cb_out_seed);

    const auto query = TensorAccessor(query_args, query_addr, tile_bytes);
    const auto key = TensorAccessor(key_args, key_addr, tile_bytes);
    const auto value = TensorAccessor(value_args, value_addr, tile_bytes);
    const auto state_acc = TensorAccessor(acc_args, state_acc_addr, fp32_bytes);
    const auto state_stats = TensorAccessor(stats_args, state_stats_addr, fp32_bytes);

    // Slot-0 addresses, the same on every core.
    const uint32_t base_query = get_write_ptr(cb_query);
    const uint32_t base_out = get_write_ptr(cb_out_seed);
    const uint32_t base_max = get_write_ptr(cb_max_seed);
    const uint32_t base_sum = get_write_ptr(cb_sum_seed);
    const uint32_t stride_query = row_tiles * tile_bytes;
    const uint32_t stride_out = row_tiles * fp32_bytes;
    const uint32_t stride_stat = Bt * fp32_bytes;
    uint32_t row_base = 0;
    uint32_t stat_base = 0;
    uint32_t col_row_base = 0;
    uint32_t col_val_base = 0;

    volatile tt_l1_ptr uint32_t* ready_imm_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_imm_sem_id[0])),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_imm_sem_id[1]))};
    volatile tt_l1_ptr uint32_t* ready_st_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_st_sem_id[0])),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_st_sem_id[1]))};
    const uint32_t scratch_l1 = get_write_ptr(cb_scratch);
    volatile tt_l1_ptr uint32_t* credit_from_prev =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_prev_sem_id));
    volatile tt_l1_ptr uint32_t* credit_from_next =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_next_sem_id));
    volatile tt_l1_ptr uint32_t* credit_from_self =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_self_sem_id));
    // The endpoint words: an endpoint writes only its own, and a consumer
    // that needs the value reads it remotely (see the backward relay).
    volatile tt_l1_ptr uint32_t* endpoint_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(endpoint1_sem_id)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(endpoint2_sem_id))};
    const uint32_t endpoint_sem_ids[2] = {endpoint1_sem_id, endpoint2_sem_id};
    const uint32_t endpoint_read_l1 = scratch_l1 + 16u;
    volatile tt_l1_ptr uint32_t* endpoint_read = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(endpoint_read_l1);

    uint32_t sent_to_prev = 0;
    uint32_t sent_to_next = 0;
    uint32_t sent_to_self = 0;

    const auto noc_xy_of = [&](uint32_t core, uint32_t& x, uint32_t& y) {
        if (core == neighbors.prev) {
            x = prev_noc_x;
            y = prev_noc_y;
        } else {
            x = next_noc_x;
            y = next_noc_y;
        }
    };
    const auto grant_credit_to = [&](uint32_t p) {
        if (p == my_core) {
            noc_semaphore_inc(get_noc_addr(get_semaphore(credit_self_sem_id)), 1u);
            return;
        }
        uint32_t x = 0;
        uint32_t y = 0;
        noc_xy_of(p, x, y);
        const uint32_t sem_id = (p == neighbors.prev) ? credit_next_sem_id : credit_prev_sem_id;
        noc_semaphore_inc(get_noc_addr(x, y, get_semaphore(sem_id)), 1u);
    };
    const auto await_credit = [&](uint32_t r) {
        WAYPOINT("CRDW");
        if (r == my_core) {
            const uint32_t want = ++sent_to_self;
            do {
                invalidate_l1_cache();
            } while ((*credit_from_self) < want);
        } else if (r == neighbors.prev) {
            const uint32_t want = ++sent_to_prev;
            do {
                invalidate_l1_cache();
            } while ((*credit_from_prev) < want);
        } else {
            const uint32_t want = ++sent_to_next;
            do {
                invalidate_l1_cache();
            } while ((*credit_from_next) < want);
        }
        WAYPOINT("CRDD");
    };

    // The state's DRAM pages for row block i: O^T tiles in the accumulator
    // scratch (the query's layout), m and l as two tiles per row tile in the
    // statistics scratch.
    const auto acc_page = [&](uint32_t i, uint32_t k) { return row_base + (i - 1u) * row_tiles + k; };
    const auto stat_page = [&](uint32_t i, uint32_t a, uint32_t which) {
        return (stat_base + (i - 1u) * Bt + a) * 2u + which;
    };

    for (uint32_t s = 0; s < slice_count; ++s) {
    const uint32_t sl = first_slice + s * slice_stride;
    const uint32_t idx = sl % heads;
    const uint32_t pair = sl / heads;
    const uint32_t bg = idx % kv_slices;
    const uint32_t sub = idx / kv_slices;
    const uint32_t bh = (bg / kv_heads) * q_heads + (bg % kv_heads) * heads_per_group + sub;
    const uint32_t row_chunk = get_arg_val<uint32_t>(pair_table_arg + 2u * pair);
    const uint32_t col_chunk = get_arg_val<uint32_t>(pair_table_arg + 2u * pair + 1u);
    row_base = (bh * chunks + row_chunk) * 2u * kCores * row_tiles;
    stat_base = (bh * chunks + row_chunk) * 2u * kCores * Bt;
    col_row_base = (bg * chunks + col_chunk) * 2u * kCores * row_tiles;
    col_val_base = (bg * chunks + col_chunk) * 2u * kCores * val_tiles;

    // Initial permissions for destination timesteps 0 and 1 (see the backward relay).
    for (uint32_t u = 0; u < 2u && u < kTimesteps; ++u) {
        const auto initial = sched.producer(my_core, u);
        if (initial.internal && initial.core != my_core) {
            grant_credit_to(initial.core);
        }
    }
    noc_async_atomic_barrier();

    // Q for timestep t: reserve its slot (unless the prefetch did), take it
    // from the previous consumer or from DRAM, hand it to the compute kernel.
    // Called for t + 1 as soon as the compute kernel has finished t, before
    // this timestep's state is forwarded: the next Q has been in L1 since
    // the predecessor forwarded it a timestep ago, and the compute kernel's
    // scores and block maxima need nothing else, so the state's forward and
    // the relay's latency run under them instead of ahead of them.
    const auto receive_query = [&](uint32_t t, uint32_t g, bool prefetched) {
        const uint32_t slot = g % 2u;
        if (!prefetched) {
            cb_reserve_back(cb_query, row_tiles);
        }
        const auto producer = sched.producer(my_core, t);
        if (producer.internal) {
            DeviceZoneScopedN("RECV-IMM");
            WAYPOINT("RDYW");
            do {
                invalidate_l1_cache();
            } while ((*ready_imm_sem[slot]) < g + 1u);
            WAYPOINT("RDYD");
        } else {
            DeviceZoneScopedN("LOAD-IMM-DRAM");
            if (!prefetched) {
                const uint32_t qs = base_query + slot * stride_query;
                const uint32_t i = sched.pair(my_core, t).i;
                for (uint32_t k = 0; k < row_tiles; ++k) {
                    noc_async_read_page(row_base + (i - 1u) * row_tiles + k, query, qs + k * tile_bytes);
                }
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_query, row_tiles);
    };

    bool prefetched_next = false;
    bool query_pushed_next = false;
    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const bool prefetched = prefetched_next;
        prefetched_next = false;
        const bool query_pushed = query_pushed_next;
        query_pushed_next = false;
        DeviceZoneScopedN("RELAY-READER-STEP");
        const auto pair_t = sched.pair(my_core, t);
        const uint32_t i = pair_t.i;
        const uint32_t j = pair_t.j;
        const uint32_t g = s * kTimesteps + t;
        const uint32_t slot = g % 2u;

        // ---- column state: K_j and V_j once per residency interval. The
        // reserve waits for the compute kernel to have popped the previous column.
        const bool column_changed = (t == 0u) || (sched.pair(my_core, t - 1u).j != j);
        if (column_changed) {
            DeviceZoneScopedN("LOAD-COL");
            read_tiles_by_row(cb_key, key, col_row_base + (j - 1u) * row_tiles, row_tiles, tile_bytes, row_tiles);
            read_tiles_by_row(cb_value, value, col_val_base + (j - 1u) * val_tiles, val_tiles, tile_bytes, val_tiles);
        }

        // ---- reserve this timestep's slot in every packet buffer.
        {
            DeviceZoneScopedN("SLOT-RESERVE");
            cb_reserve_back(cb_max_seed, Bt);
            cb_reserve_back(cb_max_plain, Bt);
            cb_reserve_back(cb_sum_seed, Bt);
            cb_reserve_back(cb_sum_plain, Bt);
            cb_reserve_back(cb_out_seed, row_tiles);
        }
        const uint32_t qs = base_query + slot * stride_query;
        const uint32_t os = base_out + slot * stride_out;
        const uint32_t ms = base_max + slot * stride_stat;
        const uint32_t ls = base_sum + slot * stride_stat;

        // ---- Q: from the previous consumer, or from DRAM at a streak start
        // (already done at the end of the previous timestep, except for the
        // first timestep of a slice).
        const auto producer = sched.producer(my_core, t);
        if (!query_pushed) {
            receive_query(t, g, prefetched);
        }

        // ---- forward Q straight away.
        const uint32_t receiver = sched.next_consumer(i, t);
        const uint32_t u = g + 1u;
        const uint32_t dst = u % 2u;
        const uint32_t dst_query = base_query + dst * stride_query;
        const uint32_t dst_out = base_out + dst * stride_out;
        const uint32_t dst_max = base_max + dst * stride_stat;
        const uint32_t dst_sum = base_sum + dst * stride_stat;
        uint32_t receiver_x = 0;
        uint32_t receiver_y = 0;
        if (receiver != kNoCore && receiver != my_core) {
            noc_xy_of(receiver, receiver_x, receiver_y);
        }
        {
            DeviceZoneScopedN("SEND-IMM");
            if (receiver == my_core) {
                cb_reserve_back(cb_query, row_tiles);
                cb_reserve_back(cb_max_seed, Bt);
                cb_reserve_back(cb_max_plain, Bt);
                cb_reserve_back(cb_sum_seed, Bt);
                cb_reserve_back(cb_sum_plain, Bt);
                cb_reserve_back(cb_out_seed, row_tiles);
                noc_async_write(qs, get_noc_addr(dst_query), stride_query);
                noc_async_write_barrier();
                noc_semaphore_set(ready_imm_sem[dst], u + 1u);
            } else if (receiver != kNoCore) {
                await_credit(receiver);
                noc_async_write(qs, get_noc_addr(receiver_x, receiver_y, dst_query), stride_query);
                noc_async_write_barrier();
                noc_inline_dw_write(
                    get_noc_addr(receiver_x, receiver_y, get_semaphore(ready_imm_sem_id[dst])), u + 1u);
            }
        }

        // ---- prefetch the next timestep's Q if it comes from DRAM.
        if (t + 1u < kTimesteps && !sched.producer(my_core, t + 1u).internal) {
            DeviceZoneScopedN("PREFETCH-IMM");
            const uint32_t ni = sched.pair(my_core, t + 1u).i;
            const uint32_t nslot = (g + 1u) % 2u;
            cb_reserve_back(cb_query, row_tiles);
            const uint32_t nqs = base_query + nslot * stride_query;
            for (uint32_t k = 0; k < row_tiles; ++k) {
                noc_async_read_page(row_base + (ni - 1u) * row_tiles + k, query, nqs + k * tile_bytes);
            }
            noc_async_read_barrier();
            prefetched_next = true;
        }

        // ---- the state: forwarded, reloaded, or nothing at a first streak.
        if (producer.internal) {
            DeviceZoneScopedN("RECV-STATE");
            WAYPOINT("STRW");
            do {
                invalidate_l1_cache();
            } while ((*ready_st_sem[slot]) < g + 1u);
            WAYPOINT("STRD");
        } else if (sched.is_later_streak_start(i, t)) {
            // Order the reload after the preceding streak's spill.
            const uint32_t e = sched.spill_endpoint(i, t);
            const uint32_t want = s * kTimesteps + sched.endpoint_threshold(i, t);
            const uint32_t endpoint_x = get_arg_val<uint32_t>(core_coords_arg + 2u * (e - 1u));
            const uint32_t endpoint_y = get_arg_val<uint32_t>(core_coords_arg + 2u * (e - 1u) + 1u);
            const uint64_t endpoint_addr =
                get_noc_addr(endpoint_x, endpoint_y, get_semaphore(endpoint_sem_ids[e - 1u]));
            WAYPOINT("ENDW");
            if (e == my_core) {
                do {
                    invalidate_l1_cache();
                } while ((*endpoint_sem[e - 1u]) < want);
            } else {
                do {
                    noc_async_read(endpoint_addr, endpoint_read_l1, sizeof(uint32_t));
                    noc_async_read_barrier();
                    invalidate_l1_cache();
                } while ((*endpoint_read) < want);
            }
            WAYPOINT("ENDD");
            DeviceZoneScopedN("LOAD-STATE-DRAM");
            for (uint32_t k = 0; k < row_tiles; ++k) {
                noc_async_read_page(acc_page(i, k), state_acc, os + k * fp32_bytes);
            }
            for (uint32_t a = 0; a < Bt; ++a) {
                noc_async_read_page(stat_page(i, a, 0u), state_stats, ms + a * fp32_bytes);
                noc_async_read_page(stat_page(i, a, 1u), state_stats, ls + a * fp32_bytes);
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_max_seed, Bt);
        cb_push_back(cb_max_plain, Bt);
        cb_push_back(cb_sum_seed, Bt);
        cb_push_back(cb_sum_plain, Bt);
        cb_push_back(cb_out_seed, row_tiles);

        // ---- this core's update closes the packet. The compute kernel hands
        // a forwarded row's state over one query tile at a time (a finished
        // row's whole, at the end), and each tile's forward is issued as it
        // arrives, so the bulk of the state is in flight under the compute
        // kernel's last columns and only the last tile's write and the
        // completion follow the timestep.
        const bool forwarded = receiver != kNoCore;
        const bool spilled = !forwarded && sched.has_later_active(i, t);
        {
            DeviceZoneScopedN("SEND-STATE");
            // Per query tile when the compute kernel hands the state over that
            // way (d <= 64, see the compute kernel's kStatePerTile), else the
            // whole state in one round.
            constexpr bool per_tile = (qWt <= 2u);
            constexpr uint32_t rounds = per_tile ? Bt : 1u;
            constexpr uint32_t tiles_per_round = per_tile ? 1u : Bt;
            const uint32_t out_round_bytes = tiles_per_round * qWt * fp32_bytes;
            const uint32_t stat_round_bytes = tiles_per_round * fp32_bytes;
            for (uint32_t r = 0; r < rounds; ++r) {
                {
                    DeviceZoneScopedN("WAIT-COMPUTE");
                    cb_wait_front(cb_out_out, (r + 1u) * tiles_per_round * qWt);
                    cb_wait_front(cb_max_out, (r + 1u) * tiles_per_round);
                    cb_wait_front(cb_sum_out, (r + 1u) * tiles_per_round);
                }
                const uint32_t o_off = r * out_round_bytes;
                const uint32_t s_off = r * stat_round_bytes;
                if (receiver == my_core) {
                    noc_async_write(os + o_off, get_noc_addr(dst_out + o_off), out_round_bytes);
                    noc_async_write(ms + s_off, get_noc_addr(dst_max + s_off), stat_round_bytes);
                    noc_async_write(ls + s_off, get_noc_addr(dst_sum + s_off), stat_round_bytes);
                } else if (forwarded) {
                    noc_async_write(os + o_off, get_noc_addr(receiver_x, receiver_y, dst_out + o_off), out_round_bytes);
                    noc_async_write(ms + s_off, get_noc_addr(receiver_x, receiver_y, dst_max + s_off), stat_round_bytes);
                    noc_async_write(ls + s_off, get_noc_addr(receiver_x, receiver_y, dst_sum + s_off), stat_round_bytes);
                } else if (spilled) {
                    // Streak end with a later streak: spill the state, then
                    // (below) complete the write and certify it for the later
                    // streak's reload.
                    for (uint32_t k = 0; k < tiles_per_round * qWt; ++k) {
                        noc_async_write_page(acc_page(i, r * tiles_per_round * qWt + k), state_acc, os + o_off + k * fp32_bytes);
                    }
                    for (uint32_t a = 0; a < tiles_per_round; ++a) {
                        noc_async_write_page(stat_page(i, r * tiles_per_round + a, 0u), state_stats, ms + s_off + a * fp32_bytes);
                        noc_async_write_page(stat_page(i, r * tiles_per_round + a, 1u), state_stats, ls + s_off + a * fp32_bytes);
                    }
                }
                // Else the row's last visit: the compute kernel finished it and
                // the write kernel stores it.
            }
            // ---- the next timestep's Q to the compute kernel before this
            // state's completion (see receive_query); its slot was popped at
            // the end of t - 1.
            if (t + 1u < kTimesteps) {
                receive_query(t + 1u, g + 1u, prefetched_next);
                query_pushed_next = true;
            }
            if (receiver == my_core) {
                noc_async_write_barrier();
                noc_semaphore_set(ready_st_sem[dst], u + 1u);
            } else if (forwarded) {
                noc_async_write_barrier();
                noc_inline_dw_write(
                    get_noc_addr(receiver_x, receiver_y, get_semaphore(ready_st_sem_id[dst])), u + 1u);
            } else if (spilled) {
                noc_async_write_barrier();
                noc_semaphore_set(endpoint_sem[my_core - 1u], g + 1u);
            }
        }
        cb_pop_front(cb_out_out, row_tiles);
        cb_pop_front(cb_max_out, Bt);
        cb_pop_front(cb_sum_out, Bt);

        // ---- release: the compute kernel has popped this slot.
        {
            DeviceZoneScopedN("RELEASE");
            cb_wait_front(cb_slot_release, 1);
            cb_pop_front(cb_slot_release, 1);
            const uint32_t released_for = t + 2u;
            if (released_for < kTimesteps) {
                const auto next_producer = sched.producer(my_core, released_for);
                if (next_producer.internal && next_producer.core != my_core) {
                    grant_credit_to(next_producer.core);
                }
            }
        }
    }
    }  // slices

    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
