// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The row-packet relay: receive a packet, hand it to the compute kernel, and
// forward it to the next consumer. Algorithm 3 of main.tex, barrier variant.
//
// The paper's two receive slots are the compute kernel's input buffers with
// room for two packets. That is not a coincidence to be worked around but the
// thing that makes this simple: the circular-buffer protocol already is the
// credit-and-readiness protocol, one core wide, and the multicast receivers
// elsewhere in tt-train use it the same way. A receiver reserves its slot,
// credits the producer, waits for the payload, and pushes; a producer waits
// for the credit, writes the payload, completes it, and publishes readiness.
//
// The packet for destination timestep u occupies slot u mod 2, which is also
// where the receiver's write pointer stands after u pushes, so a producer can
// compute the destination address without tracking the receiver: every core
// has the same buffer layout, so its own base address is the receiver's too.
//
// This kernel both receives and forwards, because forwarding needs the
// packet's immutable fields -- which live in its slots -- and the updated
// dQ, which the compute kernel produces. Splitting those across two RISCs
// would need a handoff of its own (O10 in overlaps.md). The write kernel is
// left with the column gradients and the barrier.
//
// The credit is granted where the multicast receivers elsewhere in tt-train
// grant it: immediately after the reserve that frees the slot, at the start
// of the destination timestep. The producer, still finishing the previous
// timestep, then wakes and sends. That looks late -- the packet arrives just
// in time rather than early -- but it cannot deadlock, because a receiver
// reaching its reserve never needs anything from the producer's *next*
// timestep: its column state comes from DRAM and its barrier release depends
// on the producer's write kernel, which does not wait on the relay.
//
// Granting a timestep earlier, at the release, is what the paper does and
// what the transport probe does. It is not available here: the uses of a slot
// are spread over two RISCs, and the compute kernel reads Q again for dK
// *after* it has published the updated dQ, so nothing this kernel can observe
// says the slot is free any sooner.
//
// The self-transition is the exception, and it has to be. There the producer
// and the receiver are the same core, so a credit it grants itself at the
// start of timestep t + 1 would be waited for during timestep t -- the same
// iteration, in the wrong order, which deadlocks at C = 1 immediately. It
// needs no credit: it reserves the destination slot itself, which is the same
// wait expressed locally.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/cyclic_schedule.hpp"
#include "tt-train/sources/ttml/metal/ops/cyclic_sdpa_bw/device/parity_snake.hpp"

// ENDPOINT_SYNC selects Algorithm 4: no chip-wide barrier. Within a streak
// the packet is itself the ordering token -- a consumer cannot update dQ_i
// before the previous consumer forwards it. Across a gap, two endpoint
// counters order a reload after the preceding streak's spill: every streak
// followed by a later streak ends on core 1 or core 2, so those two cores
// publish a monotone progress value after an inter-streak spill and a
// consumer waits for it.
//
// The threshold is t_prev + 1, never t. The row is inactive at t - 1 at every
// later streak start and progress is published only for spill events, so
// waiting for t waits for a publication that never comes.
//
// Publication is a set of ordered unicast writes rather than a multicast,
// which the paper explicitly allows. A multicast issued from this RISC hung
// in its write barrier: this kernel runs on the data-movement RISC whose
// default NoC is 1, the two NoCs have mirrored coordinates, and retargeting
// the multicast would mean recomputing the rectangle in the other NoC's
// frame. Unicast writes need none of that and use the same primitive the
// readiness tags already use here. They also satisfy the ordering rule the
// paper asks for, being issued in increasing value order on one NoC from one
// thread.
#ifndef ENDPOINT_SYNC
#define ENDPOINT_SYNC 0
#endif

void kernel_main() {
    uint32_t arg = 0;
    const uint32_t my_core = get_arg_val<uint32_t>(arg++);
    const uint32_t query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t value_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_output_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t lse_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t u_scalar_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_query_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_key_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t grad_value_addr = get_arg_val<uint32_t>(arg++);
    // Snake neighbours, the only cores a packet ever moves between.
    const uint32_t prev_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t prev_noc_y = get_arg_val<uint32_t>(arg++);
    const uint32_t next_noc_x = get_arg_val<uint32_t>(arg++);
    const uint32_t next_noc_y = get_arg_val<uint32_t>(arg++);
    // Every participating core's coordinates, for publishing endpoint
    // progress: (x, y) per core, cores 1..C in order.
    const uint32_t core_coords_arg = arg;

    constexpr uint32_t kCores = get_compile_time_arg_val(0);
    constexpr uint32_t qWt = get_compile_time_arg_val(1);
    constexpr uint32_t vWt = get_compile_time_arg_val(2);
    constexpr uint32_t release_sem_id = get_compile_time_arg_val(3);
    // One readiness word per slot, carrying the destination timestep's tag
    // rather than a count: a slot's uses are u, u + 2, ... so the tags
    // u + 1, u + 3, ... increase, and a stale tag from the slot's previous
    // use cannot satisfy the wait.
    constexpr uint32_t ready_sem_id[2] = {
        get_compile_time_arg_val(4), get_compile_time_arg_val(5)};
    constexpr uint32_t credit_prev_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t credit_next_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t credit_self_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t endpoint1_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t endpoint2_sem_id = get_compile_time_arg_val(10);
    constexpr auto query_args = TensorAccessorArgs<11>();
    constexpr auto key_args = TensorAccessorArgs<query_args.next_compile_time_args_offset()>();
    constexpr auto value_args = TensorAccessorArgs<key_args.next_compile_time_args_offset()>();
    constexpr auto grad_output_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
    constexpr auto lse_args = TensorAccessorArgs<grad_output_args.next_compile_time_args_offset()>();
    constexpr auto u_args = TensorAccessorArgs<lse_args.next_compile_time_args_offset()>();
    constexpr auto grad_query_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto grad_key_args = TensorAccessorArgs<grad_query_args.next_compile_time_args_offset()>();
    constexpr auto grad_value_args = TensorAccessorArgs<grad_key_args.next_compile_time_args_offset()>();

    // The packet: the compute kernel's inputs, two slots deep.
    constexpr uint32_t cb_query = tt::CBIndex::c_0;
    constexpr uint32_t cb_grad_output = tt::CBIndex::c_3;
    constexpr uint32_t cb_lse = tt::CBIndex::c_4;
    constexpr uint32_t cb_u_scalar = tt::CBIndex::c_5;
    constexpr uint32_t cb_grad_query_seed = tt::CBIndex::c_15;
    // Column state, still per timestep from DRAM, gradients included.
    constexpr uint32_t cb_key = tt::CBIndex::c_1;
    constexpr uint32_t cb_value = tt::CBIndex::c_2;
    constexpr uint32_t cb_grad_key_seed = tt::CBIndex::c_18;
    constexpr uint32_t cb_grad_value_seed = tt::CBIndex::c_21;
    // The compute kernel's updated dQ, which travels on with the packet.
    constexpr uint32_t cb_grad_query_out = tt::CBIndex::c_17;
    // The compute kernel's release token: one page per timestep, published
    // once it has popped that timestep's packet slot.
    constexpr uint32_t cb_slot_release = tt::CBIndex::c_7;
    // A local word to publish readiness tags and endpoint progress from.
    constexpr uint32_t cb_scratch = tt::CBIndex::c_24;
    constexpr uint32_t cb_column_progress = tt::CBIndex::c_26;

    using namespace ttml::metal::ops::cyclic_sdpa_bw;
    constexpr CyclicSchedule sched(kCores);
    constexpr uint32_t kTimesteps = 2u * kCores + 1u;
    const auto neighbors = snake_neighbors(kCores, my_core);

    const uint32_t tile_bytes = get_tile_size(cb_query);
    const uint32_t interm_bytes = get_tile_size(cb_lse);
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

    // Buffer bases, captured before any reserve so they are the slot-0
    // addresses -- and, because every core has the same layout, also the
    // receiver's.
    const uint32_t base_query = get_write_ptr(cb_query);
    const uint32_t base_grad_output = get_write_ptr(cb_grad_output);
    const uint32_t base_lse = get_write_ptr(cb_lse);
    const uint32_t base_u_scalar = get_write_ptr(cb_u_scalar);
    const uint32_t base_grad_query = get_write_ptr(cb_grad_query_seed);
    const uint32_t stride_query = qWt * tile_bytes;
    const uint32_t stride_grad_output = vWt * tile_bytes;
    const uint32_t stride_interm = interm_bytes;
    const uint32_t stride_grad_query = qWt * grad_bytes;

    volatile tt_l1_ptr uint32_t* release_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_sem_id));
    volatile tt_l1_ptr uint32_t* ready_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id[0])),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem_id[1]))};
    const uint32_t scratch_l1 = get_write_ptr(cb_scratch);
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
    volatile tt_l1_ptr uint32_t* credit_from_prev =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_prev_sem_id));
    volatile tt_l1_ptr uint32_t* credit_from_next =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_next_sem_id));
    volatile tt_l1_ptr uint32_t* credit_from_self =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_self_sem_id));

#if ENDPOINT_SYNC
    // Every core holds a local copy of both endpoint counters, so a consumer
    // polls its own L1 rather than a remote word.
    volatile tt_l1_ptr uint32_t* endpoint_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(endpoint1_sem_id)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(endpoint2_sem_id))};
    const uint32_t endpoint_sem_ids[2] = {endpoint1_sem_id, endpoint2_sem_id};
    // This core's write kernel says when the column gradients have landed.
    volatile tt_l1_ptr uint32_t* column_progress =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_column_progress));
#endif

    // Which columns this core owns, and whether each has been resident.
    const auto owned = sched.owned_columns(my_core);
    const uint32_t owned_column[2] = {owned.first, owned.second};
    bool visited[2] = {false, false};

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

    // A slot release grants permission to whoever produces for it next. Seen
    // from that producer, this core is its snake successor when it is this
    // core's predecessor, and the other way round.
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

    // Initial permissions, for destination timesteps 0 and 1: there is no
    // release two timesteps before those, so the receivers grant them up
    // front. Increments are atomic and order-independent, so this needs no
    // synchronisation of its own.
    for (uint32_t u = 0; u < 2u && u < kTimesteps; ++u) {
        const auto initial = sched.producer(my_core, u);
        if (initial.internal && initial.core != my_core) {
            grant_credit_to(initial.core);
        }
    }
    noc_async_atomic_barrier();

    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);
        const uint32_t i = pair.i;
        const uint32_t j = pair.j;
        const uint32_t slot = t % 2u;

        // Column state, resident for a whole residency interval: K_j and V_j
        // are read only when the column changes, which the schedule says
        // happens exactly twice per core over T + 1 timesteps, always at a
        // diagonal block. The reserve waits for the compute kernel to have
        // popped the previous column, which is the paper's rule that all
        // operations using that storage complete before it is reused.
        const bool column_changed = (t == 0u) || (sched.pair(my_core, t - 1u).j != j);
        if (column_changed) {
            read_tiles_by_row(cb_key, key, (j - 1u) * qWt, qWt, tile_bytes, qWt);
            read_tiles_by_row(cb_value, value, (j - 1u) * vWt, vWt, tile_bytes, vWt);
        }

        // Accumulated column gradients are needed only where an interval
        // starts on a column this core has been resident on before -- one
        // revisit per core, its third interval. They were stored by this
        // core's own write kernel at the end of the earlier interval, so the
        // wait is on that kernel's progress, not on anything chip-wide.
        if (column_changed) {
            const uint32_t owned_slot = (j == owned_column[0]) ? 0u : 1u;
            if (visited[owned_slot]) {
                WAYPOINT("COLW");
#if ENDPOINT_SYNC
                do {
                    invalidate_l1_cache();
                } while ((*column_progress) < t);
#else
                do {
                    invalidate_l1_cache();
                } while ((*release_sem) < t);
#endif
                WAYPOINT("COLD");
                read_tiles_by_row(cb_grad_key_seed, grad_key, (j - 1u) * qWt, qWt, grad_bytes, qWt);
                read_tiles_by_row(
                    cb_grad_value_seed, grad_value, (j - 1u) * vWt, vWt, grad_bytes, vWt);
            }
            visited[owned_slot] = true;
        }
        // Reserve this timestep's slot in every packet buffer, which is the
        // release of t - 2 becoming a credit for whoever fills it.
        cb_reserve_back(cb_query, qWt);
        cb_reserve_back(cb_grad_output, vWt);
        cb_reserve_back(cb_lse, 1);
        cb_reserve_back(cb_u_scalar, 1);
        cb_reserve_back(cb_grad_query_seed, qWt);

        const auto producer = sched.producer(my_core, t);
        if (producer.internal) {
            // Inside a streak: the packet arrives over the NoC, or locally
            // for the one self-transition. The credit for this slot was
            // granted at its release, two timesteps back, so the producer's
            // forward did not have to wait for this core to get here.
            WAYPOINT("RDYW");
            do {
                invalidate_l1_cache();
            } while ((*ready_sem[slot]) < t + 1u);
            WAYPOINT("RDYD");
        } else {
            // A streak start: load the packet from DRAM. dQ_i must carry
            // every earlier update.
#if ENDPOINT_SYNC
            if (sched.is_later_streak_start(i, t)) {
                // Order this reload after the preceding streak's spill. The
                // threshold certifies that actual spill, not the inactive
                // timestep t - 1.
                const uint32_t e = sched.spill_endpoint(i, t);
                const uint32_t want = sched.endpoint_threshold(i, t);
                WAYPOINT("ENDW");
                do {
                    invalidate_l1_cache();
                } while ((*endpoint_sem[e - 1u]) < want);
                WAYPOINT("ENDD");
            }
#endif
            const uint32_t qs = base_query + slot * stride_query;
            const uint32_t os = base_grad_output + slot * stride_grad_output;
            const uint32_t ls = base_lse + slot * stride_interm;
            const uint32_t ds = base_u_scalar + slot * stride_interm;
            const uint32_t gs = base_grad_query + slot * stride_grad_query;
            for (uint32_t k = 0; k < qWt; ++k) {
                noc_async_read_page((i - 1u) * qWt + k, query, qs + k * tile_bytes);
                noc_async_read_page((i - 1u) * qWt + k, grad_query, gs + k * grad_bytes);
            }
            for (uint32_t k = 0; k < vWt; ++k) {
                noc_async_read_page((i - 1u) * vWt + k, grad_output, os + k * tile_bytes);
            }
            noc_async_read_page(i - 1u, lse, ls);
            noc_async_read_page(i - 1u, u_scalar, ds);
            noc_async_read_barrier();
        }

        cb_push_back(cb_query, qWt);
        cb_push_back(cb_grad_output, vWt);
        cb_push_back(cb_lse, 1);
        cb_push_back(cb_u_scalar, 1);
        cb_push_back(cb_grad_query_seed, qWt);
        // The compute kernel's updated dQ closes the packet.
        cb_wait_front(cb_grad_query_out, qWt);
        const uint32_t dq_out = get_read_ptr(cb_grad_query_out);

        const uint32_t receiver = sched.next_consumer(i, t);
        if (receiver != kNoCore) {
            const uint32_t u = t + 1u;
            const uint32_t dst = u % 2u;

            const uint32_t qs = base_query + slot * stride_query;
            const uint32_t os = base_grad_output + slot * stride_grad_output;
            const uint32_t ls = base_lse + slot * stride_interm;
            const uint32_t ds = base_u_scalar + slot * stride_interm;
            const uint32_t dq_dst = base_grad_query + dst * stride_grad_query;

            if (receiver == my_core) {
                // The self-transition: reserving the destination slot is the
                // credit, and the copy is local.
                cb_reserve_back(cb_query, qWt);
                cb_reserve_back(cb_grad_output, vWt);
                cb_reserve_back(cb_lse, 1);
                cb_reserve_back(cb_u_scalar, 1);
                cb_reserve_back(cb_grad_query_seed, qWt);
                noc_async_write(qs, get_noc_addr(base_query + dst * stride_query), stride_query);
                noc_async_write(
                    os, get_noc_addr(base_grad_output + dst * stride_grad_output), stride_grad_output);
                noc_async_write(ls, get_noc_addr(base_lse + dst * stride_interm), stride_interm);
                noc_async_write(ds, get_noc_addr(base_u_scalar + dst * stride_interm), stride_interm);
                noc_async_write(dq_out, get_noc_addr(dq_dst), stride_grad_query);
                noc_async_write_barrier();
                noc_semaphore_set(ready_sem[dst], u + 1u);
            } else {
                await_credit(receiver);
                uint32_t x = 0;
                uint32_t y = 0;
                noc_xy_of(receiver, x, y);
                // Five payload writes, completed before readiness is published.
                noc_async_write(qs, get_noc_addr(x, y, base_query + dst * stride_query), stride_query);
                noc_async_write(
                    os,
                    get_noc_addr(x, y, base_grad_output + dst * stride_grad_output),
                    stride_grad_output);
                noc_async_write(ls, get_noc_addr(x, y, base_lse + dst * stride_interm), stride_interm);
                noc_async_write(
                    ds, get_noc_addr(x, y, base_u_scalar + dst * stride_interm), stride_interm);
                noc_async_write(dq_out, get_noc_addr(x, y, dq_dst), stride_grad_query);
                noc_async_write_barrier();
                *scratch = u + 1u;
                noc_semaphore_set_remote(scratch_l1, get_noc_addr(x, y, get_semaphore(ready_sem_id[dst])));
                // The tag is a 4-byte write out of that word; complete it
                // before the word is reused.
                noc_async_write_barrier();
            }
        } else {
            // Streak end: spill dQ_i and complete the write, so the reload at
            // the next streak start observes it.
            for (uint32_t k = 0; k < qWt; ++k) {
                noc_async_write_page((i - 1u) * qWt + k, grad_query, dq_out + k * grad_bytes);
            }
            noc_async_write_barrier();
#if ENDPOINT_SYNC
            // An inter-streak spill certifies itself for the later streak's
            // reload. By the endpoint spill property this core is 1 or 2, and
            // the loopback multicast writes its own copy too -- which matters,
            // since a later streak of this row may well be consumed here.
            if (sched.has_later_active(i, t)) {
                const uint32_t sem_id = endpoint_sem_ids[my_core - 1u];
                *scratch = t + 1u;
                for (uint32_t r = 1; r <= kCores; ++r) {
                    if (r == my_core) {
                        // Its own copy, which matters: a later streak of this
                        // row may well be consumed here.
                        noc_semaphore_set(endpoint_sem[my_core - 1u], t + 1u);
                        continue;
                    }
                    const uint32_t x = get_arg_val<uint32_t>(core_coords_arg + 2u * (r - 1u));
                    const uint32_t y = get_arg_val<uint32_t>(core_coords_arg + 2u * (r - 1u) + 1u);
                    noc_semaphore_set_remote(scratch_l1, get_noc_addr(x, y, get_semaphore(sem_id)));
                }
                // The value is a 4-byte write out of that word: complete every
                // publication before the word is reused.
                noc_async_write_barrier();
            }
#endif
        }

        cb_pop_front(cb_grad_query_out, qWt);

        // The compute kernel has popped this timestep's slot, so it is free
        // for destination t + 2 and its producer can be told now rather than
        // when this core reaches t + 2. That is the paper's release timing,
        // and it is what lets a producer forward as soon as its payload is
        // ready instead of waiting for its receiver to arrive.
        cb_wait_front(cb_slot_release, 1);
        cb_pop_front(cb_slot_release, 1);
        const uint32_t released_for = t + 2u;
        if (released_for < kTimesteps) {
            const auto next_producer = sched.producer(my_core, released_for);
            if (next_producer.internal && next_producer.core != my_core) {
                grant_credit_to(next_producer.core);
            }
            // A self-transition needs no credit: it reserves the destination
            // slot itself. And a DRAM load needs none, the slot being its own.
        }
    }
}
