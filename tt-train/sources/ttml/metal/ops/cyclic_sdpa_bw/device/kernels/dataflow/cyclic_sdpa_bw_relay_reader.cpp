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
    constexpr auto query_args = TensorAccessorArgs<9>();
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
    // A local word to publish readiness tags from.
    constexpr uint32_t cb_scratch = tt::CBIndex::c_24;

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

    for (uint32_t t = 0; t < kTimesteps; ++t) {
        const auto pair = sched.pair(my_core, t);
        const uint32_t i = pair.i;
        const uint32_t j = pair.j;
        const uint32_t slot = t % 2u;

        // Column state: still read from DRAM every timestep. Restoring the
        // paper's residency belongs with a later step.
        read_tiles_by_row(cb_key, key, (j - 1u) * qWt, qWt, tile_bytes, qWt);
        read_tiles_by_row(cb_value, value, (j - 1u) * vWt, vWt, tile_bytes, vWt);

        // The column gradients this core wrote at an earlier timestep. Its own
        // write kernel produced them, so the release of t - 1 is what says
        // they have landed.
        if (t > 0u) {
            WAYPOINT("BARW");
            do {
                invalidate_l1_cache();
            } while ((*release_sem) < t);
            WAYPOINT("BARD");
        }
        read_tiles_by_row(cb_grad_key_seed, grad_key, (j - 1u) * qWt, qWt, grad_bytes, qWt);
        read_tiles_by_row(cb_grad_value_seed, grad_value, (j - 1u) * vWt, vWt, grad_bytes, vWt);

        // Reserve this timestep's slot in every packet buffer, which is the
        // release of t - 2 becoming a credit for whoever fills it.
        cb_reserve_back(cb_query, qWt);
        cb_reserve_back(cb_grad_output, vWt);
        cb_reserve_back(cb_lse, 1);
        cb_reserve_back(cb_u_scalar, 1);
        cb_reserve_back(cb_grad_query_seed, qWt);

        const auto producer = sched.producer(my_core, t);
        if (producer.internal) {
            // Inside a streak: the packet arrives over the NoC, or locally for
            // the one self-transition. The reserve above freed this slot, so
            // credit its producer and then wait for the readiness tag of this
            // exact destination timestep; a stale tag cannot satisfy it.
            if (producer.core != my_core) {
                grant_credit_to(producer.core);
            }
            WAYPOINT("RDYW");
            do {
                invalidate_l1_cache();
            } while ((*ready_sem[slot]) < t + 1u);
            WAYPOINT("RDYD");
        } else {
            // A streak start: load the packet from DRAM. dQ_i must carry every
            // earlier update, which the release of t - 1 above certifies.
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
        }

        cb_pop_front(cb_grad_query_out, qWt);
    }
}
