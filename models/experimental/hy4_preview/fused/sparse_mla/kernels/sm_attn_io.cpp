// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Sparse-MLA attention I/O (data movement, on every attention core; core (b, g) is head group g of S-block b).
//   * hands the per-user counts to the attention kernel (cb_ctrl);
//   * fetches the absorbed queries Q(u, g) of every user this S-block has chunks of, and the mask of a partial
//     last chunk;
//   * moves partials along the per-user merge tree: a non-root block sends (O, max/sum) into the receive slot
//     (u, step) of its partner and raises bit u of the partner's step semaphore; the receive slots are published
//     to the attention kernel in slot order, the ones no partial is due in as dummies;
//   * writes the root partial of user u = b to the output page (u, g): O tiles | max/sum tile | status. A user
//     with no valid row gets the empty partial (O = 0, sum = 0, max = EMPTY_MAX). Status flags: 1, 2 a wait of
//     this kernel expired; 4 one of the K receiver on this core; 16 and up one of the S-block's K sender.
// The three duties are polled in one loop, so none can block another.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#include "sm_dm.hpp"

constexpr uint16_t STATUS_MAGIC_BF16 = 0x4D4B;

// The BF16 encoding of a small unsigned integer (exact below 256).
inline uint16_t bf16_of(uint32_t n) {
    if (n == 0) {
        return 0;
    }
    uint32_t p = 0;
    while ((n >> (p + 1)) != 0) {
        ++p;
    }
    const uint32_t mantissa = p <= 7 ? (n << (7 - p)) : (n >> (p - 7));
    return static_cast<uint16_t>(((127 + p) << 7) | (mantissa & 0x7F));
}

void kernel_main() {
#ifdef SM_NOP
    return;
#endif
    constexpr uint32_t cb_q = get_named_compile_time_arg_val("cb_q");
    constexpr uint32_t cb_mask = get_named_compile_time_arg_val("cb_mask");
    constexpr uint32_t cb_ctrl = get_named_compile_time_arg_val("cb_ctrl_a");
    constexpr uint32_t cb_hdr = get_named_compile_time_arg_val("cb_hdr_io");
    constexpr uint32_t cb_ms_in = get_named_compile_time_arg_val("cb_ms_in");
    constexpr uint32_t cb_out_in = get_named_compile_time_arg_val("cb_out_in");
    constexpr uint32_t cb_send_o = get_named_compile_time_arg_val("cb_send_o");
    constexpr uint32_t cb_send_ms = get_named_compile_time_arg_val("cb_send_ms");
    constexpr uint32_t cb_final_o = get_named_compile_time_arg_val("cb_final_o");
    constexpr uint32_t cb_final_ms = get_named_compile_time_arg_val("cb_final_ms");
    constexpr uint32_t USERS = get_named_compile_time_arg_val("users");
    constexpr uint32_t GROUPS = get_named_compile_time_arg_val("groups");
    constexpr uint32_t HEAD_GROUPS = get_named_compile_time_arg_val("head_groups");
    constexpr uint32_t CHUNK = get_named_compile_time_arg_val("chunk_rows");
    constexpr uint32_t DHt = get_named_compile_time_arg_val("dht");
    constexpr uint32_t vDHt = get_named_compile_time_arg_val("vdht");
    constexpr uint32_t sem_merge[sm::MERGE_STEPS] = {
        get_named_compile_time_arg_val("sem_merge0"),
        get_named_compile_time_arg_val("sem_merge1"),
        get_named_compile_time_arg_val("sem_merge2")};
    constexpr uint32_t sem_err = get_named_compile_time_arg_val("sem_err");
    constexpr uint32_t EMPTY_MAX_BF16 = get_named_compile_time_arg_val("empty_max_bf16");
    constexpr uint32_t TBL = get_named_compile_time_arg_val("tbl_io");
    constexpr uint32_t SLOTS = USERS * sm::MERGE_STEPS;

    constexpr auto sel_args = TensorAccessorArgs<0>();
    constexpr auto q_args = TensorAccessorArgs<sel_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();

    const sm::Role role =
        sm::role_from_position<get_named_compile_time_arg_val("x0_left"), get_named_compile_time_arg_val("x0_right")>();
    const uint32_t block = role.block;
    const uint32_t group = role.slot;
    const uint32_t sel_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(2);

    const auto sel = TensorAccessor(sel_args, sel_addr);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    const uint32_t tile_bytes = get_tile_size(cb_q);  // one [8, 32] BF16 tile
    const uint32_t q_bytes = DHt * tile_bytes;
    const uint32_t o_bytes = vDHt * tile_bytes;
    const uint32_t ms_bytes = tile_bytes;

    uint32_t counts[sm::MAX_USERS];
    sm::read_counts<USERS>(sel, cb_hdr, counts);
    volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_hdr));

    cb_reserve_back(cb_ctrl, 1);
    {
        volatile tt_l1_ptr uint32_t* ctrl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctrl));
        for (uint32_t w = 0; w < sm::COUNT_WORDS; ++w) {  // two counts per word
            const uint32_t lo = 2 * w < USERS ? counts[2 * w] : 0;
            const uint32_t hi = 2 * w + 1 < USERS ? counts[2 * w + 1] : 0;
            ctrl[w] = lo | (hi << 16);
        }
    }
    cb_push_back(cb_ctrl, 1);

    // ---- queries and masks, one item per user with chunks here, users ascending ----
    {
        const uint32_t q_base = get_write_ptr(cb_q);
        uint32_t items = 0;
        for (uint32_t u = 0; u < USERS; ++u) {
            if (sm::chunks_here<GROUPS, CHUNK>(counts[u], block, u)) {
                noc_async_read(q.get_noc_addr(u * HEAD_GROUPS + group), q_base + items * q_bytes, q_bytes);
                ++items;
            }
        }
        noc_async_read_barrier();
        for (uint32_t i = 0; i < items; ++i) {
            cb_reserve_back(cb_q, DHt);
            cb_push_back(cb_q, DHt);
        }
        for (uint32_t u = 0; u < USERS; ++u) {
            const uint32_t here = sm::chunks_here<GROUPS, CHUNK>(counts[u], block, u);
            const uint32_t last = sm::first_chunk<GROUPS>(block, u) + (here - 1) * GROUPS;
            if (here && last == sm::ceil_div(counts[u], CHUNK) - 1 && counts[u] % CHUNK) {
                const uint32_t valid = counts[u] % CHUNK;
                cb_reserve_back(cb_mask, 1);
                volatile tt_l1_ptr uint16_t* mask =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_mask));
                for (uint32_t i = 0; i < CHUNK; ++i) {
                    mask[i] = i < valid ? 0 : 0xFF80;  // -inf
                }
                cb_push_back(cb_mask, 1);
            }
        }
    }

    // ---- merge tree and output ----
    uint32_t partner_x[GROUPS], partner_y[GROUPS];  // head group `group` of every S-block
    for (uint32_t b = 0; b < GROUPS; ++b) {
        const sm::Coords at = sm::block_coords(TBL, b);
        partner_x[b] = at.att_x + group % 4;
        partner_y[b] = at.att_y + group / 4;
    }
    volatile tt_l1_ptr uint32_t* sem[sm::MERGE_STEPS];
    for (uint32_t s = 0; s < sm::MERGE_STEPS; ++s) {
        sem[s] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_merge[s]);
    }
    // The attention kernel pushes the partials it sends in this order: first the users it does not receive for
    // (while it attends), then the ones it merges (afterwards), users ascending in both.
    uint32_t send_user[sm::MAX_USERS];
    uint32_t n_send = 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t u = 0; u < USERS; ++u) {
            const uint32_t r = sm::rel<GROUPS>(block, u);
            const uint32_t n_act = sm::active_blocks<GROUPS, CHUNK>(counts[u]);
            if (r != 0 && r < n_act && (sm::num_receives(r, n_act) != 0) == (pass == 1)) {
                send_user[n_send++] = u;
            }
        }
    }
    const uint32_t ms_in_base = get_write_ptr(cb_ms_in);
    const uint32_t out_in_base = get_write_ptr(cb_out_in);
    const bool has_root = block < USERS;
    const uint32_t out_page = block * HEAD_GROUPS + group;

    // No partial is due on this core for any user (the common case: at most one chunk per user): the receive
    // slots are not used, and the attention kernel, which decides the same way, does not wait for them.
    uint32_t recv_k = sm::any_receive<USERS, GROUPS, CHUNK>(counts, block) ? 0 : SLOTS;
    uint32_t send_i = 0, idle = 0, status = 0;
    bool final_done = !has_root;
    if (has_root && counts[block] == 0) {
        Noc noc;
        experimental::CB final_o(cb_final_o);
        noc.async_write_zeros(final_o, o_bytes, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint16_t* ms = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_final_ms));
        for (uint32_t i = 0; i < ms_bytes / 2; ++i) {
            ms[i] = (i % 16 == 0 && i < ms_bytes / 4) ? EMPTY_MAX_BF16 : 0;  // column 0 of every row; sum = 0
        }
        noc_async_write(get_write_ptr(cb_final_o), out.get_noc_addr(out_page), o_bytes);
        noc_async_write(get_write_ptr(cb_final_ms), out.get_noc_addr(out_page, o_bytes), ms_bytes);
        final_done = true;
    }
    while (recv_k < SLOTS || send_i < n_send || !final_done) {
        bool progress = false;
        if (recv_k < SLOTS) {
            const uint32_t u = recv_k / sm::MERGE_STEPS, s = recv_k % sm::MERGE_STEPS;
            const bool due = sm::receives(sm::rel<GROUPS>(block, u), sm::active_blocks<GROUPS, CHUNK>(counts[u]), s);
            invalidate_l1_cache();
            if (!due || ((*sem[s] >> u) & 1u) || (status & 1u)) {
                cb_reserve_back(cb_ms_in, 1);
                cb_push_back(cb_ms_in, 1);
                cb_reserve_back(cb_out_in, vDHt);
                cb_push_back(cb_out_in, vDHt);
                ++recv_k;
                progress = true;
            }
        }
        if (send_i < n_send && cb_pages_available_at_front(cb_send_ms, 1) &&
            cb_pages_available_at_front(cb_send_o, vDHt)) {
            const uint32_t u = send_user[send_i];
            const uint32_t r = sm::rel<GROUPS>(block, u);
            const uint32_t s = sm::send_step(r);
            const uint32_t pb = (u + r - (1u << s)) % GROUPS;
            const uint32_t slot = u * sm::MERGE_STEPS + s;
            noc_async_write(
                get_read_ptr(cb_send_ms),
                get_noc_addr(partner_x[pb], partner_y[pb], ms_in_base + slot * ms_bytes),
                ms_bytes);
            noc_async_write(
                get_read_ptr(cb_send_o),
                get_noc_addr(partner_x[pb], partner_y[pb], out_in_base + slot * o_bytes),
                o_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(partner_x[pb], partner_y[pb], reinterpret_cast<uint32_t>(sem[s])), 1u << u);
            cb_pop_front(cb_send_ms, 1);
            cb_pop_front(cb_send_o, vDHt);
            ++send_i;
            progress = true;
        }
        if (!final_done && cb_pages_available_at_front(cb_final_ms, 1) &&
            cb_pages_available_at_front(cb_final_o, vDHt)) {
            noc_async_write(get_read_ptr(cb_final_o), out.get_noc_addr(out_page), o_bytes);
            noc_async_write(get_read_ptr(cb_final_ms), out.get_noc_addr(out_page, o_bytes), ms_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_final_ms, 1);
            cb_pop_front(cb_final_o, vDHt);
            final_done = true;
            progress = true;
        }
        if (progress) {
            idle = 0;
        } else if (++idle > sm::POLL_SPINS) {
            if (status & 1u) {
                status |= 2u;  // nothing left that this kernel can unblock
                break;
            }
            status |= 1u;  // publish the remaining receive slots as they are, so the attention kernel ends
            idle = 0;
        }
    }
    if (has_root) {
        // status page, four BF16 values: magic, wait flags (0 = no wait expired), receive slots published, sends
        volatile tt_l1_ptr uint16_t* page = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
        page[0] = STATUS_MAGIC_BF16;
        invalidate_l1_cache();
        page[1] = bf16_of((status | *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_err)) & 0xFF);
        page[2] = bf16_of(recv_k);  // SLOTS when all receive slots were published or none was needed
        page[3] = bf16_of(send_i);
        noc_async_write(reinterpret_cast<uint32_t>(scratch), out.get_noc_addr(out_page, o_bytes + ms_bytes), 8);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    // global semaphores: zero again for the next launch (every partial that was due has arrived)
    for (uint32_t s = 0; s < sm::MERGE_STEPS; ++s) {
        noc_semaphore_set(sem[s], 0);
    }
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_err), 0);
}
