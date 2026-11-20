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
//
// SM_LAYER (fused/sparse_mla/op.py LayerIo: the program is a part of a layer program): the queries are not read from
// the q tensor. The core waits (bounded) for the word l_qgo, which the landing core of its head group counts when
// the 4 chips that own the group's heads have landed their absorbed queries there, and takes 1 off it. The landing
// holds the heads as the query units pack them, [head][tile][face][user] in 32-byte pieces ([8 users, 32] tiles);
// the page (u, g) the attention kernel needs is [tile][face][head] ([8 heads, 32] tiles): 16 x DHt pieces per
// page, each read straight into its place in cb_q. The root partial is not written to the out tensor (unless
// SM_LAYER_OUT): it stays where the attention kernel packed it (cb_final_ms | cb_final_o are laid over a fixed
// address) and is counted: on merge core u when its head group is the chip's own, else in the group's word on the
// sender core. Status flag 32: no go for the queries.

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
#ifndef SM_ROWS
    constexpr uint32_t cb_final_ms = get_named_compile_time_arg_val("cb_final_ms");
#else
    constexpr uint32_t cb_final_ms = get_named_compile_time_arg_val("cb_final_ms");
#ifndef SM_LAYER
    constexpr uint32_t cb_empty = get_named_compile_time_arg_val("cb_empty");  // the empty partial's 17 tiles
#endif
#endif
#ifdef SM_LAYER_ROWS
    // SM_LAYER_ROWS (the layer form with 16 token rows): the final partial of user `block` is packed into
    // cb_final_* and the one of user `block + GROUPS` into cb_final2_*, two fixed addresses 17 tiles apart, and each
    // is counted in its own byte of the word that counts it (byte p for user block + GROUPS p). A row that is not
    // live this pass (SM_MASK) leaves no partial and is not counted. The query landing holds [16, 32] tiles.
    constexpr uint32_t fin_o[2] = {cb_final_o, get_named_compile_time_arg_val("cb_final2_o")};
    constexpr uint32_t fin_ms[2] = {cb_final_ms, get_named_compile_time_arg_val("cb_final2_ms")};
#endif
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
    [[maybe_unused]] const uint32_t q_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(2);
    [[maybe_unused]] uint32_t layer_status = 0;
#ifdef SM_LAYER
    constexpr uint32_t l_qgo = get_named_compile_time_arg_val("l_qgo");
    constexpr uint32_t l_qland = get_named_compile_time_arg_val("l_qland");
    constexpr uint32_t l_local = get_named_compile_time_arg_val("l_local");
    constexpr uint32_t l_cgroup = get_named_compile_time_arg_val("l_cgroup");
    constexpr uint32_t TBL_L = get_named_compile_time_arg_val("tbl_layer");
    const uint32_t own_group = get_common_arg_val<uint32_t>(TBL_L);
#ifdef SM_LAYER_ROWS
    // the final partial of (user block + GROUPS p, this group) is packed: count it, in byte p of the word, where it
    // is merged or sent from
    auto count_final = [&](uint32_t p = 0) {
        const uint32_t one = 1u << (8 * p);
        if (group == own_group) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_common_arg_val<uint32_t>(TBL_L + 3 + 2 * block),
                    get_common_arg_val<uint32_t>(TBL_L + 4 + 2 * block),
                    l_local),
                one);
        } else {
            noc_semaphore_inc(
                get_noc_addr(
                    get_common_arg_val<uint32_t>(TBL_L + 1),
                    get_common_arg_val<uint32_t>(TBL_L + 2),
                    l_cgroup + 4 * group),
                one);
        }
    };
#else
    // the final partial of (user `block`, this group) is packed: count it where it is merged or sent from
    auto count_final = [&]() {
        if (group == own_group) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_common_arg_val<uint32_t>(TBL_L + 3 + 2 * block),
                    get_common_arg_val<uint32_t>(TBL_L + 4 + 2 * block),
                    l_local),
                1);
        } else {
            noc_semaphore_inc(
                get_noc_addr(
                    get_common_arg_val<uint32_t>(TBL_L + 1),
                    get_common_arg_val<uint32_t>(TBL_L + 2),
                    l_cgroup + 4 * group),
                1);
        }
    };
#endif
#endif

    const auto sel = TensorAccessor(sel_args, sel_addr);
    [[maybe_unused]] const auto q = TensorAccessor(q_args, q_addr);
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
#ifdef SM_LAYER
        {
            volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l_qgo);
            if (sm::wait_min_bounded(go, 1)) {
                noc_semaphore_inc(get_noc_addr(l_qgo), 0u - 1u);
            } else {
                layer_status |= 32u;  // the pages are assembled from whatever the landing holds
            }
        }
        const uint32_t land_x = get_common_arg_val<uint32_t>(TBL_L + 3 + 2 * group);
        const uint32_t land_y = get_common_arg_val<uint32_t>(TBL_L + 4 + 2 * group);
        constexpr uint32_t PIECE = 32;  // one row of a face of an [8, 32] BF16 tile
        const uint32_t face_bytes = tile_bytes / 2;
#ifdef SM_LAYER_ROWS
        const uint32_t land_face = USERS * PIECE;  // a face of the landing's [USERS, 32] tiles
        const uint32_t land_head = 2 * DHt * land_face;
#endif
        for (uint32_t u = 0; u < USERS; ++u) {
            if (sm::chunks_here<GROUPS, CHUNK>(counts[u], block, u)) {
                const uint32_t page = q_base + items * q_bytes;
                for (uint32_t h = 0; h < 8; ++h) {
#ifdef SM_LAYER_ROWS
                    const uint32_t head = l_qland + h * land_head + u * PIECE;
                    for (uint32_t tf = 0; tf < 2 * DHt; ++tf) {
                        noc_async_read_one_packet(
                            get_noc_addr(land_x, land_y, head + tf * land_face),
                            page + tf * face_bytes + h * PIECE,
                            PIECE);
                    }
#else
                    const uint32_t head = l_qland + h * q_bytes + u * PIECE;
                    for (uint32_t tf = 0; tf < 2 * DHt; ++tf) {
                        noc_async_read_one_packet(
                            get_noc_addr(land_x, land_y, head + tf * face_bytes),
                            page + tf * face_bytes + h * PIECE,
                            PIECE);
                    }
#endif
                }
                ++items;
            }
        }
#else
        for (uint32_t u = 0; u < USERS; ++u) {
            if (sm::chunks_here<GROUPS, CHUNK>(counts[u], block, u)) {
                noc_async_read(q.get_noc_addr(u * HEAD_GROUPS + group), q_base + items * q_bytes, q_bytes);
                ++items;
            }
        }
#endif
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
#ifndef SM_ROWS
    const uint32_t ms_in_base = get_write_ptr(cb_ms_in);
    const uint32_t out_in_base = get_write_ptr(cb_out_in);
    const bool has_root = block < USERS;
    const uint32_t out_page = block * HEAD_GROUPS + group;
#else
    const uint32_t ms_in_base = get_write_ptr(cb_ms_in);
    const uint32_t out_in_base = get_write_ptr(cb_out_in);
    // The users whose partial ends on this S-block: u % GROUPS == block (one with USERS == GROUPS, more when token
    // rows queue on the S-blocks). Those with rows, in the order the attention kernel pushes their final partials:
    // first the ones it does not merge for (while it attends), then the merged ones, users ascending in both.
    uint32_t final_user[sm::MAX_USERS];
    uint32_t n_final = 0;
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t u = block; u < USERS; u += GROUPS) {
            const uint32_t n_act = sm::active_blocks<GROUPS, CHUNK>(counts[u]);
            if (counts[u] != 0 && (sm::num_receives(0, n_act) != 0) == (pass == 1)) {
                final_user[n_final++] = u;
            }
        }
    }
#ifdef SM_LAYER
#ifndef SM_LAYER_ROWS
    static_assert(USERS <= GROUPS, "the layer form hands over one final partial per attention core");
#endif
#endif
#ifdef SM_LAYER_ROWS
    static_assert(USERS == 2 * GROUPS, "two final partials per attention core");
    // which of the core's two final partials is still to come from the attention kernel
    bool fin_wait[2] = {counts[block] != 0, counts[block + GROUPS] != 0};
    n_final = (fin_wait[0] ? 1u : 0u) + (fin_wait[1] ? 1u : 0u);
#endif
#endif

    // No partial is due on this core for any user (the common case: at most one chunk per user): the receive
    // slots are not used, and the attention kernel, which decides the same way, does not wait for them.
    uint32_t recv_k = sm::any_receive<USERS, GROUPS, CHUNK>(counts, block) ? 0 : SLOTS;
#ifndef SM_ROWS
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
#if !defined(SM_LAYER) || defined(SM_LAYER_OUT)
        noc_async_write(get_write_ptr(cb_final_o), out.get_noc_addr(out_page), o_bytes);
        noc_async_write(get_write_ptr(cb_final_ms), out.get_noc_addr(out_page, o_bytes), ms_bytes);
#endif
#ifdef SM_LAYER
        count_final();
#endif
        final_done = true;
    }
    while (recv_k < SLOTS || send_i < n_send || !final_done) {
#else
    uint32_t send_i = 0, idle = 0, status = 0, final_i = 0, due_k = 0;
    const sm::DueList<USERS, GROUPS, CHUNK> due(counts, block);
    static_assert(sm::MERGE_STEPS == 3);
#ifdef SM_LAYER
#ifdef SM_LAYER_ROWS
    {
        // a LIVE row of this S-block without rows to attend: the empty partial at its fixed address, counted; a row
        // that is not live: nothing
#ifdef SM_MASK
        const uint32_t live = sm::live_rows();
#else
        const uint32_t live = 0xFFFFFFFFu;
#endif
        for (uint32_t p = 0; p < 2; ++p) {
            const uint32_t u = block + GROUPS * p;
            if (counts[u] != 0 || ((live >> u) & 1u) == 0) {
                continue;
            }
            Noc noc;
            experimental::CB final_o(fin_o[p]);
            noc.async_write_zeros(final_o, o_bytes, {.offset_bytes = 0});
            noc.write_zeros_l1_barrier();
            volatile tt_l1_ptr uint16_t* ms = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(fin_ms[p]));
            for (uint32_t i = 0; i < ms_bytes / 2; ++i) {
                ms[i] = (i % 16 == 0 && i < ms_bytes / 4) ? EMPTY_MAX_BF16 : 0;  // column 0 of every row; sum = 0
            }
#if defined(SM_LAYER_OUT)
            noc_async_write(get_write_ptr(fin_o[p]), out.get_noc_addr(u * HEAD_GROUPS + group), o_bytes);
            noc_async_write(get_write_ptr(fin_ms[p]), out.get_noc_addr(u * HEAD_GROUPS + group, o_bytes), ms_bytes);
            noc_async_write_barrier();
#endif
            count_final(p);
        }
    }
    if (false) {
#else
    if (counts[block] == 0) {  // (the layer form: the empty partial at the fixed address of cb_final_*, one user here)
#endif
        const uint32_t out_page = block * HEAD_GROUPS + group;
        Noc noc;
        experimental::CB final_o(cb_final_o);
        noc.async_write_zeros(final_o, o_bytes, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        volatile tt_l1_ptr uint16_t* ms = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_final_ms));
        for (uint32_t i = 0; i < ms_bytes / 2; ++i) {
            ms[i] = (i % 16 == 0 && i < ms_bytes / 4) ? EMPTY_MAX_BF16 : 0;  // column 0 of every row; sum = 0
        }
#if defined(SM_LAYER_OUT)
        noc_async_write(get_write_ptr(cb_final_o), out.get_noc_addr(out_page), o_bytes);
        noc_async_write(get_write_ptr(cb_final_ms), out.get_noc_addr(out_page, o_bytes), ms_bytes);
#endif
        count_final();
    }
#else
    {
        // a user of this S-block without rows: the empty partial (O = 0, sum = 0, max = EMPTY_MAX), made in a buffer of
        // its own (the attention kernel may be packing another user's final partial into cb_final_* at this moment)
        bool made = false;
        for (uint32_t u = block; u < USERS; u += GROUPS) {
            if (counts[u] != 0) {
                continue;
            }
            const uint32_t empty = get_write_ptr(cb_empty);
            if (!made) {
                cb_reserve_back(cb_empty, vDHt + 1);
                Noc noc;
                experimental::CB empty_cb(cb_empty);
                noc.async_write_zeros(empty_cb, o_bytes, {.offset_bytes = 0});
                noc.write_zeros_l1_barrier();
                volatile tt_l1_ptr uint16_t* ms = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(empty + o_bytes);
                for (uint32_t i = 0; i < ms_bytes / 2; ++i) {
                    ms[i] = (i % 16 == 0 && i < ms_bytes / 4) ? EMPTY_MAX_BF16 : 0;  // column 0 of every row; sum = 0
                }
                made = true;
            }
            noc_async_write(empty, out.get_noc_addr(u * HEAD_GROUPS + group), o_bytes);
            noc_async_write(empty + o_bytes, out.get_noc_addr(u * HEAD_GROUPS + group, o_bytes), ms_bytes);
        }
        noc_async_write_barrier();
    }
#endif
    while (recv_k < SLOTS || send_i < n_send || final_i < n_final) {
#endif
        bool progress = false;
#ifndef SM_ROWS
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
#else
        if (due_k < due.n) {
            // the next slot of the S-block's list of due pairs (sm_common.hpp DueList), in the list's order: the
            // compute kernel consumes exactly these slots in exactly this order
            invalidate_l1_cache();
            const uint32_t arrived[sm::MERGE_STEPS] = {*sem[0], *sem[1], *sem[2]};
            if (due.publishable(due_k, arrived, (status & 1u) != 0)) {
                cb_reserve_back(cb_ms_in, 1);
                cb_push_back(cb_ms_in, 1);
                cb_reserve_back(cb_out_in, vDHt);
                cb_push_back(cb_out_in, vDHt);
                ++due_k;
                if (due_k == due.n) {
                    recv_k = SLOTS;  // (the status page's "every receive slot is published")
                }
                progress = true;
            }
        }
#endif
        if (send_i < n_send && cb_pages_available_at_front(cb_send_ms, 1) &&
            cb_pages_available_at_front(cb_send_o, vDHt)) {
            const uint32_t u = send_user[send_i];
            const uint32_t r = sm::rel<GROUPS>(block, u);
            const uint32_t s = sm::send_step(r);
            const uint32_t pb = (u + r - (1u << s)) % GROUPS;
#ifndef SM_ROWS
            const uint32_t slot = u * sm::MERGE_STEPS + s;
#else
            // the partner's slot of (u, s): its place in the partner's list of due pairs
            const uint32_t slot = sm::DueList<USERS, GROUPS, CHUNK>(counts, pb).slot(u, s);
#endif
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
#ifndef SM_ROWS
        if (!final_done && cb_pages_available_at_front(cb_final_ms, 1) &&
            cb_pages_available_at_front(cb_final_o, vDHt)) {
#if !defined(SM_LAYER) || defined(SM_LAYER_OUT)
            noc_async_write(get_read_ptr(cb_final_o), out.get_noc_addr(out_page), o_bytes);
            noc_async_write(get_read_ptr(cb_final_ms), out.get_noc_addr(out_page, o_bytes), ms_bytes);
            noc_async_write_barrier();
#endif
#else
#ifdef SM_LAYER_ROWS
        for (uint32_t p = 0; p < 2; ++p) {
            if (fin_wait[p] && cb_pages_available_at_front(fin_ms[p], 1) &&
                cb_pages_available_at_front(fin_o[p], vDHt)) {
#if defined(SM_LAYER_OUT)
                const uint32_t out_page = (block + GROUPS * p) * HEAD_GROUPS + group;
                noc_async_write(get_read_ptr(fin_o[p]), out.get_noc_addr(out_page), o_bytes);
                noc_async_write(get_read_ptr(fin_ms[p]), out.get_noc_addr(out_page, o_bytes), ms_bytes);
                noc_async_write_barrier();
#endif
                count_final(p);
                cb_pop_front(fin_ms[p], 1);
                cb_pop_front(fin_o[p], vDHt);
                fin_wait[p] = false;
                ++final_i;
                progress = true;
            }
        }
        if (false) {
#else
        if (final_i < n_final && cb_pages_available_at_front(cb_final_ms, 1) &&
            cb_pages_available_at_front(cb_final_o, vDHt)) {
#endif
#if !defined(SM_LAYER) || defined(SM_LAYER_OUT)
            const uint32_t out_page = final_user[final_i] * HEAD_GROUPS + group;
            noc_async_write(get_read_ptr(cb_final_o), out.get_noc_addr(out_page), o_bytes);
            noc_async_write(get_read_ptr(cb_final_ms), out.get_noc_addr(out_page, o_bytes), ms_bytes);
            noc_async_write_barrier();
#endif
#endif
#ifdef SM_LAYER
            count_final();
#endif
#ifndef SM_ROWS
            cb_pop_front(cb_final_ms, 1);
            cb_pop_front(cb_final_o, vDHt);
            final_done = true;
            progress = true;
#else
            cb_pop_front(cb_final_ms, 1);
            cb_pop_front(cb_final_o, vDHt);
            ++final_i;
            progress = true;
#endif
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
#ifndef SM_ROWS
    if (has_root) {
        // status page, four BF16 values: magic, wait flags (0 = no wait expired), receive slots published, sends
#else
    for (uint32_t u = block; u < USERS; u += GROUPS) {
        const uint32_t out_page = u * HEAD_GROUPS + group;
        // status page, four BF16 values: magic, wait flags (0 = no wait expired), receive slots published, sends
#endif
        volatile tt_l1_ptr uint16_t* page = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch);
        page[0] = STATUS_MAGIC_BF16;
        invalidate_l1_cache();
        page[1] = bf16_of((status | layer_status | *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_err)) & 0xFF);
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
