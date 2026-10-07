// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Candidate merge + exchange of the fused decode LM head (tt/decode_terminal.py), BRISC of the exchange core, part of
// the LM-head program (experts/stream.py LinearStream with an exchange):
//   1. waits until the `n_lists` LM-head writer cores delivered their top-K lists (K order keys + K local indices
//      each, kernels/stream_linear_writer.cpp topk = 1) and merges them into this device's top-K, ordered by (value
//      descending, index ascending); writes it as the payload [K BF16 values | K UINT32 global ids] (global id =
//      id_offset + local index);
//   2. releases the fabric sender on this core (kernels/boundary_sender.cpp, program semaphore `send_sem`), which
//      writes the payload into slot `ring index` of the receive buffer on every device (atomic increment of the global
//      receive semaphore per slot);
//   3. waits for the `ring` slots, resets the receive semaphore and writes the gathered candidates as the sampler
//      input: row 0 of `values` (BF16 [1, 1, 32, ring * K] tiles, slot d -> tile d) and of `ids` (UINT32
//      [1, 1, 32, ring * K] row-major).
// With position tensors (has_pos = 1) and runtime flag inc = 1 (the model decodes with on-device sampling) it also
// advances the decode position state by one: current_pos (INT32, entries < 0 kept) and the RoPE row indices (UINT32),
// as ttnn.plus_one does. inc is a runtime argument so that both decode modes run the same program.
//
// runtime args: [lists_addr, payload_addr, slots_addr, values_addr, ids_addr, id_offset, recv_sem_addr, pos_addr,
//                rot_addr, inc]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t lists_addr = get_arg_val<uint32_t>(0);
    const uint32_t payload_addr = get_arg_val<uint32_t>(1);
    const uint32_t slots_addr = get_arg_val<uint32_t>(2);
    const uint32_t values_addr = get_arg_val<uint32_t>(3);
    const uint32_t ids_addr = get_arg_val<uint32_t>(4);
    const uint32_t id_offset = get_arg_val<uint32_t>(5);
    const uint32_t recv_sem_addr = get_arg_val<uint32_t>(6);
    const uint32_t pos_addr = get_arg_val<uint32_t>(7);
    const uint32_t rot_addr = get_arg_val<uint32_t>(8);
    const uint32_t inc = get_arg_val<uint32_t>(9);

    constexpr uint32_t n_lists = get_compile_time_arg_val(0);
    constexpr uint32_t merge_sem = get_compile_time_arg_val(1);
    constexpr uint32_t send_sem = get_compile_time_arg_val(2);
    constexpr uint32_t ring = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t has_pos = get_compile_time_arg_val(5);
    constexpr uint32_t pos_page = get_compile_time_arg_val(6);  // bytes of the current_pos page
    constexpr uint32_t pos_n = get_compile_time_arg_val(7);     // int32 entries
    constexpr uint32_t rot_page = get_compile_time_arg_val(8);
    constexpr uint32_t rot_n = get_compile_time_arg_val(9);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(10);
    constexpr auto v_args = TensorAccessorArgs<11>();
    constexpr auto i_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t payload_bytes = K * 2 + K * 4;
    static_assert(K == 32, "one tile row of candidates per device");

    // 1. merge
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(merge_sem)), n_lists);
    volatile tt_l1_ptr uint32_t* lists = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lists_addr);
    uint32_t head[n_lists];
    for (uint32_t l = 0; l < n_lists; ++l) {
        head[l] = 0;
    }
    volatile tt_l1_ptr uint16_t* out_val = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(payload_addr);
    volatile tt_l1_ptr uint32_t* out_id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(payload_addr + 2 * K);
    for (uint32_t j = 0; j < K; ++j) {
        uint32_t best = n_lists;
        uint32_t best_key = 0, best_id = 0xFFFFFFFFu;
        for (uint32_t l = 0; l < n_lists; ++l) {
            if (head[l] >= K) {
                continue;
            }
            const uint32_t key = lists[l * 2 * K + head[l]];
            const uint32_t id = lists[l * 2 * K + K + head[l]];
            if (best == n_lists || key > best_key || (key == best_key && id < best_id)) {
                best = l;
                best_key = key;
                best_id = id;
            }
        }
        head[best] += 1;
        out_val[j] = (best_key & 0x8000) ? (best_key & 0x7FFF) : (~best_key & 0xFFFF);
        out_id[j] = id_offset + best_id;
    }
    // 2. release the sender
    noc_semaphore_inc(get_noc_addr(get_semaphore(send_sem)), 1);
    noc_async_atomic_barrier();

    // position state (independent of the exchange)
    if (has_pos && inc) {
        const auto s_pos = TensorAccessor(p_args, pos_addr, pos_page);
        const auto s_rot = TensorAccessor(r_args, rot_addr, rot_page);
        cb_reserve_back(cb_scr, 1);
        const uint32_t scr = get_write_ptr(cb_scr);
        noc_async_read(s_pos.get_noc_addr(0), scr, pos_page);
        constexpr uint32_t rot_off = (pos_page + 63) / 64 * 64;
        noc_async_read(s_rot.get_noc_addr(0), scr + rot_off, rot_page);
        noc_async_read_barrier();
        volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr);
        volatile tt_l1_ptr uint32_t* rot = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + rot_off);
        for (uint32_t i = 0; i < pos_n; ++i) {
            if (pos[i] >= 0) {
                pos[i] = pos[i] + 1;
            }
        }
        for (uint32_t i = 0; i < rot_n; ++i) {
            rot[i] = rot[i] + 1;
        }
        noc_async_write(scr, s_pos.get_noc_addr(0), pos_page);
        noc_async_write(scr + rot_off, s_rot.get_noc_addr(0), rot_page);
    }

    // 3. gather
    volatile tt_l1_ptr uint32_t* recv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(recv_sem_addr);
    noc_semaphore_wait(recv_sem, ring);
    noc_semaphore_set(recv_sem, 0);
    const auto s_v = TensorAccessor(v_args, values_addr, 2048);
    const auto s_i = TensorAccessor(i_args, ids_addr, ring * K * 4);
    const uint64_t ids_dst = s_i.get_noc_addr(0);
    for (uint32_t d = 0; d < ring; ++d) {
        const uint32_t slot = slots_addr + d * payload_bytes;
        const uint64_t tile = s_v.get_noc_addr(d);
        noc_async_write(slot, tile, 32);             // values 0..15 -> face 0 row 0
        noc_async_write(slot + 32, tile + 512, 32);  // values 16..31 -> face 1 row 0
        noc_async_write(slot + 2 * K, ids_dst + d * K * 4, K * 4);
    }
    noc_async_write_barrier();
}
