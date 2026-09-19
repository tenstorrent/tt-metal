// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused anti-aliased SnakeBeta activation, writer. Once the compute kernel has produced every E/O block for this
// core, gathers the 12 down-tap tiles per output tile into CB_DN (k odd reads E at shift (k - 5) / 2, k even
// reads O at shift (k - 6) / 2; the sequence-end tiles on the first/last device apply the z clamp stick by
// stick), then streams each finished output tile to DRAM, valid sticks only on the partial last tile.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_e = get_compile_time_arg_val(3);
    constexpr uint32_t cb_o = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dn = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t cb_fl = get_compile_time_arg_val(7);
    constexpr uint32_t C = get_compile_time_arg_val(8);
    constexpr uint32_t R = get_compile_time_arg_val(9);
    constexpr uint32_t K = get_compile_time_arg_val(10);
    constexpr int32_t T_LOCAL = get_compile_time_arg_val(11);
    constexpr uint32_t OUT_PAGES = get_compile_time_arg_val(14);
    constexpr uint32_t NB_EXTRA = get_compile_time_arg_val(17);
    constexpr uint32_t STICK = C * 4;
    constexpr uint32_t PAGE = K * STICK;
    constexpr uint32_t TILE = R * STICK;

    constexpr auto out_args = TensorAccessorArgs<18>();

    const uint32_t b = get_arg_val<uint32_t>(0);
    const uint32_t o0 = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    if (n_tiles == 0) {
        return;
    }
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);

    Noc noc;
    const auto out_acc = TensorAccessor(out_args, out_addr, PAGE);
    experimental::CB e_cb(cb_e);
    experimental::CB o_cb(cb_o);
    experimental::CB dn_cb(cb_dn);
    experimental::CB out_cb(cb_out);
    experimental::CB fl_cb(cb_fl);

    fl_cb.wait_front(1);
    volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fl_cb.get_read_ptr());
    const bool is_first = flags[0] != 0;
    const bool is_last = flags[1] != 0;

    const int32_t q_lo = static_cast<int32_t>(o0) - 3;
    const uint32_t nblocks = n_tiles + NB_EXTRA;
    e_cb.wait_front(nblocks);
    o_cb.wait_front(nblocks);
    const uint32_t e_base = e_cb.get_read_ptr();
    const uint32_t o_base = o_cb.get_read_ptr();

    // Stream (E or O) and shift for down tap k: out[n] += t[k] z[2n + k - 5], z[2q] = E[q], z[2q + 1] = O[q].
    auto tap_src = [&](uint32_t k, int32_t q) -> uint32_t {
        const bool odd = (k & 1u) != 0;
        const int32_t shift = odd ? (static_cast<int32_t>(k) - 5) / 2 : (static_cast<int32_t>(k) - 6) / 2;
        return (odd ? e_base : o_base) + static_cast<uint32_t>(q + shift - q_lo) * STICK;
    };

    bool wide_state = false;
    for (uint32_t i = 0; i < n_tiles; ++i) {
        const int32_t n0 = static_cast<int32_t>(o0 + i * R);
        const int32_t n1 = n0 + static_cast<int32_t>(R);
        // A sequence-end tile needs the z clamp: m = 2n + k - 5 below 0 or above 2T - 1.
        const bool edge = (is_first && 2 * n0 - 5 < 0) || (is_last && 2 * (n1 - 1) + 6 > 2 * T_LOCAL - 1);
        dn_cb.reserve_back(12);
        if (!edge) {
            if (!wide_state) {
                experimental::set_read_state<TILE>(noc, e_base);
                wide_state = true;
            }
            for (uint32_t k = 0; k < 12; ++k) {
                experimental::read_with_state(noc, dn_cb, tap_src(k, n0), {.offset_bytes = k * TILE});
            }
        } else {
            experimental::set_read_state<STICK>(noc, e_base);
            wide_state = false;
            for (uint32_t k = 0; k < 12; ++k) {
                for (uint32_t sidx = 0; sidx < R; ++sidx) {
                    const int32_t n = n0 + static_cast<int32_t>(sidx);
                    const int32_t m = 2 * n + static_cast<int32_t>(k) - 5;
                    uint32_t src;
                    if (m < 0) {
                        src = e_base + static_cast<uint32_t>(0 - q_lo) * STICK;  // z[0] = E[0]
                    } else if (m > 2 * T_LOCAL - 1) {
                        src = o_base + static_cast<uint32_t>(T_LOCAL - 1 - q_lo) * STICK;  // z[2T-1] = O[T-1]
                    } else {
                        src = tap_src(k, n);
                    }
                    experimental::read_with_state(noc, dn_cb, src, {.offset_bytes = k * TILE + sidx * STICK});
                }
            }
        }
        noc.async_read_barrier();
        dn_cb.push_back(12);

        // The finished tile: valid sticks only, page by page (DRAM writes need 16 B alignment only).
        out_cb.wait_front(1);
        const int32_t valid = (n1 <= T_LOCAL) ? static_cast<int32_t>(R) : (T_LOCAL - n0);
        uint32_t written = 0;
        while (static_cast<int32_t>(written) < valid) {
            const uint32_t stick = static_cast<uint32_t>(n0) + written;
            const uint32_t page = b * OUT_PAGES + stick / K;
            const uint32_t in_page = stick % K;
            uint32_t count = K - in_page;
            if (static_cast<int32_t>(written + count) > valid) {
                count = valid - written;
            }
            noc.async_write(
                out_cb,
                out_acc,
                count * STICK,
                {.offset_bytes = written * STICK},
                {.page_id = page, .offset_bytes = in_page * STICK});
            written += count;
        }
        noc.async_write_barrier();
        out_cb.pop_front(1);
    }
}
