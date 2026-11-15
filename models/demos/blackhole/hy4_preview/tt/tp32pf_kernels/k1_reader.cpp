// SPDX-License-Identifier: Apache-2.0
// tp32pf K1a NCRISC (data in): plan page, then per unit: whole expert gate/up slice (768 bfp4 tiles = 8 bank runs)
// resident in cb_w (double buffered, issued one unit ahead with trid 1), then the unit's x K-blocks (pair of M-tiles x
// 24 K tiles, 48 bf16 tiles = 6 reads of 16 KB, trid 2). x layout: bank-sharded, bank b holds K tiles [24b, 24b+24) of
// every M-tile, tile index inside the bank mt*24+kl. CT: 0 cb_w, 1 cb_x, 2 cb_meta, 3 XL1 (1 = x blocks copied from a
// resident L1 staging block instead of DRAM: models x coming from L1), 4 cb_xs, TensorAccessorArgs(plan) RT: 0 me.
// Common: 0 w01 addr, 1 x addr, 2 plan addr
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t XL1 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_xs = get_compile_time_arg_val(4);
    constexpr auto p_args = TensorAccessorArgs<5>();
    constexpr uint32_t TB = 576, BLK = 96 * TB, PA = 24 * TB, XB = 48 * 2048, XP = 16384;
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t w01 = get_common_arg_val<uint32_t>(0);
    const uint32_t xaddr = get_common_arg_val<uint32_t>(1);
    const uint32_t paddr = get_common_arg_val<uint32_t>(2);

    const auto pa = TensorAccessor(p_args, paddr, 2048);
    cb_reserve_back(cb_meta, 1);
    const uint32_t ml = get_write_ptr(cb_meta);
    noc_async_read(pa.get_noc_addr(me), ml, 2048);
    noc_async_read_barrier();
    cb_push_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* plan = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ml);
    const uint32_t nu = plan[0];
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    uint32_t xs_l1 = 0;
    if (XL1) {
        cb_reserve_back(cb_xs, 48);
        xs_l1 = get_write_ptr(cb_xs);
        for (uint32_t i = 0; i < 6; ++i) {
            noc_async_read(get_noc_addr_from_bank_id<true>(me & 7, xaddr) + i * XP, xs_l1 + i * XP, XP);
        }
        noc_async_read_barrier();
    }

    auto issue_w = [&](uint32_t e) {
        cb_reserve_back(cb_w, 768);
        const uint32_t wr = get_write_ptr(cb_w);
        noc_async_read_set_trid(1);
        for (uint32_t j = 0; j < 8; ++j) {
            const uint32_t b = (j + me) & 7;
            const uint64_t base = get_noc_addr_from_bank_id<true>(b, w01 + e * BLK);
            noc_async_read_one_packet_set_state<true>(base, PA, (j + me) & 3);
            for (uint32_t i = 0; i < 4; ++i) {
                noc_async_read_one_packet_with_state_with_trid(base, i * PA, wr + b * BLK + i * PA, 1);
            }
        }
    };
    if (nu > 0) {
        issue_w(plan[1]);
    }
    for (uint32_t u = 0; u < nu; ++u) {
        noc_async_read_barrier_with_trid(1);
        cb_push_back(cb_w, 768);
        if (u + 1 < nu) {
            issue_w(plan[1 + 3 * (u + 1)]);
        }
        const uint32_t mt0 = plan[2 + 3 * u], mtc = plan[3 + 3 * u];
        for (uint32_t pr = 0; pr < mtc / 2; ++pr) {
            for (uint32_t j = 0; j < 8; ++j) {
                const uint32_t b = (j + me) & 7;
                cb_reserve_back(cb_x, 48);
                const uint32_t xw = get_write_ptr(cb_x);
                noc_async_read_set_trid(2);
                if (XL1) {
                    // local L1 -> L1 copy through the NoC (stand-in for the row gather from peer L1)
                    for (uint32_t i = 0; i < 6; ++i) {
                        noc_async_read(get_noc_addr(xs_l1 + i * XP), xw + i * XP, XP);
                    }
                    noc_async_read_barrier();
                } else {
                    const uint64_t base = get_noc_addr_from_bank_id<true>(b, xaddr);
                    noc_async_read_one_packet_set_state<true>(base, XP, j & 3);
                    for (uint32_t i = 0; i < 6; ++i) {
                        noc_async_read_one_packet_with_state_with_trid(
                            base, (mt0 + 2 * pr) * 24 * 2048 + i * XP, xw + i * XP, 2);
                    }
                    noc_async_read_barrier_with_trid(2);
                }
                cb_push_back(cb_x, 48);
            }
        }
    }
    noc_async_read_barrier();
}
