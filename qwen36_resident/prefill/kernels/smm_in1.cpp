// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Small-M streaming matmul, weight reader (NCRISC): this core's column slab of W (K x nt tiles, K-major),
// K block by K block (kb x nt tiles) into cb_in1. Block b of the slab sits in DRAM bank j % banks at
// ((b * per_bank + j / banks) * kb * nt) tiles, so the cores of a bank read neighbouring blocks; packets
// go on this RISC's NOC under the block's transaction id (BRISC reads x on the other NOC, and the two
// RISCs share a NOC's read state), up to `ring - 1` blocks ahead of the compute.
//
// It also reads this core's x block (block j, if j < blocks: Mt x kb tiles of x [M, K]) into cb_xstage and
// raises x_ready for the BRISC multicast; BRISC does no reads, so both NOCs' read state is this RISC's.
//
// Compile-time args: 0 kb, 1 nt, 2 blocks, 3 ring, 4 banks, 5 per_bank, 6 Mt, 7 Kt. Runtime args: 0 core
// j, 1 W address, 2 output columns of this core (0: no slab; the blocks are pushed unread for the
// compute's lockstep), 3 x address.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t kb = get_compile_time_arg_val(0);
    constexpr uint32_t nt = get_compile_time_arg_val(1);
    constexpr uint32_t blocks = get_compile_time_arg_val(2);
    constexpr uint32_t ring = get_compile_time_arg_val(3);
    constexpr uint32_t banks = get_compile_time_arg_val(4);
    constexpr uint32_t per_bank = get_compile_time_arg_val(5);
    constexpr uint32_t Mt = get_compile_time_arg_val(6);
    constexpr uint32_t Kt = get_compile_time_arg_val(7);
    constexpr uint32_t cb_xstage = 2, sem_x = 5, x_trid = 15;
    constexpr uint32_t cb_in1 = 1, kPacket = 4096;
    constexpr uint32_t tile = get_tile_size(cb_in1), blk = kb * nt, bytes = blk * tile;
    const uint32_t j = get_arg_val<uint32_t>(0), addr = get_arg_val<uint32_t>(1), cols = get_arg_val<uint32_t>(2);
    const uint32_t bank = j % banks, slot = j / banks;
    const uint32_t vc = j & 3;

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 0);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 1);
    // this core's x block first, under its own transaction id, collected after the first weight blocks
    const bool owns_x = j < blocks;
    if (owns_x) {
        const uint32_t xt = get_tile_size(cb_xstage);
        const InterleavedAddrGenFast<true> x{
            .bank_base_address = get_arg_val<uint32_t>(3), .page_size = xt, .data_format = DataFormat::Float16_b};
        const uint32_t stage = get_write_ptr(cb_xstage);
        noc_async_read_set_trid(x_trid, noc_index);
        for (uint32_t m = 0; m < Mt; m++) {
            for (uint32_t k = 0; k < kb; k++) {
                noc_async_read_tile(m * Kt + j * kb + k, x, stage + (m * kb + k) * xt);
            }
        }
    }
    auto x_done = [&]() {
        if (owns_x) {
            noc_async_read_barrier_with_trid(x_trid, noc_index);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x)) = 1;
        }
    };
    if (cols == 0) {
        x_done();
        for (uint32_t b = 0; b < blocks; b++) {
            cb_reserve_back(cb_in1, blk);
            cb_push_back(cb_in1, blk);
        }
        return;
    }
    auto issue = [&](uint32_t b, uint32_t dst) {
        const uint32_t trid = b % ring + 1;
        noc_async_read_set_trid(trid, 0);
        noc_async_read_set_trid(trid, 1);
        uint32_t src = (b * per_bank + slot) * bytes;
        for (uint32_t off = 0, k = 0; off < bytes; off += kPacket, k++) {
            const uint32_t size = bytes - off < kPacket ? bytes - off : kPacket;
            const uint8_t noc = k & 1;
            noc_async_read_one_packet_set_state<true>(get_noc_addr_from_bank_id<true>(bank, addr, noc), size, vc, noc);
            noc_async_read_one_packet_with_state_with_trid(addr, src + off, dst + off, trid, noc);
        }
    };
    // blocks b .. b + ring - 2 in flight; block b lands in ring slot b % ring
    const uint32_t base = get_write_ptr(cb_in1);
    cb_reserve_back(cb_in1, (ring - 1) * blk);
    for (uint32_t b = 0; b < ring - 1 && b < blocks; b++) {
        issue(b, base + (b % ring) * bytes);
    }
    x_done();
    for (uint32_t b = 0; b < blocks; b++) {
        noc_async_read_barrier_with_trid(b % ring + 1, 0);
        noc_async_read_barrier_with_trid(b % ring + 1, 1);
        cb_push_back(cb_in1, blk);
        const uint32_t nb = b + ring - 1;
        if (nb < blocks) {
            cb_reserve_back(cb_in1, (ring - 1) * blk);
            issue(nb, base + (nb % ring) * bytes);
        }
    }
}
