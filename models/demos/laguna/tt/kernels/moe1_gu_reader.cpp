// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batch-1 decode MoE gate/up reader (Laguna). Core (nt, slot group) owns output column tile nt of the up to
// SLOTS active local experts in its group. The active experts and their routing weights come from the sparsity
// row (bf16 routing weight per local expert, nonzero = active), so the program is trace-safe for any routing.
// Pushes: cb_meta <- {n, fp32 bits of each unit's routing weight}; cb_in0 <- row 0 of the Kt activation tiles
// (rows 1-31 zeroed); cb_w <- per unit Kt gate tiles then Kt up tiles from the packed [E, K, 2N] weight.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nt = get_compile_time_arg_val(1);
    constexpr uint32_t num_experts = get_compile_time_arg_val(2);
    constexpr uint32_t chunk = get_compile_time_arg_val(3);
    constexpr uint32_t in0_page = get_compile_time_arg_val(4);
    constexpr uint32_t w_page = get_compile_time_arg_val(5);
    constexpr uint32_t sp_page = get_compile_time_arg_val(6);
    constexpr uint32_t grid_x = get_compile_time_arg_val(7);
    constexpr uint32_t slot_groups = get_compile_time_arg_val(8);
    constexpr uint32_t slots = get_compile_time_arg_val(9);
    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_w = 1;
    constexpr uint32_t cb_meta = 2;
    constexpr uint32_t cb_sp = 3;
    constexpr auto in0_args = TensorAccessorArgs<10>();
    constexpr auto w_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();

    const uint32_t in0_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t core_index = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t nt = core_index / slot_groups;
    const uint32_t first_slot = (core_index % slot_groups) * slots;

    const auto in0 = TensorAccessor(in0_args, in0_addr, in0_page);
    const auto w = TensorAccessor(w_args, w_addr, w_page);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);

    const uint32_t sp_l1 = get_write_ptr(cb_sp);
    noc_async_read(sp.get_noc_addr(0), sp_l1, sp_page);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* spv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    uint32_t experts[slots];
    cb_reserve_back(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta));
    uint32_t n = 0;
    uint32_t seen = 0;
    for (uint32_t e = 0; e < num_experts && n < slots; ++e) {
        const uint32_t v = spv[e];
        if (v != 0) {
            if (seen >= first_slot) {
                meta[1 + n] = v << 16;  // bf16 -> fp32 bits
                experts[n++] = e;
            }
            ++seen;
        }
    }
    meta[0] = n;
    cb_push_back(cb_meta, 1);
    if (n == 0) {
        return;
    }

    // one token: only row 0 of each activation tile is real -- zero the CB and fetch row 0 (face 0 and face 1)
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    cb_reserve_back(cb_in0, Kt);
    const uint32_t x_l1 = get_write_ptr(cb_in0);
    for (uint32_t off = 0; off < Kt * in0_page; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, x_l1 + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        const uint64_t src = in0.get_noc_addr(kt);
        noc_async_read(src, x_l1 + kt * in0_page, 32);
        noc_async_read(src + 512, x_l1 + kt * in0_page + 512, 32);
    }
    noc_async_read_barrier();
    cb_push_back(cb_in0, Kt);

    for (uint32_t u = 0; u < n; ++u) {
        const uint32_t base = experts[u] * Kt * 2 * Nt + nt;
        for (uint32_t half = 0; half < 2; ++half) {  // gate columns, then up columns (offset Nt)
            for (uint32_t k0 = 0; k0 < Kt; k0 += chunk) {
                cb_reserve_back(cb_w, chunk);
                uint32_t wdst = get_write_ptr(cb_w);
                for (uint32_t i = 0; i < chunk; ++i) {
                    noc_async_read_tile(base + half * Nt + (k0 + i) * 2 * Nt, w, wdst);
                    wdst += w_page;
                }
                noc_async_read_barrier();
                cb_push_back(cb_w, chunk);
            }
        }
    }
}
