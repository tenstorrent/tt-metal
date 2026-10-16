// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0), a section of another program: rows [k0, k1) of this core's slice of its DRAM bank's weight tiles
// (as proj_reader.cpp reads them: row_bytes at col_offset of every bank row) straight into this core's slot of a
// persistent L1 staging tensor, row r at staging + r * row_bytes. A later BankStreamMatmul launch reads the staged
// weights from L1 (proj_staged_push.cpp) instead of DRAM, so the stream runs while DRAM would otherwise be idle.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t bank_row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_rows = get_compile_time_arg_val(2);
    constexpr uint32_t max_chunks = get_compile_time_arg_val(3);  // <= 15 trids

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t col_offset = get_arg_val<uint32_t>(3);
    const uint32_t k0 = get_arg_val<uint32_t>(4);
    const uint32_t k1 = get_arg_val<uint32_t>(5);
    const uint32_t staging = get_arg_val<uint32_t>(6);

    // The same reads as proj_reader.cpp (bank-relative src with the bank's NoC state), chunk_rows rows per trid and
    // at most max_chunks chunks in flight (as many outstanding reads as the bank streams well).
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint64_t w_noc_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    noc_async_read_one_packet_set_state<true>(w_noc_base, row_bytes, vc);
    uint32_t src = col_offset + k0 * bank_row_bytes;
    uint32_t dst = staging + k0 * row_bytes;
    uint32_t chunk = 0;
    uint32_t oldest = 0;
    for (uint32_t k = k0; k < k1; k += chunk_rows, chunk++) {
        if (chunk - oldest == max_chunks) {
            noc_async_read_barrier_with_trid((oldest % max_chunks) + 1);
            oldest++;
        }
        const uint32_t trid = (chunk % max_chunks) + 1;
        noc_async_read_set_trid(trid);
        const uint32_t end = k + chunk_rows < k1 ? k + chunk_rows : k1;
        for (uint32_t r = k; r < end; r++) {
            noc_async_read_one_packet_with_state_with_trid(w_noc_base, src, dst, trid);
            src += bank_row_bytes;
            dst += row_bytes;
        }
    }
    for (; oldest < chunk; oldest++) {
        noc_async_read_barrier_with_trid((oldest % max_chunks) + 1);
    }
}
