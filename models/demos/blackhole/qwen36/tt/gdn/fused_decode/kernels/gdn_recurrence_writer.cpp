// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode recurrence writer for one (b, vh) core. Writes the updated fp32 state
// tiles back in place (each (b,vh) owns its 16 state tiles exclusively), then writes
// row b of the gated output into the shared [1,B,value_dim] tiles as two face-row
// NoC writes per tile — rows are disjoint across cores, so no cross-core ordering is
// needed. The b==0 core also zeros the unused rows [b_rows, 32) of its head's output
// tiles so the out-projection never consumes uninitialized memory.

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_hnew = 25, cb_out = 27, cb_zero = 28;

void kernel_main() {
    constexpr uint32_t NV = get_named_compile_time_arg_val("nv");
    constexpr uint32_t DKT = get_named_compile_time_arg_val("dkt");
    constexpr uint32_t DVT = get_named_compile_time_arg_val("dvt");
    constexpr uint32_t B_ROWS = get_named_compile_time_arg_val("b_rows");
    constexpr bool state_is_dram = get_named_compile_time_arg_val("state_is_dram") == 1;
    constexpr bool out_is_dram = get_named_compile_time_arg_val("out_is_dram") == 1;

    const uint32_t state_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(1);

    const uint32_t b = get_arg_val<uint32_t>(0);
    const uint32_t vh = get_arg_val<uint32_t>(1);

    const uint32_t tf = get_tile_size(cb_hnew);  // fp32
    const auto state_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<state_is_dram>(), state_addr, tf);
    const auto out_acc = TensorAccessor(tensor_accessor::make_interleaved_dspec<out_is_dram>(), out_addr, tf);

    // fp32 32x32 tile face geometry.
    constexpr uint32_t row_bytes = 16 * 4;    // one face row
    constexpr uint32_t face_bytes = 16 * row_bytes;

    // In-place state writeback.
    constexpr uint32_t KV = DKT * DVT;
    const uint32_t state_base = (b * NV + vh) * KV;
    cb_wait_front(cb_hnew, KV);
    uint32_t l1 = get_read_ptr(cb_hnew);
    for (uint32_t t = 0; t < KV; t++) {
        noc_async_write_page(state_base + t, state_acc, l1 + t * tf, tf, 0);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_hnew, KV);

    // Row b of the gated output: two face rows per tile.
    cb_wait_front(cb_out, DVT);
    l1 = get_read_ptr(cb_out);
    const uint32_t row_off = (b < 16) ? b * row_bytes : 2 * face_bytes + (b - 16) * row_bytes;
    for (uint32_t n = 0; n < DVT; n++) {
        const uint32_t tile_id = vh * DVT + n;
        const uint32_t tl1 = l1 + n * tf;
        noc_async_write_page(tile_id, out_acc, tl1 + row_off, row_bytes, row_off);
        noc_async_write_page(tile_id, out_acc, tl1 + row_off + face_bytes, row_bytes, row_off + face_bytes);
    }

    if (b == 0 && B_ROWS < 32) {
        // cb_zero is writer-local scratch: reserved for its L1 page, never pushed (no consumer).
        cb_reserve_back(cb_zero, 1);
        auto* zp = reinterpret_cast<uint32_t*>(get_write_ptr(cb_zero));
        for (uint32_t i = 0; i < face_bytes / 4; i++) {
            zp[i] = 0;
        }
        const uint32_t zl1 = get_write_ptr(cb_zero);
        for (uint32_t n = 0; n < DVT; n++) {
            const uint32_t tile_id = vh * DVT + n;
            if constexpr (B_ROWS < 16) {
                const uint32_t off = B_ROWS * row_bytes;
                const uint32_t len = (16 - B_ROWS) * row_bytes;
                noc_async_write_page(tile_id, out_acc, zl1, len, off);
                noc_async_write_page(tile_id, out_acc, zl1, len, off + face_bytes);
                noc_async_write_page(tile_id, out_acc, zl1, face_bytes, 2 * face_bytes);
                noc_async_write_page(tile_id, out_acc, zl1, face_bytes, 3 * face_bytes);
            } else {
                const uint32_t off = 2 * face_bytes + (B_ROWS - 16) * row_bytes;
                const uint32_t len = (32 - B_ROWS) * row_bytes;
                noc_async_write_page(tile_id, out_acc, zl1, len, off);
                noc_async_write_page(tile_id, out_acc, zl1, len, off + face_bytes);
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, DVT);
}
