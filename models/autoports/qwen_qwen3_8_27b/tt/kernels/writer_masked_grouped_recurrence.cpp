// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Experimental one-step writer: output is unconditional, FP32 state commit is masked.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"
#include "api/debug/assert.h"

constexpr uint32_t cb_out = 6, cb_state = 7, cb_mask = 16;
void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    static_assert(get_compile_time_arg_val(2) == 0);
    constexpr auto o_a = TensorAccessorArgs<3>();
    constexpr auto st_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto mask_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>();
    const uint32_t first_head = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t st_addr = get_arg_val<uint32_t>(3);
    const uint32_t num_heads = get_arg_val<uint32_t>(5);
    const uint32_t mask_addr = get_arg_val<uint32_t>(6);
    ASSERT(T == 1 && num_heads > 0 && first_head / 12 == (first_head + num_heads - 1) / 12);
    const uint32_t tb = get_tile_size(cb_out);
    const auto o_acc = TensorAccessor(o_a, o_addr, tb);
    const auto st_acc = TensorAccessor(st_a, st_addr, tb);
    const auto mask_acc = TensorAccessor(mask_a, mask_addr, 2048);
    Noc noc;
    CircularBuffer cbout(cb_out), cbst(cb_state), maskcb(cb_mask);
    // [B,1,1] BF16 has one physical32x32 tile per user. Only element0 is read.
    maskcb.reserve_back(1);
    noc.async_read(mask_acc, maskcb, 32, {.page_id = first_head / 12}, {.offset_bytes = 0});
    noc.async_read_barrier();
    maskcb.push_back(1);
    maskcb.wait_front(1);
    const CoreLocalMem<volatile uint32_t> mask_word(maskcb.get_read_ptr());
    const bool active = (mask_word[0] & 0x7fff) != 0;
    maskcb.pop_front(1);
    auto write_from = [&](CircularBuffer& cb, const auto& acc, uint32_t base, uint32_t count, bool commit) {
        cb.wait_front(count);
        if (commit) {
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
            for (uint32_t t = 0; t < count; ++t) {
                noc.async_write(src, acc, tb, {.offset_bytes = t * tb}, {.page_id = base + t});
            }
            noc.async_write_barrier();
        }
        // Inactive heads must still drain their final state so compute can advance.
        cb.pop_front(count);
    };
    for (uint32_t local_head = 0; local_head < num_heads; ++local_head) {
        const uint32_t h = first_head + local_head;
        write_from(cbout, o_acc, h * Vt, Vt, true);
        write_from(cbst, st_acc, h * Kt * Vt, Kt * Vt, active);
    }
}
