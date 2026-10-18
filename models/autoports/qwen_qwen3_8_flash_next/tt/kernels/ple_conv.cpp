// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    constexpr uint32_t seq = get_compile_time_arg_val(1);
    constexpr uint32_t bytes = 10240 * 2;
    constexpr auto xa = TensorAccessorArgs<2>();
    constexpr auto pa = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<pa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto pos = TensorAccessor(pa, get_arg_val<uint32_t>(1));
    const auto slots = TensorAccessor(sa, get_arg_val<uint32_t>(2));
    const auto valid = TensorAccessor(va, get_arg_val<uint32_t>(3));
    const auto history = TensorAccessor(ha, get_arg_val<uint32_t>(4));
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(5));
    const uint32_t lane = get_arg_val<uint32_t>(6);
    CircularBuffer cb(0);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    auto words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto control = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + bytes);
    Noc noc;
    noc.async_read(pos, CoreLocalMem<uint32_t>(base + bytes), 4, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t position = control[0];
    noc.async_read(slots, CoreLocalMem<uint32_t>(base + bytes), 4, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t slot = control[0];
    noc.async_read(valid, CoreLocalMem<uint32_t>(base + bytes), 4, {.page_id = lane}, {});
    noc.async_read_barrier();
    const uint32_t length = control[0];
    ASSERT(length <= seq);
    for (uint32_t tap = 0; tap < 4; ++tap) {
        for (uint32_t t = 0; t < seq; ++t) {
            const int32_t source = int32_t(t) - 9 + int32_t(tap) * 3;
            if (t >= length || (source < 0 && position == 0)) {
                for (uint32_t j = 0; j < bytes / 4; ++j) {
                    words[j] = 0;
                }
            } else if (source < 0) {
                noc.async_read(
                    history, CoreLocalMem<uint32_t>(base), bytes, {.page_id = slot * 9 + uint32_t(source + 9)}, {});
                noc.async_read_barrier();
            } else {
                noc.async_read(x, CoreLocalMem<uint32_t>(base), bytes, {.page_id = lane * seq + uint32_t(source)}, {});
                noc.async_read_barrier();
            }
            noc.async_write(
                CoreLocalMem<uint32_t>(base), output, bytes, {}, {.page_id = tap * batch * seq + lane * seq + t});
            noc.async_write_barrier();
        }
    }
    // Advance by logical length. Ascending writes are safe for overlapping
    // cache shifts because every old source row is >= its destination row.
    for (uint32_t j = 0; j < 9; ++j) {
        const int32_t source = int32_t(length) - 9 + int32_t(j);
        if (source < 0 && position == 0) {
            for (uint32_t k = 0; k < bytes / 4; ++k) {
                words[k] = 0;
            }
        } else if (source < 0) {
            noc.async_read(
                history, CoreLocalMem<uint32_t>(base), bytes, {.page_id = slot * 9 + uint32_t(source + 9)}, {});
            noc.async_read_barrier();
        } else {
            noc.async_read(x, CoreLocalMem<uint32_t>(base), bytes, {.page_id = lane * seq + uint32_t(source)}, {});
            noc.async_read_barrier();
        }
        noc.async_write(CoreLocalMem<uint32_t>(base), history, bytes, {}, {.page_id = slot * 9 + j});
        noc.async_write_barrier();
    }
}
