// SPDX-License-Identifier: Apache-2.0
// B1/T1 head/V32 workers -> unchanged FP32 recurrence compute, Vt=1.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t qheads = get_compile_time_arg_val(2);
    constexpr auto q_a = TensorAccessorArgs<3>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto d_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t part = get_arg_val<uint32_t>(1);
    const auto q = TensorAccessor(q_a, get_arg_val<uint32_t>(2), 4096);
    const auto k = TensorAccessor(k_a, get_arg_val<uint32_t>(3), 4096);
    const auto v = TensorAccessor(v_a, get_arg_val<uint32_t>(4), 4096);
    const auto d = TensorAccessor(d_a, get_arg_val<uint32_t>(5), 4096);
    const auto b = TensorAccessor(b_a, get_arg_val<uint32_t>(6), 4096);
    const auto s = TensorAccessor(s_a, get_arg_val<uint32_t>(7), 4096);
    Noc noc;
    CircularBuffer state(5);
    state.reserve_back(Kt * Vt);
    for (uint32_t t = 0; t < Kt; ++t) {
        noc.async_read(s, state, 4096, {.page_id = h * Kt * 4 + t * 4 + part}, {.offset_bytes = t * 4096});
    }
    noc.async_read_barrier();
    state.push_back(Kt * Vt);

    // Zero the unused inner-dimension rows: they participate in k^T @ u.
    // Each source head row is in the top two FP32 faces (h<16).
    auto vector = [&](const auto& src, uint32_t cb_id, uint32_t row, uint32_t tiles, uint32_t base) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(tiles);
        noc.async_write_zeros(cb, tiles * 4096, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        for (uint32_t t = 0; t < tiles; ++t) {
            noc.async_read(src, cb, 64, {.page_id = base + t, .offset_bytes = row * 64}, {.offset_bytes = t * 4096});
            noc.async_read(
                src, cb, 64, {.page_id = base + t, .offset_bytes = 1024 + row * 64}, {.offset_bytes = t * 4096 + 1024});
        }
        noc.async_read_barrier();
        cb.push_back(tiles);
    };
    vector(q, 0, h / (12 / qheads), Kt, 0);
    vector(k, 1, h / (12 / qheads), Kt, 0);
    vector(v, 2, h, Vt, part);

    auto scalar = [&](const auto& src, uint32_t cb_id) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        noc.async_write_zeros(cb, 4096, {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        // Blackhole DRAM requires 64-byte alignment. All twelve head scalars
        // fit in this first face row; a 32-byte chunk aliases heads8..11.
        noc.async_read(src, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const CoreLocalMem<volatile uint32_t> words(cb.get_write_ptr());
        const uint32_t value = words[h];
        for (uint32_t i = 1; i < 16; ++i) {
            words[i] = 0;
        }
        words[0] = value;
        cb.push_back(1);
    };
    scalar(d, 3);
    scalar(b, 4);
}
