// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

// One work item is a (user, local value head, value-column partition).
// Q/K are either already normalized/scaled or raw vectors for the compute
// kernel's normalize_qk specialization. Gates contain exp(g) and sigmoid(beta).
// Small vectors are compact row-major DRAM pages. Only the recurrent state
// uses tiled DRAM storage. The reader constructs the broadcast views in L1.
void kernel_main() {
    constexpr uint32_t value_columns = get_compile_time_arg_val(0);
    constexpr uint32_t qk_head_repeat = get_compile_time_arg_val(1);
    static_assert(value_columns == 1 || value_columns == 2 || value_columns == 4);
    static_assert(qk_head_repeat > 0);
    constexpr uint32_t splits = 4 / value_columns;
    constexpr auto qa = TensorAccessorArgs<2>();
    constexpr auto ka = TensorAccessorArgs<qa.next_compile_time_args_offset()>();
    constexpr auto va = TensorAccessorArgs<ka.next_compile_time_args_offset()>();
    constexpr auto ga = TensorAccessorArgs<va.next_compile_time_args_offset()>();
    constexpr auto sa = TensorAccessorArgs<ga.next_compile_time_args_offset()>();
    const auto q = TensorAccessor(qa, get_arg_val<uint32_t>(0), 512);
    const auto k = TensorAccessor(ka, get_arg_val<uint32_t>(1), 512);
    const auto v = TensorAccessor(va, get_arg_val<uint32_t>(2), 512);
    const auto gates = TensorAccessor(ga, get_arg_val<uint32_t>(3), 32);
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(4), 4096);
    const uint32_t first = get_arg_val<uint32_t>(5);
    const uint32_t stride = get_arg_val<uint32_t>(6);
    const uint32_t count = get_arg_val<uint32_t>(7);
    // CB9 is reader-private scratch: no compute/writer access or CB tokens.
    const uint32_t scratch = get_write_ptr(9);
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t work = first + item * stride;
        const uint32_t head = work / splits;
        const uint32_t first_column = (work % splits) * value_columns;
        cb_reserve_back(0, 4);
        cb_reserve_back(1, 4);
        cb_reserve_back(2, value_columns);
        cb_reserve_back(3, 1);
        cb_reserve_back(4, 1);
        cb_reserve_back(5, 4 * value_columns);
        noc_async_read_page(head / qk_head_repeat, q, scratch);
        noc_async_read_page(head / qk_head_repeat, k, scratch + 512);
        noc_async_read_page(head, v, scratch + 1024);
        noc_async_read_page(head, gates, scratch + 1536);
        const uint32_t state_l1 = get_write_ptr(5);
        for (uint32_t kr = 0; kr < 4; ++kr) {
            for (uint32_t vc = 0; vc < value_columns; ++vc) {
                noc_async_read_tile(
                    head * 16 + kr * 4 + first_column + vc, state, state_l1 + (kr * value_columns + vc) * 4096);
            }
        }
        noc_async_read_barrier();
        const auto* values = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        auto* qc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
        auto* kc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
        auto* vr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
        // V's unused rows must be zero for the SFPU residual. Q/K column
        // broadcasts and scalar broadcasts never consume the other lanes.
        for (uint32_t i = 0; i < value_columns * 1024; ++i) {
            vr[i] = 0;
        }
        for (uint32_t i = 0; i < 128; ++i) {
            const uint32_t tile = i / 32;
            const uint32_t lane = i % 32;
            const uint32_t col = tile * 1024 + (lane / 16) * 512 + (lane % 16) * 16;
            qc[col] = values[i];
            kc[col] = values[128 + i];
        }
        for (uint32_t i = 0; i < value_columns * 32; ++i) {
            const uint32_t tile = i / 32;
            const uint32_t lane = i % 32;
            const uint32_t row = tile * 1024 + (lane / 16) * 256 + lane % 16;
            vr[row] = values[256 + first_column * 32 + i];
        }
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3))[0] = values[384];
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4))[0] = values[385];
        cb_push_back(0, 4);
        cb_push_back(1, 4);
        cb_push_back(2, value_columns);
        cb_push_back(3, 1);
        cb_push_back(4, 1);
        cb_push_back(5, 4 * value_columns);
    }
}
