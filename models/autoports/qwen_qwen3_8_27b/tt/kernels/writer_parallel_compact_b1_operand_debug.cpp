// SPDX-License-Identifier: Apache-2.0
// Diagnostic only: export landed reader operands before any compute.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr auto q_a = TensorAccessorArgs<2>();
    constexpr auto k_a = TensorAccessorArgs<q_a.next_compile_time_args_offset()>();
    constexpr auto v_a = TensorAccessorArgs<k_a.next_compile_time_args_offset()>();
    constexpr auto d_a = TensorAccessorArgs<v_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();
    constexpr auto s_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    const uint32_t h = get_arg_val<uint32_t>(0);
    Noc noc;
    auto copy = [&](const auto& args, uint32_t address, uint32_t cb_id, uint32_t tiles) {
        const auto out = TensorAccessor(args, address, 4096);
        CircularBuffer cb(cb_id);
        cb.wait_front(tiles);
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t t = 0; t < tiles; ++t) {
            noc.async_write(src, out, 4096, {.offset_bytes = t * 4096}, {.page_id = h * tiles + t});
        }
        noc.async_write_barrier();
        cb.pop_front(tiles);
    };
    copy(s_a, get_arg_val<uint32_t>(6), 5, Kt * Vt);
    copy(q_a, get_arg_val<uint32_t>(1), 0, Kt);
    copy(k_a, get_arg_val<uint32_t>(2), 1, Kt);
    copy(v_a, get_arg_val<uint32_t>(3), 2, Vt);
    copy(d_a, get_arg_val<uint32_t>(4), 3, 1);
    copy(b_a, get_arg_val<uint32_t>(5), 4, 1);
}
