// SPDX-License-Identifier: Apache-2.0
// Disjoint compact FP32 output head rows; state write remains head-major.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr auto o_a = TensorAccessorArgs<2>();
    constexpr auto s_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t part = get_arg_val<uint32_t>(1);
    const auto out = TensorAccessor(o_a, get_arg_val<uint32_t>(2), 4096);
    const auto state = TensorAccessor(s_a, get_arg_val<uint32_t>(3), 4096);
    Noc noc;
    CircularBuffer cbout(6), cbst(7);
    cbout.wait_front(Vt);
    auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cbout);
    for (uint32_t t = 0; t < Vt; ++t) {
        noc.async_write(src, out, 64, {.offset_bytes = t * 4096}, {.page_id = part + t, .offset_bytes = h * 64});
        noc.async_write(
            src, out, 64, {.offset_bytes = t * 4096 + 1024}, {.page_id = part + t, .offset_bytes = 1024 + h * 64});
    }
    noc.async_write_barrier();
    cbout.pop_front(Vt);
    cbst.wait_front(Kt * Vt);
    auto st = use<CircularBuffer::AddrSelector::READ_PTR>(cbst);
    for (uint32_t t = 0; t < Kt; ++t) {
        noc.async_write(st, state, 4096, {.offset_bytes = t * 4096}, {.page_id = h * Kt * 4 + t * 4 + part});
    }
    noc.async_write_barrier();
    cbst.pop_front(Kt * Vt);
}
