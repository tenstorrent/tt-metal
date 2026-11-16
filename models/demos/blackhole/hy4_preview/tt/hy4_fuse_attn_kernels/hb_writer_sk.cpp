// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B22 fused back + o-proj, back-side writer: as hb_writer (core i writes its output tile to page i of og), then
// (after the write barrier: the data is in DRAM) increments the ready semaphore of the o-proj bank core of its head h =
// i / NT. CT: 0 cb_out, 1 NT, 2 SEM, then TensorAccessorArgs(og). Common: og addr, then physical (x, y) of the 8 o-proj
// cores (index = head). RT: i.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t NT = get_compile_time_arg_val(1);
    constexpr uint32_t SEM = get_compile_time_arg_val(2);
    constexpr auto o_args = TensorAccessorArgs<3>();
    const uint32_t i = get_arg_val<uint32_t>(0);
    const uint32_t h = i / NT;
    const auto oa = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), 2048);
    CircularBuffer o(cb_out);
    o.wait_front(1);
    noc_async_write(o.get_read_ptr(), oa.get_noc_addr(i), 2048);
    noc_async_write_barrier();
    noc_semaphore_inc(
        get_noc_addr(
            get_common_arg_val<uint32_t>(1 + 2 * h), get_common_arg_val<uint32_t>(2 + 2 * h), get_semaphore(SEM)),
        1);
    noc_async_atomic_barrier();
    o.pop_front(1);
}
