// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of the decode tile's dense projection (tt/v41/decode_dense.py DecodeDense; kernels/dense_row_reader.cpp
// has the dataflow): a unit this core does not own has its fp32 partial tile written into the owner's cb_recv at
// the unit's slot and the owner's semaphore incremented; the owned units' output tiles go to pages j of ``y``
// [B, 32, N] (tile page j = b * NT + n). NoC 1, unicast writes only.
//
// compile_time_args = [cb_part, cb_recv, cb_out, PART_TILE, OUT_TILE, S, TensorAccessorArgs(y)...]
// runtime args      = [y_addr, sem_recv, n_units, n_owned, (j, owned, owner_nx, owner_ny, slot) * n_units]

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t y_addr = get_arg_val<uint32_t>(0);
    const uint32_t sem_recv_id = get_arg_val<uint32_t>(1);
    const uint32_t n_units = get_arg_val<uint32_t>(2);
    const uint32_t n_owned = get_arg_val<uint32_t>(3);
    constexpr uint32_t UNITS = 4;

    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t PART_TILE = get_compile_time_arg_val(3);
    constexpr uint32_t OUT_TILE = get_compile_time_arg_val(4);
    constexpr uint32_t S = get_compile_time_arg_val(5);
    constexpr auto y_args = TensorAccessorArgs<6>();

    const auto y = TensorAccessor(y_args, y_addr, OUT_TILE);
    DataflowBuffer part(cb_part), recv(cb_recv), out(cb_out);
    const uint32_t recv_l1 = recv.get_write_ptr();  // the CB's base: the same on every core
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t owned = get_arg_val<uint32_t>(UNITS + 5 * u + 1);
        if (S == 1 || owned) {
            continue;
        }
        const uint32_t onx = get_arg_val<uint32_t>(UNITS + 5 * u + 2), ony = get_arg_val<uint32_t>(UNITS + 5 * u + 3);
        const uint32_t slot = get_arg_val<uint32_t>(UNITS + 5 * u + 4);
        part.wait_front(1);
        noc_async_write(part.get_read_ptr(), get_noc_addr(onx, ony, recv_l1 + slot * PART_TILE), PART_TILE);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(onx, ony, get_semaphore(sem_recv_id)), 1);
        part.pop_front(1);
    }
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t j = get_arg_val<uint32_t>(UNITS + 5 * u), owned = get_arg_val<uint32_t>(UNITS + 5 * u + 1);
        if (S > 1 && !owned) {
            continue;
        }
        out.wait_front(1);
        noc_async_write(out.get_read_ptr(), y.get_noc_addr(j), OUT_TILE);
        noc_async_write_barrier();
        out.pop_front(1);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
