// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// block_pool writer, the pool phase of the block_topk_route writer (included there): each unit's DT output tiles go to
// (head h, tile row g) of q_bar [1, H, NBP, D] for Q, and to this device's shard slot of the gathered kv_g
// [1, 2H, NBG, D] for K (heads [0, H)) and V (heads [H, 2H)): tile row r * NBPT + g with r the ring index. The units
// are the reader's CB_UNITS pages, read by index once CLAIMS_SEM counts them, up to the page holding UNITS. After a K/V
// unit's write barrier, the unit's word of the flag array is incremented on every sender core; the first increment
// waits until every sender has zeroed its array (FLAGS_READY_SEM) and takes the array's address from FLAGS_ADDR_SEM.
//
// Compile-time args from CT: 0 H, 1 RT, 2 DT, 3 NBPT, 4 NX, 5 NY, 6 UNITS, 7 NCORES, 8 NBGT, then TensorAccessorArgs
// of q_bar and kv_g.
// Common runtime args from ARG: 0 q_bar address, 1 kv_g address, 2 ring index r, 3 number of senders, then their noc
// x, y.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "block_pool_units.hpp"

namespace block_pool {
namespace writer {
constexpr uint32_t cb_out = 30;
constexpr uint32_t TILE_BYTES = 2048;
}  // namespace writer

template <uint32_t CT>
void write(uint32_t ARG) {
    using namespace writer;
    constexpr uint32_t H = get_compile_time_arg_val(CT + 0);
    constexpr uint32_t RT = get_compile_time_arg_val(CT + 1);
    constexpr uint32_t DT = get_compile_time_arg_val(CT + 2);
    constexpr uint32_t NBPT = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t UNITS = get_compile_time_arg_val(CT + 6);
    constexpr uint32_t NBGT = get_compile_time_arg_val(CT + 8);
    constexpr auto q_args = TensorAccessorArgs<CT + 9>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    using Layout = block_pool::Layout<H, RT, NBPT>;
    static_assert(Layout::UNITS == UNITS, "host unit count must match the kernel layout");
    const auto q_bar = TensorAccessor(q_args, get_common_arg_val<uint32_t>(ARG + 0));
    const auto kv_g = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(ARG + 1));
    const uint32_t r = get_common_arg_val<uint32_t>(ARG + 2);
    const uint32_t n_senders = get_common_arg_val<uint32_t>(ARG + 3);

    Noc noc;
    CircularBuffer cbo(cb_out), cbu(CB_UNITS);
    const uint32_t units_base = cbu.get_read_ptr();
    volatile tt_l1_ptr uint32_t* claims = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(CLAIMS_SEM));
    uint32_t flags = 0;
    for (uint32_t i = 0;; ++i) {
        noc_semaphore_wait_min(claims, i + 1);
        const uint32_t u = CoreLocalMem<volatile uint32_t>(units_base + i * UNIT_PAGE_BYTES)[0];
        if (u >= UNITS) {
            break;
        }
        const Unit un = Layout::decode(u);
        cbo.wait_front(DT);
        const uint32_t head = un.t == 2 ? H + un.h : un.h;
        const uint32_t first = un.t == 0 ? (head * NBPT + un.g) * DT : (head * NBGT + r * NBPT + un.g) * DT;
        for (uint32_t n = 0; n < DT; ++n) {
            if (un.t == 0) {
                noc.async_write(cbo, q_bar, TILE_BYTES, {.offset_bytes = n * TILE_BYTES}, {.page_id = first + n});
            } else {
                noc.async_write(cbo, kv_g, TILE_BYTES, {.offset_bytes = n * TILE_BYTES}, {.page_id = first + n});
            }
        }
        noc.async_write_barrier();
        cbo.pop_front(DT);
        if (u < Layout::KV_UNITS) {
            if (flags == 0) {
                noc_semaphore_wait_min(
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FLAGS_READY_SEM)), n_senders);
                flags = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FLAGS_ADDR_SEM));
            }
            for (uint32_t s = 0; s < n_senders; ++s) {
                const uint32_t x = get_common_arg_val<uint32_t>(ARG + 4 + 2 * s);
                const uint32_t y = get_common_arg_val<uint32_t>(ARG + 5 + 2 * s);
                noc_semaphore_inc(get_noc_addr(x, y, flags + 4 * u), 1);
            }
        }
    }
    noc_async_atomic_barrier();
}

}  // namespace block_pool
