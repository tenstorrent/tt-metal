// SPDX-License-Identifier: Apache-2.0
//
// Final fused-SwiGLU reader: transpose-eliminating + expert-skipping + trace-safe.
//
// Reads ONLY the n_active experts (ids from a device tensor idx[1,1,1,NACT] uint32,
// produced from the router top-k each token) directly from the raw fused gate/up
// sparse_matmul output [1,E,1,2I] (padded [1,E,32,2I]) in native expert-major tile
// layout. This removes BOTH the reshape/transpose/slice chain AND the 28 inactive
// experts' work.
//
// Raw page math: expert e, tile-col j at page e*Wt2 + j.
//   gate half = tile-cols [0,Ht);  up half = tile-cols [Ht,2Ht) (Ht=I/32, Wt2=2I/32).
// Compact active-tile index a in [0, NACT*Ht): slot=a/Ht, ti=a%Ht, e=idx[slot].
//   gate page = e*Wt2 + ti ;  up page = e*Wt2 + Ht + ti.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_gate_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_up_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_idx_id = get_compile_time_arg_val(2);
constexpr uint32_t Ht = get_compile_time_arg_val(3);
constexpr uint32_t Wt2 = get_compile_time_arg_val(4);
constexpr uint32_t nact = get_compile_time_arg_val(5);
constexpr uint32_t ct_idx_raw = 6;
constexpr uint32_t ct_idx_idx = TensorAccessorArgs<ct_idx_raw>::next_compile_time_args_offset();

void kernel_main() {
    const uint32_t raw_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t n_tiles = get_arg_val<uint32_t>(3);

    constexpr auto raw_args = TensorAccessorArgs<ct_idx_raw>();
    constexpr auto idx_args = TensorAccessorArgs<ct_idx_idx>();
    const auto raw = TensorAccessor(raw_args, raw_addr);
    const auto idxt = TensorAccessor(idx_args, idx_addr);

    Noc noc;
    DataflowBuffer cb_gate(cb_gate_id);
    DataflowBuffer cb_up(cb_up_id);
    DataflowBuffer cb_idx(cb_idx_id);
    const uint32_t gate_page = get_local_cb_interface(cb_gate_id).fifo_page_size;
    const uint32_t up_page = get_local_cb_interface(cb_up_id).fifo_page_size;
    const uint32_t idx_page = get_local_cb_interface(cb_idx_id).fifo_page_size;

    // Load the active-expert id list (one page: [1,1,1,NACT] uint32) into L1.
    cb_idx.reserve_back(1);
    noc.async_read(idxt, cb_idx, idx_page, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_idx.push_back(1);
    volatile tt_l1_ptr uint32_t* exp = (volatile tt_l1_ptr uint32_t*)get_local_cb_interface(cb_idx_id).fifo_rd_ptr;

    constexpr uint32_t BATCH = 8;
    uint32_t t = 0;
    for (; t + BATCH <= n_tiles; t += BATCH) {
        cb_gate.reserve_back(BATCH);
        cb_up.reserve_back(BATCH);
        for (uint32_t b = 0; b < BATCH; ++b) {
            const uint32_t a = start_tile + t + b;
            const uint32_t slot = a / Ht;
            const uint32_t ti = a % Ht;
            const uint32_t e = exp[slot];
            const uint32_t gpage = e * Wt2 + ti;
            const uint32_t upage = e * Wt2 + Ht + ti;
            noc.async_read(raw, cb_gate, gate_page, {.page_id = gpage}, {.offset_bytes = b * gate_page});
            noc.async_read(raw, cb_up, up_page, {.page_id = upage}, {.offset_bytes = b * up_page});
        }
        noc.async_read_barrier();
        cb_gate.push_back(BATCH);
        cb_up.push_back(BATCH);
    }
    for (; t < n_tiles; ++t) {
        const uint32_t a = start_tile + t;
        const uint32_t slot = a / Ht;
        const uint32_t ti = a % Ht;
        const uint32_t e = exp[slot];
        const uint32_t gpage = e * Wt2 + ti;
        const uint32_t upage = e * Wt2 + Ht + ti;
        cb_gate.reserve_back(1);
        cb_up.reserve_back(1);
        noc.async_read(raw, cb_gate, gate_page, {.page_id = gpage}, {.offset_bytes = 0});
        noc.async_read(raw, cb_up, up_page, {.page_id = upage}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_gate.push_back(1);
        cb_up.push_back(1);
    }
}
