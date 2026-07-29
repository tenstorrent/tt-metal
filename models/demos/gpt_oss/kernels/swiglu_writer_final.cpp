// SPDX-License-Identifier: Apache-2.0
//
// Scatter writer for the final fused SwiGLU: writes each compact active-slot
// output tile to its REAL expert slot in the [1,E,1,I] down_input tensor.
//
// Compact active-tile index a in [0, NACT*Ht): slot=a/Ht, ti=a%Ht, e=idx[slot].
//   out page = e*Ht + ti.
// Inactive expert slots are left untouched (the down sparse_matmul skips them via
// its sparsity mask, so their contents don't matter).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_idx_id = get_compile_time_arg_val(1);
constexpr uint32_t Ht = get_compile_time_arg_val(2);
constexpr uint32_t nact = get_compile_time_arg_val(3);
constexpr uint32_t ct_idx_out = 4;
constexpr uint32_t ct_idx_idx = TensorAccessorArgs<ct_idx_out>::next_compile_time_args_offset();

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);
    const uint32_t n_tiles = get_arg_val<uint32_t>(3);

    constexpr auto out_args = TensorAccessorArgs<ct_idx_out>();
    constexpr auto idx_args = TensorAccessorArgs<ct_idx_idx>();
    const auto outt = TensorAccessor(out_args, out_addr);
    const auto idxt = TensorAccessor(idx_args, idx_addr);

    Noc noc;
    DataflowBuffer cb_out(cb_out_id);
    DataflowBuffer cb_idx(cb_idx_id);
    const uint32_t out_page = get_local_cb_interface(cb_out_id).fifo_page_size;
    const uint32_t idx_page = get_local_cb_interface(cb_idx_id).fifo_page_size;

    // Load active-expert id list into L1 (separate scratch CB from the reader's).
    cb_idx.reserve_back(1);
    noc.async_read(idxt, cb_idx, idx_page, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    cb_idx.push_back(1);
    volatile tt_l1_ptr uint32_t* exp = (volatile tt_l1_ptr uint32_t*)get_local_cb_interface(cb_idx_id).fifo_rd_ptr;

    for (uint32_t t = 0; t < n_tiles; ++t) {
        const uint32_t a = start_tile + t;
        const uint32_t slot = a / Ht;
        const uint32_t ti = a % Ht;
        const uint32_t e = exp[slot];
        const uint32_t page = e * Ht + ti;
        cb_out.wait_front(1);
        noc.async_write(cb_out, outt, out_page, {}, {.page_id = page});
        noc.async_write_barrier();
        cb_out.pop_front(1);
    }
}
