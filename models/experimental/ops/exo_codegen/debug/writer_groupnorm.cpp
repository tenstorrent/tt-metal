#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
    const uint32_t packed_scaler = get_arg_val<uint32_t>(0);
    const uint32_t packed_scaler_global = get_arg_val<uint32_t>(1);
    const uint32_t packed_eps = get_arg_val<uint32_t>(2);
    const uint32_t dst_addr = get_arg_val<uint32_t>(3);
    const uint32_t block_hw = get_arg_val<uint32_t>(4);
    const uint32_t start_id = get_arg_val<uint32_t>(5);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_scaler = 2;
    constexpr uint32_t cb_eps = 3;
    constexpr uint32_t cb_scaler_global = 4;
    constexpr uint32_t cb_input_mask = 28;
    constexpr uint32_t cb_out0 = 16;

    // Generate reduce scaler (1.0) — used for local reduction
    generate_reduce_scaler(cb_scaler, packed_scaler);

    // Generate global reduce scaler (1/N) — used for mean/variance
    generate_reduce_scaler(cb_scaler_global, packed_scaler_global);

    // Generate epsilon tile
    generate_bcast_col_scalar(cb_eps, packed_eps);

    // Generate input mask tile (all 1.0 for aligned groups)
    {
        constexpr uint32_t packed_one = 0x3F803F80;  // bfloat16 1.0 x2
        cb_reserve_back(cb_input_mask, 1);
        uint32_t mask_addr = get_write_ptr(cb_input_mask);
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_addr);
        for (uint32_t i = 0; i < 512; ++i) {
            ptr[i] = packed_one;
        }
        cb_push_back(cb_input_mask, 1);
    }

    // Write output tiles to DRAM
    const uint32_t page_bytes = get_local_cb_interface(cb_out0).fifo_page_size;
    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

    for (uint32_t i = 0; i < block_hw; ++i) {
        cb_wait_front(cb_out0, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out0);
        noc_async_write_page(start_id + i, s, l1_read_addr);
        noc_async_writes_flushed();
        cb_pop_front(cb_out0, 1);
    }
    noc_async_write_barrier();
}
