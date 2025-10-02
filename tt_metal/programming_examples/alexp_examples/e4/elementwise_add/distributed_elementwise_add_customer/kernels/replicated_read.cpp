#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);
    uint32_t r_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, tile_size_bytes);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, tile_size_bytes);

    // const InterleavedAddrGenFast<true> a = {
    //     .bank_base_address = a_addr,
    //     .page_size = tile_size_bytes,
    //     .data_format = DataFormat::Float32,
    // };

    // const InterleavedAddrGenFast<true> b = {
    //     .bank_base_address = b_addr,
    //     .page_size = tile_size_bytes,
    //     .data_format = DataFormat::Float32,
    // };

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        noc_async_read_tile(i, a, cb_in0_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);

        for (uint32_t j = 0; j < r_tiles; j++) {
            cb_reserve_back(cb_in1, 1);
            uint32_t cb_in1_addr = get_write_ptr(cb_in1);
            noc_async_read_tile(j, b, cb_in1_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in1, 1);
        }
    }
    }
