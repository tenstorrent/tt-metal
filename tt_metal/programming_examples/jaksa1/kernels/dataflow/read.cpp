#include <cstdint>
#include "api/debug/dprint.h"
// #include "api/debug/dprint_pages.h"
#include "api/dataflow/dataflow_api.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        DPRINT_DATA0({ DPRINT << r << " " << TileSlice(cb_id, tile_id, sr, true, untilize) << ENDL(); });
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    uint32_t in_a_addr = get_arg_val<uint32_t>(0);
    uint32_t in_b_addr = get_arg_val<uint32_t>(1);
    uint32_t in_c_addr = get_arg_val<uint32_t>(2);
    uint32_t Mt = get_arg_val<uint32_t>(3);
    uint32_t Kt = get_arg_val<uint32_t>(4);
    uint32_t Nt = get_arg_val<uint32_t>(5);
    uint32_t work_per_core = get_arg_val<uint32_t>(6);
    uint32_t work_offset = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_in_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_in_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_in_c = tt::CBIndex::c_2;

    const uint32_t tile_size_bytes = get_tile_size(cb_in_a);

    constexpr auto in_a_offset = TensorAccessorArgs<0>();
    constexpr auto in_b_offset = TensorAccessorArgs<in_a_offset.next_compile_time_args_offset()>();
    constexpr auto in_c_offset = TensorAccessorArgs<in_b_offset.next_compile_time_args_offset()>();

    const auto in_a = TensorAccessor(in_a_offset, in_a_addr, tile_size_bytes);
    const auto in_b = TensorAccessor(in_b_offset, in_b_addr, tile_size_bytes);
    const auto in_c = TensorAccessor(in_c_offset, in_c_addr, tile_size_bytes);

    for (uint32_t i = work_offset; i < work_offset + work_per_core; ++i) {
        // DPRINT << "READ - TILE " << i << " START" << ENDL();
        uint32_t mt = i / Nt;
        uint32_t nt = i % Nt;

        for (uint32_t kt = 0; kt < Kt; kt++) {
            cb_reserve_back(cb_in_a, 1);
            uint32_t a_l1 = get_write_ptr(cb_in_a);
            noc_async_read_tile(mt * Kt + kt, in_a, a_l1);
            noc_async_read_barrier();
            cb_push_back(cb_in_a, 1);

            cb_reserve_back(cb_in_b, 1);
            uint32_t b_l1 = get_write_ptr(cb_in_b);
            noc_async_read_tile(kt * Nt + nt, in_b, b_l1);
            noc_async_read_barrier();
            cb_push_back(cb_in_b, 1);
        }

        cb_reserve_back(cb_in_c, 1);
        uint32_t c_l1 = get_write_ptr(cb_in_c);
        noc_async_read_tile(mt * Nt + nt, in_c, c_l1);
        noc_async_read_barrier();
        cb_push_back(cb_in_c, 1);
        DPRINT << "READ - TILE " << i << " END" << ENDL();
    }
}
