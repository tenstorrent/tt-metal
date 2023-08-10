#include <cstdint>
#include "dataflow_api.h"
// #include "debug_print.h"

// SliceRange sr = SliceRange{ .h0 = 0, .h1 = 1, .hs = 8, .w0 = 0, .w1 = 32, .ws = 1 };
// SliceRange sr = SliceRange{ .h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8 };

// Fill an L1 buffer with the given val
// WARNING: Use with caution as there's no memory protection. Make sure size is within limits
bool fill_with_val(uint32_t begin_addr, uint32_t n, uint16_t val) {
    // simplest impl:
    uint16_t* ptr = reinterpret_cast<uint16_t*>(begin_addr);
    for (uint32_t i = 0; i < n; ++ i) {
        ptr[i] = val;
    }
    return true;
}

/**
 * Max-pool 2D. Highly Unoptimized!!
 * TODO [AS]: reuse data moved to L1 instead of reading every time
 */
void kernel_main() {
    kernel_profiler::mark_time(7);

    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t window_h = get_arg_val<uint32_t>(2);
    const uint32_t window_w = get_arg_val<uint32_t>(3);
    const uint32_t window_hw = get_arg_val<uint32_t>(4);
    const uint32_t window_hw_padded = get_arg_val<uint32_t>(5);
    const uint32_t stride_h = get_arg_val<uint32_t>(6);
    const uint32_t stride_w = get_arg_val<uint32_t>(7);
    const int32_t pad_h = get_arg_val<int32_t>(8);
    const int32_t pad_w = get_arg_val<int32_t>(9);
    const int32_t out_h = get_arg_val<int32_t>(10);
    const int32_t out_w = get_arg_val<int32_t>(11);
    const uint32_t in_nbytes_c = get_arg_val<uint32_t>(14);
    const int32_t in_h = get_arg_val<int32_t>(16);
    const int32_t in_w = get_arg_val<int32_t>(17);
    const int32_t in_c = get_arg_val<int32_t>(19);
    const uint32_t bf16_one_u32 = get_arg_val<uint32_t>(22);
    constexpr bool is_in_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t in_cb_id = tt::CB::c_in0;
    constexpr uint32_t in_scalar_cb_id = tt::CB::c_in1;

    constexpr uint32_t TILE_HW = 1024;

    // ROW_MAJOR input
    const InterleavedAddrGen<is_in_dram> s_in = {
        .bank_base_address = in_addr,
        .page_size = in_nbytes_c
    };

    // Reduce scalar = 1
    cb_reserve_back(in_scalar_cb_id, 1);
    uint16_t* in_scalar_cb_write_ptr = reinterpret_cast<uint16_t*>(get_write_ptr(in_scalar_cb_id));
    constexpr uint32_t tile_size = TILE_HW;
    uint16_t bf16_one_u16 = bf16_one_u32 >> 16;
    for (uint32_t i = 0; i < tile_size; ++ i) {
        in_scalar_cb_write_ptr[i] = (uint16_t) bf16_one_u16;
    }
    cb_push_back(in_scalar_cb_id, 1);

    uint32_t start_in_row_id = 0;
    uint32_t out_row_id = 0;

    // fill in_cb_id rows with -inf
    uint32_t in_l1_write_addr = get_write_ptr(in_cb_id);
    fill_with_val(in_l1_write_addr, window_hw_padded * in_c, 0xff7f);

    int32_t start_h = - pad_h;
    // for every output row (across all channels)
    for (int32_t out_h_i = 0; out_h_i < out_h; ++ out_h_i) {
        int32_t start_w = - pad_w;
        // for every output col
        for (int32_t out_w_i = 0; out_w_i < out_w; ++ out_w_i) {
            // start = {start_h, start_w}
            int32_t end_h = start_h + window_h;
            int32_t end_w = start_w + window_w;
            int32_t start_h_max = start_h < 0 ? 0 : start_h;
            int32_t start_w_max = start_w < 0 ? 0 : start_w;
            int32_t end_h_min = end_h < in_h ? end_h : in_h;
            int32_t end_w_min = end_w < in_w ? end_w : in_w;

            // make sure cb is available to fill
            cb_reserve_back(in_cb_id, 1);

            // kernel_profiler::mark_time(10);

            // read at most window_hw input rows into CB
            uint32_t read_rows = 0;
            uint32_t curr_in_l1_write_addr = get_write_ptr(in_cb_id);
            uint32_t in_hw_row_id_base = in_w * start_h_max;  // TODO: get rid of *
            for (int32_t h = start_h_max; h < end_h_min; ++ h) {
                for (int32_t w = start_w_max; w < end_w_min; ++ w) {
                    uint32_t in_hw_row_id = in_hw_row_id_base + w;
                    uint64_t in_noc_addr = get_noc_addr(in_hw_row_id, s_in);
                    noc_async_read(in_noc_addr, curr_in_l1_write_addr, in_nbytes_c);
                    curr_in_l1_write_addr += in_nbytes_c;
                    ++ read_rows;
                }
                in_hw_row_id_base += in_w;
            }
            noc_async_read_barrier();

            if (read_rows != window_hw) {
                // if needed, fill the remainining (window_hw - read_row_id) with -INF
                fill_with_val(curr_in_l1_write_addr, (window_hw - read_rows) * in_c, 0xff7f);   // TODO: get rid of *
            }

            // kernel_profiler::mark_time(11);

            // input for current output index (out_h_i, out_w_i) are ready for this block to be consumed by triscs
            cb_push_back(in_cb_id, 1);
            start_w += stride_w;
        }
        start_h += stride_h;
    }
} // kernel_main()
