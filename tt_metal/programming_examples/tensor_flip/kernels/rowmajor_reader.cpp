#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Compile time arguments
    constexpr bool src_is_dram = static_cast<bool>(get_compile_time_arg_val(0));
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t row_width = get_compile_time_arg_val(2);
    constexpr uint32_t rank = get_compile_time_arg_val(3);

    // Runtime arguments
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t end_row = get_arg_val<uint32_t>(2);

    uint32_t dims_to_flip[rank];
    for (uint32_t i = 0; i < rank; i++) {
        dims_to_flip[i] = get_arg_val<uint32_t>(i + 3);
    }

    constexpr uint32_t cb_id = tt::CBIndex::c_0;
    const InterleavedAddrGen<src_is_dram> s0 = {.bank_base_address = src_addr, .page_size = page_size};

    for (uint32_t row_id = start_row; row_id < end_row; ++row_id) {
        cb_reserve_back(cb_id, 1);
        uint32_t l1_buffer_addr = get_write_ptr(cb_id);
        uint64_t read_noc_addr = get_noc_addr(row_id, s0);
        noc_async_read(read_noc_addr, l1_buffer_addr, page_size);
        noc_async_read_barrier();

        // for (uint32_t col_id = 0; col_id < row_width; ++col_id) {
        //     DPRINT << uint32_t(reinterpret_cast<uint32_t*>(l1_buffer_addr)[col_id]) << ", ";
        // }
        // DPRINT << ENDL();

        cb_push_back(cb_id, 1);
    }
}
