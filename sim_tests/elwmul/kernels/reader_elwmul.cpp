#include <cstdint>

void kernel_main() {
    uint32_t in0_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    const InterleavedAddrGenFast<true> in0 = {
        .bank_base_address = in0_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Int8,
    };
    const InterleavedAddrGenFast<true> in1 = {
        .bank_base_address = in1_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Int8,
    };

    cb_reserve_back(cb_in0, 1);
    uint32_t cb_in0_addr = get_write_ptr(cb_in0);
    noc_async_read_tile(0, in0, cb_in0_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in0, 1);

    cb_reserve_back(cb_in1, 1);
    uint32_t cb_in1_addr = get_write_ptr(cb_in1);
    noc_async_read_tile(0, in1, cb_in1_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in1, 1);
}
