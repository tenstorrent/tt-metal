#include <cstdint>

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    const InterleavedAddrGenFast<true> dst = {
        .bank_base_address = dst_addr,
        .page_size = tile_size_bytes,
        .data_format = DataFormat::Int8,
    };

    cb_wait_front(cb_out0, 1);
    uint32_t cb_out0_addr = get_read_ptr(cb_out0);
    noc_async_write_tile(0, dst, cb_out0_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_out0, 1);
}
