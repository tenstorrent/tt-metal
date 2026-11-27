#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t MP = get_compile_time_arg_val(1);
    constexpr auto ta = TensorAccessorArgs<2>();
    constexpr auto ma = TensorAccessorArgs<ta.next_compile_time_args_offset()>();
    auto tensor = TensorAccessor(ta, get_arg_val<uint32_t>(0), 4096);
    auto map = TensorAccessor(ma, get_arg_val<uint32_t>(1), MP);
    uint32_t core = get_arg_val<uint32_t>(2);
    uint32_t sp = get_write_ptr(0), dp = sp + 4096, mp = get_write_ptr(1);
    noc_async_read(map.get_noc_addr(0), mp, MP);
    noc_async_read_barrier();
    auto mapping = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mp);
    for (uint32_t w = core; w < W; w += 64) {
        noc_async_read(tensor.get_noc_addr(w), sp, 4096);
        noc_async_read_barrier();
        auto s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp);
        auto d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dp);
        for (uint32_t row = 0; row < 32; ++row) {
            uint32_t source = mapping[row];
            uint32_t sr = (source / 16) * 512 + (source % 16) * 16;
            uint32_t dr = (row / 16) * 512 + (row % 16) * 16;
            for (uint32_t j = 0; j < 16; ++j) {
                d[dr + j] = s[sr + j];
                d[dr + 256 + j] = s[sr + 256 + j];
            }
        }
        noc_async_write(dp, tensor.get_noc_addr(w), 4096);
        noc_async_write_barrier();
    }
}
