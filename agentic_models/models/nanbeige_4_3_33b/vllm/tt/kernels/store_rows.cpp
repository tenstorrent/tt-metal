#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t W = get_compile_time_arg_val(0);
    constexpr uint32_t MP = get_compile_time_arg_val(1);
    constexpr auto sa = TensorAccessorArgs<2>();
    constexpr auto da = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    auto src = TensorAccessor(sa, get_arg_val<uint32_t>(0), 2048);
    auto dst = TensorAccessor(da, get_arg_val<uint32_t>(1), 2048);
    auto map = TensorAccessor(ma, get_arg_val<uint32_t>(2), MP);
    uint32_t core = get_arg_val<uint32_t>(3);
    uint32_t sp = get_write_ptr(0), dp = sp + 2048, mp = get_write_ptr(1);
    noc_async_read(map.get_noc_addr(0), mp, MP);
    noc_async_read_barrier();
    auto mapping = reinterpret_cast<volatile tt_l1_ptr int32_t*>(mp);
    for (uint32_t w = core; w < W; w += 64) {
        noc_async_read(src.get_noc_addr(w), sp, 2048);
        noc_async_read(dst.get_noc_addr(w), dp, 2048);
        noc_async_read_barrier();
        auto s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp);
        auto d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dp);
        for (uint32_t row = 0; row < 32; ++row) {
            int32_t dest = mapping[row];
            if (dest < 0) {
                continue;
            }
            uint32_t sr = (row / 16) * 256 + (row % 16) * 8;
            uint32_t dr = (dest / 16) * 256 + (dest % 16) * 8;
            for (uint32_t j = 0; j < 8; ++j) {
                d[dr + j] = s[sr + j];
                d[dr + 128 + j] = s[sr + 128 + j];
            }
        }
        noc_async_write(dp, dst.get_noc_addr(w), 2048);
        noc_async_write_barrier();
    }
}
