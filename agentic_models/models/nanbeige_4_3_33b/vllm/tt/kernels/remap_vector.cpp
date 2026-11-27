#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0), D = get_compile_time_arg_val(1);
    constexpr uint32_t SP = get_compile_time_arg_val(2), DP = get_compile_time_arg_val(3);
    constexpr auto sa = TensorAccessorArgs<4>();
    constexpr auto da = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    constexpr auto ma = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    auto src = TensorAccessor(sa, get_arg_val<uint32_t>(0), SP);
    auto dst = TensorAccessor(da, get_arg_val<uint32_t>(1), DP);
    auto map = TensorAccessor(ma, get_arg_val<uint32_t>(2), 128);
    uint32_t s = get_write_ptr(0), d = s + 128, m = d + 128;
    noc_async_read(src.get_noc_addr(0), s, SP);
    noc_async_read(dst.get_noc_addr(0), d, DP);
    noc_async_read(map.get_noc_addr(0), m, 128);
    noc_async_read_barrier();
    auto source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s);
    auto dest = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(d);
    auto mapping = reinterpret_cast<volatile tt_l1_ptr int32_t*>(m);
    for (uint32_t i = 0; i < D; ++i) {
        int32_t row = mapping[i];
        if (row >= 0 && row < int32_t(S)) {
            dest[i] = source[row];
        }
    }
    noc_async_write(d, dst.get_noc_addr(0), DP);
    noc_async_write_barrier();
}
