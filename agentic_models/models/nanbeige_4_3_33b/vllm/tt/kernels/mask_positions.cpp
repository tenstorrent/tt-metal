#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0), P = get_compile_time_arg_val(1);
    constexpr auto a = TensorAccessorArgs<2>();
    constexpr auto b = TensorAccessorArgs<a.next_compile_time_args_offset()>();
    constexpr auto c = TensorAccessorArgs<b.next_compile_time_args_offset()>();
    auto x = TensorAccessor(a, get_arg_val<uint32_t>(0), P);
    auto m = TensorAccessor(b, get_arg_val<uint32_t>(1), P);
    auto y = TensorAccessor(c, get_arg_val<uint32_t>(2), P);
    uint32_t ptr = get_write_ptr(0);
    noc_async_read(x.get_noc_addr(0), ptr, P);
    noc_async_read(m.get_noc_addr(0), ptr + P, P);
    noc_async_read_barrier();
    auto pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(ptr);
    auto active = reinterpret_cast<volatile tt_l1_ptr int32_t*>(ptr + P);
    for (uint32_t i = 0; i < N; i++) {
        if (!active[i]) {
            pos[i] = -1;
        }
    }
    noc_async_write(ptr, y.get_noc_addr(0), P);
    noc_async_write_barrier();
}
