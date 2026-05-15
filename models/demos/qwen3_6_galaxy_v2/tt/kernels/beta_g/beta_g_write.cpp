// beta_g_write
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
    int32_t v1 = 0;
    int32_t v2 = 2048;
    int32_t v3 = 1;
    size_t v4 = 0;
    size_t v5 = 1;
    experimental::CircularBuffer cb_ctarg_5(get_compile_time_arg_val(5));
    experimental::CircularBuffer cb_ctarg_6(get_compile_time_arg_val(6));
    cb_ctarg_5.wait_front(v3);
    cb_ctarg_6.wait_front(v3);
    int32_t v6 = get_common_arg_val<uint32_t>(v4);
    auto tensor_accessor_args_11 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 7>(), 0>();
    TensorAccessor v7 = TensorAccessor(tensor_accessor_args_11, v6, v2);
    noc_async_write_tile(v1, v7, cb_ctarg_5.get_read_ptr());
    int32_t v8 = get_common_arg_val<uint32_t>(v5);
    auto tensor_accessor_args_19 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 7>(), 1>();
    TensorAccessor v9 = TensorAccessor(tensor_accessor_args_19, v8, v2);
    noc_async_write_tile(v1, v9, cb_ctarg_6.get_read_ptr());
    noc_async_write_barrier();
    cb_ctarg_6.pop_front(v3);
    cb_ctarg_5.pop_front(v3);
    return;
}
