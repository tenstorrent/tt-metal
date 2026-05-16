// recurrent_write
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
    int32_t v1 = 0;
    int32_t v2 = 4096;
    int32_t v3 = 1;
    size_t v4 = 1;
    size_t v5 = 0;
    size_t v6 = 4;
    experimental::CircularBuffer cb_ctarg_7(get_compile_time_arg_val(7));
    experimental::CircularBuffer cb_ctarg_6(get_compile_time_arg_val(6));
    size_t v7 = get_absolute_logical_x();
    size_t v8 = get_absolute_logical_y();
    size_t v9 = v8 * v6;
    for (size_t i10 = v5; i10 < v6; i10 += v4) {
        cb_ctarg_6.wait_front(v3);
        size_t v11 = v9 + i10;
        int32_t v12 = get_common_arg_val<uint32_t>(v4);
        auto tensor_accessor_args_32 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 10>(), 1>();
        TensorAccessor v13 = TensorAccessor(tensor_accessor_args_32, v12, v2);
        size_t v14 = v11 * v6;
        size_t v15 = v14 + v7;
        ptrdiff_t v16 = (ptrdiff_t)v15;
        int32_t v17 = (int32_t)v16;
        noc_async_write_tile(v17, v13, cb_ctarg_6.get_read_ptr());
        noc_async_write_barrier();
        cb_ctarg_6.pop_front(v3);
    }
    cb_ctarg_7.wait_front(v3);
    int32_t v18 = get_common_arg_val<uint32_t>(v5);
    auto tensor_accessor_args_24 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<7, 10>(), 0>();
    TensorAccessor v19 = TensorAccessor(tensor_accessor_args_24, v18, v2);
    size_t v20 = v9 + v7;
    ptrdiff_t v21 = (ptrdiff_t)v20;
    int32_t v22 = (int32_t)v21;
    noc_async_write_tile(v22, v19, cb_ctarg_7.get_read_ptr());
    noc_async_write_barrier();
    cb_ctarg_7.pop_front(v3);
    return;
}
