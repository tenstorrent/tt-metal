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
    for (size_t i7 = v5; i7 < v6; i7 += v4) {
        for (size_t j8 = v5; j8 < v6; j8 += v4) {
            cb_ctarg_6.wait_front(v3);
            int32_t v9 = get_common_arg_val<uint32_t>(v4);
            auto tensor_accessor_args_23 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 9>(), 1>();
            TensorAccessor v10 = TensorAccessor(tensor_accessor_args_23, v9, v2);
            size_t v11 = j8 * v6;
            size_t v12 = v11 + i7;
            ptrdiff_t v13 = (ptrdiff_t)v12;
            int32_t v14 = (int32_t)v13;
            noc_async_write_tile(v14, v10, cb_ctarg_6.get_read_ptr());
            noc_async_write_barrier();
            cb_ctarg_6.pop_front(v3);
        }
    }
    for (size_t i15 = v5; i15 < v6; i15 += v4) {
        cb_ctarg_7.wait_front(v3);
        int32_t v16 = get_common_arg_val<uint32_t>(v5);
        auto tensor_accessor_args_18 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<7, 9>(), 0>();
        TensorAccessor v17 = TensorAccessor(tensor_accessor_args_18, v16, v2);
        ptrdiff_t v18 = (ptrdiff_t)i15;
        int32_t v19 = (int32_t)v18;
        noc_async_write_tile(v19, v17, cb_ctarg_7.get_read_ptr());
        noc_async_write_barrier();
        cb_ctarg_7.pop_front(v3);
    }
    return;
}
