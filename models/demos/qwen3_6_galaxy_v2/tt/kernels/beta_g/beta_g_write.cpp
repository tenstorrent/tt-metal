// beta_g_write
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
    int32_t v1 = 0;
    int32_t v2 = 2048;
    int32_t v3 = 1;
    size_t v4 = 1;
    size_t v5 = 0;
    size_t v6 = 2;
    experimental::CircularBuffer cb_ctarg_5(get_compile_time_arg_val(5));
    experimental::CircularBuffer cb_ctarg_6(get_compile_time_arg_val(6));
    for (size_t i7 = v5; i7 < v6; i7 += v4) {
        for (size_t j8 = v5; j8 < v6; j8 += v4) {
            cb_ctarg_5.wait_front(v3);
            cb_ctarg_6.wait_front(v3);
            int32_t v9 = get_common_arg_val<uint32_t>(v5);
            auto tensor_accessor_args_18 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 7>(), 0>();
            TensorAccessor v10 = TensorAccessor(tensor_accessor_args_18, v9, v2);
            size_t v11 = i7 * v6;
            size_t v12 = v11 + j8;
            ptrdiff_t v13 = (ptrdiff_t)v12;
            int32_t v14 = (int32_t)v13;
            noc_async_write_tile(v14, v10, cb_ctarg_5.get_read_ptr());
            int32_t v15 = get_common_arg_val<uint32_t>(v4);
            auto tensor_accessor_args_33 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<6, 7>(), 1>();
            TensorAccessor v16 = TensorAccessor(tensor_accessor_args_33, v15, v2);
            noc_async_write_tile(v14, v16, cb_ctarg_6.get_read_ptr());
            noc_async_write_barrier();
            cb_ctarg_6.pop_front(v3);
            cb_ctarg_5.pop_front(v3);
        }
    }
    return;
}
