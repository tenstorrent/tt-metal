// beta_g_read
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
    int32_t v1 = 4;
    size_t v2 = 4;
    int32_t v3 = 3;
    size_t v4 = 3;
    int32_t v5 = 2;
    int32_t v6 = 0;
    int32_t v7 = 2048;
    int32_t v8 = 1;
    size_t v9 = 1;
    size_t v10 = 0;
    size_t v11 = 2;
    experimental::CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
    experimental::CircularBuffer cb_ctarg_3(get_compile_time_arg_val(3));
    experimental::CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
    experimental::CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
    experimental::CircularBuffer cb_ctarg_4(get_compile_time_arg_val(4));
    for (size_t i12 = v10; i12 < v11; i12 += v9) {
        for (size_t j13 = v10; j13 < v11; j13 += v9) {
            cb_ctarg_0.reserve_back(v8);
            cb_ctarg_1.reserve_back(v8);
            cb_ctarg_2.reserve_back(v8);
            cb_ctarg_3.reserve_back(v8);
            cb_ctarg_4.reserve_back(v8);
            int32_t v14 = get_common_arg_val<uint32_t>(v11);
            auto tensor_accessor_args_53 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 7>(), 2>();
            TensorAccessor v15 = TensorAccessor(tensor_accessor_args_53, v14, v7);
            size_t v16 = i12 * v11;
            size_t v17 = v16 + j13;
            ptrdiff_t v18 = (ptrdiff_t)v17;
            int32_t v19 = (int32_t)v18;
            noc_async_read_tile(v19, v15, cb_ctarg_0.get_write_ptr());
            int32_t v20 = get_common_arg_val<uint32_t>(v9);
            auto tensor_accessor_args_68 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 7>(), 1>();
            TensorAccessor v21 = TensorAccessor(tensor_accessor_args_68, v20, v7);
            noc_async_read_tile(v19, v21, cb_ctarg_1.get_write_ptr());
            int32_t v22 = get_common_arg_val<uint32_t>(v4);
            auto tensor_accessor_args_76 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 7>(), 3>();
            TensorAccessor v23 = TensorAccessor(tensor_accessor_args_76, v22, v7);
            noc_async_read_tile(v19, v23, cb_ctarg_2.get_write_ptr());
            int32_t v24 = get_common_arg_val<uint32_t>(v10);
            auto tensor_accessor_args_84 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 7>(), 0>();
            TensorAccessor v25 = TensorAccessor(tensor_accessor_args_84, v24, v7);
            noc_async_read_tile(v19, v25, cb_ctarg_3.get_write_ptr());
            int32_t v26 = get_common_arg_val<uint32_t>(v2);
            auto tensor_accessor_args_92 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 7>(), 4>();
            TensorAccessor v27 = TensorAccessor(tensor_accessor_args_92, v26, v7);
            noc_async_read_tile(v19, v27, cb_ctarg_4.get_write_ptr());
            noc_async_read_barrier();
            cb_ctarg_4.push_back(v8);
            cb_ctarg_3.push_back(v8);
            cb_ctarg_2.push_back(v8);
            cb_ctarg_1.push_back(v8);
            cb_ctarg_0.push_back(v8);
        }
    }
    return;
}
