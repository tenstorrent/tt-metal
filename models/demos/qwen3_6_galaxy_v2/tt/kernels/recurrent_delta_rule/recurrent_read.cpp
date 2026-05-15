// recurrent_read
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
    int32_t v1 = 4;
    int32_t v2 = 2;
    size_t v3 = 2;
    int32_t v4 = 3;
    size_t v5 = 3;
    int32_t v6 = 0;
    int32_t v7 = 4096;
    int32_t v8 = 1;
    size_t v9 = 1;
    size_t v10 = 0;
    size_t v11 = 4;
    experimental::CircularBuffer cb_ctarg_5(get_compile_time_arg_val(5));
    experimental::CircularBuffer cb_ctarg_4(get_compile_time_arg_val(4));
    experimental::CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
    experimental::CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
    experimental::CircularBuffer cb_ctarg_3(get_compile_time_arg_val(3));
    cb_ctarg_4.reserve_back(v8);
    cb_ctarg_5.reserve_back(v8);
    int32_t v12 = get_common_arg_val<uint32_t>(v9);
    auto tensor_accessor_args_23 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 9>(), 1>();
    TensorAccessor v13 = TensorAccessor(tensor_accessor_args_23, v12, v7);
    noc_async_read_tile(v6, v13, cb_ctarg_4.get_write_ptr());
    int32_t v14 = get_common_arg_val<uint32_t>(v10);
    auto tensor_accessor_args_31 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 9>(), 0>();
    TensorAccessor v15 = TensorAccessor(tensor_accessor_args_31, v14, v7);
    noc_async_read_tile(v6, v15, cb_ctarg_5.get_write_ptr());
    noc_async_read_barrier();
    cb_ctarg_5.push_back(v8);
    cb_ctarg_4.push_back(v8);
    for (size_t i16 = v10; i16 < v11; i16 += v9) {
        for (size_t j17 = v10; j17 < v11; j17 += v9) {
            cb_ctarg_0.reserve_back(v8);
            cb_ctarg_2.reserve_back(v8);
            cb_ctarg_3.reserve_back(v8);
            int32_t v18 = get_common_arg_val<uint32_t>(v5);
            auto tensor_accessor_args_70 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 9>(), 3>();
            TensorAccessor v19 = TensorAccessor(tensor_accessor_args_70, v18, v7);
            size_t v20 = j17 * v11;
            size_t v21 = v20 + i16;
            ptrdiff_t v22 = (ptrdiff_t)v21;
            int32_t v23 = (int32_t)v22;
            noc_async_read_tile(v23, v19, cb_ctarg_0.get_write_ptr());
            int32_t v24 = get_common_arg_val<uint32_t>(v3);
            auto tensor_accessor_args_85 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 9>(), 2>();
            TensorAccessor v25 = TensorAccessor(tensor_accessor_args_85, v24, v7);
            ptrdiff_t v26 = (ptrdiff_t)j17;
            int32_t v27 = (int32_t)v26;
            noc_async_read_tile(v27, v25, cb_ctarg_2.get_write_ptr());
            int32_t v28 = get_common_arg_val<uint32_t>(v11);
            auto tensor_accessor_args_96 =
                TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 9>(), 4>();
            TensorAccessor v29 = TensorAccessor(tensor_accessor_args_96, v28, v7);
            ptrdiff_t v30 = (ptrdiff_t)i16;
            int32_t v31 = (int32_t)v30;
            noc_async_read_tile(v31, v29, cb_ctarg_3.get_write_ptr());
            noc_async_read_barrier();
            cb_ctarg_3.push_back(v8);
            cb_ctarg_2.push_back(v8);
            cb_ctarg_0.push_back(v8);
        }
    }
    return;
}
