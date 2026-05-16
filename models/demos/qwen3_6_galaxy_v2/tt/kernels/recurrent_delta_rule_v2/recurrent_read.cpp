// recurrent_read
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"
void kernel_main() {
    int32_t v1 = 3;
    size_t v2 = 3;
    int32_t v3 = 5;
    size_t v4 = 5;
    int32_t v5 = 2;
    size_t v6 = 2;
    int32_t v7 = 4;
    int32_t v8 = 0;
    int32_t v9 = 4096;
    int32_t v10 = 1;
    size_t v11 = 1;
    size_t v12 = 0;
    size_t v13 = 4;
    experimental::CircularBuffer cb_ctarg_5(get_compile_time_arg_val(5));
    experimental::CircularBuffer cb_ctarg_4(get_compile_time_arg_val(4));
    experimental::CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
    experimental::CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
    experimental::CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
    experimental::CircularBuffer cb_ctarg_3(get_compile_time_arg_val(3));
    size_t v14 = get_absolute_logical_x();
    cb_ctarg_4.reserve_back(v10);
    cb_ctarg_5.reserve_back(v10);
    int32_t v15 = get_common_arg_val<uint32_t>(v11);
    auto tensor_accessor_args_29 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 10>(), 1>();
    TensorAccessor v16 = TensorAccessor(tensor_accessor_args_29, v15, v9);
    noc_async_read_tile(v8, v16, cb_ctarg_4.get_write_ptr());
    int32_t v17 = get_common_arg_val<uint32_t>(v12);
    auto tensor_accessor_args_37 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 10>(), 0>();
    TensorAccessor v18 = TensorAccessor(tensor_accessor_args_37, v17, v9);
    noc_async_read_tile(v8, v18, cb_ctarg_5.get_write_ptr());
    noc_async_read_barrier();
    cb_ctarg_5.push_back(v10);
    cb_ctarg_4.push_back(v10);
    for (size_t i19 = v12; i19 < v13; i19 += v11) {
        cb_ctarg_0.reserve_back(v10);
        cb_ctarg_2.reserve_back(v10);
        cb_ctarg_3.reserve_back(v10);
        cb_ctarg_1.reserve_back(v10);
        int32_t v20 = get_common_arg_val<uint32_t>(v13);
        auto tensor_accessor_args_79 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 10>(), 4>();
        TensorAccessor v21 = TensorAccessor(tensor_accessor_args_79, v20, v9);
        size_t v22 = i19 * v13;
        size_t v23 = v22 + v14;
        ptrdiff_t v24 = (ptrdiff_t)v23;
        int32_t v25 = (int32_t)v24;
        noc_async_read_tile(v25, v21, cb_ctarg_0.get_write_ptr());
        int32_t v26 = get_common_arg_val<uint32_t>(v6);
        auto tensor_accessor_args_94 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 10>(), 2>();
        TensorAccessor v27 = TensorAccessor(tensor_accessor_args_94, v26, v9);
        ptrdiff_t v28 = (ptrdiff_t)i19;
        int32_t v29 = (int32_t)v28;
        noc_async_read_tile(v29, v27, cb_ctarg_2.get_write_ptr());
        int32_t v30 = get_common_arg_val<uint32_t>(v4);
        auto tensor_accessor_args_105 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 10>(), 5>();
        TensorAccessor v31 = TensorAccessor(tensor_accessor_args_105, v30, v9);
        ptrdiff_t v32 = (ptrdiff_t)v14;
        int32_t v33 = (int32_t)v32;
        noc_async_read_tile(v33, v31, cb_ctarg_3.get_write_ptr());
        int32_t v34 = get_common_arg_val<uint32_t>(v2);
        auto tensor_accessor_args_116 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 10>(), 3>();
        TensorAccessor v35 = TensorAccessor(tensor_accessor_args_116, v34, v9);
        noc_async_read_tile(v29, v35, cb_ctarg_1.get_write_ptr());
        noc_async_read_barrier();
        cb_ctarg_1.push_back(v10);
        cb_ctarg_3.push_back(v10);
        cb_ctarg_2.push_back(v10);
        cb_ctarg_0.push_back(v10);
    }
    return;
}
