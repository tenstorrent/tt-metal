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
    size_t v15 = get_absolute_logical_y();
    cb_ctarg_4.reserve_back(v10);
    cb_ctarg_5.reserve_back(v10);
    int32_t v16 = get_common_arg_val<uint32_t>(v11);
    auto tensor_accessor_args_31 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 10>(), 1>();
    TensorAccessor v17 = TensorAccessor(tensor_accessor_args_31, v16, v9);
    ptrdiff_t v18 = (ptrdiff_t)v15;
    int32_t v19 = (int32_t)v18;
    noc_async_read_tile(v19, v17, cb_ctarg_4.get_write_ptr());
    int32_t v20 = get_common_arg_val<uint32_t>(v12);
    auto tensor_accessor_args_42 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<5, 10>(), 0>();
    TensorAccessor v21 = TensorAccessor(tensor_accessor_args_42, v20, v9);
    noc_async_read_tile(v19, v21, cb_ctarg_5.get_write_ptr());
    noc_async_read_barrier();
    cb_ctarg_5.push_back(v10);
    cb_ctarg_4.push_back(v10);
    size_t v22 = v15 * v13;
    for (size_t i23 = v12; i23 < v13; i23 += v11) {
        cb_ctarg_0.reserve_back(v10);
        cb_ctarg_2.reserve_back(v10);
        cb_ctarg_3.reserve_back(v10);
        cb_ctarg_1.reserve_back(v10);
        size_t v24 = v22 + i23;
        int32_t v25 = get_common_arg_val<uint32_t>(v13);
        auto tensor_accessor_args_92 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 10>(), 4>();
        TensorAccessor v26 = TensorAccessor(tensor_accessor_args_92, v25, v9);
        size_t v27 = v24 * v13;
        size_t v28 = v27 + v14;
        ptrdiff_t v29 = (ptrdiff_t)v28;
        int32_t v30 = (int32_t)v29;
        noc_async_read_tile(v30, v26, cb_ctarg_0.get_write_ptr());
        int32_t v31 = get_common_arg_val<uint32_t>(v6);
        auto tensor_accessor_args_107 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 10>(), 2>();
        TensorAccessor v32 = TensorAccessor(tensor_accessor_args_107, v31, v9);
        ptrdiff_t v33 = (ptrdiff_t)v24;
        int32_t v34 = (int32_t)v33;
        noc_async_read_tile(v34, v32, cb_ctarg_2.get_write_ptr());
        int32_t v35 = get_common_arg_val<uint32_t>(v4);
        auto tensor_accessor_args_118 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 10>(), 5>();
        TensorAccessor v36 = TensorAccessor(tensor_accessor_args_118, v35, v9);
        size_t v37 = v22 + v14;
        ptrdiff_t v38 = (ptrdiff_t)v37;
        int32_t v39 = (int32_t)v38;
        noc_async_read_tile(v39, v36, cb_ctarg_3.get_write_ptr());
        int32_t v40 = get_common_arg_val<uint32_t>(v2);
        auto tensor_accessor_args_131 =
            TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 10>(), 3>();
        TensorAccessor v41 = TensorAccessor(tensor_accessor_args_131, v40, v9);
        noc_async_read_tile(v34, v41, cb_ctarg_1.get_write_ptr());
        noc_async_read_barrier();
        cb_ctarg_1.push_back(v10);
        cb_ctarg_3.push_back(v10);
        cb_ctarg_2.push_back(v10);
        cb_ctarg_0.push_back(v10);
    }
    return;
}
