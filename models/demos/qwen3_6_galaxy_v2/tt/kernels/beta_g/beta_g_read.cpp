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
    size_t v8 = 2;
    int32_t v9 = 1;
    size_t v10 = 0;
    size_t v11 = 1;
    experimental::CircularBuffer cb_ctarg_1(get_compile_time_arg_val(1));
    experimental::CircularBuffer cb_ctarg_3(get_compile_time_arg_val(3));
    experimental::CircularBuffer cb_ctarg_0(get_compile_time_arg_val(0));
    experimental::CircularBuffer cb_ctarg_2(get_compile_time_arg_val(2));
    experimental::CircularBuffer cb_ctarg_4(get_compile_time_arg_val(4));
    cb_ctarg_0.reserve_back(v9);
    cb_ctarg_1.reserve_back(v9);
    cb_ctarg_2.reserve_back(v9);
    cb_ctarg_3.reserve_back(v9);
    cb_ctarg_4.reserve_back(v9);
    int32_t v12 = get_common_arg_val<uint32_t>(v8);
    auto tensor_accessor_args_23 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<0, 7>(), 2>();
    TensorAccessor v13 = TensorAccessor(tensor_accessor_args_23, v12, v7);
    noc_async_read_tile(v6, v13, cb_ctarg_0.get_write_ptr());
    int32_t v14 = get_common_arg_val<uint32_t>(v11);
    auto tensor_accessor_args_31 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<1, 7>(), 1>();
    TensorAccessor v15 = TensorAccessor(tensor_accessor_args_31, v14, v7);
    noc_async_read_tile(v6, v15, cb_ctarg_1.get_write_ptr());
    int32_t v16 = get_common_arg_val<uint32_t>(v4);
    auto tensor_accessor_args_39 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<2, 7>(), 3>();
    TensorAccessor v17 = TensorAccessor(tensor_accessor_args_39, v16, v7);
    noc_async_read_tile(v6, v17, cb_ctarg_2.get_write_ptr());
    int32_t v18 = get_common_arg_val<uint32_t>(v10);
    auto tensor_accessor_args_47 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<3, 7>(), 0>();
    TensorAccessor v19 = TensorAccessor(tensor_accessor_args_47, v18, v7);
    noc_async_read_tile(v6, v19, cb_ctarg_3.get_write_ptr());
    int32_t v20 = get_common_arg_val<uint32_t>(v2);
    auto tensor_accessor_args_55 =
        TensorAccessorArgs<tensor_accessor::detail::get_tensor_accessor_args_cta_offset<4, 7>(), 4>();
    TensorAccessor v21 = TensorAccessor(tensor_accessor_args_55, v20, v7);
    noc_async_read_tile(v6, v21, cb_ctarg_4.get_write_ptr());
    noc_async_read_barrier();
    cb_ctarg_4.push_back(v9);
    cb_ctarg_3.push_back(v9);
    cb_ctarg_2.push_back(v9);
    cb_ctarg_1.push_back(v9);
    cb_ctarg_0.push_back(v9);
    return;
}
