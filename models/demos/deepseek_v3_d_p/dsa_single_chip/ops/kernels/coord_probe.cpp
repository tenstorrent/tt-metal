// Probe: every core writes [noc_x, noc_y, logical_x, logical_y] (noc coords on the NOC this RISC uses) to its page.
// compile-time: [GRID_Y] + TensorAccessorArgs(out) (page = 32 bytes)   common runtime: [out_addr]   cb0 = 32 bytes
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "split_work.h"

void kernel_main() {
    constexpr uint32_t GRID_Y = get_compile_time_arg_val(0);
    constexpr auto out_args = TensorAccessorArgs<1>();
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const auto s = TensorAccessor(out_args, out_addr, 32);
    const uint32_t l1 = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    p[0] = my_x[noc_index];
    p[1] = my_y[noc_index];
    p[2] = my_logical_x_;
    p[3] = my_logical_y_;
    const uint32_t idx = my_logical_x_ * GRID_Y + my_logical_y_;
    noc_async_write(l1, s.get_noc_addr(idx), 32);
    noc_async_write_barrier();
}
