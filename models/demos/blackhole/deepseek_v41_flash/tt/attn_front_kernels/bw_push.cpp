// pushes the (uninitialised) in0 / weight tiles for bw_compute.cpp. CT: 0 cb_in0, 1 cb_w, 2 ct
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t ct = get_compile_time_arg_val(2);
    cb_reserve_back(cb_in0, 80);
    cb_push_back(cb_in0, 80);
    cb_reserve_back(cb_w, 80 * ct);
    cb_push_back(cb_w, 80 * ct);
}
