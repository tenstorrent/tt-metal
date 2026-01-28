// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

namespace {
void kernel_main() {
    volatile uint32_t tt_l1_ptr* results = (volatile uint32_t tt_l1_ptr*)RESULTS_ADDR;

#ifdef TEST_DEFINES
    results[0] = DEFINES_0;
    results[1] = DEFINES_1;
    results[2] = DEFINES_2;
#elif defined(TEST_COMPILE_ARGS)
    results[0] = get_compile_time_arg_val(0);
    results[1] = get_compile_time_arg_val(1);
    results[2] = get_compile_time_arg_val(2);
    results[3] = get_compile_time_arg_val(3);
    results[4] = get_compile_time_arg_val(4);

#elif defined(TEST_NAMED_COMPILE_ARGS)
    results[0] = get_named_compile_time_arg_val("NAMED_COMPILE_ARGS_0");
    results[1] = get_named_compile_time_arg_val("NAMED_COMPILE_ARGS_1");
    results[2] = get_named_compile_time_arg_val("NAMED_COMPILE_ARGS_2");
    results[3] = get_named_compile_time_arg_val("NAMED_COMPILE_ARGS_3");
    results[4] = get_named_compile_time_arg_val("NAMED_COMPILE_ARGS_4");

#elif defined(TEST_RUNTIME_ARGS)
    uint32_t i = 0;
#ifdef NUM_COMMON_RUNTIME_ARGS
    for (i = 0; i < NUM_COMMON_RUNTIME_ARGS; i++) {
        results[i] = get_common_arg_val<uint32_t>(i);
    }
#endif

#ifdef NUM_UNIQUE_RUNTIME_ARGS
    for (; i < NUM_UNIQUE_RUNTIME_ARGS; i++) {
        results[i] = get_arg_val<uint32_t>(i);
    }
#endif

#endif
}
}  // namespace
