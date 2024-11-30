// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "Kernel Compile Time Args" << ENDL();
    DPRINT << get_compile_time_arg_val(0) << ENDL();
    DPRINT << get_compile_time_arg_val(1) << ENDL();
    DPRINT << get_compile_time_arg_val(2) << ENDL();
    DPRINT << get_compile_time_arg_val(3) << ENDL();
}
