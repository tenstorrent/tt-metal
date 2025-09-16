// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t buffer_size = get_named_compile_time_arg_val("buffer_size");
    constexpr uint32_t special_chars = get_named_compile_time_arg_val("!@#$%^&*()");
    constexpr uint32_t empty_string = get_named_compile_time_arg_val("");
    constexpr uint32_t huge_string = get_named_compile_time_arg_val(
        "very_long_parameter_name_that_someone_could_potentially_use_to_try_to_break_the_kernel");

    volatile uint32_t tt_l1_ptr* l1_ptr = (volatile uint32_t tt_l1_ptr*)WRITE_ADDRESS;

    l1_ptr[0] = special_chars;
    l1_ptr[1] = huge_string;
    l1_ptr[2] = buffer_size;
    l1_ptr[3] = empty_string;
}
