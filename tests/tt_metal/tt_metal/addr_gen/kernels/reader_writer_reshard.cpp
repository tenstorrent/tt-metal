// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    const uint32_t bank_base_address = get_arg_val<uint32_t>(0);

    constexpr uint32_t dim1 = get_compile_time_arg_val(0);
    constexpr uint32_t dim2 = get_compile_time_arg_val(1);
    constexpr uint32_t dim3 = get_compile_time_arg_val(2);

    DPRINT << "HERE" << ENDL();
    DPRINT << "bank_base_address: " << bank_base_address << ENDL();
    DPRINT << "dim1: " << dim1 << ENDL();
    DPRINT << "dim2: " << dim2 << ENDL();
    DPRINT << "dim3: " << dim3 << ENDL();

    constexpr uint32_t rank = 3;
    constexpr uint32_t num_banks = 4;
}
