// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
    uint32_t table_address = get_arg_val<uint32_t>(0);
    uint32_t a = get_arg_val<uint32_t>(1);
    uint32_t b = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* table_address_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_address);

    *table_address_ptr = a + b;
}
