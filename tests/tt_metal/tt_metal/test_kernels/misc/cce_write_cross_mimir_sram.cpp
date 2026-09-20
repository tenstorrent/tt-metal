// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/cce_gddr.h"
#include "risc_common.h"

void kernel_main() {
    const uint32_t dst0 = get_arg_val<uint32_t>(0);
    const uint32_t dst1 = get_arg_val<uint32_t>(1);
    const uint32_t offset = get_arg_val<uint32_t>(2);
    const uint32_t magic_base = get_arg_val<uint32_t>(3);

    experimental::cce_sram_write_uint32(dst0, offset, magic_base | dst0);
    experimental::cce_sram_write_uint32(dst1, offset, magic_base | dst1);
}
