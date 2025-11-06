// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "risc_common.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t value = get_common_arg_val<uint32_t>(0);
    uint32_t buffer_size = get_named_ct_arg("buffer_size");
    DPRINT << "buffer_size: " << buffer_size << ENDL();
    *((uint32_t*)((uint64_t)dst_addr)) = value;  // use cache write-around for now, in the future use cache flush
    flush_l2_cache(dst_addr);
}
