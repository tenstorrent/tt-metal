// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#if defined(ARCH_QUASAR)
#include "risc_common.h"
#endif

void kernel_main() {
    constexpr uint32_t l1_address = get_compile_time_arg_val(0);
    constexpr uint32_t value = get_compile_time_arg_val(1);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<tt_l1_ptr uint32_t*>(l1_address);
    result[0] = value;
#if defined(ARCH_QUASAR)
    // The host reads L1 memory, which a cached store reaches only after an L2 flush. The flush probes L1 D$ for the
    // dirty line, so no separate L1 flush is needed.
    flush_l2_cache_range(l1_address, sizeof(uint32_t));
#endif
}
