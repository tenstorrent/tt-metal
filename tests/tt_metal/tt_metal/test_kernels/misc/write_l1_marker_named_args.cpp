// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Writes one runtime-arg word at a runtime-arg L1 address: the smallest observable kernel.
// Metal 2.0 version of write_l1_marker.cpp (named runtime args "address" and "value"); runs on WH/BH and Quasar.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#if defined(ARCH_QUASAR)
#include "api/core_local_mem.h"
#include "risc_common.h"
#endif

void kernel_main() {
    const uintptr_t addr = get_arg(args::address);
    const uint32_t value = get_arg(args::value);
#if defined(ARCH_QUASAR)
    // Quasar L1 is cacheable: write through CoreLocalMem, then flush the cache line so the host sees the value.
    CoreLocalMem<uint32_t> buffer(addr);
    buffer[0] = value;
    flush_l2_cache_line(addr);
#else
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr) = value;
#endif
}
