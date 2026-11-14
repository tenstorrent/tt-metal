// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Quasar DM probe: records the DFB config base this hart's firmware walked (g_dfb_config_base_addr,
// a cached TL1 address) to L1 through the uncached alias, so the host can check its cache-line alignment.
// pad_nops grows the binary (4B per nop) so the host can shift where fast dispatch places the next config.

#include "api/dataflow/dataflow_buffer.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t pad_nops = get_arg(args::pad_nops);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);

    asm volatile(".rept %0\n\tnop\n\t.endr" : : "i"(pad_nops));

    DataflowBuffer dfb(dfb::out);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_addr + MEM_L1_UNCACHED_BASE) =
        static_cast<uint32_t>(g_dfb_config_base_addr);
    dfb.finish();
}
