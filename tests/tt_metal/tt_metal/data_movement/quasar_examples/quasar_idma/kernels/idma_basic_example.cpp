// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Basic IDMA copy: transfers num_elements * elem_size bytes from src to dst
// in a single IDMA transaction using direct NOC address registers (no addrgen).
//
// Sequence:
//   1. Configure cmdbuf_0 for IDMA copy
//   2. Set src and dst addresses directly via set_src / set_dest
//   3. Set transfer length and issue one transaction
//   4. Wait for IDMA ack

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include <cstdint>

constexpr uint32_t num_elements = 16;
constexpr uint32_t elem_size = 8;
constexpr uint32_t total_bytes = num_elements * elem_size;  // 128

void kernel_main() {
    const uint32_t src_addr = get_compile_time_arg_val(0);
    const uint32_t dst_addr = get_compile_time_arg_val(1);

    reset_cmdbuf_0();

    /* CMD Misc register, only difference to NOC*/
    idma_setup_as_copy_cmdbuf_0(false);
    /* Vcs = IDMA channel*/
    setup_vcs_cmdbuf_0(false);

    set_src_cmdbuf_0(src_addr);
    set_dest_cmdbuf_0(dst_addr);
    set_len_cmdbuf_0(total_bytes);

    issue_cmdbuf_0();

    /* wait on IDMA to finish */
    while (!idma_acked_cmdbuf_0());

    DPRINT << "IDMA basic done: " << DEC() << num_elements << " elements (" << total_bytes << " B)" << ENDL();
}
