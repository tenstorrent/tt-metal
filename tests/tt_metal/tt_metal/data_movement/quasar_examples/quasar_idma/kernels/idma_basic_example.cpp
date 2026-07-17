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
#include "api/debug/device_print.h"
#include "experimental/kernel_args.h"
#include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include <cstdint>

using namespace overlay;

constexpr uint32_t num_elements = 16;
constexpr uint32_t elem_size = 8;
constexpr uint32_t total_bytes = num_elements * elem_size;  // 128

void kernel_main() {
    constexpr uint32_t src_addr = get_arg(args::src_addr);
    constexpr uint32_t dst_addr = get_arg(args::dst_addr);

    reset_cmdbuf_0();
    DEVICE_PRINT("HERE \n");

    /* CMD Misc register, only difference to NOC*/
    idma_setup_as_copy_cmdbuf_0(false);
    DEVICE_PRINT("HERE 2\n");
    /* Vcs = IDMA channel*/
    setup_vcs_cmdbuf_0(false);
    DEVICE_PRINT("HERE 3\n");

    set_src_cmdbuf_0(src_addr);
    DEVICE_PRINT("HERE 4\n");
    set_dest_cmdbuf_0(dst_addr);
    DEVICE_PRINT("HERE 5\n");
    set_len_cmdbuf_0(total_bytes);
    DEVICE_PRINT("HERE 6\n");

    issue_cmdbuf_0();
    DEVICE_PRINT("HERE 7\n");

    /* wait on IDMA to finish */
    while (!idma_acked_cmdbuf_0()) {
    }

    DEVICE_PRINT("IDMA 1D strided done: {} elements", num_elements);
}
