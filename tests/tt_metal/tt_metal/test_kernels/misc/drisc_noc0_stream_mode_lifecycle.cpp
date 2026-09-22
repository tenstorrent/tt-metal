// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "experimental/drisc_mode.h"

// Records six NOC-mode observations at status[0..5], transfers one aligned
// Tensix word into status[8], waits for the host release at status[9], and
// restores NOC0 before returning. All addresses are byte addresses.
void kernel_main() {
    constexpr uint32_t status_addr = get_compile_time_arg_val(0);
    constexpr uint32_t tensix_noc_x = get_compile_time_arg_val(1);
    constexpr uint32_t tensix_noc_y = get_compile_time_arg_val(2);
    constexpr uint32_t tensix_l1_addr = get_compile_time_arg_val(3);
    constexpr uint32_t release_value = get_compile_time_arg_val(4);

    volatile tt_l1_ptr uint32_t* const status =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(status_addr);
    status[0] = experimental::drisc_is_noc2axi_mode(0);
    status[1] = experimental::drisc_is_noc2axi_mode(1);

    experimental::drisc_set_stream_mode(0);
    status[2] = !experimental::drisc_is_noc2axi_mode(0);
    status[3] = experimental::drisc_is_noc2axi_mode(1);

    Noc noc;
    UnicastEndpoint source;
    CoreLocalMem<uint32_t> destination(status_addr + 8u * sizeof(uint32_t));
    noc.async_read(
        source,
        destination,
        sizeof(uint32_t),
        {.noc_x = tensix_noc_x, .noc_y = tensix_noc_y, .addr = tensix_l1_addr},
        {});
    noc.async_read_barrier();

    while (true) {
        invalidate_l1_cache();
        if (status[9] == release_value) {
            break;
        }
    }

    experimental::drisc_set_noc2axi_mode(0);
    status[4] = experimental::drisc_is_noc2axi_mode(0);
    status[5] = experimental::drisc_is_noc2axi_mode(1);
}
