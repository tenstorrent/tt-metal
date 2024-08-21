// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    std::uint32_t num_bytes = get_arg_val<uint32_t>(0);

    eth_wait_for_bytes(num_bytes);
    eth_receiver_done();
}
