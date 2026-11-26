// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// attn_out_writer.cpp for the T tokens of a verify pass (attn_out_rows_reader.cpp): token t's dense row -> row
// (runtime arg 1 + t) of this core's own L1 shard of the o-projection input.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t nh = get_named_compile_time_arg_val("nh");
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t cb_out = 16;
    const uint32_t shard = get_arg_val<uint32_t>(0);
    for (uint32_t t = 0; t < tokens; ++t) {
        const uint32_t roff = get_arg_val<uint32_t>(1 + t) * 32;  // face row offset in bytes
        cb_wait_front(cb_out, 1);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t j = 0; j < nh * 4; ++j) {
            noc_async_write(src + j * 64, get_noc_addr(shard + j * 2048 + roff), 32);
            noc_async_write(src + j * 64 + 32, get_noc_addr(shard + j * 2048 + 512 + roff), 32);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
